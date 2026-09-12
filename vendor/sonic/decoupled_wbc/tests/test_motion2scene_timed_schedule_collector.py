"""Preparation and multi-beam measurement checks, without a simulator."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from decoupled_wbc.tests.test_motion2scene_course import beams, payload
from decoupled_wbc.tests.test_motion2scene_timed_history_collector import inputs

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/research"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "schedule_collector", SCRIPTS / "motion2scene_collect_timed_schedules.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def args_fixture(tmp_path, count=2):
    args = inputs(tmp_path)
    # The actual shell resolves its file arguments without invoking this interpreter.
    template = json.loads(args.template.read_text())
    template["implementation"]["python"] = sys.executable
    args.template.write_text(json.dumps(template))
    definition = json.loads(args.scene_definition.read_text())
    definition.update(
        schema=collector.SCENE_SCHEMA, beams=beams()[:count], beam_collision_enabled=[True] * count
    )
    definition.pop("beam")
    args.scene_definition.write_text(json.dumps(definition))
    args.preferred_option_id = "neutral"
    args.preferred_reference_id = "sustained"
    return args


def test_preparation_binds_all_beam_counterparts_and_fixed_history(tmp_path):
    args = args_fixture(tmp_path)
    collector.prepare(args)
    manifest, bank, scene = collector.verify_manifest(args.out)
    assert len(manifest["cells"]) == len(bank.option_ids) == 3
    assert manifest["phase_ticks"] == [15]
    assert manifest["sensor"]["history_seconds"] == 2
    assert manifest["sensor"]["history_frames"] == 101
    assert manifest["expected_recorded_control_steps"] == 298
    assert manifest["expected_physics_steps"] == 1192
    assert len(manifest["environment_beam_paths"]) == 2
    for cell in manifest["cells"]:
        overrides = cell["hydra_overrides"]
        assert (
            '++manager_env.config.environment_beam_paths=["/World/ground/terrain/CounterfactualBeam",'
            '"/World/ground/terrain/CounterfactualBeam_01"]' in overrides
        )
        assert cell["command"][cell["command"].index("--max-steps") + 1] == "299"
        assert not Path(cell["output"]).exists()
    assert scene["beams"][1]["center_xy_m"] == [0.8, 0]
    args.request.write_text("modified request")
    with pytest.raises(ValueError, match="hash mismatch"):
        collector.verify_manifest(args.out)


def test_bad_constant_and_unbound_evaluation_fail_before_registration(tmp_path):
    args = args_fixture(tmp_path)
    args.policy_mode, args.preferred_option_id = "constant_option", "unsupported_late"
    with pytest.raises(ValueError, match="qualified option ID"):
        collector.prepare(args)
    assert not args.out.exists()
    args.policy_mode = "always_walk"
    definition = json.loads(args.scene_definition.read_text())
    definition["split"] = "reserved_evaluation_v3"
    args.scene_definition.write_text(json.dumps(definition))
    with pytest.raises(ValueError, match="artifact path"):
        collector.prepare(args)
    assert not args.out.exists()


def shapes_for(definitions):
    shapes = []
    for i, beam in enumerate(definitions):
        matrix = np.eye(4)
        matrix[:3, :3] = np.diag([beam["length_m"], beam["width_m"], beam["thickness_m"]])
        matrix[3, :3] = [*beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2]
        shapes.append(
            dict(
                path=f"/World/ground/terrain/{collector.beam_prim_name(i)}",
                type="Cube",
                collision=True,
                rigid_body=True,
                attributes={"size": "1", "physics:collisionEnabled": "True"},
                local_to_world_at_capture_start=matrix.tolist(),
            )
        )
    return shapes


def test_second_native_beam_transform_and_enabled_attribute_are_checked():
    scene = dict(beams=beams(), beam_collision_enabled=[True, True])
    shapes = shapes_for(scene["beams"])
    assert len(collector.audit_scene_geometry(shapes, scene)) == 2
    bad = copy.deepcopy(shapes)
    bad[1]["local_to_world_at_capture_start"][0][0] *= 2
    with pytest.raises(ValueError, match="dimensions"):
        collector.audit_scene_geometry(bad, scene)
    bad = copy.deepcopy(shapes)
    bad[1]["attributes"]["physics:collisionEnabled"] = "False"
    with pytest.raises(ValueError, match="collision setting"):
        collector.audit_scene_geometry(bad, scene)
    with pytest.raises(ValueError, match="native beam set"):
        collector.audit_scene_geometry(
            shapes, dict(beams=beams()[:1], beam_collision_enabled=[True])
        )


def test_second_beam_unreached_and_second_beam_contact_remain_failures():
    scene = dict(beams=beams(), beam_collision_enabled=[True, True])
    force = np.zeros((101, 2, 1, 3))
    bank = SimpleNamespace(frame_count=102)
    assert collector.score_scene(payload(), force, scene, bank, commands_valid=True)["pass"]
    force[50, 1, 0, 0] = 2
    result = collector.score_scene(payload(), force, scene, bank, commands_valid=True)
    assert not result["pass"] and result["max_force_by_beam_n"] == [0, 2]
    force[:] = 0
    scene["beams"][1]["center_xy_m"][0] = 3
    result = collector.score_scene(payload(), force, scene, bank, commands_valid=True)
    assert not result["pass"] and result["completed_beams"] == 1
    assert "safe_captured_reference_horizon_before_course_completion" in result["failure_reasons"]
    json.dumps(result)


def test_forced_smoke_subset_is_explicit_and_never_fills_missing_actions(tmp_path):
    args = args_fixture(tmp_path, count=1)
    args.forced_option_ids = ["neutral"]
    collector.prepare(args)
    manifest, bank, _ = collector.verify_manifest(args.out)
    assert len(manifest["option_ids"]) == len(bank.option_ids) == 3
    assert [cell["forced_option_id"] for cell in manifest["cells"]] == ["neutral"]
    assert not Path(manifest["cells"][0]["output"]).exists()


def test_script_settings_are_explicit_and_bound_through_actual_command(tmp_path):
    from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_script import (
        DEFAULT_CONFIG,
    )

    args = args_fixture(tmp_path, count=1)
    args.policy_mode = "scripted_multi"
    with pytest.raises(ValueError, match="immutable script"):
        collector.prepare(args)
    assert not args.out.exists()
    args.script_parameters = tmp_path / "settings.json"
    args.script_parameters.write_text(json.dumps(DEFAULT_CONFIG))
    args.policy_id = "development_script_defaults"
    collector.prepare(args)
    manifest, bank, scene = collector.verify_manifest(args.out)
    cell = manifest["cells"][0]
    assert cell["policy_id"] == args.policy_id
    assert manifest["script_parameters"] == collector.artifact(args.script_parameters)
    assert len(manifest["runtime_artifacts"]) > 1
    assert len(manifest["scoring_artifacts"]) > 1
    bad = copy.deepcopy(cell)
    bad["hydra_overrides"] = [
        item.replace("++seed=8731", "++seed=8732") for item in bad["hydra_overrides"]
    ]
    # Even consistent --extra text cannot alter the separately registered seed.
    bad["command"][bad["command"].index("--extra") + 1] = " ".join(bad["hydra_overrides"])
    # Fixture seed may differ; replace its effective seed deterministically.
    for i, item in enumerate(bad["hydra_overrides"]):
        if item.lstrip("+").startswith("seed="):
            bad["hydra_overrides"][i] = "++seed=999999"
    bad["command"][bad["command"].index("--extra") + 1] = " ".join(bad["hydra_overrides"])
    with pytest.raises(ValueError, match="effective policy/seed"):
        collector.validate_collection_context(bad, manifest, bank, scene)
    args.script_parameters.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        collector.verify_manifest(args.out)


def test_reserved_proposal_is_rejected_before_creating_registration(tmp_path):
    args = args_fixture(tmp_path, count=1)
    protocol = tmp_path / "proposed.json"
    protocol.write_text(
        json.dumps({"schema": "motion2scene_reserved_traversal_protocol_v3", "status": "PROPOSED"})
    )
    scene = json.loads(args.scene_definition.read_text())
    scene.update(split="reserved_evaluation_v3", evaluation_protocol=collector.artifact(protocol))
    args.scene_definition.write_text(json.dumps(scene))
    with pytest.raises(ValueError, match="cannot enable evaluation"):
        collector.prepare(args)
    assert not args.out.exists()


def test_protocol_context_is_derived_from_actual_bound_command(tmp_path, monkeypatch):
    args = args_fixture(tmp_path, count=1)
    args.policy_mode = "always_walk"
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    scene = json.loads(args.scene_definition.read_text())
    scene.update(split="reserved_evaluation_v3", evaluation_protocol=collector.artifact(protocol))
    args.scene_definition.write_text(json.dumps(scene))
    contexts = []
    monkeypatch.setattr(
        collector, "validate_reserved_execution", lambda scene, ctx: contexts.append(ctx)
    )
    collector.prepare(args)
    manifest, bank, scene = collector.verify_manifest(args.out)
    assert len(contexts) == 2  # Independent prepare and verification hooks.
    assert contexts[0]["physics_seed"] == args.seed
    assert contexts[0]["registry"] == manifest["registry"]
    assert contexts[0]["runtime_artifacts"] == manifest["runtime_artifacts"]
    assert contexts[0]["feature_names"] == manifest["feature_names"]
    assert contexts[0]["controller"] == bank.request["controller"]
    assert contexts[0]["model"] is None


def test_missing_later_phase_and_raw_abort_preserve_verified_failure(tmp_path):
    import pickle

    from decoupled_wbc.tests.test_motion2scene_timed_outcome import fixture

    args = args_fixture(tmp_path, count=1)
    collector.prepare(args)
    manifest, bank, scene = collector.verify_manifest(args.out)
    cell = manifest["cells"][0]
    folder = Path(cell["output"]) / "trajectories"
    folder.mkdir(parents=True)
    data, observations, _ = fixture(30)
    data["root_pos_w"][20:, 2] = 0.4
    for i in range(20, 30):
        observations[i]["root_pos_w"][2] = 0.4
    raw = {key: value for key, value in data.items() if key != "fps"}
    with (folder / "aborted_raw_recording.pkl").open("wb") as handle:
        pickle.dump({0: raw}, handle)
    (folder / "aborted_interface.json").write_text(
        json.dumps({"observations": observations, "switches": []})
    )
    np.savez(folder / "aborted_physics.npz", physics_steps=np.arange(1, 121))
    row = collector.analyze_incomplete(cell, manifest, bank, scene, {"exit_status": 1}, "aborted")
    assert row["outcome"]["task_outcome"] == "failure"
    assert row["task_outcome_admitted"] and not row["measurement_admitted"]
    assert row["physics_steps"] == 120
    assert row["outcome"]["assigned_slots"] == 1
    # Phase50 is absent; no privileged scene or default packet fills it.
    visibility = collector.phase_visibility(
        observations, {"packet_eligible": [True] * 30}, scene, [50]
    )
    assert visibility[0]["available"] is False and visibility[0]["beams"] is None
    json.dumps(row)
