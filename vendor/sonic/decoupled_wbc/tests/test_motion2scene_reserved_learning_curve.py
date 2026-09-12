"""Expanded reserved allocation and charged execution; synthetic physics only."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))
import motion2scene_reserved_learning_curve as runner  # noqa: E402
from test_motion2scene_development_learning_curve import declaration_fixture, save  # noqa: E402
from test_motion2scene_primary_controller import FakeBackend, fixture  # noqa: E402


def build_proposal(tmp_path, monkeypatch):
    curve_root, plan, options = declaration_fixture(tmp_path, monkeypatch)
    curve = json.loads((curve_root / "study.json").read_text())
    study_path = Path(curve["anchor"]["path"])
    study = json.loads(study_path.read_text())
    # Retain actual reserved geometry/seed validation; fake only development
    # model paths and bank loading. No scene is prepared and no physics runs.
    previous = json.loads(
        (ROOT / "docs/motion2scene/TRAVERSAL_PROTOCOL_AMENDMENT_V6_PROPOSED.json").read_text()
    )
    options = previous["schedules"]["option_ids"]
    scenes = {
        r["scene_id"]: {k: r[k] for k in ("scene_id", "scene_definition")}
        for r in study["assignments"]
    }
    study["assignments"] = runner.development.anchor.assignments(
        plan["runs"], list(scenes.values()), options
    )
    monkeypatch.setattr(
        runner.development,
        "load_verified_registry",
        lambda *args: SimpleNamespace(option_ids=options),
    )
    previous["implementation"]["registry"] = study["registry"]
    study["request"] = previous["implementation"]["request"]
    study["script"] = save(tmp_path / "script.json", {"fixture": True})
    curve["anchor"] = save(study_path, study)
    curve["assignments"] = study["assignments"]
    save(curve_root / "study.json", curve)
    source = save(tmp_path / "previous.json", previous)
    out = tmp_path / "reserved"
    runner.declare(Path(source["path"]), curve_root, out)
    protocol = json.loads((out / "proposal.json").read_text())
    return out, protocol, plan, options


@pytest.fixture
def proposal(tmp_path, monkeypatch):
    return build_proposal(tmp_path, monkeypatch)


def test_complete1908_pairing_derives_from_current_inventory_and_original_geometry(
    proposal, tmp_path
):
    out, protocol, plan, _ = proposal
    rows = runner.assignments(protocol, tmp_path / "execution")
    assert (
        len(rows)
        == len({(r["policy_id"], r["layout_id"], r["physics_seed"]) for r in rows})
        == 1908
    )
    assert len({r["policy_id"] for r in rows}) == 53
    assert len({r["layout_id"] for r in rows}) == 18
    assert {r["physics_seed"] for r in rows} == {94301, 94302}
    assert {r["variant_id"] for r in rows} == {"nominal"}
    assert len(protocol["scenes"]) == 162
    assert protocol["execution_contract"]["maximum_physics_steps"] == 2274336
    assert len(runner.design_models(plan)) == 45
    assert {p["acquisition_assigned_maximum_steps"] for p in runner.design_models(plan)} == {
        84632,
        160920,
        313496,
    }
    assert not (out / "execution").exists()
    protocol["policies"].reverse()
    protocol["scenes"].reverse()
    assert runner.assignments(protocol, tmp_path / "execution") == rows


@pytest.mark.parametrize(
    "change", ["missing_policy", "early_checkpoint", "script", "stress", "geometry", "scoring"]
)
def test_scope_and_information_contract_cannot_silently_change(proposal, change):
    out, protocol, _, _ = proposal
    if change == "missing_policy":
        protocol["policies"].pop()
    elif change == "early_checkpoint":
        next(p for p in protocol["policies"] if p["mode"] == "learned")["checkpoint"] = 4
    elif change == "script":
        protocol["policies"][0]["script_parameters"] = {"path": "different", "sha256": "different"}
    elif change == "stress":
        protocol["enabled_variant_ids"].append("yaw_rad_plus")
    elif change == "geometry":
        protocol["scenes"][0]["beams"][0]["underside_m"] += 0.001
    else:
        protocol["scoring"]["environment_normal_force_threshold_n"] = 2.0
    save(out / "proposal.json", protocol)
    with pytest.raises(ValueError):
        runner.read_proposal(out / "proposal.json")


def test_incomplete_development_cannot_freeze_or_prepare_reserved_outputs(proposal):
    out, _, _, _ = proposal
    with pytest.raises(FileNotFoundError, match="M32"):
        runner.freeze_inventory(out / "proposal.json", out / "absent_stats.json", out / "freeze")
    assert not (out / "freeze").exists()
    with pytest.raises(ValueError, match="proposal cannot"):
        runner.prepare(out / "proposal.json", out / "execution")
    assert not (out / "execution").exists()
    with pytest.raises(FileNotFoundError, match="M32"):
        runner.adopt(out / "proposal.json", None, None, None, out / "adoption")
    assert not (out / "adoption").exists()


def controller_fixture(proposal, tmp_path, monkeypatch):
    _, protocol, _, _ = proposal
    context = fixture(tmp_path / "primary")
    output = tmp_path / "synthetic_evaluation"
    table = runner.assignments(protocol, output)
    source = context["learner"]["training_implementation"]
    execution = dict(
        shared_lock_path=str(context["execution_root"] / ".primary-acquisition.lock"),
        environment=context["runtime_ref"],
        runtime_assets_declaration=context["runtime_ref"],
        implementation=source,
        batch_implementation=source,
    )
    policies = {p["policy_id"]: p for p in protocol["policies"]}
    definitions = {}
    for row in table:
        if row["scene_id"] not in definitions:
            definitions[row["scene_id"]] = save(
                output / "definitions" / (row["scene_id"] + ".json"),
                {k: row[k] for k in ("layout_id", "scene_id", "variant_id")},
            )
    children = []
    for row in table:
        p = policies[row["policy_id"]]
        folder = Path(row["collection_directory"])
        cell = dict(
            cell_id="fixture_cell",
            output=str(folder / "rollouts/fixture_cell"),
            policy_id=row["policy_id"],
            runtime_seed=row["physics_seed"],
            timed_schedule_mode=p["mode"],
            forced_option_id=p.get("forced_option_id", "neutral"),
        )
        cell["command"] = ["fixture_only", cell["output"], "student"]
        manifest = dict(
            cells=[cell],
            split="reserved_evaluation_v3",
            scene_definition=definitions[row["scene_id"]],
            policy=p["model"],
            script_parameters=p.get("script_parameters"),
            expected_physics_steps=1192,
            limits=dict(minimum_free_gpu_mib=7500, timeout_s=375),
        )
        children.append(dict(index=row["index"], manifest=save(folder / "manifest.json", manifest)))
    spec = save(output / "spec.json", dict(collections=[dict(index=i) for i in range(len(table))]))
    batch_plan = save(
        output / "batch_plan.json",
        dict(
            specification=spec,
            expected_episodes=len(table),
            planned_maximum_physics_steps=len(table) * 1192,
            budget_physics_steps=len(table) * 1192,
            implementation=source,
        ),
    )
    registration = save(
        output / "batch/registration.json", dict(plan=batch_plan, children=children)
    )
    protocol_ref = save(output / "protocol.json", protocol)
    save(
        output / "registration.json",
        dict(
            protocol=protocol_ref,
            assignments=save(output / "assignments.json", table),
            batch_registration=registration,
            execution_contract=protocol["execution_contract"],
        ),
    )

    def verify(folder):
        manifest = runner.read_bound(runner.artifact(folder / "manifest.json"))
        return manifest, None, runner.read_bound(manifest["scene_definition"])

    monkeypatch.setattr(runner.collector, "verify_manifest", verify)
    return dict(
        protocol=protocol,
        protocol_ref=protocol_ref,
        execution=execution,
        learner=context["learner"],
        output=output,
        table=table,
        inventory=dict(
            models={
                p["policy_id"]: dict(
                    acquisition_cost=dict(total_recorded_steps=(7 + 8 * p["checkpoint"]) * 1192)
                )
                for p in protocol["policies"]
                if p["mode"] == "learned"
            }
        ),
    )


def test_serial_pause_resume_and_costs_keep_the_full1908_assignment(
    proposal, tmp_path, monkeypatch
):
    evaluation = controller_fixture(proposal, tmp_path, monkeypatch)
    backend = FakeBackend()
    first = runner.EvaluationController(evaluation, backend).run(1)
    assert first["status"] == "paused" and len(backend.launches) == 1
    assert first["report"]["assigned_episodes"] == 1908
    assert first["ledger"]["reserved_future_assigned_steps"] == 1907 * 1192
    second = runner.EvaluationController(evaluation, backend).run(1)
    assert second["status"] == "paused" and len(backend.launches) == 2
    assert second["ledger"]["actual_recorded_physics_steps"] == 2384
    assert sum(r["status"] == "not_run" for r in second["report"]["rows"]) == 1906
    learned = next(r for r in second["report"]["rows"] if r["policy_mode"] == "learned")
    assert learned["acquisition_actual_physics_steps"] == (7 + 8 * learned["checkpoint"]) * 1192
    fixed = next(r for r in second["report"]["rows"] if r["policy_mode"] == "forced")
    assert fixed["acquisition_actual_physics_steps"] is None


def test_interrupted_attempt_is_charged_and_never_retried(proposal, tmp_path, monkeypatch):
    evaluation = controller_fixture(proposal, tmp_path, monkeypatch)
    backend = FakeBackend()
    backend.crash_on_launch = True
    first = runner.EvaluationController(evaluation, backend).run()
    assert first["status"] == "paused" and len(backend.launches) == 1
    assert first["ledger"]["unknown_attempt_reserved_steps"] == 1192
    backend.crash_on_launch = False
    second = runner.EvaluationController(evaluation, backend).run()
    assert second["status"] == "paused" and len(backend.launches) == 1
    assert second["report"]["rows"][0]["status"] == "technical_missing"


def test_wrong_fixed_schedule_is_detected_before_first_launch(proposal, tmp_path, monkeypatch):
    evaluation = controller_fixture(proposal, tmp_path, monkeypatch)
    row = next(r for r in evaluation["table"] if r["policy_id"].startswith("fixed_"))
    path = Path(row["collection_directory"]) / "manifest.json"
    manifest = json.loads(path.read_text())
    changed = copy.deepcopy(manifest)
    changed["cells"][0]["forced_option_id"] = "some_other_schedule"
    with pytest.raises(ValueError, match="fixed schedule"):
        runner.check_assignment(row, changed, evaluation["protocol"])
    # Even the final child is verified before dispatching the first one.
    last = Path(evaluation["table"][-1]["collection_directory"]) / "manifest.json"
    last.write_text("changed")
    backend = FakeBackend()
    with pytest.raises(ValueError):
        runner.EvaluationController(evaluation, backend).run()
    assert not backend.launches
