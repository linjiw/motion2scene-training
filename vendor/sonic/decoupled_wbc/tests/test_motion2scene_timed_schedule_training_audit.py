"""Adversarial trainer history/provenance tests with a stubbed physical analyzer."""

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from decoupled_wbc.tests.test_motion2scene_timed_schedule_policy import bank_fixture

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))
SPEC = importlib.util.spec_from_file_location(
    "schedule_training_audit", ROOT / "scripts/research/motion2scene_train_timed_schedules.py"
)
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return TRAINER.artifact(path)


@pytest.fixture
def collection(tmp_path, monkeypatch):
    bank = bank_fixture()
    phases, masks, entries = TRAINER.schedule_layout(bank)
    names = TRAINER.expected_feature_names(7)
    registry = save(tmp_path / "registry.json", {"synthetic": True})
    scene_asset = save(tmp_path / "scene.usda", {"synthetic": "declared room"})
    scene = dict(split="development", scene_id="test_scene", scene=scene_asset)
    scene_ref = save(tmp_path / "scene.json", scene)
    rows, cells, actual = [], [], {}
    for index, option_id in enumerate(bank.option_ids):
        folder = tmp_path / "rollouts" / option_id
        refs = {
            key: save(folder / "trajectories" / (key + ".json"), {"fixture": key})
            for key in TRAINER.ARTIFACT_KEYS
        }
        command = ["bash", "fixture.sh", "--extra", "++seed=8731"]
        refs["attempt"] = save(folder / "attempt.json", {"exit_status": 0, "command": command})
        save(
            folder / "success_manifest.json",
            {
                "capture_context": {
                    "scene_id": "test_scene",
                    "scene": {"hash": scene_asset["sha256"], "resolved": scene_asset["path"]},
                }
            },
        )
        row = dict(
            cell_id=option_id,
            forced_option_id=option_id,
            mode="forced",
            measurement_admitted=True,
            **{"pass": index == 6},
            costs={"passage_time_s": 3.0 if index == 6 else None},
            schedule_audit={"valid": True},
            observation_timing={"fixture": True},
            physics_steps=1192,
            **refs
        )
        rows.append(row)
        cells.append(
            dict(
                cell_id=option_id,
                forced_option_id=option_id,
                timed_schedule_mode="forced",
                runtime_seed=8731,
                hydra_overrides=["++seed=8731"],
                command=command,
                output=str(folder),
            )
        )
        payload = {key: np.zeros((298, 3)) for key in TRAINER.PREFIX_KEYS if key != "motion_time_s"}
        # Actual recorder arithmetic, including tick35's tiny .02-vs-/50 residual.
        payload["motion_time_s"] = np.arange(298) * 0.02
        entry = entries[index]
        ret = None if entry is None else bank.request["options"][index - 1]["return_tick"]
        if entry is not None:
            payload["dof_pos"][entry:ret] = index
        observations = []
        for tick in range(1, 299):
            before = option_id if entry is not None and entry < tick <= ret else "neutral"
            after = option_id if entry is not None and entry <= tick < ret else "neutral"
            active = bank.option_ids.index(before)
            legal = np.eye(7, dtype=bool)[active]
            if before == "neutral" and tick in phases:
                legal = masks[np.flatnonzero(phases == tick)[0]]
            x = np.zeros(114)
            x[names.index("phase_s")] = tick / 50
            x[100 + active] = 1
            x[-7:] = legal
            observations.append(
                dict(
                    tick=tick,
                    active_before=before,
                    active=after,
                    selected_option_id=after,
                    state={"fixture_joint": payload["dof_pos"][tick - 1].tolist()},
                    root_pos_w=[0, 0, 0.8],
                    root_quat_w=[1, 0, 0, 0],
                    features=x.tolist(),
                    legal_mask=legal.tolist(),
                    measurements=[{"distance_m": 2.0}] * 65,
                    normal_known_mask=[True] * 65,
                    capture_elapsed_s=(tick - 1) / 50,
                    delivered_capture_elapsed_s=(tick - 1) / 50,
                )
            )
        actual[option_id] = (copy.deepcopy(row), payload, {"observations": observations})
    manifest = dict(
        schema=TRAINER.COLLECTION_SCHEMA,
        split="development",
        registry=registry,
        request_digest=TRAINER.definition_digest(bank.request),
        scene_definition=scene_ref,
        sensor=dict(rays_per_tick=65, history_seconds=2.0, history_frames=101, delay_s=0),
        cells=cells,
    )
    manifest_ref = save(tmp_path / "manifest.json", manifest)
    result = dict(schema=TRAINER.COLLECTION_SCHEMA, manifest=manifest_ref, rows=rows)
    path = tmp_path / "result.json"
    save(path, result)
    monkeypatch.setattr(TRAINER, "analyze_cell", lambda cell, *_: actual[cell["forced_option_id"]])
    # This test isolates teacher/provenance logic; full artifact/schema validation
    # is separately covered by collector tests with real synthetic registries.
    monkeypatch.setattr(
        TRAINER, "verify_manifest", lambda *_args, **_kwargs: (manifest, bank, scene), raising=False
    )
    return dict(
        path=path,
        bank=bank,
        registry=registry,
        actual=actual,
        manifest=manifest,
        result=result,
        root=tmp_path,
    )


def audit(fixture):
    return TRAINER.audit_collection(fixture["path"], fixture["bank"], fixture["registry"])


def test_exact_preaction_prefix_allows_current_postdecision_difference_and_future_wait(collection):
    result = audit(collection)
    assert all(row["matched"] for row in result["prefix_comparisons"])
    assert [row["teacher_action"] for row in result["targets"]] == [0, 0, 6]
    assert result["targets"][0]["waiting_has_future_adaptation"]
    assert result["targets"][0]["expected_continuation_counts"][0] == 4
    assert result["targets"][2]["complete_legal_action_table"]
    assert result["physics_steps"] == 7 * 1192
    json.dumps(result)


def test_old_sensor_history_difference_invalidates_wait_even_if_current_features_match(collection):
    latest = collection["actual"][collection["bank"].option_ids[6]][2]
    latest["observations"][0]["measurements"] = [{"distance_m": 1.8}] * 65
    result = audit(collection)
    assert result["targets"][0]["teacher_action"] is None
    assert result["targets"][0]["admitted"][0] is False
    assert result["targets"][2]["admitted"][6] is False
    assert result["targets"][2]["teacher_action"] is None


def test_current_postaction_is_excluded_but_current_preaction_state_is_bound(collection):
    _, payload, interface = collection["actual"]["neutral"]
    before = TRAINER.history_id(["group"], payload, interface, 70)
    interface["observations"][69]["active"] = "arbitrary_postdecision_placeholder"
    interface["observations"][69]["selected_option_id"] = "arbitrary_postdecision_placeholder"
    assert TRAINER.history_id(["group"], payload, interface, 70) == before
    interface["observations"][69]["state"]["fixture_joint"][0] = 0.1
    assert TRAINER.history_id(["group"], payload, interface, 70) != before
    interface["observations"][69]["active_before"] = collection["bank"].option_ids[1]
    with pytest.raises(ValueError, match="neutral preaction"):
        TRAINER.history_id(["group"], payload, interface, 70)


def test_different_applied_action_prefix_cannot_become_a_matching_future_branch(collection):
    collection["actual"][collection["bank"].option_ids[6]][1]["applied_joint_action"][30, 0] = 0.1
    result = audit(collection)
    assert result["targets"][0]["complete_legal_action_table"]
    assert result["targets"][1]["teacher_action"] is None
    assert result["targets"][2]["teacher_action"] is None


def test_changed_recorded_label_or_seed_is_rejected(collection):
    value = copy.deepcopy(collection["result"])
    value["rows"][0]["pass"] = True
    save(collection["path"], value)
    with pytest.raises(ValueError, match="stored pass"):
        audit(collection)
    save(collection["path"], collection["result"])
    cell = collection["manifest"]["cells"][0]
    cell["runtime_seed"] = 8732
    ref = save(collection["root"] / "manifest.json", collection["manifest"])
    collection["result"]["manifest"] = ref
    save(collection["path"], collection["result"])
    with pytest.raises(ValueError, match="common physics seed"):
        audit(collection)


def test_duplicate_and_missing_forced_branch_cannot_synthesize_failed_targets(collection):
    result = copy.deepcopy(collection["result"])
    result["rows"][-1] = copy.deepcopy(result["rows"][0])
    save(collection["path"], result)
    with pytest.raises(ValueError, match="unique forced branch"):
        audit(collection)
    result["rows"] = result["rows"][:-1]
    save(collection["path"], result)
    with pytest.raises(ValueError, match="unique forced branch"):
        audit(collection)


def test_changed_usd_is_rejected_despite_unchanged_recorded_runtime_hash(collection):
    scene_ref = collection["manifest"]["scene_definition"]
    scene = json.loads(Path(scene_ref["path"]).read_text())
    Path(scene["scene"]["path"]).write_text("changed native geometry")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit(collection)


def test_archival_source_verification_failure_and_wrong_sensor_cannot_be_ignored(
    collection, monkeypatch
):
    def rejected(*_args, **_kwargs):
        raise ValueError("registered source snapshot hash mismatch")

    monkeypatch.setattr(TRAINER, "verify_manifest", rejected)
    with pytest.raises(ValueError, match="source snapshot"):
        audit(collection)
    scene = json.loads(Path(collection["manifest"]["scene_definition"]["path"]).read_text())
    monkeypatch.setattr(
        TRAINER,
        "verify_manifest",
        lambda *_args, **_kwargs: (collection["manifest"], collection["bank"], scene),
    )
    collection["manifest"]["sensor"]["history_seconds"] = 0.5
    collection["result"]["manifest"] = save(
        collection["root"] / "manifest.json", collection["manifest"]
    )
    save(collection["path"], collection["result"])
    with pytest.raises(ValueError, match="archival"):
        audit(collection)


def make_partial(fixture, monkeypatch, *, verified_failure=False):
    """Stub only native re-audit; exercise real raw-file/prefix/teacher dispatch."""
    import pickle

    option = fixture["bank"].option_ids[-1]
    original, payload, interface = fixture["actual"][option]
    cell = fixture["manifest"]["cells"][-1]
    folder = Path(cell["output"])
    attempt = save(folder / "attempt.json", dict(exit_status=1, command=cell["command"]))
    refs = {}
    availability = [
        dict(tick=t, sensor_preaction_available=verified_failure and t == 15) for t in (15, 50, 70)
    ]
    if verified_failure:
        # Align the synthetic recorded poses with its sensor log on every branch.
        for _, data, captured in fixture["actual"].values():
            data["root_pos_w"][:] = [0, 0, 0.8]
            data["root_quat_w"] = np.tile([1, 0, 0, 0], (298, 1))
            for packet in captured["observations"]:
                packet["time_s"] = packet["tick"] / 50
            data["fps"] = 50
        partial_payload = {
            key: (value[:30].copy() if isinstance(value, np.ndarray) else value)
            for key, value in payload.items()
        }
        partial_payload["root_pos_w"][20:, 2] = 0.4
        captured = copy.deepcopy(interface["observations"][:30])
        for row in captured[20:]:
            row["root_pos_w"][2] = 0.4
        path = folder / "aborted_raw_recording.pkl"
        with path.open("wb") as handle:
            pickle.dump({0: {k: v for k, v in partial_payload.items() if k != "fps"}}, handle)
        refs["aborted_raw_recording"] = TRAINER.artifact(path)
        refs["sensor"] = save(folder / "aborted_interface.json", dict(observations=captured))
    row = dict(
        cell_id=option,
        forced_option_id=option,
        mode="forced",
        status="failed_attempt",
        measurement_admitted=False,
        task_outcome_admitted=verified_failure,
        **{"pass": False},
        physics_steps=120 if verified_failure else 0,
        costs={"passage_time_s": None},
        raw_artifacts=refs,
        attempt=attempt,
        analysis_errors=["registered technical or early-terminal attempt"],
        outcome=dict(
            task_outcome="failure" if verified_failure else "unknown",
            phase_availability=availability,
        )
    )
    fixture["result"]["rows"][-1] = row
    save(fixture["path"], fixture["result"])
    monkeypatch.setattr(TRAINER, "validate_collection_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(TRAINER, "analyze_incomplete", lambda *_args, **_kwargs: copy.deepcopy(row))
    return row


def test_unknown_technical_branch_retains_group_cost_and_incomplete_wait_targets(
    collection, monkeypatch
):
    make_partial(collection, monkeypatch)
    result = audit(collection)
    assert result["assigned_branches"] == 7 and len(result["branch_assessments"]) == 7
    assert result["physics_steps"] == 6 * 1192
    missing = result["branch_assessments"][-1]
    assert missing["task_outcome"] == "unknown" and not missing["task_outcome_admitted"]
    assert missing["available_prefix_ticks"] == []
    assert all(row["available"] for row in result["targets"])  # Neutral history exists.
    assert all(row["teacher_action"] is None for row in result["targets"])
    assert result["targets"][0]["admitted"][0] is False
    assert result["targets"][0]["admitted_continuation_counts"][0] == 3
    assert result["targets"][0]["expected_continuation_counts"][0] == 4


def test_early_verified_failure_can_label_matched_earlier_future_but_never_later_prefix(
    collection, monkeypatch
):
    make_partial(collection, monkeypatch, verified_failure=True)
    result = audit(collection)
    branch = result["branch_assessments"][-1]
    assert branch["task_outcome"] == "failure" and branch["task_outcome_admitted"]
    assert not branch["measurement_admitted"] and branch["available_prefix_ticks"] == [15]
    assert result["physics_steps"] == 6 * 1192 + 120
    assert result["targets"][0]["complete_legal_action_table"]
    assert result["targets"][0]["admitted_continuation_counts"][0] == 4
    assert not result["targets"][1]["complete_legal_action_table"]
    assert result["targets"][2]["admitted"][6] is False


def test_missing_neutral_later_history_emits_unavailable_targets_without_default_features(
    collection,
):
    row, payload, interface = collection["actual"]["neutral"]
    for key, value in list(payload.items()):
        payload[key] = value[:30]
    interface["observations"] = interface["observations"][:30]
    result = audit(collection)
    assert result["targets"][0]["available"] is True
    for target in result["targets"][1:]:
        assert target["available"] is False
        assert target["features"] is target["pass_labels"] is target["legal_mask"] is None
        assert target["teacher_action"] is None
    json.dumps(result)


def test_no_model_receipt_and_teachers_survive_missing_phase_evidence(collection, monkeypatch):
    make_partial(collection, monkeypatch)
    monkeypatch.setattr(TRAINER, "load_verified_registry", lambda *_args: collection["bank"])
    out = collection["root"] / "fit_incomplete"
    TRAINER.run(Path(collection["registry"]["path"]), [collection["path"]], out, 1e-6)
    result = json.loads((out / "result.json").read_text())
    teachers = json.loads((out / "teachers.json").read_text())
    assert result["status"] == "insufficient_phase_evidence" and result["policy"] is None
    assert result["assigned_branches"] == 7 and result["audited_groups"] == 1
    assert result["source_physics_steps"] == 6 * 1192
    assert len(teachers) == 1 and len(teachers[0]["branch_assessments"]) == 7
    assert not (out / "policy.npz").exists()


def test_hard_audit_failure_has_explicit_no_model_and_assigned_group_receipt(
    collection, monkeypatch
):
    monkeypatch.setattr(TRAINER, "load_verified_registry", lambda *_args: collection["bank"])

    def rejected(*_args):
        raise ValueError("source hash mismatch")

    monkeypatch.setattr(TRAINER, "audit_collection", rejected)
    out = collection["root"] / "fit_bad_source"
    TRAINER.run(Path(collection["registry"]["path"]), [collection["path"]], out, 1e-6)
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "audit_failed" and result["assigned_groups"] == 1
    assert result["assigned_branches"] == 7 and result["audited_groups"] == 0
    assert result["audit_errors"][0]["reason"] == "source hash mismatch"
    assert json.loads((out / "teachers.json").read_text()) == []
    assert not (out / "policy.npz").exists()


def test_complete_successful_training_retains_policy_and_teacher_format(collection, monkeypatch):
    monkeypatch.setattr(TRAINER, "load_verified_registry", lambda *_args: collection["bank"])
    out = collection["root"] / "fit_complete"
    TRAINER.run(Path(collection["registry"]["path"]), [collection["path"]], out, 1e-6)
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "complete" and result["policy"] == TRAINER.artifact(
        out / "policy.npz"
    )
    assert result["fit"]["supervised_decisions"] == 3
    assert result["assigned_branches"] == 7 and result["source_physics_steps"] == 7 * 1192
    assert len(json.loads((out / "teachers.json").read_text())[0]["targets"]) == 3


def test_failed_fit_still_preserves_audited_teachers_without_a_model(collection, monkeypatch):
    monkeypatch.setattr(TRAINER, "load_verified_registry", lambda *_args: collection["bank"])

    def rejected(*_args, **_kwargs):
        raise ValueError("unsupported phase head")

    monkeypatch.setattr(TRAINER, "fit_timed_schedule_policy", rejected)
    out = collection["root"] / "fit_rejected"
    TRAINER.run(Path(collection["registry"]["path"]), [collection["path"]], out, 1e-6)
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "fit_validation_failed" and result["policy"] is None
    assert result["fit_error"] == "unsupported phase head"
    assert len(json.loads((out / "teachers.json").read_text())) == 1
    assert not (out / "policy.npz").exists()
