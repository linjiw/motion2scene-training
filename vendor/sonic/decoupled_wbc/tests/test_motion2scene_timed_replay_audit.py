"""Offline replay rejects altered actual invocations and student outcome artifacts."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from decoupled_wbc.tests.test_motion2scene_timed_schedule_training_audit import (
    TRAINER,
    collection,  # noqa: F401 - imported shared synthetic physical fixture.
    save,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "timed_replay_audit", ROOT / "scripts/research/motion2scene_build_timed_replay.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


@pytest.fixture
def actual_student(collection, monkeypatch):  # noqa: F811 - shared pytest fixture injection.
    item = collection
    manifest = item["manifest"]
    manifest["cells"] = [manifest["cells"][0]]
    cell = manifest["cells"][0]
    cell["timed_schedule_mode"] = "learned"
    manifest["policy"] = save(item["root"] / "policy.npz", {"synthetic": "frozen_student"})
    item["result"]["rows"] = [item["result"]["rows"][0]]
    row = item["result"]["rows"][0]
    row.update(mode="learned", task_outcome_admitted=True)
    row["outcome"] = {
        "task_outcome": "failure",
        "phase_availability": [
            {"tick": t, "sensor_preaction_available": True, "packet_index": t - 1}
            for t in (15, 50, 70)
        ],
    }
    verified = copy.deepcopy(row)
    _, payload, interface = item["actual"]["neutral"]
    item["result"]["manifest"] = save(item["root"] / "manifest.json", manifest)
    save(item["path"], item["result"])
    scene = json.loads(Path(manifest["scene_definition"]["path"]).read_text())
    monkeypatch.setattr(
        REPLAY, "verify_manifest", lambda *_args, **_kwargs: (manifest, item["bank"], scene)
    )
    monkeypatch.setattr(REPLAY, "analyze_cell", lambda *_args: (verified, payload, interface))
    # Native/source checks are covered by collector tests; retain real invocation
    # validation and exact ten-field physical/sensor history hashing here.
    monkeypatch.setattr(REPLAY, "validate_collection_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(REPLAY, "partial_source_bound", lambda *_args: True)
    return item


def audit(item):
    return REPLAY.audit_students(item["path"], item["bank"], item["registry"])


def test_actual_full_failure_and_three_neutral_prefixes_are_retained(actual_student):
    rows, captures, unknown = audit(actual_student)
    assert len(rows) == len(captures) == 1 and not unknown
    assert rows[0]["task_outcome_admitted"] and rows[0]["passed"] is False
    assert set(rows[0]["phase_records"]) == {15, 50, 70}
    assert captures[0]["physics_steps"] == 1192
    original = actual_student["actual"]["neutral"]
    group = REPLAY.group_identity(
        actual_student["registry"], actual_student["manifest"]["scene_definition"], 8731
    )
    assert rows[0]["phase_records"][15]["recorded_history_sha256"] == TRAINER.history_id(
        group, original[1], original[2], 15
    )


@pytest.mark.parametrize("fault", ["pass", "cost", "artifact", "attempt", "seed", "scene"])
def test_recorded_student_identity_and_outcome_tampering_is_rejected(actual_student, fault):
    item = actual_student
    row = item["result"]["rows"][0]
    cell = item["manifest"]["cells"][0]
    if fault == "pass":
        row["pass"] = True
    elif fault == "cost":
        row["costs"]["passage_time_s"] = 2.0
    elif fault == "artifact":
        row["features"] = save(item["root"] / "other_features.json", {"other": True})
    elif fault == "attempt":
        path = Path(cell["output"]) / "attempt.json"
        body = json.loads(path.read_text())
        body["command"] = ["a_different_program"]
        row["attempt"] = save(path, body)
    elif fault == "seed":
        cell["runtime_seed"] = 8732
        item["result"]["manifest"] = save(item["root"] / "manifest.json", item["manifest"])
    else:
        path = Path(cell["output"]) / "success_manifest.json"
        body = json.loads(path.read_text())
        body["capture_context"]["scene_id"] = "different_scene"
        save(path, body)
    save(item["path"], item["result"])
    with pytest.raises(ValueError):
        audit(item)


def test_archival_source_failure_is_not_downgraded_to_unknown_weight(actual_student, monkeypatch):
    def invalid(*_args, **_kwargs):
        raise ValueError("archived source snapshot hash differs")

    monkeypatch.setattr(REPLAY, "verify_manifest", invalid)
    with pytest.raises(ValueError, match="source snapshot"):
        audit(actual_student)


def test_missing_registered_student_row_remains_an_unknown_attempt(actual_student):
    actual_student["result"]["rows"] = []
    save(actual_student["path"], actual_student["result"])
    rows, captures, unknown = audit(actual_student)
    assert len(rows) == 1 and not captures and len(unknown) == 1
    assert rows[0]["phase_records"] == {}
    assert rows[0]["task_outcome_admitted"] is False


def test_duplicate_student_rows_are_not_silently_collapsed(actual_student):
    actual_student["result"]["rows"] *= 2
    save(actual_student["path"], actual_student["result"])
    with pytest.raises(ValueError, match="unique registered"):
        audit(actual_student)


def verification_fixture(tmp_path, monkeypatch):
    teacher_path = tmp_path / "teachers.json"
    collection_ref = save(tmp_path / "collection.json", {"synthetic": True})
    groups = [
        {
            "collection": collection_ref,
            "targets": [{"phase_tick": 15, "recorded_history_sha256": "matched"}],
        }
    ]
    teacher_ref = save(teacher_path, groups)
    student_ref = save(tmp_path / "student.json", {"synthetic": True})
    registry_ref = save(tmp_path / "registry.json", {"synthetic": True})
    registration_ref = save(
        tmp_path / "registration.json",
        {
            "schema": REPLAY.SCHEMA,
            "teachers": teacher_ref,
            "student_results": [student_ref],
            "registry": registry_ref,
            "rule": REPLAY.DEFAULT_RULE,
            "implementation": [],
            "acquisition_receipt": None,
        },
    )
    core = {
        "schema": REPLAY.SCHEMA,
        "registry": registry_ref,
        "records": [
            {
                "encounter_id": "group",
                "phase_tick": 15,
                "teacher_collection": collection_ref,
                "recorded_history_sha256": "matched",
            }
        ],
        "replay": {"weights": [1.0], "distribution": {"maximum_weight": 1}},
        "accounting": {"new_physics_steps": 0},
        "online_acquisition": None,
    }
    evidence_ref = save(tmp_path / "evidence.json", {**core, "registration": registration_ref})
    weights_path = tmp_path / "weights.json"
    weights = {
        "schema": REPLAY.SCHEMA,
        "evidence": evidence_ref,
        "rows": [{**core["records"][0], "weight": 1.0}],
    }
    save(weights_path, weights)
    monkeypatch.setattr(REPLAY, "compute_evidence", lambda *_args: copy.deepcopy(core))
    return weights_path, weights, groups, registry_ref


def test_weight_verifier_recomputes_and_aligns_exact_teacher_slots(tmp_path, monkeypatch):
    path, _, groups, registry = verification_fixture(tmp_path, monkeypatch)
    report = REPLAY.verify_weights(path, groups, registry)
    assert report["weights_in_group_target_order"] == [1]
    assert report["new_training_runs"] == report["new_physics_steps"] == 0


@pytest.mark.parametrize("fault", ["weight", "phase", "teacher", "rehash_forged_evidence"])
def test_weight_verifier_rejects_edits_even_with_fresh_content_hashes(tmp_path, monkeypatch, fault):
    path, weights, groups, registry = verification_fixture(tmp_path, monkeypatch)
    if fault == "weight":
        weights["rows"][0]["weight"] = 0.4
    elif fault == "phase":
        weights["rows"][0]["phase_tick"] = 50
    elif fault == "teacher":
        groups[0]["targets"][0]["recorded_history_sha256"] = "other"
    else:
        evidence = json.loads(Path(weights["evidence"]["path"]).read_text())
        evidence["replay"]["weights"] = [0.4]
        weights["evidence"] = save(Path(weights["evidence"]["path"]), evidence)
        weights["rows"][0]["weight"] = 0.4
    save(path, weights)
    with pytest.raises(ValueError):
        REPLAY.verify_weights(path, groups, registry)


def online_fixture(tmp_path, monkeypatch):
    from gear_sonic.dataset_generation.hallucination import motion2scene_acquisition_plan as plans

    # The separate plan tests validate adoption/markers. These tests isolate
    # replay's independent model-training and current-group exclusion checks.
    monkeypatch.setattr(plans, "validate_completed_replay_order", lambda *_args: {})
    registry = save(tmp_path / "registry.json", {"synthetic": True})
    plan = save(tmp_path / "plan.json", {"synthetic": "adoption checked separately"})
    groups, students, entries = [], [], []
    for index in range(3):
        teacher = save(tmp_path / f"teacher_{index}.json", {"round": index})
        group = {
            "encounter_id": str(index),
            "collection": teacher,
            "targets": [{"round": index}],
            "trajectory_identities": [],
            "physics_steps": 8344,
        }
        groups.append(group)
        if index == 0:
            entries.append({"round_index": 0, "teacher_collection": teacher})
            continue
        policy = save(tmp_path / f"model_{index-1}.npz", {"synthetic": f"model{index-1}"})
        trained = save(tmp_path / f"model_{index-1}_teachers.json", groups[:index])
        registration = save(
            tmp_path / f"model_{index-1}_registration.json",
            {
                "registry": registry,
                "collections": [g["collection"] for g in groups[:index]],
                "l2": 0.1,
            },
        )
        training = save(
            tmp_path / f"model_{index-1}_result.json",
            {
                "registration": registration,
                "policy": policy,
                "teachers": trained,
            },
        )
        student = save(tmp_path / f"student_{index}.json", {"round": index})
        students.append(
            {"encounter_id": str(index), "collection": student, "manifest": {"policy": policy}}
        )
        entries.append(
            {
                "round_index": index,
                "teacher_collection": teacher,
                "student_collection": student,
                "student_model": policy,
                "model_training_result": training,
            }
        )
    receipt_path = tmp_path / "completion.json"
    save(receipt_path, {"plan": plan, "run_id": "synthetic_run", "entries": entries})
    return receipt_path, groups, students, registry


def test_ordered_online_models_report_historical_gap_age(tmp_path, monkeypatch):
    path, groups, students, registry = online_fixture(tmp_path, monkeypatch)
    result = REPLAY.validate_online_acquisition(path, groups, students, registry)
    assert result["rounds"]["1"]["gap_age_in_completed_rounds"] == 1
    assert result["rounds"]["2"]["gap_age_in_completed_rounds"] == 0
    assert result["rounds"]["2"]["current_group_labels_excluded"]


@pytest.mark.parametrize("fault", ["current_labels", "actual_model", "different_encounter"])
def test_online_mode_rejects_current_labels_or_a_different_executed_model(
    tmp_path, monkeypatch, fault
):
    path, groups, students, registry = online_fixture(tmp_path, monkeypatch)
    if fault == "actual_model":
        students[1]["manifest"]["policy"] = students[0]["manifest"]["policy"]
    elif fault == "different_encounter":
        students[1]["encounter_id"] = "another_scene_or_seed"
    else:
        receipt = json.loads(path.read_text())
        entry = receipt["entries"][2]
        result_path = Path(entry["model_training_result"]["path"])
        training = json.loads(result_path.read_text())
        registration_path = Path(training["registration"]["path"])
        registration = json.loads(registration_path.read_text())
        registration["collections"].append(groups[2]["collection"])
        training["registration"] = save(registration_path, registration)
        entry["model_training_result"] = save(result_path, training)
        save(path, receipt)
    with pytest.raises(ValueError, match="pre-update model"):
        REPLAY.validate_online_acquisition(path, groups, students, registry)
