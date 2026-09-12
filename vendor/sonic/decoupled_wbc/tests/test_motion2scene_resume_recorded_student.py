import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_resume_recorded_student as recovery


def save(path, value):
    path.write_text(json.dumps(value))
    return recovery.artifact(path)


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    out = tmp_path / "student"
    out.mkdir()
    intent = save(tmp_path / "intent.json", dict(attempt_id="round005_student_learned_neutral"))
    attempt = save(tmp_path / "attempt.json", dict(intent=intent, exit_status=None))
    manifest = save(tmp_path / "manifest.json", dict(policy=dict(id="prior_M4")))
    declaration = dict(run_id="reference", round_index=5, attempt=attempt, manifest=manifest)
    row = dict(outcome=dict(task_outcome="unknown", exit_status=None), physics_steps=1192)
    save(out / "result.json", dict(rows=[row]))
    path = tmp_path / "reconciliation.json"
    save(path, declaration)
    monkeypatch.setattr(recovery, "audit", lambda path: (declaration, out, row))
    controller = recovery.primary.Controller(
        dict(run_root=tmp_path, run=dict(run_id="reference")), backend=object()
    )
    monkeypatch.setattr(controller, "ledger", lambda: None)
    return controller, path, out, declaration, row


def test_reuses_recorded_unknown_without_launch_or_imputing_exit(recorded, monkeypatch):
    controller, path, out, declaration, row = recorded

    def forbidden(*args, **kwargs):
        pytest.fail("a recorded episode must never be relaunched")

    monkeypatch.setattr(recovery.primary.Controller, "collect", forbidden)
    with recovery.recorded_student(path):
        result = controller.collect(
            dict(student_directory=str(out), round_index=5), "student", dict(id="prior_M4")
        )
    assert recovery.read_bound(result)["rows"] == [row]
    assert recovery.read_bound(declaration["attempt"])["exit_status"] is None
    assert recovery.primary.Controller.collect is forbidden


def test_other_assignments_use_original_dispatch(recorded, monkeypatch):
    controller, path, out, _, _ = recorded
    calls = []
    monkeypatch.setattr(
        recovery.primary.Controller, "collect", lambda *args: calls.append(args) or "original"
    )
    with recovery.recorded_student(path):
        assert (
            controller.collect(dict(teacher_directory=str(out / "other")), "teacher") == "original"
        )
    assert len(calls) == 1


def test_changed_policy_rejected_and_controller_restored(recorded):
    controller, path, out, _, _ = recorded
    original = recovery.primary.Controller.collect
    with pytest.raises(ValueError, match="another policy or slot"):
        with recovery.recorded_student(path):
            controller.collect(
                dict(student_directory=str(out), round_index=5), "student", dict(id="M5")
            )
    assert recovery.primary.Controller.collect is original


def test_does_not_relabel_missing_exit_as_completed_process(recorded, monkeypatch):
    controller, path, _, declaration, _ = recorded
    monkeypatch.setattr(
        recovery.primary.Controller,
        "accounting",
        lambda self: dict(
            actual_recorded_physics_steps=1192,
            attempts=[
                dict(
                    attempt_id="round005_student_learned_neutral",
                    actual_process_receipt=declaration["attempt"],
                    actual_physics_steps=1192,
                )
            ],
        ),
    )
    with recovery.recorded_student(path):
        report = controller.accounting()
    assert report["actual_recorded_physics_steps"] == 1192
    assert report["attempts"][0]["actual_process_receipt"] is None
    assert report["attempts"][0]["launch_state"] == "native_recording_verified_process_exit_unknown"


def test_changed_published_outcome_rejected(recorded):
    _, path, out, _, _ = recorded
    save(out / "result.json", dict(rows=[dict(outcome=dict(task_outcome="pass"))]))
    with pytest.raises(ValueError, match="published student differs"):
        with recovery.recorded_student(path):
            pytest.fail("changed outcome must not be admitted")
