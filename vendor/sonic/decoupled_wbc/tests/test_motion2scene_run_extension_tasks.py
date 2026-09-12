"""Simulator attempts are charged once, including timeout and interruption."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_run_extension_tasks as runner


def fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.collection, "validate_collection_context", lambda *a: None)
    monkeypatch.setattr(runner.collection, "free_gpu_mib", lambda: 10000)
    cell = dict(command=["simulator"], output=str(tmp_path / "rollout"))
    common = dict(limits=dict(minimum_free_gpu_mib=7500, timeout_s=375))
    return cell, common, None, None, tmp_path / "launch.json"


def test_complete_attempt_is_never_run_twice(monkeypatch, tmp_path):
    args = fixture(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        runner.collection, "run_with_process_group", lambda *a, **kw: calls.append(a) or 0
    )
    first = runner.execute_cell(*args)
    assert first["exit_status"] == 0
    assert runner.execute_cell(*args) == first and len(calls) == 1
    assert json.loads(args[-1].read_text())["charged_maximum_physics_steps"] == 1192


def test_timeout_retains_attempt_and_charge_without_retry(monkeypatch, tmp_path):
    args = fixture(monkeypatch, tmp_path)

    def timeout(command, **kw):
        raise runner.subprocess.TimeoutExpired(command, kw["timeout"])

    monkeypatch.setattr(runner.collection, "run_with_process_group", timeout)
    assert runner.execute_cell(*args)["exit_status"] == 124
    assert runner.execute_cell(*args)["exit_status"] == 124


def test_interruption_records_incomplete_attempt_then_propagates(monkeypatch, tmp_path):
    args = fixture(monkeypatch, tmp_path)

    def interrupt(*a, **kw):
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner.collection, "run_with_process_group", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.execute_cell(*args)
    assert json.loads((tmp_path / "rollout/attempt.json").read_text())["exit_status"] == -1
    assert args[-1].exists()


def test_capacity_wait_does_not_charge_or_launch(monkeypatch, tmp_path):
    args = fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(runner.collection, "free_gpu_mib", lambda: 5000)
    with pytest.raises(RuntimeError, match="resource wait before launch"):
        runner.execute_cell(*args)
    assert not args[-1].exists() and not (tmp_path / "rollout").exists()


def test_unfinished_capture_and_changed_command_are_retained(monkeypatch, tmp_path):
    args = fixture(monkeypatch, tmp_path)
    folder = tmp_path / "rollout"
    folder.mkdir()
    with pytest.raises(RuntimeError, match="no automatic retry"):
        runner.execute_cell(*args)
    (folder / "attempt.json").write_text(json.dumps(dict(command=["different"], exit_status=0)))
    with pytest.raises(ValueError, match="differs from its assignment"):
        runner.execute_cell(*args)
