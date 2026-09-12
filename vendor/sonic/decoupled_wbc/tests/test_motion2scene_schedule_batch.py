"""Batch ownership, CPU-only resume and full-branch budget regression checks."""

import json
from pathlib import Path
import sys
import threading

import pytest

from decoupled_wbc.tests.test_motion2scene_timed_schedule_collector import args_fixture

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

import motion2scene_collect_schedule_batch as batch  # noqa: E402
from motion2scene_timing_diagnostic import artifact, write_new  # noqa: E402


def registered_child(tmp_path, *, attempted=False, completed=False):
    child = tmp_path / "child"
    child.mkdir()
    output = child / "rollouts/cell"
    output.mkdir(parents=True)
    manifest = {"cells": [{"cell_id": "cell", "output": str(output)}]}
    write_new(child / "manifest.json", manifest)
    if attempted:
        write_new(output / "attempt.json", {"exit_status": 0})
    write_new(tmp_path / "spec.json", {"test_fixture_only": True})
    write_new(
        tmp_path / "plan.json",
        dict(
            implementation=[],
            specification=artifact(tmp_path / "spec.json"),
            budget_physics_steps=1192,
            expected_episodes=1,
            scope="synthetic batch review",
        ),
    )
    ref = artifact(child / "manifest.json")
    write_new(
        tmp_path / "registration.json",
        dict(
            plan=artifact(tmp_path / "plan.json"),
            children=[{"index": 0, "manifest": ref}],
            minimum_free_gpu_mib=7500,
        ),
    )
    result = dict(
        manifest=ref,
        rows=[{"cell_id": "cell"}],
        physics_steps=1192,
        unmeasured_failed_attempts=0,
    )
    if completed:
        write_new(child / "result.json", result)
    return child, manifest, result


def test_completed_physics_can_resume_cpu_analysis_with_busy_gpu(tmp_path, monkeypatch):
    child, manifest, result = registered_child(tmp_path, attempted=True)
    calls = []
    monkeypatch.setattr(batch, "verify_manifest", lambda *a, **k: (manifest, None, None))

    def no_gpu_query():
        pytest.fail("completed physical attempts need no GPU preflight")

    def analyze_only(folder):
        calls.append(folder)
        write_new(folder / "result.json", result)

    monkeypatch.setattr(batch, "free_gpu_mib", no_gpu_query)
    monkeypatch.setattr(batch, "run_collection", analyze_only)
    batch.run(tmp_path)
    final = json.loads((tmp_path / "result.json").read_text())
    assert calls == [child] and final["measured_physics_steps"] == 1192


def test_concurrent_controllers_allow_only_one_collection_launch(tmp_path, monkeypatch):
    _, manifest, result = registered_child(tmp_path)
    entered, release = threading.Event(), threading.Event()
    calls, failures = [], []
    monkeypatch.setattr(batch, "verify_manifest", lambda *a, **k: (manifest, None, None))
    monkeypatch.setattr(batch, "free_gpu_mib", lambda: 10000)

    def held_collection(folder):
        calls.append(folder)
        entered.set()
        assert release.wait(timeout=5)
        write_new(folder / "result.json", result)

    def controller():
        try:
            batch.run(tmp_path)
        except Exception as error:
            failures.append(error)

    monkeypatch.setattr(batch, "run_collection", held_collection)
    worker = threading.Thread(target=controller)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        with pytest.raises(RuntimeError, match="another controller already owns"):
            batch.run(tmp_path)
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive() and not failures and len(calls) == 1
    assert json.loads((tmp_path / "result.json").read_text())["episodes"] == 1


def test_resource_pause_releases_ownership_and_keeps_unattempted_cell(tmp_path, monkeypatch):
    child, manifest, result = registered_child(tmp_path)
    calls = []
    monkeypatch.setattr(batch, "verify_manifest", lambda *a, **k: (manifest, None, None))
    monkeypatch.setattr(batch, "free_gpu_mib", lambda: 1000)

    def run_once(folder):
        calls.append(folder)
        write_new(folder / "result.json", result)

    monkeypatch.setattr(batch, "run_collection", run_once)
    batch.run(tmp_path)
    assert not calls and not (tmp_path / "result.json").exists()
    assert not (child / "rollouts/cell/attempt.json").exists()
    monkeypatch.setattr(batch, "free_gpu_mib", lambda: 10000)
    batch.run(tmp_path)
    assert calls == [child]
    with pytest.raises(ValueError, match="must not be rerun"):
        batch.run(tmp_path)


def test_complete_forced_branch_set_is_charged_before_preparation(tmp_path, monkeypatch):
    args = args_fixture(tmp_path, count=1)
    spec = dict(
        schema=batch.SCHEMA,
        registry=artifact(args.registry),
        request=artifact(args.request),
        template=artifact(args.template),
        collections=[
            dict(scene_definition=artifact(args.scene_definition), policy_mode=mode, seed=8731)
            for mode in ("forced", "always_walk")
        ],
        maximum_physics_steps=4 * 1192 - 1,
        scope="synthetic three-option complete-bank budget test",
    )
    path = tmp_path / "spec.json"
    write_new(path, spec)
    out = tmp_path / "batch"
    with pytest.raises(ValueError, match="all declared branches"):
        batch.prepare(path, out)
    assert not out.exists()
    spec["maximum_physics_steps"] += 1
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(batch, "closure", lambda _: {Path(batch.__file__)})
    batch.prepare(path, out)
    plan = json.loads((out / "plan.json").read_text())
    assert plan["expected_episodes"] == 4
    assert plan["planned_maximum_physics_steps"] == plan["budget_physics_steps"] == 4768
    children = json.loads((out / "registration.json").read_text())["children"]
    assert [
        len(json.loads(Path(c["manifest"]["path"]).read_text())["cells"]) for c in children
    ] == [3, 1]
