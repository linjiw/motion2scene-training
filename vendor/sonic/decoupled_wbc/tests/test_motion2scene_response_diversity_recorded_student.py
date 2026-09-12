import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_response_diversity_recorded_student as adapter


def write(path, value):
    path.write_text(json.dumps(value))
    return adapter.original.artifact(path)


@pytest.fixture
def capture(tmp_path, monkeypatch):
    scene = dict(path="assigned_scene", sha256="fixture")
    manifest = write(
        tmp_path / "manifest.json",
        dict(split="development", scene_definition=scene, cells=[dict(runtime_seed=93201)]),
    )
    row = dict(
        outcome=dict(task_outcome="unknown"), task_outcome_admitted=False, physics_steps=1192
    )
    result = write(
        tmp_path / "result.json", dict(manifest=manifest, rows=[row], physics_steps=1192)
    )
    declaration = dict(manifest=manifest, run_id="reference", round_index=5)
    path = tmp_path / "declaration.json"
    write(path, declaration)
    monkeypatch.setattr(adapter.recovery, "audit", lambda p: (declaration, tmp_path, row))
    return path, tmp_path, scene, declaration, row, result


def test_original_rejects_and_adapter_keeps_unknown_and_measured_cost(capture):
    path, out, scene, declaration, row, result = capture
    with pytest.raises(ValueError, match="unresolved physical outcome"):
        adapter.original.measured_collection(result, scene, 93201, ["neutral"], student=True)
    original_reader = adapter.original.measured_collection
    with adapter.retained_unknown_student(path):
        recovered, _ = adapter.original.measured_collection(result, scene, 93201, [], student=True)
        assert recovered["rows"][0]["outcome"]["task_outcome"] == "unknown"
        assert recovered["physics_steps"] == 1192
    assert adapter.original.measured_collection is original_reader


@pytest.mark.parametrize("seed", [8732, 93202])
def test_other_seed_rejected(capture, seed):
    path, _, scene, _, _, result = capture
    with adapter.retained_unknown_student(path):
        with pytest.raises(ValueError, match="assigned scene, seed"):
            adapter.original.measured_collection(result, scene, seed, [], student=True)


def test_unknown_teacher_still_uses_original_rejection(capture):
    path, out, scene, _, _, _ = capture
    row = dict(forced_option_id="neutral", outcome=dict(task_outcome="unknown"))
    manifest = adapter.original.artifact(out / "manifest.json")
    result = write(out / "result.json", dict(manifest=manifest, rows=[row], physics_steps=1192))
    with adapter.retained_unknown_student(path):
        with pytest.raises(ValueError, match="unresolved physical outcome"):
            adapter.original.measured_collection(result, scene, 93201, ["neutral"])


def test_changed_student_cost_is_not_accepted(capture):
    path, out, scene, declaration, row, _ = capture
    result = write(
        out / "result.json", dict(manifest=declaration["manifest"], rows=[row], physics_steps=0)
    )
    with adapter.retained_unknown_student(path):
        with pytest.raises(ValueError, match="unknown and cost"):
            adapter.original.measured_collection(result, scene, 93201, [], student=True)


def test_published_unknown_is_explicit_and_tasks_unchanged(capture, monkeypatch):
    path, _, _, _, _, _ = capture
    task = dict(round=5, student_steps=1192, student_outcome="unknown")
    value = dict(
        schema="motion2scene_response_diversity_v1",
        implementation=dict(id="original"),
        corpora=[dict(run_id="reference", tasks=[task])],
    )
    monkeypatch.setattr(adapter.original, "write_new", lambda p, v: v)
    with adapter.retained_unknown_student(path):
        result = adapter.original.write_new(Path("result.json"), value)
    assert result["corpora"] == value["corpora"]
    assert result["retained_unknown_students"] == [dict(run_id="reference", round=5, steps=1192)]
    assert result["original_implementation"] == dict(id="original")
