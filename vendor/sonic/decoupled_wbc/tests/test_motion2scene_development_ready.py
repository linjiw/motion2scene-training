"""Early development execution preserves the full panel and never repeats attempts."""

from collections import Counter
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_development_ready as ready


def study():
    api = ready.panel_api
    runs = [
        dict(seed=s, arm=a, run_id=f"seed{s}_{a}")
        for s in (93201, 93202, 93203)
        for a in sorted(api.ARMS)
    ]
    scenes = [
        dict(scene_id=s, scene_definition=dict(path=s, sha256=s)) for s in sorted(api.CONTEXTS)
    ]
    return dict(
        assignments=api.assignments(runs, scenes, ["neutral"] + [f"o{i}" for i in range(6)]),
        models=[dict(run_id=r["run_id"]) for r in runs],
    )


def test_comparators_are_eligible_without_any_m8_model():
    source = study()
    rows = ready.eligible_assignments(source, {})
    assert len(rows) == 48
    assert Counter(r["mode"] for r in rows) == dict(forced=42, scripted_multi=6)
    assert rows == [r for r in source["assignments"] if r["mode"] != "learned"]
    assert len(source["assignments"]) == 138


def test_only_completed_model_adds_its_six_assignments():
    source = study()
    name = source["models"][0]["run_id"]
    rows = ready.eligible_assignments(source, {name: dict(policy="frozen")})
    assert len(rows) == 54
    assert {r["run_id"] for r in rows if r["mode"] == "learned"} == {name}
    assert len(ready.eligible_assignments(source, {name: {}}, comparators_only=True)) == 48


def stored_result(tmp_path, outcome, wrong_manifest=False):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    row = dict(assignment_id="episode_000", collection=ready.artifact(manifest))
    value = dict(
        manifest={} if wrong_manifest else row["collection"],
        rows=[dict(outcome=dict(task_outcome=outcome))],
    )
    (tmp_path / "result.json").write_text(json.dumps(value))
    return row


@pytest.mark.parametrize("outcome", ["pass", "failure"])
def test_completed_passes_and_failures_are_reused_once(tmp_path, monkeypatch, outcome):
    row = stored_result(tmp_path, outcome)
    (tmp_path / "readiness.json").write_text("{}")
    monkeypatch.setattr(ready, "prepare_available", lambda *_: (study(), {}, [row]))
    monkeypatch.setattr(
        ready.panel_api, "execute_cell", lambda *_: pytest.fail("duplicate attempt")
    )
    for _ in range(2):
        result = ready.run_available(tmp_path)
        assert len(result["assignments"]) == 1
        assert result["complete_panel"] is False
        assert result["assigned_panel_episodes"] == 138


def test_unknown_is_retained_and_never_retried(tmp_path, monkeypatch):
    row = stored_result(tmp_path, "unknown")
    monkeypatch.setattr(ready, "prepare_available", lambda *_: (study(), {}, [row]))
    monkeypatch.setattr(ready.panel_api, "execute_cell", lambda *_: pytest.fail("retried unknown"))
    with pytest.raises(RuntimeError, match="no automatic retry"):
        ready.run_available(tmp_path)
    assert (tmp_path / "result.json").exists()


def test_completed_capture_must_belong_to_assignment(tmp_path):
    row = stored_result(tmp_path, "pass", wrong_manifest=True)
    with pytest.raises(ValueError, match="another assignment"):
        ready.completed_result(row)


def prepared_fixture(tmp_path, monkeypatch):
    row = dict(
        assignment_id="episode_000",
        mode="forced",
        option_id="neutral",
        scene_definition=dict(path="scene"),
    )
    source = dict(registry=dict(path="bank"), script=dict(path="script"))
    common = dict(
        split="development",
        registry=source["registry"],
        scene_definition=row["scene_definition"],
        expected_physics_steps=1192,
        policy=None,
        script_parameters=None,
        cells=[
            dict(
                runtime_seed=ready.panel_api.PHYSICS_SEED,
                timed_schedule_mode="forced",
                forced_option_id="neutral",
            )
        ],
    )
    target = tmp_path / "episodes" / row["assignment_id"]
    target.mkdir(parents=True)
    (target / "manifest.json").write_text("{}")
    (tmp_path / "ready").mkdir()
    monkeypatch.setattr(
        ready.panel_api.collection, "verify_manifest", lambda *_: (common, None, None)
    )
    monkeypatch.setattr(
        ready.panel_api.collection, "prepare", lambda *_: pytest.fail("reprepared capture")
    )
    return row, source, common, target


def test_resume_preparation_checks_native_assignment(tmp_path, monkeypatch):
    row, source, common, _ = prepared_fixture(tmp_path, monkeypatch)
    first = ready.prepare_one(tmp_path, {}, source, row, {})
    assert ready.prepare_one(tmp_path, {}, source, row, {}) == first
    common["cells"][0]["runtime_seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        ready.prepare_one(tmp_path, {}, source, row, {})


def test_partial_preparation_without_manifest_is_not_overwritten(tmp_path, monkeypatch):
    row, source, _, target = prepared_fixture(tmp_path, monkeypatch)
    (target / "manifest.json").unlink()
    with pytest.raises(ValueError, match="partial preparation retained"):
        ready.prepare_one(tmp_path, {}, source, row, {})
    assert target.is_dir()


def test_changed_ready_binding_is_rejected(tmp_path, monkeypatch):
    row, source, _, _ = prepared_fixture(tmp_path, monkeypatch)
    ready.prepare_one(tmp_path, dict(sha256="first"), source, row, {})
    with pytest.raises(ValueError, match="existing preparation"):
        ready.prepare_one(tmp_path, dict(sha256="changed"), source, row, {})


def test_partial_panel_cannot_finalize(tmp_path, monkeypatch):
    source = study()
    monkeypatch.setattr(
        ready, "prepare_available", lambda *_: (source, {}, source["assignments"][:48])
    )
    with pytest.raises(ValueError, match="full M8 model inventory"):
        ready.finalize(tmp_path)
    assert not (tmp_path / "prepared.json").exists()


def test_finalization_matches_original_schema_and_all_assignments(tmp_path, monkeypatch):
    source = study()
    models = {m["run_id"]: dict(policy=dict(path=m["run_id"])) for m in source["models"]}
    rows = [dict(row, collection=dict(path=row["assignment_id"])) for row in source["assignments"]]
    (tmp_path / "study.json").write_text(json.dumps(source))
    monkeypatch.setattr(ready, "prepare_available", lambda *_: (source, models, rows))
    monkeypatch.setattr(ready.panel_api, "bind_models", lambda *_: models)
    ref = ready.finalize(tmp_path)
    assert ready.finalize(tmp_path) == ref
    prepared = ready.read_checked(ref)
    assert set(prepared) == {"study", "models", "assignments"}
    assert prepared["models"] == models
    assert [
        {k: v for k, v in r.items() if k != "collection"} for r in prepared["assignments"]
    ] == source["assignments"]


def test_uncommitted_models_are_not_eligible(tmp_path, monkeypatch):
    source = dict(
        expanded_plan={},
        checkpoint=8,
        models=[dict(run_id="one", expected_training_result=str(tmp_path / "model.json"))],
    )
    monkeypatch.setattr(ready, "read_checked", lambda *_: dict(execution_root=str(tmp_path)))
    monkeypatch.setattr(ready.panel_api, "bind_models", lambda *_: pytest.fail("uncommitted model"))
    assert ready.available_models(source) == {}
