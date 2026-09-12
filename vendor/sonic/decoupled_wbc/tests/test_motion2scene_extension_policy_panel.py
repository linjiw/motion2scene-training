"""Every arm/held-task assignment must survive preparation and execution."""

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_extension_policy_panel as panel
from motion2scene_extension_teaching import longitudinal_folds

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import write_new


def study_inputs():
    tasks = [
        dict(task_id=f"task_{i}", center_xy_m=[x, -0.1], length_m=length, underside_m=height)
        for i, (x, length, height) in enumerate(
            (x, length, height)
            for x in (1.7, 2.25, 2.8)
            for length in (0.1, 0.75)
            for height in (1.24, 1.30)
        )
    ]
    study = dict(
        folds=longitudinal_folds(tasks),
        arms=["generated", "authored"],
        policy_evaluation_seed=97001,
    )
    prepared = dict(
        tasks=[
            dict(
                task_id=t["task_id"],
                collection=dict(path=f"/{t['task_id']}/manifest.json", sha256="test"),
            )
            for t in tasks
        ]
    )
    return study, prepared


def test_exactly_twenty_four_pairs_use_their_assigned_model_and_held_center():
    study, prepared = study_inputs()
    rows = panel.assignments(Path("/fit"), study, prepared)
    assert len(rows) == 24
    assert rows == panel.assignments(Path("/fit"), study, prepared)
    assert len({(r["arm"], r["task_id"]) for r in rows}) == 24
    assert len({r["expected_model_result"] for r in rows}) == 6
    for row in rows:
        fold = study["folds"][row["fold_index"]]
        assert row["task_id"] in fold["evaluation_task_ids"]
        assert row["task_id"] not in fold["training_task_ids"]
        assert row["physics_seed"] == 97001
    assert len({r["assignment_id"] for r in rows}) == 24


@pytest.mark.parametrize("change", ["missing_arm", "duplicate_held", "overlap", "seed"])
def test_incomplete_or_leaking_assignment_is_rejected(change):
    study, prepared = study_inputs()
    if change == "missing_arm":
        study["arms"] = ["generated"]
    elif change == "duplicate_held":
        study["folds"][1] = copy.deepcopy(study["folds"][0])
    elif change == "overlap":
        study["folds"][0]["training_task_ids"][0] = study["folds"][0]["evaluation_task_ids"][0]
    else:
        study["policy_evaluation_seed"] = 97002
    with pytest.raises(ValueError):
        panel.assignments(Path("/fit"), study, prepared)


def test_prepared_rows_cannot_drop_or_swap_fitted_models():
    fitting, original = study_inputs()
    rows = panel.assignments(Path("/fit"), fitting, original)
    models = {
        r["expected_model_result"]: dict(path=r["expected_model_result"], sha256="model")
        for r in rows
    }
    study = dict(assignments=rows)
    ref = dict(path="/panel/study.json", sha256="study")
    prepared = dict(
        study=ref,
        models=models,
        assignments=[
            dict(
                r,
                model_result=models[r["expected_model_result"]],
                episode=dict(path=f"/{r['assignment_id']}/manifest.json", sha256="episode"),
            )
            for r in rows
        ],
    )
    panel.check_prepared(prepared, ref, study, models)
    altered = copy.deepcopy(prepared)
    altered["assignments"].pop()
    with pytest.raises(ValueError, match="every original assignment"):
        panel.check_prepared(altered, ref, study, models)
    prepared["assignments"][0]["model_result"]["sha256"] = "wrong"
    # Build an independent expected binding: the intentionally shared test dict
    # above also changed its model lookup; production artifact reads are distinct.
    expected_models = {key: dict(path=key, sha256="model") for key in models}
    with pytest.raises(ValueError, match="every original assignment"):
        panel.check_prepared(prepared, ref, study, expected_models)


def test_preparation_waits_for_all_models_before_creating_episodes(monkeypatch, tmp_path):
    monkeypatch.setattr(panel, "verify_study", lambda p: ({}, {}, {}))

    def missing(*args):
        raise FileNotFoundError("model incomplete")

    monkeypatch.setattr(panel, "bind_models", missing)
    monkeypatch.setattr(
        panel.episode, "prepare", lambda *args: pytest.fail("premature episode preparation")
    )
    with pytest.raises(FileNotFoundError, match="model incomplete"):
        panel.prepare(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("first_outcome", ["pass", "failure", "unknown"])
def test_serial_execution_retains_failures_and_stops_on_unknown(
    monkeypatch, tmp_path, first_outcome
):
    fitting, original = study_inputs()
    assigned = panel.assignments(Path("/fit"), fitting, original)
    models = {
        r["expected_model_result"]: dict(path=r["expected_model_result"], sha256="model")
        for r in assigned
    }
    study = dict(assignments=assigned)
    study_ref = write_new(tmp_path / "study.json", study)
    rows, manifests = [], {}
    for row in assigned:
        folder = tmp_path / "episodes" / row["assignment_id"]
        model = models[row["expected_model_result"]]
        manifest = dict(model_result=model, nominal_template=row["nominal_template"])
        ref = write_new(folder / "manifest.json", manifest)
        manifests[folder] = manifest
        rows.append(dict(row, model_result=model, episode=ref))
    write_new(tmp_path / "prepared.json", dict(study=study_ref, models=models, assignments=rows))
    monkeypatch.setattr(panel, "verify_study", lambda _: (study_ref, study, fitting))
    monkeypatch.setattr(panel, "bind_models", lambda *args: models)
    monkeypatch.setattr(panel, "resource_ready", lambda _: True)
    monkeypatch.setattr(
        panel.episode, "verify", lambda out, **kwargs: (manifests[out], None, None, None)
    )
    calls = []

    def execute(out):
        outcome = first_outcome if not calls else "pass"
        calls.append(out)
        return write_new(
            out / "result.json",
            dict(
                manifest=panel.artifact(out / "manifest.json"),
                row=dict(outcome=dict(task_outcome=outcome), physics_steps=1192),
                complete_policy_trace_verified=outcome == "pass",
            ),
        )

    monkeypatch.setattr(panel.episode, "run", execute)
    if first_outcome == "unknown":
        with pytest.raises(RuntimeError, match="unresolved physical outcome"):
            panel.run(tmp_path)
        assert len(calls) == 1
        assert not (tmp_path / "result.json").exists()
    else:
        result = panel.read_checked(panel.run(tmp_path))
        assert len(calls) == len(result["results"]) == 24
        assert result["measured_physics_steps"] == 24 * 1192
        assert result["complete_policy_traces"] == (24 if first_outcome == "pass" else 23)
