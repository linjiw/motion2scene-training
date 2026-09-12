"""Equivalent-teaching fitting excludes held-center and opposite-arm outcomes."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_extension_teaching import arm_view, longitudinal_folds
import motion2scene_fit_extension_teaching as fitting
from motion2scene_fit_extension_teaching import fit_fold, select_constant, validate_design
from motion2scene_tune_schedule_learner import arrays
from test_motion2scene_extension_teaching import bank, observation

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (
    fit_timed_schedule_policy,
    schedule_layout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (
    expected_feature_names,
)


def result(b, states, times):
    return dict(
        rows=[
            dict(
                forced_option_id=name,
                cell_id=f"branch_{i}",
                physics_steps=1192,
                outcome=dict(task_outcome={1: "pass", 0: "failure", -1: "unknown"}[states[i]]),
                **{"pass": states[i] == 1},
                costs=dict(passage_time_s=times[i]),
            )
            for i, name in enumerate(b.option_ids)
        ]
    )


def training_fold(b):
    fold = dict(
        training_task_ids=[f"train_{i}" for i in range(8)],
        evaluation_task_ids=[f"test_{i}" for i in range(4)],
    )
    _, _, entries = schedule_layout(b)
    training = {}
    for task_id in fold["training_task_ids"]:
        measured = result(b, [1] * 5, [4, 2, 3, 0.2, 0.1])
        targets = []
        for tick in (15, 50):
            names, x, legal = observation(b, tick)
            targets.append(
                dict(
                    phase_tick=tick,
                    available=True,
                    recorded_history_sha256=f"prefix{tick}",
                    features=x.tolist(),
                    feature_names=list(names),
                    legal_mask=legal.tolist(),
                )
            )
        group = dict(
            scene_id=task_id,
            physics_seed=1,
            option_ids=list(b.option_ids),
            collection=dict(path=f"/{task_id}/result.json", sha256="test-reference"),
            targets=targets,
            branch_assessments=[
                dict(
                    option_id=r["forced_option_id"],
                    cell_id=r["cell_id"],
                    physics_steps=r["physics_steps"],
                    task_outcome=r["outcome"]["task_outcome"],
                )
                for r in measured["rows"]
            ],
            prefix_comparisons=[
                dict(phase_tick=tick, option_id=b.option_ids[i], matched=True)
                for tick in (15, 50)
                for i, entry in entries.items()
                if entry is None or entry >= tick
            ],
        )
        training[task_id] = (group, measured)
    return fold, training


def test_constant_prefers_passage_then_time_then_original_order():
    b = bank()
    view, _ = arm_view(b, "generated")
    rows = [
        result(b, [1, 1, 1, 1, 1], [5, 2, 0.01, 0.001, 0.001]),
        result(b, [1, 1, 0, 1, 1], [5, 2, None, 0.001, 0.001]),
    ]
    selected = select_constant(view, rows)
    assert selected["selected_option_id"] == "generated_00_e15"
    tied = select_constant(view, [result(b, [1] * 5, [2] * 5)])
    assert tied["selected_option_id"] == "neutral"


def test_missing_own_constant_outcome_is_not_counted_as_failure():
    b = bank()
    view, _ = arm_view(b, "generated")
    with pytest.raises(ValueError, match="known physical"):
        select_constant(view, [result(b, [1, -1, 1, 1, 1], [2, None, 3, 1, 1])])
    # Other-arm unknown outcomes have no role in this arm's constant choice.
    assert (
        select_constant(view, [result(b, [1, 1, 1, -1, -1], [4, 2, 3, None, None])])[
            "selected_option_id"
        ]
        == "generated_00_e15"
    )


def test_fitting_uses_the_unchanged_ridge_and_only_own_training_outcomes():
    b = bank()
    fold, training = training_fold(b)
    model, report = fit_fold(b, "generated", fold, training)
    view, _ = arm_view(b, "generated")
    expected, _ = fit_timed_schedule_policy(
        view,
        feature_names=expected_feature_names(3),
        l2=10,
        allow_measured_tie_initialization=True,
        **arrays(report["targets"]),
    )
    for key in model:
        np.testing.assert_array_equal(model[key], expected[key])
    assert report["constant"]["selected_option_id"] == "generated_00_e15"
    assert report["assigned_training_branches"] == 24  # Synthetic neutral + two own schedules.
    assert report["measured_training_physics_steps"] == 24 * 1192
    assert report["training_branches_with_unknown_step_count"] == 0
    assert all("test_" not in r["path"] for r in report["training_collections"])
    changed = deepcopy(training)
    for _, measured in changed.values():
        for row in measured["rows"][3:]:
            row["costs"]["passage_time_s"] = 1000.0
    other, analysis = fit_fold(b, "generated", fold, changed)
    for key in model:
        np.testing.assert_array_equal(model[key], other[key])
    assert report["constant"] == analysis["constant"]


def test_extra_held_task_or_overlapping_fold_is_rejected_before_fitting():
    b = bank()
    fold, training = training_fold(b)
    training["test_0"] = training["train_0"]
    with pytest.raises(ValueError, match="without held-center"):
        fit_fold(b, "generated", fold, training)
    del training["test_0"]
    fold["evaluation_task_ids"][0] = "train_0"
    with pytest.raises(ValueError, match="without held-center"):
        fit_fold(b, "generated", fold, training)


def test_changed_scene_identity_and_duplicate_branch_are_rejected():
    b = bank()
    fold, training = training_fold(b)
    training["train_0"][0]["scene_id"] = "test_0"
    with pytest.raises(ValueError, match="task identity"):
        fit_fold(b, "generated", fold, training)
    measured = result(b, [1] * 5, [1] * 5)
    measured["rows"].append(deepcopy(measured["rows"][0]))
    with pytest.raises(ValueError, match="exactly one"):
        select_constant(arm_view(b, "generated")[0], [measured])


def test_declared_design_rejects_changed_folds_or_learner():
    tasks = [
        dict(task_id=str(i), center_xy_m=[x, -0.1], length_m=length, underside_m=height)
        for i, (x, length, height) in enumerate(
            (x, length, height)
            for x in (1.7, 2.25, 2.8)
            for length in (0.1, 0.75)
            for height in (1.24, 1.30)
        )
    ]
    registry = dict(path="/registry.json", sha256="test")
    study = dict(
        schema="motion2scene_extension_equivalent_teaching_v1",
        registry=registry,
        folds=longitudinal_folds(tasks),
        arms=["generated", "authored"],
        l2=10.0,
        allow_measured_tie_initialization=True,
        replay="uniform",
        parameter_search=False,
        fitting_models=6,
        assigned_future_policy_episodes=24,
        maximum_future_policy_physics_steps=28608,
        policy_evaluation_seed=97001,
    )
    validate_design(study, tasks, registry)
    for key, changed in (
        ("l2", 1.0),
        ("policy_evaluation_seed", 97002),
        ("folds", study["folds"][:2]),
    ):
        with pytest.raises(ValueError, match="allocation changed"):
            validate_design({**study, key: changed}, tasks, registry)


def test_watcher_does_not_fit_before_physical_panel_completes(monkeypatch, tmp_path):
    prepared_ref = dict(path=str(tmp_path / "panel" / "prepared.json"), sha256="test")
    prepared = dict(
        tasks=[dict(collection=dict(path=str(tmp_path / "collection" / "manifest.json")))]
    )
    monkeypatch.setattr(fitting, "artifact", lambda path: dict(path=str(path)))
    monkeypatch.setattr(
        fitting,
        "read_checked",
        lambda ref: (
            dict(prepared=prepared_ref) if ref["path"].endswith("study.json") else prepared
        ),
    )
    monkeypatch.setattr(fitting, "run", lambda out: pytest.fail("fit before assigned physics"))

    def stop(seconds):
        assert seconds == 30
        raise TimeoutError("verified waiting test")

    monkeypatch.setattr(fitting.time, "sleep", stop)
    with pytest.raises(TimeoutError, match="verified waiting"):
        fitting.watch(tmp_path)
