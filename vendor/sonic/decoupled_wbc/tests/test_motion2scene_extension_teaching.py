"""Restricted banks must not inherit the other arm's continuation supervision."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_extension_teaching import (
    arm_view,
    choose_arm_schedule,
    longitudinal_folds,
    outcomes_from_audited_group,
    project_observation,
    restricted_teacher,
)
from motion2scene_tune_schedule_learner import arrays

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (
    TimedScheduledOutcome,
    fit_timed_schedule_policy,
    schedule_layout,
    timed_schedule_teacher,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (
    expected_feature_names,
    validate_schedule_policy,
)


def bank():
    options = [
        dict(option_id=f"{arm}_00_e{tick}", reference_id=f"{arm}_00", entry_tick=tick)
        for arm in ("generated", "authored")
        for tick in (15, 50)
    ]
    return SimpleNamespace(
        online_verified=True,
        option_ids=("neutral", *[o["option_id"] for o in options]),
        request=dict(
            max_entries_per_episode=1,
            options=options,
            references=[dict(reference_id=r) for r in ("neutral", "generated_00", "authored_00")],
        ),
    )


def observation(b, tick=15, legal=None, active=0):
    phases, masks, _ = schedule_layout(b)
    if legal is None:
        legal = masks[list(phases).index(tick)]
    names = expected_feature_names(len(b.option_ids))
    x = np.zeros(len(names))
    x[:95] = np.arange(95) / 95
    x[95:100] = [tick / 50, active != 0, 0.0, legal[0], legal[1:].any()]
    x[100 : 100 + len(b.option_ids)] = np.eye(len(b.option_ids))[active]
    x[100 + len(b.option_ids) :] = legal
    return names, x, legal


def branches(states, times=None):
    if times is None:
        times = [2.0 if s == 1 else None for s in states]
    return [
        TimedScheduledOutcome(
            branch_id=f"branch_{i}",
            option_index=i,
            physics_seed=1,
            passed=s == 1,
            passage_time_s=t,
            admitted=s != -1,
            prefix_hash_by_tick={15: "prefix15", 50: "prefix50"},
            physics_steps=1192,
        )
        for i, (s, t) in enumerate(zip(states, times, strict=True))
    ]


def teacher(b, arm, outcomes, tick=15):
    return restricted_teacher(b, arm, outcomes, tick, f"prefix{tick}", 1, *observation(b, tick))


def test_wait_cannot_borrow_the_other_arms_passing_late_motion():
    b = bank()
    outcomes = branches([0, 1, 0, 0, 1], [None, 2, None, None, 1])
    full = timed_schedule_teacher(b, outcomes, 15, "prefix15", 1, observation(b)[2])
    assert full["teacher_action"] == 0  # Best full-bank WAIT uses authored late.
    generated = teacher(b, "generated", outcomes)
    authored = teacher(b, "authored", outcomes)
    assert generated["teacher_action"] == 1
    assert generated["pass_labels"][0] is False
    assert generated["expected_continuation_counts"][0] == 2
    assert authored["teacher_action"] == 0
    assert authored["continuation_branch_ids"][0] == "branch_4"


def test_missing_own_continuation_withholds_but_missing_other_arm_does_not():
    b = bank()
    outcomes = branches([0, 1, 0, -1, -1])
    assert teacher(b, "generated", outcomes)["complete_legal_action_table"]
    outcomes[2] = replace(outcomes[2], admitted=False)
    target = teacher(b, "generated", outcomes)
    assert not target["complete_legal_action_table"]
    assert target["teacher_action"] is None


def test_unmatched_own_prefix_withholds_instead_of_using_scene_wise_outcome():
    b = bank()
    outcomes = branches([0, 1, 1, 1, 1])
    outcomes[2] = replace(outcomes[2], prefix_hash_by_tick={15: "different", 50: "prefix50"})
    target = teacher(b, "generated", outcomes)
    assert not target["complete_legal_action_table"]
    assert target["teacher_action"] is None


def test_projection_preserves_sensor_state_and_updates_aggregate_legality():
    b = bank()
    view, indices = arm_view(b, "generated")
    names, x, legal = observation(b, legal=np.array([1, 0, 0, 1, 0], bool))
    projected_names, projected, projected_legal, active = project_observation(
        b, indices, names, x, legal, 0
    )
    np.testing.assert_array_equal(projected[:98], x[:98])
    assert x[99] == 1 and projected[99] == 0
    assert projected_names == expected_feature_names(len(view.option_ids))
    assert active == 0 and projected_legal.tolist() == [True, False, False]
    assert len(b.request["options"]) == 4  # No mutation of the qualified registry.


def test_common_ridge_fits_projected_targets_and_runtime_returns_original_identity():
    b = bank()
    outcomes = branches([1, 1, 1, 1, 1], [4, 2, 3, 0.5, 0.4])
    rows = [teacher(b, "generated", outcomes, tick) for tick in (15, 50)]
    view, _ = arm_view(b, "generated")
    model, _ = fit_timed_schedule_policy(
        view, feature_names=expected_feature_names(3), l2=10, **arrays(rows)
    )
    validate_schedule_policy(model, view)
    selected, record = choose_arm_schedule(b, "generated", model, *observation(b), 0, 15, None)
    assert selected == "generated_00_e15"
    assert record["original_option_indices"] == [0, 1, 2]
    names, x, legal = observation(b, tick=265, legal=np.array([1, 1, 0, 0, 0], bool), active=1)
    selected, _ = choose_arm_schedule(b, "generated", model, names, x, legal, 1, 265, 0)
    assert selected == "neutral"


def test_other_arm_active_motion_and_inconsistent_recording_are_rejected():
    b = bank()
    _, indices = arm_view(b, "generated")
    names, x, legal = observation(b)
    with pytest.raises(ValueError, match="admissible arm"):
        project_observation(b, indices, names, x, legal, active=3)
    x[-1] = 1
    with pytest.raises(ValueError, match="disagree"):
        project_observation(b, indices, names, x, legal, active=0)


def test_each_arm_must_support_every_original_decision_phase():
    b = bank()
    b.request["options"][1]["entry_tick"] = 15
    with pytest.raises(ValueError, match="every original phase"):
        arm_view(b, "generated")


def test_folds_hold_out_whole_longitudinal_centers_before_outcomes():
    tasks = [
        dict(task_id=str(i), center_xy_m=[x, -0.1], length_m=length, underside_m=height)
        for i, (x, length, height) in enumerate(
            (x, length, height)
            for x in (1.7, 2.25, 2.8)
            for length in (0.1, 0.75)
            for height in (1.24, 1.30)
        )
    ]
    folds = longitudinal_folds(tasks)
    assert len(folds) == 3
    for fold in folds:
        assert len(fold["training_task_ids"]) == 8
        assert len(fold["evaluation_task_ids"]) == 4
        assert not set(fold["training_task_ids"]) & set(fold["evaluation_task_ids"])
    assert sorted(t for f in folds for t in f["evaluation_task_ids"]) == sorted(
        t["task_id"] for t in tasks
    )
    tasks[0]["underside_m"] = 1.30
    with pytest.raises(ValueError, match="center-by-length-by-height"):
        longitudinal_folds(tasks)


def test_branch_reconstruction_uses_only_audited_matched_history():
    b = bank()
    original = branches([0, 1, 0, 0, 1])
    _, _, entries = schedule_layout(b)
    rows = [
        dict(
            forced_option_id=b.option_ids[o.option_index],
            cell_id=o.branch_id,
            physics_steps=o.physics_steps,
            outcome=dict(task_outcome="pass" if o.passed else "failure"),
            **{"pass": o.passed},
            costs=dict(passage_time_s=o.passage_time_s),
        )
        for o in original
    ]
    group = dict(
        option_ids=list(b.option_ids),
        physics_seed=1,
        targets=[
            dict(phase_tick=tick, available=True, recorded_history_sha256=f"prefix{tick}")
            for tick in (15, 50)
        ],
        branch_assessments=[
            dict(
                option_id=r["forced_option_id"],
                cell_id=r["cell_id"],
                physics_steps=r["physics_steps"],
                task_outcome=r["outcome"]["task_outcome"],
            )
            for r in rows
        ],
        prefix_comparisons=[
            dict(phase_tick=tick, option_id=b.option_ids[i], matched=True)
            for tick in (15, 50)
            for i, entry in entries.items()
            if entry is None or entry >= tick
        ],
    )
    recovered = outcomes_from_audited_group(b, group, dict(rows=rows))
    assert teacher(b, "generated", recovered) == teacher(b, "generated", original)
    # A physically successful branch is not usable at an unmatched decision prefix.
    group["prefix_comparisons"][1]["matched"] = False
    recovered = outcomes_from_audited_group(b, group, dict(rows=rows))
    assert teacher(b, "generated", recovered)["teacher_action"] is None
    rows[1]["outcome"]["task_outcome"] = "failure"
    with pytest.raises(ValueError, match="physical labels differ"):
        outcomes_from_audited_group(b, group, dict(rows=rows))
