"""Keep the learner fixed while changing privileged continuation targets."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_information_policy import passage_first_information_teacher
from motion2scene_information_regret import expand_group_targets, fit_explicit_regret, group_regret

from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_imitation import (
    physical_regret,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (
    fit_timed_schedule_policy,
    schedule_layout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (
    expected_feature_names,
)


def bank(entries):
    ids = tuple(["neutral"] + [f"option_{i}" for i in range(1, len(entries))])
    return SimpleNamespace(
        online_verified=True,
        option_ids=ids,
        request=dict(
            max_entries_per_episode=1,
            options=[dict(option_id=ids[i], entry_tick=t) for i, t in enumerate(entries) if i],
        ),
    )


def observations(b, encounters):
    phases, legal, _ = schedule_layout(b)
    names = expected_feature_names(len(b.option_ids))
    x = np.zeros((encounters, len(phases), len(names)))
    x[:, :, names.index("phase_s")] = phases[None] / 50
    x[:, :, names.index("active_option_0")] = 1
    for a in range(len(b.option_ids)):
        x[:, :, names.index(f"option_{a}_legal")] = legal[None, :, a]
    return x, names, np.broadcast_to(legal, x.shape[:2] + (len(b.option_ids),)).copy()


def choice(model, x, tick, legal):
    k = list(model["phase_ticks"]).index(tick)
    scores = ((x - model["mean"][k]) / model["std"][k]) @ model["weights"][k] + model["bias"][k]
    return int(np.where(legal, scores, -np.inf).argmax())


def test_singleton_targets_and_ridge_coefficients_match_original_learner():
    b = bank([None, 15, 15])
    x, names, legal = observations(b, 4)
    x[:, 0, 0] = np.arange(4)
    passed = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0]], bool)
    times = np.where(passed, [[4, 0, 3], [0, 2, 0], [3, 4, 2], [3, 0, 0]], np.nan)
    admitted = np.ones_like(passed)
    rows = passage_first_information_teacher(
        [[str(i)] for i in range(4)], [15], [None, 15, 15], passed, times, admitted
    )
    targets, mask = expand_group_targets(rows, x, [15], 3)
    expected, _ = physical_regret(passed, times, admitted, legal[:, 0])
    np.testing.assert_array_equal(targets[:, 0], expected)
    model = fit_explicit_regret(b, x[:, 0], names, np.full(4, 15), targets[:, 0], mask[:, 0], l2=10)
    original, _ = fit_timed_schedule_policy(
        b, x[:, 0], names, np.full(4, 15), passed, times, admitted, legal[:, 0], l2=10
    )
    for key in ("mean", "std", "weights", "bias"):
        np.testing.assert_allclose(model[key], original[key], rtol=0, atol=1e-13)


def test_same_ridge_learns_two_passages_instead_of_oracle_wait_one():
    b = bank([None, 15, 50, 50, 50])
    x, names, legal = observations(b, 3)
    passed = np.array([[0, 1, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 1]], bool)
    times = np.where(passed, [[0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [0, 0, 0, 0, 1]], np.nan)
    rows = passage_first_information_teacher(
        [["same", "same"]] * 3,
        [15, 50],
        [None, 15, 50, 50, 50],
        passed,
        times,
        np.ones_like(passed),
    )
    target, mask = expand_group_targets(rows, x, [15, 50], 5)
    # Scene-wise WAIT can choose a different passing late schedule in every scene.
    old_pass = np.repeat(passed[:, None], 2, axis=1)
    old_times = np.repeat(times[:, None], 2, axis=1)
    old_pass[:, 0, 0], old_times[:, 0, 0] = True, 1
    old_regret, _ = physical_regret(
        old_pass.reshape(-1, 5),
        old_times.reshape(-1, 5),
        legal.reshape(-1, 5),
        legal.reshape(-1, 5),
    )
    ticks = np.tile([15, 50], 3)
    old = fit_explicit_regret(
        b, x.reshape(-1, len(names)), names, ticks, old_regret, legal.reshape(-1, 5)
    )
    new = fit_explicit_regret(
        b, x.reshape(-1, len(names)), names, ticks, target.reshape(-1, 5), mask.reshape(-1, 5)
    )
    old_first = choice(old, x[0, 0], 15, legal[0, 0])
    old_late = choice(old, x[0, 1], 50, legal[0, 1])
    new_first = choice(new, x[0, 0], 15, legal[0, 0])
    assert old_first == 0 and passed[:, old_late].sum() == 1
    assert new_first == 1 and passed[:, new_first].sum() == 2


def test_time_cannot_override_the_maximal_passage_target():
    row = dict(
        complete=True,
        has_passing_supervision=True,
        maximum_causal_passages=2,
        action_values={
            0: dict(passage_count=2, successful_time_sum_s=1000),
            1: dict(passage_count=2, successful_time_sum_s=1),
            2: dict(passage_count=1, successful_time_sum_s=0.001),
        },
    )
    regret = group_regret(row)
    assert regret[1] == 0 and regret[0] < regret[2]
    row["action_values"][1]["successful_time_sum_s"] = 0
    assert group_regret(row)[0] < group_regret(row)[2]


def test_group_targets_reject_different_student_information():
    b = bank([None, 15])
    x, _, _ = observations(b, 2)
    x[1, 0, 0] = 1
    rows = passage_first_information_teacher(
        [["same"]] * 2,
        [15],
        [None, 15],
        np.ones((2, 2), bool),
        np.array([[2, 3], [2, 3]]),
        np.ones((2, 2), bool),
    )
    with pytest.raises(ValueError, match="identical student inputs"):
        expand_group_targets(rows, x, [15], 2)
