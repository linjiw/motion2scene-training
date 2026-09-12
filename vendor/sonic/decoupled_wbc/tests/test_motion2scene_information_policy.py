"""The finite teacher optimizes an executable shared policy, not scene-wise WAIT."""

from itertools import product
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_information_policy import (
    evaluate_finite_policy,
    passage_first_information_teacher,
)


def teach(keys, phases, entries, passed, times=None, admitted=None):
    passed = np.asarray(passed, bool)
    times = np.where(passed, 3.0, np.nan) if times is None else np.asarray(times, float)
    admitted = np.ones_like(passed) if admitted is None else admitted
    return passage_first_information_teacher(keys, phases, entries, passed, times, admitted)


def test_incompatible_wait_has_lower_count_than_common_early_commitment():
    # Each scene has a faster late solution, but no later observation can tell
    # the three scenes apart. The early option passes two; any late option one.
    passed = [[0, 1, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 1]]
    times = [
        [np.nan, 2, 1, np.nan, np.nan],
        [np.nan, 2, np.nan, 1, np.nan],
        [np.nan, np.nan, np.nan, np.nan, 1],
    ]
    rows = teach([["same", "same"]] * 3, [15, 50], [None, 15, 50, 50, 50], passed, times)
    assert rows[0]["universal_success_actions"] == []
    assert rows[0]["teacher_action"] == 1
    assert rows[0]["maximum_causal_passages"] == 2
    assert rows[0]["action_values"][0]["passage_count"] == 1
    assert rows[0]["information_gap"] == 1


def test_future_information_makes_wait_realize_all_three_passages():
    passed = [[0, 1, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 1]]
    rows = teach(
        [["same", key] for key in ("a", "b", "c")], [15, 50], [None, 15, 50, 50, 50], passed
    )
    assert rows[0]["teacher_action"] == 0
    assert rows[0]["maximum_causal_passages"] == 3
    assert rows[0]["selected_schedule_by_encounter"] == {0: 2, 1: 3, 2: 4}


def test_more_successes_beat_arbitrarily_faster_fewer_successes():
    rows = teach(
        [["same"], ["same"]], [15], [None, 15], [[1, 1], [1, 0]], [[100, 0.01], [100, np.nan]]
    )
    assert rows[0]["teacher_action"] == 0
    assert rows[0]["maximum_causal_passages"] == 2


def test_equal_success_count_uses_successful_time_and_keeps_tied_targets():
    passed = [[0, 1, 0, 0], [0, 0, 1, 1]]
    rows = teach(
        [["same"], ["same"]],
        [15],
        [None, 15, 15, 15],
        passed,
        [[np.nan, 3, np.nan, np.nan], [np.nan, np.nan, 2, 2]],
    )
    assert rows[0]["passage_optimal_actions"] == [1, 2, 3]
    assert rows[0]["lexicographic_optimal_actions"] == [2, 3]
    assert rows[0]["teacher_action"] == 2


def test_all_failed_group_has_no_passing_supervision_or_fake_success_time():
    row = teach([["same"]], [15], [None, 15], [[0, 0]])[0]
    assert row["maximum_causal_passages"] == 0 and not row["has_passing_supervision"]
    assert row["action_values"][row["teacher_action"]]["mean_successful_time_s"] is None


def test_unknown_outcomes_withhold_parent_and_child_targets():
    rows = teach(
        [["same", "same"]],
        [15, 50],
        [None, 15, 50],
        [[0, 1, 1]],
        admitted=np.array([[True, True, False]]),
    )
    assert all(not r["complete"] and r["teacher_action"] is None for r in rows)
    assert all(r["maximum_causal_passages"] is None for r in rows)


def test_refining_information_is_required():
    with pytest.raises(ValueError, match="refining"):
        teach([["a", "same"], ["b", "same"]], [15, 50], [None, 15, 50], [[0, 1, 0]] * 2)


def test_dynamic_program_cost_and_count_match_exhaustive_policy_enumeration():
    phases, entries = [15, 50, 70], [None, 15, 15, 50, 70]
    keys = np.array([["root", "a", "a1"], ["root", "a", "a2"], ["root", "b", "b1"]])
    nodes = [(k, key) for k in range(3) for key in sorted(set(keys[:, k]))]
    legal = [[0] + [a for a, tick in enumerate(entries) if tick == phases[k]] for k, _ in nodes]
    rng = np.random.default_rng(20260909)
    for _ in range(50):
        passed = rng.integers(0, 2, size=(3, 5)).astype(bool)
        times = np.where(passed, rng.integers(1, 100, size=(3, 5)) / 10, np.nan)
        best = None
        for actions in product(*legal):
            policy = dict(zip(nodes, actions, strict=True))
            selected = [
                next((policy[k, keys[i, k]] for k in range(3) if policy[k, keys[i, k]]), 0)
                for i in range(3)
            ]
            succeeded = [i for i, a in enumerate(selected) if passed[i, a]]
            value = (-len(succeeded), math.fsum(times[i, selected[i]] for i in succeeded))
            if best is None or value < best:
                best = value
        rows = teach(keys, phases, entries, passed, times)
        actual = evaluate_finite_policy(rows, keys, phases, passed, times)
        assert actual["passage_count"] == -best[0]
        assert actual["successful_time_sum_s"] == pytest.approx(best[1], abs=1e-12)
        assert rows[0]["maximum_causal_passages"] == actual["passage_count"]
