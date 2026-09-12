"""Capability bounds must be exact and cannot call unobserved branches failures."""

from itertools import product
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_extension_coverage import summarize, union_bounds

IDS = ["neutral", "generated_0", "generated_1", "authored_0", "authored_1"]


def condition(states, task_id="task", times=None):
    return dict(
        task_id=task_id,
        states=list(states),
        times_s=times if times is not None else [3.0 if s == 1 else None for s in states],
    )


def test_all_unknown_stays_in_denominator_and_is_not_zero_capability():
    result = summarize(IDS, [condition([-1] * 5)])
    assert result["assigned_tasks"] == 1 and result["unknown_outcomes"] == 5
    assert result["arms"]["generated"]["capability_bounds"] == [0, 1]
    assert result["neutral_capability_bounds"] == [0, 1]
    assert result["mean_paired_best_time_difference_s"] is None


def test_observed_success_proves_union_despite_unknown_other_branches():
    assert union_bounds([0, -1, 1]) == [1, 1]
    result = summarize(IDS, [condition([0, 1, -1, 0, 0])])
    assert result["generated_only_bounds"] == [1, 1]
    assert result["arms"]["generated"]["incremental_coverage_bounds"] == [1, 1]
    assert result["mutually_solvable_complete_tasks"] == 0


def test_shared_neutral_success_prevents_false_exclusive_coverage():
    result = summarize(IDS, [condition([1, 1, -1, 0, -1])])
    assert result["generated_only_bounds"] == result["authored_only_bounds"] == [0, 0]
    assert all(a["capability_bounds"] == [1, 1] for a in result["arms"].values())


def test_bounds_match_exhaustive_completion_of_every_five_branch_pattern():
    for states in product((-1, 0, 1), repeat=5):
        result = summarize(IDS, [condition(states)])
        unknown = [i for i, s in enumerate(states) if s == -1]
        metrics = []
        for replacements in product((0, 1), repeat=len(unknown)):
            completed = list(states)
            for i, value in zip(unknown, replacements, strict=True):
                completed[i] = value
            neutral = completed[0]
            g, a = max(completed[:3]), max(completed[0], *completed[3:])
            metrics.append([neutral, g, a, g - neutral, a - neutral, int(g > a), int(a > g)])
        expected = [[min(v[i] for v in metrics), max(v[i] for v in metrics)] for i in range(7)]
        actual = [
            result["neutral_capability_bounds"],
            result["arms"]["generated"]["capability_bounds"],
            result["arms"]["authored"]["capability_bounds"],
            result["arms"]["generated"]["incremental_coverage_bounds"],
            result["arms"]["authored"]["incremental_coverage_bounds"],
            result["generated_only_bounds"],
            result["authored_only_bounds"],
        ]
        assert actual == expected


def test_oracle_time_comparison_requires_complete_mutually_solvable_tasks():
    result = summarize(
        IDS,
        [
            condition([0, 1, 0, 1, 0], "a", [None, 3, None, 4, None]),
            condition([0, 1, -1, 1, 0], "b", [None, 2, None, 6, None]),
            condition([0, 1, 0, 0, 0], "c", [None, 2, None, None, None]),
        ],
    )
    assert result["assigned_tasks"] == 3
    assert result["mutually_solvable_complete_tasks"] == 1
    assert result["mean_paired_best_time_difference_s"] == -1


@pytest.mark.parametrize(
    "states,times", [([0] * 5, [None, 2, None, None, None]), ([0, 1, 0, 0, 0], [None] * 5)]
)
def test_failed_or_missing_costs_are_not_used_as_success_times(states, times):
    with pytest.raises(ValueError):
        summarize(IDS, [condition(states, times=times)])
