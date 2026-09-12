"""Motion candidates, not entry schedules, are the units of extension yield."""

from itertools import product
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_extension_candidate_coverage import added_bounds, summarize

OPTIONS = [
    dict(option_id=f"{candidate}_e{entry}", reference_id=candidate)
    for candidate in ("generated_00", "generated_01", "authored_00")
    for entry in (15, 50)
]


def condition(states, task_id="task", times=None):
    return dict(
        task_id=task_id,
        states=list(states),
        times_s=times if times is not None else [3.0 if s == 1 else None for s in states],
    )


def test_entries_are_grouped_into_one_candidate_and_unique_requires_other_failures():
    result = summarize(OPTIONS, [condition([0, 1, 0, 0, 0, 1, 1])])
    assert result["arms"]["generated"]["assigned_candidates"] == 2
    assert result["arms"]["generated"]["candidates_with_incremental_coverage_bounds"] == [1, 1]
    candidate = result["candidates"]["generated_00"]
    assert candidate["assigned_branches"] == 2
    assert candidate["unique_within_arm_bounds"] == [1, 1]
    assert candidate["incremental_over_neutral_bounds"] == [1, 1]
    # Passing options from the other arm do not erase this arm's unique contribution.
    assert result["candidates"]["authored_00"]["unique_within_arm_bounds"] == [1, 1]


def test_redundant_candidates_are_not_unique_and_neutral_prevents_incremental_claim():
    result = summarize(
        OPTIONS,
        [condition([0, 1, 0, 1, 0, 0, 0], "a"), condition([1, 1, 1, 0, 0, 0, 0], "b")],
    )
    candidate = result["candidates"]["generated_00"]
    assert candidate["candidate_alone_bounds"] == [2, 2]
    assert candidate["incremental_over_neutral_bounds"] == [1, 1]
    assert candidate["unique_within_arm_bounds"] == [0, 0]


def test_all_missing_outcomes_keep_full_denominators():
    result = summarize(OPTIONS, [condition([-1] * 7)])
    candidate = result["candidates"]["generated_00"]
    assert candidate["assigned_tasks"] == 1 and candidate["unknown_outcomes"] == 2
    assert candidate["candidate_alone_bounds"] == [0, 1]
    assert candidate["unique_within_arm_bounds"] == [0, 1]
    assert candidate["mean_paired_best_time_difference_s"] is None
    assert result["arms"]["generated"]["candidates_with_incremental_coverage_bounds"] == [0, 2]


def test_increment_bounds_match_all_completions_of_every_five_branch_pattern():
    for states in product((-1, 0, 1), repeat=5):
        unknown = [i for i, s in enumerate(states) if s == -1]
        values = []
        for replacements in product((0, 1), repeat=len(unknown)):
            complete = list(states)
            for i, value in zip(unknown, replacements, strict=True):
                complete[i] = value
            values.append(int(max(complete[:2]) > max(complete[2:])))
        assert added_bounds(states[:2], states[2:]) == [min(values), max(values)]


def test_time_is_paired_with_neutral_only_for_complete_mutually_passing_banks():
    result = summarize(
        OPTIONS,
        [
            condition([1, 1, 0, 0, 0, 0, 0], "a", [5, 3, None, None, None, None, None]),
            condition([1, 1, -1, 0, 0, 0, 0], "b", [5, 2, None, None, None, None, None]),
            condition([0, 1, 0, 0, 0, 0, 0], "c"),
        ],
    )
    candidate = result["candidates"]["generated_00"]
    assert candidate["mutually_passing_complete_tasks"] == 1
    assert candidate["mean_paired_best_time_difference_s"] == -2


def test_wrong_reference_assignment_is_rejected():
    options = [dict(o) for o in OPTIONS]
    options[0]["reference_id"] = "authored_00"
    with pytest.raises(ValueError, match="declared candidate"):
        summarize(options, [condition([0] * 7)])


def test_output_is_invariant_to_schedule_permutation_with_corresponding_outcomes():
    states = [0, 1, 0, 0, 1, 1, 0]
    original = summarize(OPTIONS, [condition(states)])
    perm = [5, 4, 3, 2, 1, 0]
    reordered = summarize(
        [OPTIONS[i] for i in perm], [condition([states[0], *[states[i + 1] for i in perm]])]
    )
    for candidate, row in original["candidates"].items():
        other = reordered["candidates"][candidate]
        assert set(row["schedule_ids"]) == set(other["schedule_ids"])
        assert {k: v for k, v in row.items() if k != "schedule_ids"} == {
            k: v for k, v in other.items() if k != "schedule_ids"
        }
