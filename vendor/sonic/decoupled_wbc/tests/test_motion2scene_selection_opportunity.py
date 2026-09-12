"""Passage first does not conflate a fast failed policy with efficient coverage."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_selection_opportunity import summarize


def example():
    return ["slow", "fast"], [
        dict(task_id="a", passed=[True, True], times_s=[5, 2]),
        dict(task_id="b", passed=[True, False], times_s=[5, None]),
    ]


def test_fast_policy_loses_passage_while_oracle_exposes_cost_opportunity():
    ids, tasks = example()
    result = summarize(ids, tasks, {"learner": {"a": 1, "b": 1}})
    assert result["best_fixed"]["option_id"] == "slow"
    assert result["privileged_fastest_passing"]["passages"] == 2
    assert result["privileged_fastest_passing"]["mean_paired_time_minus_best_fixed_s"] == -1.5
    learner = result["selections"]["learner"]
    assert learner["passage_difference_from_best_fixed"] == -1
    assert learner["mutually_successful_with_best_fixed"] == 1
    assert learner["mean_paired_time_minus_best_fixed_s"] == -3


def test_constant_is_selected_by_success_before_time():
    ids, tasks = example()
    tasks[0]["times_s"][0] = 100
    assert summarize(ids, tasks, {})["best_fixed"]["option_id"] == "slow"


def test_equal_passage_count_uses_time_then_declared_order():
    ids, tasks = example()
    tasks[1].update(passed=[True, True], times_s=[5, 2])
    assert summarize(ids, tasks, {})["best_fixed"]["option_id"] == "fast"
    for t in tasks:
        t["times_s"] = [2, 2]
    assert summarize(ids, tasks, {})["best_fixed"]["option_id"] == "slow"


def test_unsolvable_task_stays_in_denominator():
    ids, tasks = example()
    tasks.append(dict(task_id="c", passed=[False, False], times_s=[None, None]))
    result = summarize(ids, tasks, {})
    assert result["assigned_tasks"] == 3 and result["bank_capability"] == 2
    assert result["privileged_fastest_passing"]["assigned_tasks"] == 3
    assert result["privileged_fastest_passing"]["selected_schedule_counts"]["bank_unsolvable"] == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t[0]["times_s"].__setitem__(0, float("nan")),
        lambda t: t[1]["times_s"].__setitem__(1, 2),
        lambda t: t[0]["passed"].__setitem__(0, None),
        lambda t: t.append(t[0]),
    ],
)
def test_incomplete_or_invalid_measurements_are_rejected(change):
    ids, tasks = example()
    change(tasks)
    with pytest.raises(ValueError):
        summarize(ids, tasks, {})


@pytest.mark.parametrize("choices", [{"a": 0}, {"a": 0, "b": 7}, {"a": None, "b": 0}])
def test_missing_or_unsupported_policy_choices_are_rejected(choices):
    with pytest.raises(ValueError):
        summarize(*example(), {"learner": choices})
