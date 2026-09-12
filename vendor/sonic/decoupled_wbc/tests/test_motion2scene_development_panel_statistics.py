"""Matched capability, selection and cost must remain separate under missing data."""

from itertools import product
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_development_panel_statistics import (
    construction_comparisons,
    measure,
    paired_time,
    summarize,
)


def row(policy, scene, status, cost=3):
    return dict(
        policy_id=policy,
        scene_id=scene,
        physics_seed=8732,
        status=status,
        time_s=cost if status == "pass" else None,
    )


def test_capability_and_selection_have_separate_denominators():
    rows = [
        row("fixed_a", "a", "pass"),
        row("fixed_b", "a", "failure"),
        row("script", "a", "failure"),
        row("fixed_a", "b", "failure"),
        row("fixed_b", "b", "failure"),
        row("script", "b", "failure"),
    ]
    result = summarize(rows, ["fixed_a", "fixed_b"])
    assert result["capability_count_lower"] == result["capability_count_upper"] == 1
    script = result["policies"]["script"]
    assert script["assigned"] == 2 and script["failure"] == 2
    assert (
        script["selection_failures"] == 1 and script["complete_matched_capability_conditions"] == 2
    )
    assert script["measured_capability_minus_policy_passages"] == 1


def test_policy_only_success_is_visible_not_clipped_or_rejected():
    result = summarize([row("fixed", "a", "failure"), row("script", "a", "pass")], ["fixed"])
    script = result["policies"]["script"]
    assert script["selection_failures"] == 0 and script["policy_only_successes"] == 1
    assert script["measured_capability_minus_policy_passages"] == -1


def test_missing_and_unexecuted_branches_keep_capability_uncertain():
    result = summarize(
        [
            row("fixed_a", "a", "technical_missing"),
            row("fixed_b", "a", "not_run"),
            row("script", "a", "failure"),
        ],
        ["fixed_a", "fixed_b"],
    )
    assert not result["complete"]
    assert (result["capability_count_lower"], result["capability_count_upper"]) == (0, 1)
    assert result["policies"]["script"]["measured_capability_minus_policy_passages"] is None
    assert result["policies"]["fixed_a"]["assigned"] == 1
    assert result["policies"]["fixed_a"]["complete_outcome_mean"] is None


def test_capability_can_be_known_while_best_time_remains_unknown():
    result = summarize(
        [
            row("fixed_a", "a", "pass", 2),
            row("fixed_b", "a", "not_run"),
            row("script", "a", "failure"),
        ],
        ["fixed_a", "fixed_b"],
    )
    assert result["capability_count_lower"] == result["capability_count_upper"] == 1
    assert result["capability"][0]["best_passing_time_s"] is None


def test_cost_is_paired_not_each_methods_own_success_mean():
    a = {"a": row("a", "a", "pass", 2), "b": row("a", "b", "pass", 100)}
    b = {"a": row("b", "a", "pass", 3), "b": row("b", "b", "failure")}
    assert paired_time(a, b) == dict(mutually_successful_contexts=1, mean_time_difference_s=-1)
    with pytest.raises(ValueError, match="identical"):
        paired_time(a, {"a": b["a"]})


def test_no_cost_for_failure_or_unknown_and_no_duplicate_or_missing_cells():
    rows = [row("fixed", "a", "pass"), row("script", "a", "pass")]
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows + rows[:1], ["fixed"])
    with pytest.raises(ValueError, match="rectangular"):
        summarize(rows + [row("script", "b", "pass")], ["fixed"])
    rows[0]["status"] = "failure"
    with pytest.raises(ValueError, match="null time"):
        summarize(rows, ["fixed"])


def test_measured_partial_failure_is_not_a_missing_outcome():
    value = measure(
        dict(
            outcome=dict(task_outcome="failure", physics_steps_recorded=756),
            **{"pass": False},
            costs=dict(passage_time_s=None),
            measurement_admitted=False,
        )
    )
    assert value["status"] == "failure" and value["recorded_physics_steps"] == 756


def test_capability_bounds_match_all_completions_of_small_missing_tables():
    for statuses in product(("pass", "failure", "not_run"), repeat=4):
        rows = [row("fixed_" + str(i // 2), str(i % 2), state) for i, state in enumerate(statuses)]
        rows += [row("script", str(i), "failure") for i in range(2)]
        result = summarize(rows, ["fixed_0", "fixed_1"])
        missing = [i for i, s in enumerate(statuses) if s == "not_run"]
        counts = []
        for values in product(("pass", "failure"), repeat=len(missing)):
            completed = list(statuses)
            for i, value in zip(missing, values, strict=True):
                completed[i] = value
            counts.append(
                sum(completed[i] == "pass" or completed[i + 2] == "pass" for i in range(2))
            )
        assert result["capability_count_lower"] == min(counts)
        assert result["capability_count_upper"] == max(counts)


def test_construction_differences_are_paired_by_training_seed():
    arms = (
        "uniform",
        "target_only",
        "analytic_contrast",
        "observation_curriculum",
        "reference_contrast",
    )
    models = [dict(arm=a, seed=s, run_id=f"{a}_{s}") for a in arms for s in (93201, 93202, 93203)]
    rows = [row(m["run_id"], "a", "failure" if m["arm"] == "uniform" else "pass") for m in models]
    result = construction_comparisons(rows, models)
    values = result["analytic_contrast_minus_uniform"]["corpora"]
    assert [r["passage_count_difference"] for r in values] == [1, 1, 1]
    assert all(r["mutually_successful_contexts"] == 0 for r in values)
    assert (
        result["observation_curriculum_minus_analytic_contrast"][
            "descriptive_passage_difference_variation"
        ]["mean"]
        == 0
    )
    rows[0]["status"] = "technical_missing"
    result = construction_comparisons(rows, models)
    assert (
        result["analytic_contrast_minus_uniform"]["descriptive_passage_difference_variation"]
        is None
    )
