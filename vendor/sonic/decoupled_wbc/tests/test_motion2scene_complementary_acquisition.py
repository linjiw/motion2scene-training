"""Controls against false complementarity and ineffective targeted selection."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_complementary_acquisition import (
    common_response,
    complementary_queue,
    paired_selection,
)

OPTIONS = ["neutral", "prior", "sustained"]


def task(outcomes, times):
    return dict(outcomes=dict(zip(OPTIONS, outcomes)), passage_time_s=dict(zip(OPTIONS, times)))


def candidates():
    return [
        dict(
            candidate_id=f"c{i}",
            candidate_index=i,
            stratum="early_constraint" if i == 0 else "short_then_sustained",
            assigned_positive_option_id="sustained",
            future_physics_seed=12,
            geometry_key=f"g{i}",
        )
        for i in range(3)
    ]


def geometry():
    minimum = np.array([[-0.1, -0.1, 0.05], [-0.1, -0.1, 0.05], [-0.1, -0.1, 0.04]])
    counts = np.array([[1, 1, 81]] * 3)
    negative = np.array([[-0.5, 0.1, 0.1], [-0.5, 0.1, 0.1], [-0.5, -0.025, 0.1]])
    return minimum, counts, negative


def test_all_failing_and_unknown_do_not_manufacture_complementarity():
    rows = [
        task(["failure", "pass", "pass"], [None, 4.0, 5.0]),
        task(["failure", "failure", "failure"], [None] * 3),
        task(["unknown", "failure", "pass"], [None, None, 3.0]),
    ]
    result = common_response(rows, OPTIONS)
    assert result["assigned_tasks"] == 3
    assert result["common_passing_schedules"] == ["prior", "sustained"]
    assert result["selected_fixed_schedule"] == "prior"
    assert result["all_failing_task_indices"] == [1]
    assert result["incomplete_task_indices"] == [2]


def test_no_solvable_task_has_no_selected_response():
    result = common_response([task(["failure"] * 3, [None] * 3)], OPTIONS)
    assert result["selected_fixed_schedule"] is None
    assert result["common_passing_schedules"] == []


def test_genuine_disjoint_passing_sets_are_not_dominated():
    rows = [
        task(["failure", "pass", "failure"], [None, 4.0, None]),
        task(["failure", "failure", "pass"], [None, None, 5.0]),
    ]
    assert common_response(rows, OPTIONS)["selected_fixed_schedule"] is None


def test_time_ties_use_registry_order():
    row = task(["failure", "pass", "pass"], [None, 4.0, 4.0])
    assert common_response([row], OPTIONS)["selected_fixed_schedule"] == "prior"
    assert common_response([row], list(reversed(OPTIONS)))["selected_fixed_schedule"] == "sustained"


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -1, True])
def test_common_passing_response_requires_real_time(invalid):
    with pytest.raises(ValueError, match="passage time"):
        common_response([task(["failure", "pass", "pass"], [None, invalid, 4.0])], OPTIONS)


def test_enumerates_target_pair_even_when_original_best_pair_has_other_negative():
    rows = complementary_queue(candidates(), OPTIONS, *geometry(), "prior")
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "c2"
    assert rows[0]["negative_option_id"] == "prior"
    assert rows[0]["positive_option_id"] == "sustained"
    assert rows[0]["geometric_slack_m"] == pytest.approx(0.015)
    assert rows[0]["physical_passage"] is None


def test_paired_continuation_retains_ordinary_and_changes_only_second_selection():
    result = paired_selection(
        candidates(), OPTIONS, *geometry(), "prior", [], proposals_per_stratum=2
    )
    first, second = result["rounds"]
    assert first["ordinary"] == first["complementary"]
    assert first["ordinary"]["candidate_id"] == "c0"
    assert second["ordinary"]["candidate_id"] == "c1"
    assert second["complementary"]["candidate_id"] == "c2"
    assert second["distinct_intervention"]
    assert result["assigned_proposal_budget"] == 4
    assert result["proposals_considered"] == 3


def test_exclusion_consumes_draw_budget_without_refill():
    result = paired_selection(
        candidates(), OPTIONS, *geometry(), "prior", ["g1"], proposals_per_stratum=1
    )
    second = result["rounds"][1]
    assert second["proposed_candidate_ids"] == ["c1"]
    assert second["ordinary"] is None
    assert second["complementary"] is None
    assert not result["matched_physical_pair_ready"]


def test_no_target_uses_explicit_inactive_ordinary_fallback():
    result = paired_selection(
        candidates(), OPTIONS, *geometry(), "prior", ["g2"], proposals_per_stratum=2
    )
    second = result["rounds"][1]
    assert second["ordinary"] == second["complementary"]
    assert second["ordinary_fallback"]
    assert not second["targeted_selection"]
    assert not second["distinct_intervention"]


def test_incomplete_robust_positive_is_rejected():
    minimum, counts, negative = geometry()
    counts[2, 2] = 1
    with pytest.raises(ValueError, match="complete positives"):
        complementary_queue(candidates(), OPTIONS, minimum, counts, negative, "prior")


def test_unknown_target_schedule_is_rejected():
    with pytest.raises(ValueError, match="fixed bank"):
        complementary_queue(candidates(), OPTIONS, *geometry(), "new_motion")


def test_duplicate_candidate_identity_is_rejected():
    rows = candidates()
    rows[2]["candidate_id"] = rows[1]["candidate_id"]
    with pytest.raises(ValueError, match="unique candidate"):
        paired_selection(rows, OPTIONS, *geometry(), "prior", [])
