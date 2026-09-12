import copy

import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_coverage import (
    ARMS,
    COURSES,
    SINGLES,
    coverage_selection,
)


def queues():
    rows = [
        dict(candidate_id=f"candidate{i}", stratum=stratum, positive_option_id="arbitrary")
        for i, stratum in enumerate(("sustained", "sustained", *SINGLES, *COURSES))
    ]
    return {arm: copy.deepcopy(rows) for arm in ARMS}


def test_course_precedes_repeated_single_and_rotation_is_seed_only():
    first = coverage_selection(queues(), 93201)
    second = coverage_selection(queues(), 93202)
    assert first["selected_strata_in_order"] == [*SINGLES, "short_then_short"]
    assert second["selected_strata_in_order"] == [*SINGLES, "short_then_sustained"]
    assert first["selected"]["uniform"][1]["original_queue_rank"] == 0
    assert first["omitted_common_strata"] == ["short_then_sustained"]


def test_missing_early_stratum_is_conditioned_out_for_all_arms_and_reported():
    data = queues()
    for arm in ("analytic_contrast", "observation_curriculum"):
        data[arm] = [r for r in data[arm] if r["stratum"] != "early_constraint"]
    result = coverage_selection(data, 93203)
    assert result["selected_strata_in_order"] == ["short", "sustained", *COURSES]
    assert result["excluded_from_common_support_by_arm"]["uniform"] == ["early_constraint"]
    assert result["unsupported_common_strata"] == ["early_constraint"]


def test_no_sensor_outcomes_or_positive_witness_preference_and_no_hidden_refill():
    data = queues()
    for rows in data.values():
        for row in rows:
            row["physical_passage"] = True
            row["observation_availability"] = "irrelevant"
            row["positive_option_id"] = "changed_witness"
    result = coverage_selection(data, 93201)
    assert [r["candidate_id"] for r in result["selected"]["uniform"]] == [
        r["candidate_id"] for r in coverage_selection(queues(), 93201)["selected"]["uniform"]
    ]
    for arm in data:
        data[arm] = [r for r in data[arm] if r["stratum"] == "sustained"]
    result = coverage_selection(data, 93201)
    assert result["planned_group_shortfall"] == 3
    assert len(result["selected"]["uniform"]) == 1


def test_different_analytic_replay_scene_is_rejected():
    data = queues()
    data["observation_curriculum"][0]["candidate_id"] = "different_scene"
    with pytest.raises(ValueError, match="identical geometric"):
        coverage_selection(data, 93201)
