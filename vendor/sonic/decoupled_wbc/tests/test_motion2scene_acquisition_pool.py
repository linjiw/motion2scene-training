"""Proposal draws and constructor distinctions without physical/sensor labels."""

import copy

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_pool import (
    acquisition_queues,
    budget_prefixes,
    draw_pool,
)


def domain():
    return dict(
        single_families_in_order=[
            dict(name="short", station_range=[0.26, 0.77], length_range_m=[0.1, 0.25]),
            dict(name="sustained", station_range=[0.32, 0.68], length_range_m=[0.5, 0.9]),
            dict(name="early_constraint", station_range=[0.15, 0.35], length_range_m=[0.1, 0.25]),
        ],
        course_station_ranges=[[0.24, 0.35], [0.65, 0.77]],
        common_ranges=dict(
            underside_m=[1.14, 1.4],
            width_m=[1.15, 1.55],
            thickness_m=[0.08, 0.16],
            lateral_offset_m=[-0.08, 0.08],
            yaw_offset_rad=[-0.06, 0.06],
        ),
    )


def test_frozen_draws_separate_geometry_targets_and_future_physics_seed():
    options = ("neutral", "short", "sustained")
    first = draw_pool(domain(), 93201, 3, options)
    assert first == draw_pool(domain(), 93201, 3, options)
    assert len(first) == 15 and len({c["candidate_id"] for c in first}) == 15
    assert all(c["future_physics_seed"] == 93201 for c in first)
    assert all(sorted(c["future_branch_order"]) == sorted(options) for c in first)
    second = draw_pool(domain(), 93202, 3, options)
    assert first[0]["beam_specifications"] != second[0]["beam_specifications"]
    reordered = draw_pool(domain(), 93201, 3, ("neutral", "sustained", "short"))
    assert [c["beam_specifications"] for c in first] == [
        c["beam_specifications"] for c in reordered
    ]
    assert [c["stratum"] for c in first[:5]] == [
        "short",
        "sustained",
        "early_constraint",
        "short_then_short",
        "short_then_sustained",
    ]


def test_target_only_is_independent_of_all_negative_outcomes_and_uses_assignment():
    ids = ("neutral", "short", "prior")
    candidates = draw_pool(domain(), 93201, 1, ids)[:2]
    for c in candidates:
        c["assigned_positive_option_id"] = "short"
    minimum = np.array([[-0.06, 0.04, 0.03], [-0.06, 0.012, 0.06]])
    counts = np.full_like(minimum, 81, dtype=int)
    counts[:, 0] = 1
    no_contrast = np.ones_like(minimum) * 0.1
    contrast = no_contrast.copy()
    contrast[:, 0] = -0.05
    first = acquisition_queues(candidates, ids, minimum, counts, no_contrast)
    second = acquisition_queues(candidates, ids, minimum, counts, contrast)
    assert first["target_only"] == second["target_only"]
    assert first["uniform"] == second["uniform"]
    assert [r["candidate_index"] for r in first["uniform"]] == [0, 1]
    assert [r["candidate_index"] for r in first["target_only"]] == [1, 0]
    assert not first["analytic_contrast"] and len(second["analytic_contrast"]) == 2
    for row in second["analytic_contrast"]:
        assert row["positive_option_id"] != row["negative_option_id"]
        assert row["negative_nominal_clearance_m"] <= -0.01


def test_partial_positive_cannot_be_admitted_and_nominal_rejection_is_retained():
    ids = ("neutral", "short")
    candidates = draw_pool(domain(), 93201, 1, ids)[:1]
    candidates[0]["assigned_positive_option_id"] = "short"
    minimum = np.array([[0.03, -0.02]])
    counts = np.array([[81, 1]])
    queues = acquisition_queues(candidates, ids, minimum, counts, np.array([[0.03, -0.02]]))
    assert len(queues["uniform"]) == 1 and not queues["target_only"]
    incomplete = counts.copy()
    incomplete[0, 0] = 1
    with pytest.raises(ValueError, match="complete positives"):
        acquisition_queues(candidates, ids, minimum, incomplete, minimum)


def test_observation_curriculum_stays_pending_and_never_uses_geometry_as_visibility():
    ids = ("neutral", "prior")
    candidates = draw_pool(domain(), 93201, 1, ids)[:1]
    minimum, counts = np.array([[-0.02, 0.04]]), np.array([[1, 81]])
    queues = acquisition_queues(candidates, ids, minimum, counts, minimum)
    assert queues["observation_curriculum"] == queues["analytic_contrast"]
    row = queues["observation_curriculum"][0]
    assert not row["curriculum_ready"]
    assert row["observation_availability"] is None and row["verified_learner_gap"] is None
    assert row["physical_passage"] is None and row["physical_measurement_admitted"] is None


def test_budget_prefixes_never_overspend_or_invent_missing_groups():
    rows = budget_prefixes(20, 7, 299)
    assert [r["maximum_complete_groups"] for r in rows] == [5, 17, 53]
    assert [r["available_prefix_groups"] for r in rows] == [5, 17, 20]
    assert rows[-1]["candidate_shortfall"] == 33
    assert all(r["planned_prefix_maximum_steps"] <= r["budget_physics_steps"] for r in rows)
    assert all(r["acquired_groups"] == 0 for r in rows)
    with pytest.raises(ValueError):
        budget_prefixes(20, 7, 299, (150000, 50000))


def test_coverage_order_is_stable_and_target_change_cannot_change_scene_draws():
    original = draw_pool(domain(), 93203, 2, ("neutral", "a"))
    changed = draw_pool(domain(), 93203, 2, ("neutral", "a", "b", "c"))
    assert [c["beam_specifications"] for c in original] == [
        c["beam_specifications"] for c in changed
    ]
    broken = copy.deepcopy(domain())
    broken["common_ranges"]["underside_m"] = [1.4, 1.14]
    with pytest.raises(ValueError, match="ordered proposal bounds"):
        draw_pool(broken, 93203, 2, ("neutral", "a"))
