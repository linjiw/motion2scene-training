from __future__ import annotations

import numpy as np
import pytest

from motion2scene.dataset.schema import SemanticStatus
from motion2scene.motion.paired_semantics import (
    PairedSemanticPolicy,
    assess_controller_retention,
    assess_paired_reduction,
)

POLICY = PairedSemanticPolicy(minimum_effect=0.05)


def profiles(depth: float, *, start: float = 0.3, stop: float = 0.7):
    progress = np.linspace(0.0, 1.0, 101)
    walk = np.ones_like(progress)
    target = walk.copy()
    target[(progress >= start) & (progress <= stop)] -= depth
    return progress, target, walk


def test_small_paired_change_is_s0_absent() -> None:
    progress, target, walk = profiles(0.04)
    result = assess_paired_reduction(progress, target, walk, route_valid=True, policy=POLICY)
    assert result.semantic_status == SemanticStatus.ABSENT


def test_sustained_change_is_s1_but_not_localized() -> None:
    progress, target, walk = profiles(0.08, start=0.0, stop=1.0)
    result = assess_paired_reduction(progress, target, walk, route_valid=True, policy=POLICY)
    assert result.semantic_status == SemanticStatus.ELICITED


def test_local_event_on_invalid_route_stops_at_s2() -> None:
    progress, target, walk = profiles(0.08)
    result = assess_paired_reduction(progress, target, walk, route_valid=False, policy=POLICY)
    assert result.semantic_status == SemanticStatus.LOCALIZED


def test_valid_local_event_is_s3() -> None:
    progress, target, walk = profiles(0.08)
    result = assess_paired_reduction(progress, target, walk, route_valid=True, policy=POLICY)
    assert result.semantic_status == SemanticStatus.ROUTE_ALIGNED
    assert result.event_interval == pytest.approx((0.3, 0.69))


def test_achieved_event_needs_magnitude_and_location_for_s4() -> None:
    progress, target, walk = profiles(0.08)
    reference = assess_paired_reduction(progress, target, walk, route_valid=True, policy=POLICY)
    _, achieved_target, achieved_walk = profiles(0.06, start=0.35, stop=0.72)
    achieved = assess_paired_reduction(
        progress, achieved_target, achieved_walk, route_valid=True, policy=POLICY
    )
    retained = assess_controller_retention(reference, achieved, policy=POLICY)
    assert retained.semantic_status == SemanticStatus.CONTROLLER_RETAINED


def test_controller_flattens_event_without_claiming_s4() -> None:
    progress, target, walk = profiles(0.10)
    reference = assess_paired_reduction(progress, target, walk, route_valid=True, policy=POLICY)
    _, achieved_target, achieved_walk = profiles(0.055)
    achieved = assess_paired_reduction(
        progress, achieved_target, achieved_walk, route_valid=True, policy=POLICY
    )
    retained = assess_controller_retention(reference, achieved, policy=POLICY)
    assert retained.semantic_status == SemanticStatus.ROUTE_ALIGNED


def test_profiles_fail_closed_on_nonmonotonic_progress() -> None:
    progress, target, walk = profiles(0.08)
    progress[50] = -1.0
    with pytest.raises(ValueError, match="monotonic"):
        assess_paired_reduction(progress, target, walk, route_valid=True, policy=POLICY)
