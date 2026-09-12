"""Independent synthetic checks of the provisional seven-schedule sensor script."""

import itertools

import numpy as np
import pytest

from decoupled_wbc.tests.test_motion2scene_timed_schedule_policy import bank_fixture
from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_script import (
    BANDS,
    choose_sensor_schedule,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (
    TimedOptionState,
    apply_timed_request,
    legal_timed_actions,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (
    expected_feature_names,
)


def bank():
    value = bank_fixture()
    for option in value.request["options"]:
        if option["reference_id"] == "prior":
            option["reference_id"] = "prior_splice"
    return value


def observation(hazards=(), *, gap=None):
    names = expected_feature_names(7)
    values = np.zeros(len(names))
    for index, (start, end) in enumerate(BANDS):
        prefix = f"corridor_{start:g}_{end:g}_"
        values[names.index(prefix + "unknown_fraction")] = 1
        if index in hazards:
            values[names.index(prefix + "upper_hit")] = 1
        if gap is not None:
            values[names.index(prefix + "floor_fraction")] = 0.1
            values[names.index(prefix + "ceiling_fraction")] = 0.1
            values[names.index(prefix + "minimum_ceiling_m")] = gap - 0.8
            values[names.index(prefix + "maximum_floor_m")] = -0.8
    return names, values


def choose(tick, hazards=(), **kwargs):
    value = bank()
    mask, _ = legal_timed_actions(value, TimedOptionState(), tick, np.zeros(7), np.zeros(7))
    selected, report = choose_sensor_schedule(*observation(hazards, **kwargs), mask, tick, value)
    return selected, report


def test_all_hazard_patterns_select_only_current_qualified_schedules():
    value = bank()
    for tick, bits in itertools.product((15, 50, 70), itertools.product((False, True), repeat=4)):
        mask, _ = legal_timed_actions(value, TimedOptionState(), tick, np.zeros(7), np.zeros(7))
        selected, _ = choose_sensor_schedule(*observation(np.flatnonzero(bits)), mask, tick, value)
        assert mask[value.option_ids.index(selected)]
        state, transition = apply_timed_request(
            value, TimedOptionState(), tick, selected, np.zeros(7), np.zeros(7)
        )
        assert transition["allowed"] and state.entries <= 1


def test_default_reference_order_and_observed_gap():
    assert choose(15, (0,))[0] == "prior_15"
    assert choose(15, (1,))[0] == "neutral"
    assert choose(50, (1,))[0] == "prior_50"
    assert choose(70, (1,))[0] == "short_70"
    assert choose(70, (1, 2))[0] == "sustained_70"
    assert choose(50, (0,), gap=1.40)[0] == "neutral"
    assert choose(50, (0,), gap=1.20)[0] == "prior_50"


def test_unknown_is_wait_and_does_not_invent_a_stop():
    for tick in (15, 50, 70):
        selected, report = choose(tick)
        assert selected == "neutral" and not any(report["hazard_bands"])


def test_unavailable_preferred_schedule_is_never_replaced_by_illegal_entry():
    value = bank()
    mask = np.zeros(7, bool)
    mask[0] = True
    selected, report = choose_sensor_schedule(*observation((0,)), mask, 50, value)
    assert selected == "neutral" and report["preferred_reference"] == "prior_splice"
    mask[0] = False
    with pytest.raises(ValueError, match="neutral legal"):
        choose_sensor_schedule(*observation((0,)), mask, 50, value)


def test_missing_surface_does_not_dismiss_an_upper_hit():
    value = bank()
    names, features = observation((0,), gap=1.40)
    features[names.index("corridor_0_0.75_ceiling_fraction")] = 0
    mask, _ = legal_timed_actions(value, TimedOptionState(), 50, np.zeros(7), np.zeros(7))
    assert choose_sensor_schedule(names, features, mask, 50, value)[0] == "prior_50"
    with pytest.raises(ValueError, match="registered neutral decision"):
        choose_sensor_schedule(names, features, mask, 49, value)
