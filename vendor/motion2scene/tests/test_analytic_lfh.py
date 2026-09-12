import numpy as np
import pytest

from motion2scene.inverse import solve_overhead_interval


def solve(**overrides):
    values = {
        "target_motion_id": "crouch_40",
        "weaker_motion_id": "neutral",
        "target_reach_m": 1.20,
        "weaker_reach_m": 1.30,
        "safety_margin_m": 0.01,
        "strike_margin_m": 0.02,
    }
    values.update(overrides)
    return solve_overhead_interval(**values)


def test_solves_target_clear_weaker_strike_interval():
    interval = solve()

    assert interval.nonempty
    assert interval.raw_gap_m == pytest.approx(0.10)
    assert interval.lower_m == pytest.approx(1.21)
    assert interval.upper_m == pytest.approx(1.28)
    assert interval.width_m == pytest.approx(0.07)
    assert interval.clearance_m(1.21) == pytest.approx(0.01)
    assert interval.weaker_deficit_m(1.28) == pytest.approx(0.02)


def test_collapsed_interval_is_returned_and_cannot_be_sampled():
    interval = solve(target_reach_m=1.29)

    assert not interval.nonempty
    assert interval.width_m < 0.0
    with pytest.raises(ValueError, match="empty critical interval"):
        interval.sample_stratified(4, seed=7)


def test_wrong_motion_order_remains_visible_as_empty():
    interval = solve(target_reach_m=1.31, weaker_reach_m=1.30, safety_margin_m=0.0)

    assert not interval.nonempty
    assert interval.raw_gap_m == pytest.approx(-0.01)


def test_stratified_samples_are_deterministic_and_cover_every_bin():
    interval = solve()
    first = interval.sample_stratified(8, seed=19)
    second = interval.sample_stratified(8, seed=19)

    assert np.array_equal(first, second)
    assert all(interval.contains(value) for value in first)
    normalized = np.sort((first - interval.lower_m) / interval.width_m)
    assert np.all(normalized >= np.arange(8) / 8)
    assert np.all(normalized < (np.arange(8) + 1) / 8)


@pytest.mark.parametrize("field", ["target_reach_m", "safety_margin_m", "strike_margin_m"])
@pytest.mark.parametrize("value", [-0.1, np.inf, np.nan])
def test_rejects_invalid_geometry(field, value):
    with pytest.raises(ValueError, match=field):
        solve(**{field: value})


def test_rejects_invalid_sample_counts():
    interval = solve()
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            interval.sample_stratified(value, seed=0)
