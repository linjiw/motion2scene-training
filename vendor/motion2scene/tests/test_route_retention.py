from __future__ import annotations

import numpy as np
import pytest

from motion2scene.motion.route_retention import assess_route_retention, compare_routes


def straight(length: float = 4.0, frames: int = 120) -> np.ndarray:
    return np.column_stack([np.linspace(0.0, length, frames), np.zeros(frames)])


def test_translation_is_removed_but_no_other_route_error_is_hidden() -> None:
    reference = straight()
    achieved = reference + np.array([4.2, -1.7])
    result = compare_routes(
        reference,
        achieved,
        expected_route="straight",
        reference_fps=30.0,
        achieved_fps=30.0,
    )

    assert result.initial_offset_m == pytest.approx(np.hypot(4.2, 1.7))
    assert result.path_length_ratio == pytest.approx(1.0)
    assert result.endpoint_error_m == pytest.approx(0.0, abs=1e-12)
    assert result.route_shape_rmse_m == pytest.approx(0.0, abs=1e-12)


def test_short_execution_preserves_along_track_deficit() -> None:
    result = compare_routes(
        straight(4.0),
        straight(3.0, frames=200),
        expected_route="straight",
        reference_fps=30.0,
        achieved_fps=50.0,
    )

    assert result.path_length_ratio == pytest.approx(0.75, rel=1e-4)
    assert result.net_displacement_ratio == pytest.approx(0.75, rel=1e-4)
    assert result.endpoint_error_m == pytest.approx(1.0)
    assert result.endpoint_along_chord_error_m == pytest.approx(-1.0)
    assert result.endpoint_cross_chord_error_m == pytest.approx(0.0)
    assert result.along_track_rmse_m > 0.5
    assert result.cross_track_rmse_m == pytest.approx(0.0, abs=1e-12)


def test_lateral_shape_error_is_measured_even_with_matching_endpoints() -> None:
    reference = straight()
    achieved = reference.copy()
    achieved[:, 1] = 0.2 * np.sin(np.linspace(0.0, np.pi, len(achieved)))
    result = compare_routes(
        reference,
        achieved,
        expected_route="straight",
        reference_fps=30.0,
        achieved_fps=30.0,
    )

    assert result.endpoint_error_m == pytest.approx(0.0, abs=1e-12)
    assert result.cross_track_rmse_m > 0.1
    assert result.cross_track_max_abs_error_m == pytest.approx(0.2, abs=2e-3)


def test_relative_gate_accepts_small_controller_route_error() -> None:
    reference = straight()
    achieved = straight(3.8, frames=200)
    achieved[:, 1] = 0.04 * np.sin(np.linspace(0.0, np.pi, len(achieved)))
    result = compare_routes(
        reference,
        achieved,
        expected_route="straight",
        reference_fps=30.0,
        achieved_fps=50.0,
    )

    decision = assess_route_retention(result)
    assert decision.retained
    assert all(decision.checks.values())


def test_relative_gate_rejects_short_and_laterally_wrong_routes() -> None:
    reference = straight()
    achieved = straight(3.2, frames=200)
    achieved[:, 1] = 0.3 * np.sin(np.linspace(0.0, np.pi, len(achieved)))
    result = compare_routes(
        reference,
        achieved,
        expected_route="straight",
        reference_fps=30.0,
        achieved_fps=50.0,
    )

    decision = assess_route_retention(result)
    assert not decision.retained
    assert "path_length_ratio_lower" in decision.failure_reasons
    assert "cross_track_rmse" in decision.failure_reasons
    assert "cross_track_max" in decision.failure_reasons


def test_relative_gate_requires_an_independently_valid_reference() -> None:
    reference = straight(0.5)
    result = compare_routes(
        reference,
        reference,
        expected_route="straight",
        reference_fps=30.0,
        achieved_fps=30.0,
    )

    decision = assess_route_retention(result)
    assert not decision.retained
    assert decision.failure_reasons == ("reference_route_valid",)


def test_bad_paths_and_sampling_fail_closed() -> None:
    with pytest.raises(ValueError, match="reference_path_xy"):
        compare_routes(
            np.zeros((2, 2)),
            straight(),
            expected_route="straight",
            reference_fps=30.0,
            achieved_fps=30.0,
        )
    with pytest.raises(ValueError, match="station_count"):
        compare_routes(
            straight(),
            straight(),
            expected_route="straight",
            reference_fps=30.0,
            achieved_fps=30.0,
            station_count=2,
        )
    with pytest.raises(ValueError, match="zero arclength"):
        compare_routes(
            np.zeros((10, 2)),
            straight(),
            expected_route="straight",
            reference_fps=30.0,
            achieved_fps=30.0,
        )
