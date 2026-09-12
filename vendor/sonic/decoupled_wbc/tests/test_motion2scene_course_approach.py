"""Course approach diagnostics summarize every beam, including earlier ones."""

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_course import score_course


def course_capture(stations, yaw):
    normal = np.array([np.cos(yaw), np.sin(yaw)])
    across = np.array([-np.sin(yaw), np.cos(yaw)])
    progress = np.linspace(0.0, 2.0, 101)
    positions = np.zeros((101, 2, 3))
    positions[:, :, 2] = 0.8
    positions[:, :, :2] = (progress[:, None] + np.array([-0.04, 0.04]))[
        :, :, None
    ] * normal + np.array([-0.01, 0.01])[None, :, None] * across
    payload = {
        "body_pos_w": positions,
        "root_pos_w": positions.mean(1),
        "projected_gravity_b": np.tile([0.0, 0.0, -1.0], (101, 1)),
        "motion_time_s": np.arange(101) / 50,
        "fps": 50,
    }
    beams = [
        {
            "center_xy_m": (station * normal).tolist(),
            "yaw_rad": yaw,
            "length_m": 0.2,
            "width_m": 1.2,
            "thickness_m": 0.1,
            "underside_m": 1.25,
        }
        for station in stations
    ]
    return payload, np.zeros((101, 2, 2, 3)), beams


@pytest.mark.parametrize("yaw", [0.0, 0.7])
@pytest.mark.parametrize(
    "stations,expected",
    [
        ((-0.4, 0.6), [False, True]),  # Earlier beam starts wholly behind the body.
        ((0.0, 0.6), [False, True]),  # Earlier beam initially straddles the body.
        ((0.6, -0.4), [True, False]),  # Converse: only the final beam starts behind.
        ((0.6, 0.0), [True, False]),  # Converse: only the final beam is straddled.
        ((0.4, 0.8), [True, True]),
        ((-0.4, 0.0), [False, False]),
    ],
)
def test_scalar_approach_diagnostics_aggregate_all_beams(stations, expected, yaw):
    result = score_course(*course_capture(stations, yaw), commands_valid=True, timeout_s=3)
    assert result["initially_upstream_by_beam"] == expected
    assert result["initially_upstream"] is all(expected)
    assert result["invalid_initial_approach"] is not all(expected)
    if all(expected):
        assert result["pass"]
        assert "invalid_initial_approach" not in result["failure_reasons"]
    else:
        assert not result["pass"]
        assert "invalid_initial_approach" in result["failure_reasons"]
    assert result["maximum_beam_normal_force_n_through_passage"] == 0.0
    assert result["no_fall_through_course"]
