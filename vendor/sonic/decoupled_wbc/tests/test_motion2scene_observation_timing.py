"""Actual measurement delivery must precede the legal adaptation deadline."""

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_observation_timing import (
    audit_decision_visibility,
    observed_beam_faces,
    transition_timing_screen,
)

BEAM = dict(
    center_xy_m=[2, 0], underside_m=1.2, thickness_m=0.1, length_m=0.5, width_m=1.2, yaw_rad=0
)
RAY = dict(
    origin_w=[2, 0, 1],
    direction_w=[0, 0, 1],
    range_m=4,
    hit_distance_m=0.2,
    hit_normal_w=[0, 0, -1],
)


def rows(delay=0):
    return [
        dict(
            tick=i + 1,
            capture_elapsed_s=i * 0.02,
            delivered_capture_elapsed_s=None if i < delay else (i - delay) * 0.02,
            measurements=(
                [RAY] if i == 1 else [{**RAY, "hit_distance_m": None, "hit_normal_w": None}]
            ),
        )
        for i in range(6)
    ]


def test_surface_binding_and_disabled_geometry_do_not_invent_visibility():
    assert observed_beam_faces([RAY], BEAM) == {"beam_surface_rays": 1, "underside_rays": 1}
    miss = {**RAY, "origin_w": [0, 0, 1]}
    assert observed_beam_faces([miss], BEAM)["beam_surface_rays"] == 0
    result = audit_decision_visibility(rows(), BEAM, 3, collision_enabled=False)
    assert result["surface"]["first_capture_elapsed_s"] is None
    assert not result["surface"]["delivered_by_entry"]


def test_delivery_not_capture_controls_deadline_and_history_retains_early_hit():
    immediate = audit_decision_visibility(rows(), BEAM, 3)
    delayed = audit_decision_visibility(rows(2), BEAM, 3)
    assert immediate["surface"]["delivered_by_entry"]
    assert immediate["surface"]["first_capture_elapsed_s"] == 0.02
    assert delayed["surface"]["first_capture_elapsed_s"] == 0.02
    assert delayed["surface"]["first_delivery_elapsed_s"] == 0.06
    assert not delayed["surface"]["delivered_by_entry"]


def test_future_delivery_is_rejected():
    values = rows()
    values[0]["delivered_capture_elapsed_s"] = 0.02
    with pytest.raises(ValueError, match="available captured"):
        audit_decision_visibility(values, BEAM, 3)


def test_transition_needs_stable_low_height_and_timing_margin():
    visibility = audit_decision_visibility(rows(), BEAM, 3)
    times = np.arange(12) * 0.02
    height = np.array([1.3, 1.3, 1.1, 1.3, 1.1, 1.1, 1.1, 1.1, 1.1, 1.3, 1.3, 1.3])
    result = transition_timing_screen(height, times, 0.04, 1.2, 0.20, visibility)
    assert result["ready_elapsed_s"] == 0.08
    assert result["transition_s"] == 0.04
    assert result["eligible"]
    assert not transition_timing_screen(height, times, 0.04, 1.2, 0.15, visibility)["eligible"]
    assert not transition_timing_screen(
        height, times, 0.04, 1.2, 0.20, audit_decision_visibility(rows(2), BEAM, 3)
    )["eligible"]


def test_visibility_audit_is_json_serializable():
    import json

    beam = {
        "center_xy_m": [1.0, 0.0],
        "yaw_rad": 0.0,
        "length_m": 0.2,
        "width_m": 1.0,
        "thickness_m": 0.1,
        "underside_m": 1.0,
    }
    measurement = {
        "origin_w": [0.0, 0.0, 1.05],
        "direction_w": [1.0, 0.0, 0.0],
        "range_m": 4.0,
        "hit_distance_m": 0.9,
        "hit_normal_w": [-1.0, 0.0, 0.0],
    }
    rows = [
        {
            "tick": tick,
            "capture_elapsed_s": (tick - 1) / 50,
            "delivered_capture_elapsed_s": (tick - 1) / 50,
            "measurements": [measurement],
        }
        for tick in range(1, 16)
    ]
    report = audit_decision_visibility(rows, beam, 15)
    assert json.loads(json.dumps(report))["surface"]["delivered_by_entry"] is True
