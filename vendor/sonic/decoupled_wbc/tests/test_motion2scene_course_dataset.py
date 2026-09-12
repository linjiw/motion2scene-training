"""Portable course records must retain both constraints and causal input privacy."""

import numpy as np
import pytest

from scripts.research.motion2scene_export_course_demonstration import (
    check_sensor_privacy,
    course_channels,
)
from scripts.research.motion2scene_export_option_dataset import sensor_packets


def contacts():
    force = np.zeros((8, 2, 3, 3))
    force[1, 1, 2, 0] = 9.0
    return {
        "physics_force_w": force,
        "control_force_w": force[[3, 7]].copy(),
        "physics_steps": np.arange(8),
        "control_steps": np.array([3, 7]),
        "physics_dt": np.array(0.005),
        "beam_names": np.array(["CounterfactualBeam", "CounterfactualBeam_01"]),
        "filter_paths": np.array(["body0", "body1", "body2"]),
    }


def test_second_beam_substep_collision_survives_portable_synchronization():
    values = contacts()
    forces, error = course_channels(values, {"beams": [{}, {}]}, control_frames=2, physics_steps=8)
    assert error == 0
    assert not values["control_force_w"].any()
    assert forces.shape == (2, 2, 3, 3)
    assert forces[0, 1, 2, 0] == 9.0
    assert not forces[:, 0].any()


@pytest.mark.parametrize("corruption", ["reverse", "drop", "body", "frame"])
def test_course_export_rejects_missing_or_reordered_contact_streams(corruption):
    values = contacts()
    if corruption == "reverse":
        values["beam_names"] = values["beam_names"][::-1]
    elif corruption == "drop":
        values["physics_force_w"] = values["physics_force_w"][:, :1]
    elif corruption == "body":
        values["filter_paths"] = values["filter_paths"][:2]
    else:
        values["control_steps"] = values["control_steps"][:1]
    with pytest.raises(ValueError, match="multi-beam contact channels"):
        course_channels(values, {"beams": [{}, {}]}, control_frames=2, physics_steps=8)


def test_course_measurements_strip_second_beam_identity_and_outcomes():
    raw = {
        "observations": [
            {
                "time_s": 0.02,
                "beam": "second",
                "pass": True,
                "measurements": [
                    {
                        "origin_w": [0, 0, 1],
                        "direction_w": [1, 0, 0],
                        "hit_distance_m": 1.2,
                        "hit_normal_w": [0, 0, -1],
                        "object_id": 27,
                        "path": "/World/second",
                    }
                ],
            }
        ]
    }
    clean = sensor_packets(raw)
    check_sensor_privacy(clean)
    assert clean[0]["measurements"][0]["hit_distance_m"] == 1.2
    with pytest.raises(ValueError, match="privileged"):
        check_sensor_privacy(raw["observations"])
