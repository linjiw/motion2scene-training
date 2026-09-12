import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_sensor_alignment import (
    audit_sensor_alignment,
)


def capture():
    return {
        "fps": 50,
        "motion_time_s": np.array([0, 0.02, 0.04]),
        "root_pos_w": np.array([[0.1, 0, 1], [0.2, 0, 1], [0.3, 0, 1]]),
        "root_quat_w": np.tile([1, 0, 0, 0], (3, 1)),
        "dof_pos": np.zeros((3, 2)),
    }


def test_post_recording_reset_invalidates_sensor_not_valid_physical_row():
    p = capture()
    packets = [
        {
            "time_s": (i + 1) / 50,
            "capture_elapsed_s": i / 50,
            "root_pos_w": p["root_pos_w"][i].tolist(),
            "root_quat_w": p["root_quat_w"][i].tolist(),
            "state": {"dof_pos": p["dof_pos"][i].tolist()},
        }
        for i in range(3)
    ]
    packets[-1]["time_s"] = 0.02
    packets[-1]["root_pos_w"] = [0, 0, 1]
    result = audit_sensor_alignment(p, packets, reference_frames=3)
    assert result["physical_first_episode_frames"] == 3
    assert result["packet_eligible"] == [True, True, False]
    assert result["packets"][-1]["mismatched_state_fields"] == ["root_pos_w"]


def test_legacy_packets_do_not_invent_pose_verification():
    packets = [{"time_s": t} for t in [0.02, 0.04, 0.02]]
    result = audit_sensor_alignment(capture(), packets, reference_frames=3)
    assert result["packet_eligible"] == [True, True, False]
    assert result["pose_unavailable_packets"] == 3
    assert result["packets"][0]["confidence"] == "phase_only_pose_unavailable"


def test_actual_physical_reset_excludes_later_episode_packets():
    p = capture()
    p["motion_time_s"][-1] = 0
    result = audit_sensor_alignment(
        p, [{"time_s": t} for t in [0.02, 0.04, 0.02]], reference_frames=5
    )
    assert result["physical_first_episode_frames"] == 2
    assert result["packet_eligible"] == [True, True, False]
    assert "after_physical_first_episode" in result["packets"][-1]["reasons"]


def test_clocks_and_capture_counts_are_checked():
    with pytest.raises(ValueError, match="capture counts"):
        audit_sensor_alignment(capture(), [], reference_frames=3)
    p = capture()
    p["motion_time_s"][1] = 0.01
    with pytest.raises(ValueError, match="reference clock"):
        audit_sensor_alignment(p, [{"time_s": 0.02}] * 3, reference_frames=3)
