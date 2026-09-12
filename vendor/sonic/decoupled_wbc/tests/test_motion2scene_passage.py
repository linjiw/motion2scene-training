"""Passage requires a genuine upstream approach before a stable crossing."""

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_passage import (
    score_passage,
)


def capture(initial=(-0.4, -0.2), *, yaw=0.0, frames=25):
    beam = {"center_xy_m": [1.7, -0.9], "yaw_rad": yaw, "length_m": 0.2}
    normal = np.array([np.cos(yaw), np.sin(yaw)])
    across = np.array([-np.sin(yaw), np.cos(yaw)])
    positions = np.zeros((frames, 2, 3))
    positions[:, :, 2] = 0.8
    for i in range(frames):
        projection = np.asarray(initial if i == 0 else (0.4, 0.6))
        positions[i, :, :2] = (
            beam["center_xy_m"]
            + projection[:, None] * normal
            + np.array([-0.03, 0.03])[:, None] * across
        )
    payload = {
        "body_pos_w": positions,
        "root_pos_w": positions.mean(1),
        "projected_gravity_b": np.tile([0.0, 0.0, -1.0], (frames, 1)),
        "motion_time_s": np.arange(frames) / 50,
        "fps": 50,
    }
    return payload, np.zeros((frames, 2, 3)), beam


@pytest.mark.parametrize("initial", [(0.4, 0.6), (-0.3, 0.05)])
def test_initially_downstream_or_straddling_cannot_count_as_passage(initial):
    payload, forces, beam = capture(initial)
    result = score_passage(payload, forces, beam)
    assert not result["pass"]
    assert not result["initially_upstream"]
    assert result["invalid_initial_approach"]
    # Preserve measured downstream completion; admission reports a separate failure.
    assert result["passage_finish_frame_exclusive"] is not None
    assert not result["incomplete_crossing"]
    assert not result["stabilization_failed"]
    assert not result["observed_beam_contact"]


def test_exact_upstream_edge_is_not_strictly_upstream():
    payload, forces, beam = capture()
    beam["center_xy_m"] = [0.0, 0.0]
    payload["body_pos_w"][0, :, :2] = [[-0.3, 0.0], [-0.1, 0.0]]
    payload["root_pos_w"][0, :2] = [-0.2, 0.0]
    result = score_passage(payload, forces, beam)
    assert not result["pass"]
    assert result["invalid_initial_approach"]


@pytest.mark.parametrize("yaw", [0.0, np.pi / 2, -0.7, np.pi])
def test_strict_upstream_approach_is_checked_in_beam_frame(yaw):
    result = score_passage(*capture(yaw=yaw))
    assert result["pass"]
    assert result["initially_upstream"]
    assert not result["invalid_initial_approach"]
    assert result["passage_finish_frame_exclusive"] == 17
    assert result["stabilization_seconds"] == 0.3
    assert result["maximum_beam_normal_force_n_through_passage"] == 0.0


def test_yawed_straddling_approach_is_rejected():
    result = score_passage(*capture((-0.3, 0.05), yaw=-0.7))
    assert result["invalid_initial_approach"] and not result["pass"]


def test_contact_failure_and_force_counts_are_preserved():
    payload, forces, beam = capture()
    forces[5, 0, 0] = 2.0
    result = score_passage(payload, forces, beam)
    assert result["initially_upstream"] and not result["pass"]
    assert result["observed_beam_contact"]
    assert result["maximum_beam_normal_force_n_through_passage"] == 2.0
    assert result["force_threshold_frame_counts"] == {"0.1": 1, "1": 1, "10": 0}


def test_first_episode_and_completion_contact_horizon_are_preserved():
    payload, forces, beam = capture()
    payload["motion_time_s"][20:] = np.arange(5) / 50
    forces[21, 0, 0] = 100.0
    result = score_passage(payload, forces, beam)
    assert result["pass"]
    assert result["first_episode_frames"] == 20
    assert result["capture_frames"] == 25
    assert result["reset_count"] == 1
    assert result["passage_finish_frame_exclusive"] == 17
    assert result["maximum_beam_normal_force_n_through_passage"] == 0.0


@pytest.mark.parametrize("initial,upstream", [((-0.4, -0.2), True), ((0.4, 0.6), False)])
def test_one_frame_keeps_incomplete_stabilization(initial, upstream):
    result = score_passage(*capture(initial, frames=1))
    assert not result["pass"] and result["stabilization_failed"]
    assert result["passage_finish_frame_exclusive"] is None
    assert result["initially_upstream"] is upstream
    assert result["invalid_initial_approach"] is not upstream


def test_zero_frames_keep_existing_validation_error():
    with pytest.raises(ValueError, match="complete finite synchronized"):
        score_passage(*capture(frames=0))


@pytest.mark.parametrize(
    "key",
    ["body_pos_w", "root_pos_w", "projected_gravity_b", "motion_time_s", "forces"],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_capture_keeps_existing_validation_error(key, value):
    payload, forces, beam = capture()
    target = forces if key == "forces" else payload[key]
    target.flat[0] = value
    with pytest.raises(ValueError, match="complete finite synchronized"):
        score_passage(payload, forces, beam)


@pytest.mark.parametrize("fps", [0.0, -1.0, np.nan, np.inf])
def test_invalid_fps_keeps_existing_validation_error(fps):
    payload, forces, beam = capture()
    payload["fps"] = fps
    with pytest.raises(ValueError, match="complete finite synchronized"):
        score_passage(payload, forces, beam)
