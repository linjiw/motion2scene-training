"""Deployment contract checks without ONNX inference, a GPU, or simulator."""

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_planner_adapter import (
    deployment_context,
    kinematic_diagnostics,
    make_inputs,
    named_qpos_to_mujoco,
    valid_output,
    validate_inputs,
)
from gear_sonic.dataset_generation.kimodo_motion_adapter import (
    G1_ISAACLAB_JOINT_NAMES,
    KIMODO_G1_JOINT_NAMES,
)


def reference(frames=60):
    qpos = np.zeros((frames, 36))
    qpos[:, 0] = np.arange(frames) / 50
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1
    return qpos


def packet():
    context, _ = deployment_context(reference(), 15)
    return make_inputs(context, mode=1, target_vel=0.6, height=-1, seed=1234, heading_rad=0.2)


def test_named_permutation_uses_every_named_joint_once():
    qpos = reference()
    for index, name in enumerate(G1_ISAACLAB_JOINT_NAMES):
        qpos[:, 7 + index] = KIMODO_G1_JOINT_NAMES.index(name)
    converted = named_qpos_to_mujoco(qpos, G1_ISAACLAB_JOINT_NAMES)
    np.testing.assert_array_equal(converted[:, :7], qpos[:, :7])
    np.testing.assert_array_equal(converted[0, 7:], np.arange(29))
    with pytest.raises(ValueError, match="bijection"):
        named_qpos_to_mujoco(qpos, [G1_ISAACLAB_JOINT_NAMES[0]] * 29)


def test_context_matches_cpp_future_clock_and_shortest_arc_quaternions():
    qpos = reference()
    yaw = np.arange(len(qpos)) / 50
    qpos[:, 3] = np.cos(yaw / 2)
    qpos[:, 6] = np.sin(yaw / 2)
    qpos[::2, 3:7] *= -1  # Equivalent signs must not introduce a long rotation.
    context, metadata = deployment_context(qpos, 15)
    expected = np.array([0.34, 0.34 + 1 / 30, 0.34 + 2 / 30, 0.44])
    np.testing.assert_allclose(metadata["sample_times_s"], expected, atol=1e-12)
    np.testing.assert_allclose(context[0, :, 0], expected, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(context[0, :, 3:7], axis=1), 1, atol=1e-7)
    actual_yaw = 2 * np.arctan2(context[0, :, 6], context[0, :, 3])
    np.testing.assert_allclose(np.exp(1j * actual_yaw), np.exp(1j * expected), atol=1e-6)
    assert metadata["achieved_history"] is False


def test_context_refuses_to_repeat_missing_future_samples():
    with pytest.raises(ValueError, match="no padding"):
        deployment_context(reference(21), 15)
    with pytest.raises(ValueError, match="nonnegative integer"):
        deployment_context(reference(), -1)


def test_inputs_bind_dtypes_mask_seed_and_common_modes():
    inputs = packet()
    assert inputs["context_mujoco_qpos"].dtype == np.float32
    assert inputs["random_seed"].item() == 1234
    assert inputs["has_specific_target"].item() == 0
    inputs["mode"][0] = 22
    with pytest.raises(ValueError, match="common enum"):
        validate_inputs(inputs)
    inputs = packet()
    inputs["allowed_pred_num_tokens"][:] = 0
    with pytest.raises(ValueError, match="nonempty"):
        validate_inputs(inputs)
    inputs = packet()
    inputs["random_seed"] = inputs["random_seed"].astype(np.int32)
    with pytest.raises(ValueError, match="int64"):
        validate_inputs(inputs)


def test_only_valid_native_frames_are_motion_and_tail_is_ignored():
    values = reference(64).astype(np.float32)[None]
    values[:, 36:] = np.nan  # Padding is preserved raw, never used or repaired.
    output = {"mujoco_qpos": values, "num_pred_frames": np.array([36], dtype=np.int32)}
    candidate = valid_output(output, packet())
    assert candidate.shape == (36, 36)
    assert np.isfinite(candidate).all()
    output["num_pred_frames"] = np.array([36], dtype=np.int64)
    with pytest.raises(ValueError, match="int32"):
        valid_output(output, packet())


@pytest.mark.parametrize("count", [0, 23, 24, 35, 48, 68])
def test_invalid_or_disabled_output_counts_are_rejected(count):
    output = {
        "mujoco_qpos": reference(64).astype(np.float32)[None],
        "num_pred_frames": np.array([count], dtype=np.int32),
    }
    with pytest.raises(ValueError):
        valid_output(output, packet())


def test_nonfinite_valid_frame_is_rejected_and_limits_are_reported_without_clipping():
    values = reference(64).astype(np.float32)[None]
    values[0, 20, 7] = np.nan
    output = {"mujoco_qpos": values, "num_pred_frames": np.array([36], dtype=np.int32)}
    with pytest.raises(ValueError, match="NaN"):
        valid_output(output, packet())
    candidate = reference(36)
    candidate[20, 7] = 2.0
    report = kinematic_diagnostics(
        candidate,
        packet()["context_mujoco_qpos"],
        KIMODO_G1_JOINT_NAMES,
        np.tile([-1.0, 1.0], (29, 1)),
    )
    assert report["joint_limit_violation_cells"] == 1
    assert report["max_joint_limit_violation_rad"] == 1.0
    assert report["physical_qualification"] == "not_run"
    assert report["source_fps"] == 30.0
    assert report["sample_span_s"] == 35 / 30
    assert candidate[20, 7] == 2.0
