"""File-only contract for the installed SONIC planner; no robot or simulator imports.

The context follows UpdateContextFromMotion, not an achieved-state history.
Output stays at the graph's native 30 Hz; padded output never becomes motion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from gear_sonic.dataset_generation.deployable_retiming import slerp
from gear_sonic.dataset_generation.kimodo_motion_adapter import (
    KIMODO_G1_JOINT_NAMES,
    validate_kimodo_qpos,
)

PLANNER_FPS = 30.0
REFERENCE_FPS = 50.0
COMMON_MODES = {0: "IDLE", 1: "SLOW_WALK", 2: "WALK", 4: "IDLE_SQUAT"}
ACTIVE_TOKEN_MASK = (0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0)
INPUT_SPEC = {
    "context_mujoco_qpos": ("float32", (1, 4, 36)),
    "target_vel": ("float32", (1,)),
    "mode": ("int64", (1,)),
    "movement_direction": ("float32", (1, 3)),
    "facing_direction": ("float32", (1, 3)),
    "random_seed": ("int64", (1,)),
    "has_specific_target": ("int64", (1, 1)),
    "specific_target_positions": ("float32", (1, 4, 3)),
    "specific_target_headings": ("float32", (1, 4)),
    "allowed_pred_num_tokens": ("int64", (1, 11)),
    "height": ("float32", (1,)),
}
OUTPUT_SPEC = {
    "mujoco_qpos": ("float32", (1, 64, 36)),
    "num_pred_frames": ("int32", (1,)),
}


def named_qpos_to_mujoco(qpos: np.ndarray, joint_names: Sequence[str]) -> np.ndarray:
    """Require a bijection of exact G1 joint names, preserving XYZ/WXYZ."""
    names = tuple(str(name) for name in joint_names)
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(KIMODO_G1_JOINT_NAMES):
        raise ValueError("joint names must be a bijection of the 29 named G1 joints")
    values = validate_kimodo_qpos(qpos)
    indices = [names.index(name) for name in KIMODO_G1_JOINT_NAMES]
    return np.concatenate([values[:, :7], values[:, 7:][:, indices]], axis=1)


def deployment_context(qpos_50hz: np.ndarray, current_frame: int) -> tuple[np.ndarray, dict]:
    """Four future planned samples, +2 50 Hz ticks then 30 Hz spacing.

    Unlike the deployment wrapper's endpoint clamp, this offline gate rejects
    missing source samples. There is no padding or reference-duration extension.
    """
    values = validate_kimodo_qpos(qpos_50hz)
    if isinstance(current_frame, bool) or int(current_frame) != current_frame or current_frame < 0:
        raise ValueError("current_frame must be a nonnegative integer")
    times = (current_frame + 2) / REFERENCE_FPS + np.arange(4) / PLANNER_FPS
    coordinates = times * REFERENCE_FPS
    if coordinates[-1] > len(values) - 1 + 1e-10:
        raise ValueError(
            "future planned context exceeds the available reference; no padding allowed"
        )
    left = np.floor(coordinates + 1e-10).astype(int)
    right = np.minimum(left + 1, len(values) - 1)
    fraction = np.clip(coordinates - left, 0.0, 1.0)
    result = values[left] * (1.0 - fraction[:, None]) + values[right] * fraction[:, None]
    result[:, 3:7] = slerp(values[left, 3:7], values[right, 3:7], fraction)
    result = validate_kimodo_qpos(result).astype(np.float32)[None]
    return result, {
        "source": "deployment_parity_future_planned_reference",
        "achieved_history": False,
        "current_frame_50hz": int(current_frame),
        "lookahead_frames_50hz": 2,
        "sample_times_s": times.tolist(),
        "reference_frames": len(values),
        "reference_fps": REFERENCE_FPS,
        "output_anchor_reference_time_s": float(times[0]),
        "interpolation": "linear XYZ/joints, shortest-arc WXYZ slerp",
        "endpoint_padding": False,
    }


def validate_inputs(inputs: Mapping[str, np.ndarray]) -> None:
    if set(inputs) != set(INPUT_SPEC):
        raise ValueError("planner inputs must contain exactly the eleven graph inputs")
    for name, (dtype, shape) in INPUT_SPEC.items():
        value = np.asarray(inputs[name])
        if value.dtype != np.dtype(dtype) or value.shape != shape:
            raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"nonfinite input {name}")
    validate_kimodo_qpos(inputs["context_mujoco_qpos"][0])
    if int(inputs["mode"][0]) not in COMMON_MODES:
        raise ValueError("mode is outside the audited common enum subset")
    mask = inputs["allowed_pred_num_tokens"]
    if not np.all(np.isin(mask, [0, 1])) or not np.any(mask):
        raise ValueError("token mask must be nonempty and binary")
    if int(inputs["has_specific_target"][0, 0]) not in (0, 1):
        raise ValueError("has_specific_target must be binary")
    for name in ("movement_direction", "facing_direction"):
        if not np.isclose(np.linalg.norm(inputs[name][0]), 1.0, atol=1e-5):
            raise ValueError(f"{name} must be a unit world vector")
        if abs(float(inputs[name][0, 2])) > 1e-6:
            raise ValueError(f"{name} must be horizontal for this audited adapter")


def make_inputs(
    context: np.ndarray,
    *,
    mode: int,
    target_vel: float,
    height: float,
    seed: int,
    heading_rad: float,
    token_mask: Sequence[int] = ACTIVE_TOKEN_MASK,
) -> dict[str, np.ndarray]:
    """Build one explicit packet, with specific world target keyframes disabled.

    Nonpositive speed and height select mode defaults in the wrapper. They do
    not assert a protective stop or arbitrary height control in walking modes.
    """
    direction = [[np.cos(heading_rad), np.sin(heading_rad), 0.0]]
    values = {
        "context_mujoco_qpos": context,
        "target_vel": [target_vel],
        "mode": [mode],
        "movement_direction": direction,
        "facing_direction": direction,
        "random_seed": [seed],
        "has_specific_target": [[0]],
        "specific_target_positions": np.zeros((1, 4, 3)),
        "specific_target_headings": np.zeros((1, 4)),
        "allowed_pred_num_tokens": [token_mask],
        "height": [height],
    }
    result = {name: np.asarray(value, dtype=INPUT_SPEC[name][0]) for name, value in values.items()}
    validate_inputs(result)
    return result


def valid_output(outputs: Mapping[str, np.ndarray], inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Validate the graph's int32 valid count and discard only the padded tail."""
    validate_inputs(inputs)
    if set(outputs) != set(OUTPUT_SPEC):
        raise ValueError("planner outputs do not match the graph contract")
    for name, (dtype, shape) in OUTPUT_SPEC.items():
        value = np.asarray(outputs[name])
        if value.dtype != np.dtype(dtype) or value.shape != shape:
            raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
    count = int(outputs["num_pred_frames"][0])
    if count < 24 or count > 64 or count % 4:
        raise ValueError("valid frame count must be 24..64 and divisible by four")
    if inputs["allowed_pred_num_tokens"][0, count // 4 - 6] != 1:
        raise ValueError("returned frame count is disabled by the registered token mask")
    result = outputs["mujoco_qpos"][0, :count].copy()
    validate_kimodo_qpos(result)
    return result


def kinematic_diagnostics(
    candidate: np.ndarray,
    context: np.ndarray,
    joint_limit_names: Sequence[str],
    joint_limits: np.ndarray,
) -> dict:
    """Report continuity/ranges without clipping or declaring physical feasibility."""
    values = validate_kimodo_qpos(candidate)
    source = validate_kimodo_qpos(np.asarray(context).reshape(4, 36))
    names = tuple(joint_limit_names)
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(KIMODO_G1_JOINT_NAMES):
        raise ValueError("joint-limit XML must cover every named G1 joint exactly once")
    limits = np.asarray(joint_limits, dtype=float)[[names.index(n) for n in KIMODO_G1_JOINT_NAMES]]
    if (
        limits.shape != (29, 2)
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("invalid joint limits")
    excess = np.maximum(np.maximum(limits[:, 0] - values[:, 7:], values[:, 7:] - limits[:, 1]), 0)
    dots = np.abs(np.sum(values[:4, 3:7] * source[:, 3:7], axis=1))
    dots /= np.linalg.norm(values[:4, 3:7], axis=1) * np.linalg.norm(source[:, 3:7], axis=1)
    return {
        "physical_qualification": "not_run",
        "reference_frames": len(values),
        "source_fps": PLANNER_FPS,
        "sample_span_s": (len(values) - 1) / PLANNER_FPS,
        "padded_frames_excluded": 64 - len(values),
        "finite_valid_values": True,
        "max_quaternion_norm_error": float(np.max(abs(np.linalg.norm(values[:, 3:7], axis=1) - 1))),
        "context_overlap_root_max_l2_m": float(
            np.max(np.linalg.norm(values[:4, :3] - source[:, :3], axis=1))
        ),
        "context_overlap_joint_max_abs_rad": float(np.max(abs(values[:4, 7:] - source[:, 7:]))),
        "context_overlap_rotation_max_rad": float(np.max(2 * np.arccos(np.clip(dots, 0, 1)))),
        "max_adjacent_root_step_m": float(
            np.max(np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1))
        ),
        "max_adjacent_joint_step_rad": float(np.max(abs(np.diff(values[:, 7:], axis=0)))),
        "context_to_new_root_step_m": float(np.linalg.norm(values[4, :3] - values[3, :3])),
        "context_to_new_joint_step_rad": float(np.max(abs(values[4, 7:] - values[3, 7:]))),
        "root_height_range_m": [float(values[:, 2].min()), float(values[:, 2].max())],
        "root_displacement_m": (values[-1, :3] - values[0, :3]).tolist(),
        "joint_limit_violation_cells": int(np.count_nonzero(excess > 1e-6)),
        "max_joint_limit_violation_rad": float(excess.max()),
        "joint_limit_max_violation_by_name_rad": dict(
            zip(KIMODO_G1_JOINT_NAMES, excess.max(axis=0).tolist(), strict=True)
        ),
        "limit_scope": "static MJCF ranges only; no physical/contact/stability test",
        "coordinates": "world metres, Z up, +X forward, +Y left; WXYZ; named MuJoCo joints in radians",
        "resampled": False,
        "clipped": False,
    }
