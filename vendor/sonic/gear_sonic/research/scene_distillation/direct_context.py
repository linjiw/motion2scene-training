"""Explicit known-map task labels at measured robot poses; no reference-derived pose."""

import numpy as np

from gear_sonic.research.scene_distillation.observations import navigation_observation


def task_context(task, root_xyz, root_wxyz):
    observation = navigation_observation(
        proprio=np.zeros(930, np.float32),
        root_xyz=root_xyz,
        root_wxyz=root_wxyz,
        start_xyz=task["start_xyz"],
        goal_xyz=task["goal_xyz"],
        obstacles=task["obstacles"],
    )
    # 6 body-frame start/goal coordinates, tolerance, terminal speed, hold seconds,
    # explicit complete-known-map bit. This bit is NOT a camera visibility mask.
    nav = np.concatenate(
        [
            observation.pop("start_goal_body"),
            [task["goal_tolerance_m"], task["terminal_speed_mps"], task["hold_ticks"] * 0.02, 1.0],
        ]
    ).astype(np.float32)
    return {
        "navigation_context": nav,
        "obstacles_body": observation["obstacles_body"],
        "obstacle_mask": observation["obstacle_mask"],
    }


def score_navigation_task(task, root_xyz, speed, undesired_force, *, fell=False):
    """Task scoring independent of any reference pose or reference tracking success.

    Each row is an uncensored pre-reset control-step measurement. Contact rows
    are max undesired pair-resolved normal force over that control interval.
    """
    root, speed, force = np.asarray(root_xyz), np.asarray(speed), np.asarray(undesired_force)
    n = len(root)
    if root.shape != (n, 3) or speed.shape != (n,) or force.shape != (n,) or n < 1:
        raise ValueError("Invalid task evidence shape")
    if not all(np.isfinite(x).all() for x in (root, speed, force)):
        raise ValueError("Nonfinite task evidence")
    if (speed < 0).any() or (force < 0).any():
        raise ValueError("Speed and contact magnitude must be nonnegative")
    distance = np.linalg.norm(root - np.asarray(task["goal_xyz"]), axis=-1)
    good = (distance <= task["goal_tolerance_m"]) & (speed <= task["terminal_speed_mps"])
    run, best = 0, 0
    for valid in good:
        run = run + 1 if valid else 0
        best = max(best, run)
    hold = best >= task["hold_ticks"]
    contacts = bool((force <= 1.0).all())
    return {
        "navigation_success": bool(hold and contacts and not fell),
        "goal_ever_reached": bool((distance <= task["goal_tolerance_m"]).any()),
        "terminal_hold": bool(hold),
        "collision_free": contacts,
        "fell": bool(fell),
        "max_hold_ticks": best,
        "final_goal_distance_m": float(distance[-1]),
        "max_undesired_force_n": float(force.max()),
        "control_steps": n,
        "reference_tracking_required": False,
    }
