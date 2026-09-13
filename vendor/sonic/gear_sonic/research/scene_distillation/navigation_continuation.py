"""Bounded command-space continuation candidates; physics must establish support."""

import numpy as np

from gear_sonic.research.hindsight_training.observations import rotation_wxyz
from gear_sonic.research.scene_distillation.direct_context import score_navigation_task


def validate_continuation(config):
    expected = {
        "kind",
        "position_gain",
        "velocity_gain",
        "max_speed_mps",
        "max_shift_m",
        "activation_radius_m",
    }
    if set(config) != expected or config["kind"] not in (
        "nominal",
        "goal_velocity",
        "goal_velocity_keypoints",
    ):
        raise ValueError("Unknown continuation contract")
    for key in expected - {"kind"}:
        if not isinstance(config[key], (int, float)) or not np.isfinite(config[key]):
            raise ValueError("Nonfinite continuation parameter")
    if not (
        0 <= config["position_gain"] <= 2
        and 0 <= config["velocity_gain"] <= 2
        and 0 < config["max_speed_mps"] <= 0.5
        and 0 <= config["max_shift_m"] <= 0.25
        and 0 < config["activation_radius_m"] <= 1
    ):
        raise ValueError("Continuation exceeds bounded pilot limits")
    return config


def _limit(vector, maximum):
    return vector * min(1.0, maximum / max(float(np.linalg.norm(vector)), 1e-12))


def continuation_commands(nominal, position, quaternion, reference_anchor, goal, velocity, config):
    """Regulate horizontal goal error with causal velocity, never a hidden deadline.

    Joint targets, joint velocities, heading, height and relative orientation stay
    nominal. Optional keypoint translation is bounded and is only a candidate:
    neither kinematic consistency nor recoverability is assumed before execution.
    All vectors except causal body velocity are expressed in the registered world.
    """
    validate_continuation(config)
    nominal = np.asarray(nominal)
    position, reference_anchor, goal, velocity = map(
        np.asarray, (position, reference_anchor, goal, velocity)
    )
    if (
        nominal.shape != (114,)
        or velocity.shape != (4,)
        or any(x.shape != (3,) for x in (position, reference_anchor, goal))
        or not all(
            np.isfinite(x).all() for x in (nominal, position, reference_anchor, goal, velocity)
        )
        or velocity[3] not in (0, 1)
    ):
        raise ValueError("Invalid continuation observation")
    rotation = rotation_wxyz(quaternion)
    result = nominal.copy()
    error = goal - position
    error[2] = 0
    distance = float(np.linalg.norm(error))
    if config["kind"] == "nominal" or not velocity[3]:
        return result
    # Continuous spatial gate: no reference phase, intervention clock or urgency.
    weight = float(np.clip(2 * (1 - distance / config["activation_radius_m"]), 0, 1))
    world_velocity = rotation @ velocity[:3]
    desired = config["position_gain"] * error - config["velocity_gain"] * world_velocity
    desired[2] = 0
    desired = _limit(desired, config["max_speed_mps"])
    result[2:5] = (1 - weight) * nominal[2:5] + weight * (rotation.T @ desired)
    if config["kind"] == "goal_velocity_keypoints":
        shift = goal - reference_anchor
        shift[2] = 0
        shift = weight * _limit(shift, config["max_shift_m"])
        result[8:50] = (nominal[8:50].reshape(14, 3) + rotation.T @ shift).reshape(-1)
    return result


def continuation_outcomes(task, trace, switch_tick):
    """Keep local stabilization and original-deadline success separate.

    Accepts diagnostic traces extending beyond the deadline, but never promotes
    their late holds to timely support. A shorter observed suffix is censored;
    failure within that suffix is not evidence of global unstabilizability.
    """
    n = len(trace["speed"])
    if not isinstance(switch_tick, int) or not 0 <= switch_tick <= n:
        raise ValueError("Invalid continuation entry")

    def score(start, end):
        if end <= start:
            return None
        return score_navigation_task(
            task,
            trace["root_xyz"][start:end],
            trace["speed"][start:end],
            trace["undesired_force"][start:end],
            fell=bool((np.asarray(trace["root_xyz"])[start:end, 2] < 0.25).any()),
        )

    deadline = min(n, task["deadline_ticks"])
    local, timely, whole = score(switch_tick, n), score(switch_tick, deadline), score(0, deadline)
    prefix_safe = not bool(
        (np.asarray(trace["undesired_force"])[:switch_tick] > 1).any()
        or (np.asarray(trace["root_xyz"])[:switch_tick, 2] < 0.25).any()
    )
    return dict(
        local_stabilized=bool(local and local["navigation_success"]),
        original_deadline_success=bool(whole and whole["navigation_success"]),
        timely_suffix_supported=bool(prefix_safe and timely and timely["navigation_success"]),
        observed_suffix_ticks=n - switch_tick,
        original_remaining_ticks=max(0, task["deadline_ticks"] - switch_tick),
        diagnostic_extension_ticks=max(0, n - task["deadline_ticks"]),
        local_score=local,
        timely_suffix_score=timely,
    )
