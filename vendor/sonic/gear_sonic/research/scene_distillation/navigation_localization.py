"""Causal pose-difference velocity for the explicitly localized known-map actor."""

from collections import deque

import numpy as np

from gear_sonic.research.hindsight_training.observations import rotation_wxyz


class CausalLocalization:
    """Five-control-interval backward estimate, rotated into the current body frame.

    Input timestamps are measurement times, not reference phase. The first five
    decisions are invalid/zero. Reset at every episode and goal change. This
    pilot assumes zero-delay localized poses; camera estimation is not supplied.
    """

    def __init__(self, lag=5):
        if not isinstance(lag, int) or lag < 1:
            raise ValueError("Localization lag must be positive")
        self.lag = lag
        self.samples = deque(maxlen=lag + 1)
        self.goal = None

    def update(self, position, quaternion, timestamp, goal):
        position, goal = np.asarray(position), np.asarray(goal)
        rotation = rotation_wxyz(quaternion)
        if (
            position.shape != (3,)
            or goal.shape != (3,)
            or not all(np.isfinite(v).all() for v in (position, goal, timestamp))
        ):
            raise ValueError("Invalid localization measurement")
        if self.goal is None or not np.array_equal(goal, self.goal):
            self.samples.clear()
            self.goal = goal.copy()
        if self.samples and timestamp <= self.samples[-1][0]:
            raise ValueError("Localization timestamps must increase; reset each episode")
        self.samples.append((float(timestamp), position.copy()))
        if len(self.samples) < self.lag + 1:
            return np.zeros(4, np.float32)
        old_time, old_position = self.samples[0]
        velocity = rotation.T @ ((position - old_position) / (timestamp - old_time))
        return np.asarray([*velocity, 1.0], np.float32)


def episode_localization(arrays, goal):
    """Reconstruct before masking rows, preserving actual chronological gaps."""
    n = len(arrays["proprio"])
    for key, shape in {
        "measured_root_xyz": (n, 3),
        "measured_root_wxyz": (n, 4),
        "observation_time_s": (n,),
    }.items():
        if key not in arrays or tuple(arrays[key].shape) != shape:
            raise ValueError(f"Missing causal localization evidence: {key}")
    estimator = CausalLocalization()
    return np.stack(
        [
            estimator.update(p, q, float(t), goal)
            for p, q, t in zip(
                arrays["measured_root_xyz"].numpy(),
                arrays["measured_root_wxyz"].numpy(),
                arrays["observation_time_s"].numpy(),
            )
        ]
    )
