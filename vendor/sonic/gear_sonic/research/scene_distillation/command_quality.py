"""Measure commanded behavior separately from exact reference-pose reproduction."""

import numpy as np


def command_metrics(command, measured, valid_mask, *, dt=0.02):
    """Summarize synchronized [vx, vy, yaw rate, height] before any reset/censoring.

    These measurements do not certify navigation success or collision-free locomotion.
    A termination policy must be declared alongside the recorded trace.
    """
    command, measured, mask = np.asarray(command), np.asarray(measured), np.asarray(valid_mask)
    if command.ndim != 2 or command.shape[1] != 4 or measured.shape != command.shape:
        raise ValueError("Expected T,4 command and measured traces")
    if mask.shape != (len(command),) or mask.dtype != bool or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid trace availability or timing")
    if not mask.any() or not all(np.isfinite(x[mask]).all() for x in (command, measured)):
        raise ValueError("No finite uncensored command measurements")
    error = measured[mask] - command[mask]
    return {
        "measured_steps": int(mask.sum()),
        "measured_seconds": float(mask.sum() * dt),
        "velocity_xy_rmse_mps": float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))),
        "yaw_rate_rmse_radps": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "height_rmse_m": float(np.sqrt(np.mean(error[:, 3] ** 2))),
        "navigation_success": None,
    }
