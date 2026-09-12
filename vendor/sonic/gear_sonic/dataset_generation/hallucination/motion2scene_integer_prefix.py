"""Exact matched histories with numerical validation of the recorded 50 Hz clock."""

import numpy as np

PREFIX_KEYS = (
    "dof_pos",
    "dof_vel",
    "root_pos_w",
    "root_quat_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "applied_joint_action",
    "action_motion_token",
    "reference_g1_qpos",
    "motion_time_s",
)


def paired_prefix_on_ticks(first, second, decision_time_s):
    """Validate integer grid identities; retain exact cross-branch recorded equality."""
    scaled = float(decision_time_s) * 50
    if not np.isfinite(scaled) or abs(scaled - round(scaled)) > 1e-8 or scaled < 1:
        raise ValueError("decision must be a positive exact50Hz tick")
    count = int(round(scaled))
    residuals = []
    for payload in (first, second):
        times = np.asarray(payload["motion_time_s"], dtype=float).reshape(-1)[:count]
        if len(times) != count or not np.isfinite(times).all():
            raise ValueError("incomplete pre-decision clock")
        ticks = np.rint(times * 50).astype(np.int64)
        residual = float(np.max(np.abs(times - ticks / 50)))
        if not np.array_equal(ticks, np.arange(count)) or residual > 1e-9:
            raise ValueError("incomplete or off-grid pre-decision clock")
        residuals.append(residual)
    errors = {}
    for key in PREFIX_KEYS:
        a, b = np.asarray(first[key])[:count], np.asarray(second[key])[:count]
        if (
            a.shape != b.shape
            or len(a) != count
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
        ):
            raise ValueError("malformed paired prefix")
        errors[key] = float(np.max(np.abs(a - b)))
    return {
        "frames": count,
        "max_abs_errors": errors,
        "exact_match": all(value == 0 for value in errors.values()),
        "clock_validation": {
            "rule": "consecutive integer50Hz tick identity plus <=1e-9s numerical residual",
            "maximum_residual_s_by_branch": residuals,
            "cross_branch_recorded_arrays": "unchanged exact equality, including recorded times",
        },
    }
