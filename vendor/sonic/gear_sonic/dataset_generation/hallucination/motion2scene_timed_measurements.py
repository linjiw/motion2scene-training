"""Physical contact admission for explicit finite schedule qualification."""

import numpy as np

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def audit_whole_body_contacts(capture, expected_physics_steps, threshold_n=1.0):
    """Audit recorded normal-force sums and pair-resolved support contacts.

    This detects non-foot net contact and foot contact unexplained by the support
    plane. It does not recover every contact point or tangential friction force;
    distant-wall clearance must be checked independently from actual geometry.
    """
    net = np.asarray(capture["net_force_w"])
    left = np.asarray(capture["left_floor_force_w"])
    right = np.asarray(capture["right_floor_force_w"])
    names = [str(name) for name in capture["body_names"]]
    steps = np.asarray(capture["physics_steps"])
    controls = np.asarray(capture["control_steps"])
    n = len(steps)
    if (
        n == 0
        or net.shape != (n, len(names), 3)
        or left.shape != (n, 1, 3)
        or right.shape != (n, 1, 3)
        or len(set(names)) != len(names)
        or any(foot not in names for foot in FOOT_BODIES)
        or not np.isfinite(threshold_n)
        or threshold_n <= 0
        or not all(np.isfinite(value).all() for value in (net, left, right))
    ):
        raise ValueError("complete finite all-body and both foot-to-floor force arrays required")
    nonfoot = [i for i, name in enumerate(names) if name not in FOOT_BODIES]
    nonfoot_norm = np.linalg.norm(net[:, nonfoot], axis=-1)
    residuals = np.stack(
        [
            net[:, names.index(FOOT_BODIES[0])] - left[:, 0],
            net[:, names.index(FOOT_BODIES[1])] - right[:, 0],
        ],
        axis=1,
    )
    residual_norm = np.linalg.norm(residuals, axis=-1)
    complete = (
        n == expected_physics_steps
        and np.array_equal(steps, np.arange(1, n + 1))
        and np.array_equal(controls, np.arange(4, n + 1, 4))
    )
    maximum_nonfoot = float(nonfoot_norm.max(initial=0))
    maximum_residual = float(residual_norm.max(initial=0))
    return {
        "complete_synchronized_streams": bool(complete),
        "recorded_physics_steps": n,
        "threshold_n": threshold_n,
        "maximum_nonfoot_normal_force_n": maximum_nonfoot,
        "maximum_foot_force_unexplained_by_floor_n": maximum_residual,
        "nonfoot_exceeding_physics_steps": int((nonfoot_norm > threshold_n).any(1).sum()),
        "foot_residual_exceeding_physics_steps": int((residual_norm > threshold_n).any(1).sum()),
        "no_undesired_measured_contact": maximum_nonfoot <= threshold_n
        and maximum_residual <= threshold_n,
        "scope": (
            "200Hz per-body net normal forces and pair-resolved foot/support-plane forces; "
            "no individual contact-point or friction-force guarantee"
        ),
    }
