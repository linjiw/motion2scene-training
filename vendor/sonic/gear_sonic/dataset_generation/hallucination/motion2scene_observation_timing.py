"""Audit when actual range measurements expose a feasible adaptation decision.

Scene geometry is privileged audit input only. No scene fields are appended to
student observations. This finite timing check is not a sensor-noise guarantee.
"""

import numpy as np

from .motion2scene_course import validate_beams
from .motion2scene_observation_history import SensorRay


def observed_beam_faces(measurements, beam, tolerance_m=1e-4):
    """Associate recorded nearest hit points with a native-audited beam surface."""
    validate_beams([beam])
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("finite nonnegative surface tolerance required")
    c, s = np.cos(beam["yaw_rad"]), np.sin(beam["yaw_rad"])
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    center = np.r_[beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2]
    half = np.array([beam["length_m"], beam["width_m"], beam["thickness_m"]]) / 2
    hits, underside = 0, 0
    for ray in measurements:
        SensorRay(**ray)
        distance = ray["hit_distance_m"]
        if distance is None:
            continue
        endpoint = np.asarray(ray["origin_w"]) + distance * np.asarray(ray["direction_w"])
        local = (endpoint - center) @ rotation
        if (np.abs(local) <= half + tolerance_m).all() and np.isclose(
            np.abs(local), half, atol=tolerance_m, rtol=0
        ).any():
            hits += 1
            normal = ray.get("hit_normal_w")
            underside += int(
                abs(local[2] + half[2]) <= tolerance_m
                and normal is not None
                and np.asarray(normal)[2] < -0.5
            )
    return {"beam_surface_rays": hits, "underside_rays": underside}


def audit_decision_visibility(observations, beam, entry_tick, *, collision_enabled=True):
    """Separate capture visibility from causal delivery before the legal entry."""
    if not observations or type(entry_tick) is not int or entry_tick <= 0:
        raise ValueError("nonempty sensor history and positive command entry tick required")
    ticks = np.array([row["tick"] for row in observations])
    elapsed = np.array([row["capture_elapsed_s"] for row in observations], dtype=float)
    if not np.isfinite(elapsed).all() or not np.all(np.diff(elapsed) > 0):
        raise ValueError("strictly increasing finite sensor capture chronology required")
    entry = np.flatnonzero(ticks == entry_tick)
    if len(entry) != 1:
        raise ValueError("entry must occur exactly once in captured command history")
    faces = [observed_beam_faces(row["measurements"], beam) for row in observations]
    if not collision_enabled:
        faces = [{"beam_surface_rays": 0, "underside_rays": 0} for _ in observations]
    result = {"entry_tick": entry_tick, "entry_capture_elapsed_s": float(elapsed[entry[0]])}
    for name, key in (("surface", "beam_surface_rays"), ("underside", "underside_rays")):
        capture = [i for i, value in enumerate(faces) if value[key] > 0]
        first_delivery = None
        for i, row in enumerate(observations):
            delivered = row["delivered_capture_elapsed_s"]
            if delivered is None:
                continue
            source = np.flatnonzero(np.isclose(elapsed, delivered, atol=1e-8, rtol=0))
            if len(source) != 1 or source[0] > i:
                raise ValueError("delivery must bind exactly one available captured sensor frame")
            if faces[int(source[0])][key] > 0:
                first_delivery = i
                break
        result[name] = {
            "first_capture_elapsed_s": None if not capture else float(elapsed[capture[0]]),
            "first_delivery_elapsed_s": (
                None if first_delivery is None else float(elapsed[first_delivery])
            ),
            "delivered_by_entry": bool(first_delivery is not None and first_delivery <= entry[0]),
            "captured_ray_hits": sum(value[key] for value in faces),
        }
    result["scope"] = (
        "Native nearest ray endpoints geometrically associated with audited static box; "
        "underside additionally requires measured downward normal. No semantic identity "
        "or scene parameters are supplied to the student. Visibility does not prove "
        "distinguishability of all option costs or noise robustness."
    )
    return result


def transition_timing_screen(
    height_m,
    elapsed_s,
    entry_elapsed_s,
    underside_m,
    obstacle_elapsed_s,
    visibility,
    *,
    clearance_margin_m=0.01,
    timing_margin_s=0.1,
    stable_samples=5
):
    """Measured complete-body low-height onset plus causal sensing deadline.

    The separately checked full-trajectory geometry screen remains necessary:
    passing this onset test alone does not prove that adaptation lasts long enough.
    """
    heights, times = np.asarray(height_m), np.asarray(elapsed_s)
    numbers = [
        entry_elapsed_s,
        underside_m,
        obstacle_elapsed_s,
        clearance_margin_m,
        timing_margin_s,
    ]
    if (
        heights.ndim != 1
        or heights.shape != times.shape
        or len(times) < stable_samples
        or not np.isfinite(heights).all()
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
        or not np.isfinite(numbers).all()
        or clearance_margin_m < 0
        or timing_margin_s < 0
        or type(stable_samples) is not int
        or stable_samples <= 0
    ):
        raise ValueError("finite aligned motion and valid timing/margin requirements needed")
    low = (heights <= underside_m - clearance_margin_m) & (times >= entry_elapsed_s)
    starts = [
        i for i in range(len(times) - stable_samples + 1) if low[i : i + stable_samples].all()
    ]
    ready = None if not starts else float(times[starts[0]])
    seen = visibility["surface"]["first_delivery_elapsed_s"]
    transition = None if ready is None else ready - entry_elapsed_s
    deadline = None if seen is None or transition is None else seen + transition + timing_margin_s
    return {
        "ready_elapsed_s": ready,
        "transition_s": transition,
        "stable_samples": stable_samples,
        "clearance_margin_m": clearance_margin_m,
        "timing_margin_s": timing_margin_s,
        "obstacle_elapsed_s": obstacle_elapsed_s,
        "seen_plus_transition_plus_margin_s": deadline,
        "eligible": bool(
            visibility["surface"]["delivered_by_entry"]
            and deadline is not None
            and deadline <= obstacle_elapsed_s
        ),
        "scope": (
            "Finite measured onset and actual delivered sensor cue; "
            "full trajectory solution screen still required"
        ),
    }
