"""Audit explicit contact partners without confusing self-contact with obstacles."""

import numpy as np

FOOT_BODIES = {"left_ankle_roll_link", "right_ankle_roll_link"}
STRUCTURE = "/World/ground/terrain/Structure"
STRUCTURE_PATHS = [
    f"{STRUCTURE}/{name}" for name in ("Floor", "WallWest", "WallEast", "WallSouth", "WallNorth")
]
BEAM_PATH = "/World/ground/terrain/CounterfactualBeam"
EXTERNAL_PATHS = STRUCTURE_PATHS + [BEAM_PATH]


def validated_beam_paths(beam_paths=None):
    """Allow a nonempty consecutive set of the explicitly authored beam prims."""
    if beam_paths is None:
        return [BEAM_PATH]
    if isinstance(beam_paths, str):
        raise ValueError("beam paths must be a nonempty sequential list")
    paths = list(beam_paths)
    expected = [BEAM_PATH] + [f"{BEAM_PATH}_{index:02d}" for index in range(1, len(paths))]
    if not paths or paths != expected:
        raise ValueError("beam paths must be the exact consecutive CounterfactualBeam prims")
    return paths


def environment_counterpart_paths(beam_paths=None):
    return STRUCTURE_PATHS + validated_beam_paths(beam_paths)


def audit_environment_contacts(
    pair_capture,
    net_capture,
    mapping,
    expected_physics_steps,
    threshold_n=1.0,
    neutral_self_pairs=None,
    beam_paths=None,
):
    """Separate stream completeness, environment failure, and self-contact reports.

    All vectors are measured normal contact forces. Foot-floor support is allowed;
    nonfoot-floor, wall and beam contact share the same declared force threshold.
    Self-contact is reported, including pairs absent from the supplied neutral set.
    A physically failed passage can have complete valid measurement streams.
    """
    result = {
        "schema": "motion2scene_environment_contact_audit_v1",
        "complete_synchronized_streams": False,
        "no_undesired_measured_contact": False,
        "recorded_physics_steps": len(pair_capture.get("physics_steps", [])),
        "normal_force_threshold_n": float(threshold_n),
        "normal_forces_only": True,
    }
    required = {"force_w", "physics_steps", "physics_dt_s", "body_names"}
    if not required.issubset(pair_capture) or not {
        "net_force_w",
        "physics_steps",
        "physics_dt_s",
        "body_names",
    }.issubset(net_capture):
        return {**result, "measurement_error": "missing required contact arrays"}
    names = list(pair_capture["body_names"])
    net_names = list(net_capture["body_names"])
    values = np.asarray(pair_capture["force_w"])
    net = np.asarray(net_capture["net_force_w"])
    count = len(names)
    external_paths = environment_counterpart_paths(beam_paths)
    expected_paths = [f"/World/envs/env_0/Robot/{body}" for body in names] + external_paths
    clocks_ok = (
        np.array_equal(pair_capture["physics_steps"], np.arange(1, expected_physics_steps + 1))
        and np.array_equal(pair_capture["physics_steps"], net_capture["physics_steps"])
        and float(pair_capture["physics_dt_s"]) == float(net_capture["physics_dt_s"]) == 0.005
    )
    mapping_ok = (
        len(set(names)) == count
        and set(net_names) == set(names)
        and mapping.get("pair_subject_body_names") == names
        and mapping.get("robot_articulation_body_names") == names
        and mapping.get("net_body_names") == net_names
        and mapping.get("net_native_body_paths")
        == [f"/World/envs/env_0/Robot/{body}" for body in net_names]
    )
    for body in names:
        sensor = mapping.get("sensors", {}).get(body, {})
        mapping_ok &= (
            sensor.get("sensor_body_names") == [body]
            and sensor.get("native_body_paths") == [f"/World/envs/env_0/Robot/{body}"]
            and sensor.get("filter_paths") == expected_paths
            and sensor.get("native_filter_count") == count + len(external_paths)
        )
    shapes_ok = values.shape == (
        expected_physics_steps,
        count,
        count + len(external_paths),
        3,
    ) and net.shape == (
        expected_physics_steps,
        count,
        3,
    )
    if not (clocks_ok and mapping_ok and shapes_ok and np.isfinite(values).all()):
        return {
            **result,
            "measurement_error": "incomplete clock, invalid native mapping, shape or forces",
            "clocks_valid": bool(clocks_ok),
            "mapping_valid": bool(mapping_ok),
            "shapes_valid": bool(shapes_ok),
        }
    aligned_net = net[:, [net_names.index(body) for body in names]]
    residual = np.linalg.norm(values.sum(axis=2) - aligned_net, axis=-1)
    max_residual = float(residual.max(initial=0))
    complete = np.isfinite(aligned_net).all() and max_residual <= 0.001
    norms = np.linalg.norm(values, axis=-1)
    external = norms[:, :, count:].copy()
    for i, name in enumerate(names):
        if name in FOOT_BODIES:
            external[:, i, 0] = 0
    undesired = external.max(axis=(1, 2))
    neutral_set = {tuple(sorted(pair)) for pair in neutral_self_pairs or []}
    self_rows = []
    for i, body in enumerate(names):
        for j in range(i + 1, count):
            maximum = float(max(norms[:, i, j].max(), norms[:, j, i].max()))
            if maximum == 0:
                continue
            pair = tuple(sorted((body, names[j])))
            self_rows.append(
                {
                    "body_pair": list(pair),
                    "maximum_force_n": maximum,
                    "physics_steps_above_threshold": int(
                        np.count_nonzero(np.maximum(norms[:, i, j], norms[:, j, i]) > threshold_n)
                    ),
                    "maximum_equal_opposite_residual_n": float(
                        np.linalg.norm(values[:, i, j] + values[:, j, i], axis=-1).max()
                    ),
                    "present_in_declared_neutral_pairs": pair in neutral_set,
                }
            )
    external_rows = []
    for i, name in enumerate(names):
        for j, path in enumerate(external_paths):
            maximum = float(norms[:, i, count + j].max())
            if maximum > 0:
                external_rows.append(
                    {
                        "body": name,
                        "counterpart": path,
                        "maximum_force_n": maximum,
                        "allowed_foot_floor_support": name in FOOT_BODIES and j == 0,
                    }
                )
    return {
        **result,
        "complete_synchronized_streams": bool(complete),
        "no_undesired_measured_contact": bool(complete and (undesired <= threshold_n).all()),
        "maximum_pair_sum_net_residual_n": max_residual,
        "maximum_undesired_environment_force_n": float(undesired.max(initial=0)),
        "physics_steps_with_undesired_environment_contact": int(
            np.count_nonzero(undesired > threshold_n)
        ),
        "self_contacts": self_rows,
        "unexpected_self_contact_pairs": [
            row["body_pair"] for row in self_rows if not row["present_in_declared_neutral_pairs"]
        ],
        "external_contacts": external_rows,
        "scope": (
            "Every robot body against exact robot/floor/four-wall/beam paths; "
            "self-contact is retained separately, not classified as an environment collision."
        ),
    }
