"""Preserve each beam's worst substep contact from explicit native counterpart data."""

import re

import numpy as np

BEAM_PATTERN = re.compile(r"/World/ground/terrain/CounterfactualBeam(?:_\d{2})?")


def synchronize_pair_beam_forces(pair_capture, mapping, beam_paths, control_steps):
    """Return [control, beam, body, xyz] without cancellation across substeps.

    Each subject's native filter mapping determines its columns. The output beam
    order is the explicit caller order. All captured beam counterparts must be
    included: silently omitting an encountered second beam is an error.
    """
    names = [str(name) for name in pair_capture["body_names"]]
    beams = list(beam_paths)
    force = np.asarray(pair_capture["force_w"])
    steps = np.asarray(pair_capture["physics_steps"])
    controls = np.asarray(control_steps)
    if (
        not beams
        or len(beams) != len(set(beams))
        or any(BEAM_PATTERN.fullmatch(path) is None for path in beams)
        or len(set(names)) != len(names)
        or not names
        or mapping.get("pair_subject_body_names") != names
        or force.ndim != 4
        or force.shape[1] != len(names)
        or force.shape[-1] != 3
        or not np.isfinite(force).all()
        or not len(force)
        or len(force) % 4
        or float(pair_capture["physics_dt_s"]) != 0.005
        or steps.dtype.kind not in "iu"
        or controls.dtype.kind not in "iu"
        or not np.array_equal(steps, np.arange(1, len(force) + 1))
        or not np.array_equal(controls, np.arange(4, len(force) + 1, 4))
    ):
        raise ValueError("complete ordered200Hz pairs and four-substep control blocks required")
    extracted = np.empty((len(force), len(beams), len(names), 3), dtype=force.dtype)
    columns = {}
    for body_index, body in enumerate(names):
        sensor = mapping.get("sensors", {}).get(body, {})
        paths = sensor.get("filter_paths", [])
        if (
            sensor.get("sensor_body_names") != [body]
            or sensor.get("native_body_paths") != [f"/World/envs/env_0/Robot/{body}"]
            or len(paths) != force.shape[2]
            or len(set(paths)) != len(paths)
            or sensor.get("native_filter_count") != len(paths)
            or {path for path in paths if BEAM_PATTERN.fullmatch(path)} != set(beams)
        ):
            raise ValueError("explicit per-body native filters must include every declared beam")
        indices = [paths.index(beam) for beam in beams]
        extracted[:, :, body_index] = force[:, body_index, indices]
        columns[body] = indices
    blocks = extracted.reshape(len(controls), 4, len(beams), len(names), 3)
    peak = np.linalg.norm(blocks, axis=-1).argmax(axis=1)
    values = np.take_along_axis(blocks, peak[:, None, :, :, None], axis=1)[:, 0]
    return values, {
        "schema": "motion2scene_timed_course_contact_synchronization_v1",
        "physics_steps": len(force),
        "control_steps": len(controls),
        "beam_paths": beams,
        "body_names": names,
        "native_columns_by_body": columns,
        "shape": list(values.shape),
        "maximum_force_n_by_beam": np.linalg.norm(values, axis=-1).max(axis=(0, 2)).tolist(),
        "reduction": "retain maximum-normal-force-norm vector per beam/body/four-substep block",
        "scope": "synchronized normal contacts; external contact and passage audits remain separate",
    }
