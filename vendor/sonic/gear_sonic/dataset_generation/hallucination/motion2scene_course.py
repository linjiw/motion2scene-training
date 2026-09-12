"""Static overhead course authoring and first-episode whole-course scoring."""

import math
import re

import numpy as np

from .motion2scene_passage import score_passage

COURSE_SCHEMA = "motion2scene_development_course_v1"


def reference_frame_budget(source_frames, source_fps):
    """Match the installed loader's exclusive-end interpolation, without padding."""
    if (
        type(source_frames) is not int
        or source_frames < 2
        or not math.isfinite(source_fps)
        or source_fps <= 0
    ):
        raise ValueError("finite source frame count and positive fps required")
    source_duration = (source_frames - 1) / source_fps
    # MotionLibRobot skips interpolation at 50 Hz; otherwise interploate_pose
    # uses arange(0, source endpoint duration, 1/50), excluding the endpoint.
    frames = (
        source_frames
        if source_fps == 50
        else len(np.arange(0, source_duration, 1 / 50, dtype=np.float32))
    )
    if frames < 2:
        raise ValueError("source must yield at least two installed reference frames")
    return {
        "frames": frames,
        "fps": 50,
        "duration_s": (frames - 1) / 50,
        "source_frames": source_frames,
        "source_fps": source_fps,
        "source_endpoint_duration_s": source_duration,
        "resampling": "existing exclusive-end MotionLibRobot interpolation; source assets unchanged",
    }


def validate_beams(beams):
    if not isinstance(beams, list) or not beams:
        raise ValueError("at least one overhead constraint required")
    for beam in beams:
        xy = np.asarray(beam["center_xy_m"])
        values = [beam[k] for k in ("yaw_rad", "length_m", "width_m", "thickness_m", "underside_m")]
        if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(values).all():
            raise ValueError("finite beam pose and dimensions required")
        if min(values[1:]) <= 0:
            raise ValueError("positive beam dimensions and overhead clearance required")


def validate_course_definition(definition):
    """This initial adapter admits fresh development definitions only."""
    if (
        definition.get("schema") != COURSE_SCHEMA
        or definition.get("split") != "development"
        or not isinstance(definition.get("course_id"), str)
        or re.fullmatch(r"[a-zA-Z0-9_-]+", definition["course_id"]) is None
        or definition.get("motion_mode") != "one_authored_reference_pass"
    ):
        raise ValueError("requires an explicit fresh-development course schema")
    validate_beams(definition["beams"])
    budget = definition["reference_budget"]
    frames, fps = budget["frames"], budget["fps"]
    if type(frames) is not int or frames < 2 or fps != 50:
        raise ValueError("requires a finite 50 Hz reference frame budget")
    if not math.isclose(budget["duration_s"], (frames - 1) / fps, abs_tol=1e-9):
        raise ValueError("reference duration and frame count disagree")
    if (
        not math.isfinite(definition["timeout_s"])
        or not 0 < definition["timeout_s"] <= budget["duration_s"]
    ):
        raise ValueError("course timeout must fit the existing reference duration")


def beam_prim_name(index):
    if type(index) is not int or index < 0:
        raise ValueError("beam index must be a nonnegative integer")
    return "CounterfactualBeam" if index == 0 else f"CounterfactualBeam_{index:02d}"


def author_course(template_usda, beams, *, course_id=None):
    """Keep the shared room; author every beam with its actual length/thickness.

    The template must contain the inherited final single-beam block. No geometry
    screen or physical feasibility is implied by creating this USD scene.
    """
    validate_beams(beams)
    marker = '    def Cube "CounterfactualBeam"'
    if template_usda.count(marker) != 1:
        raise ValueError("requires the shared room template with one final beam block")
    prefix, tail = template_usda.split(marker)
    # Only replace a final Cube in the supplied single-root room template. A
    # mismatched tail must fail instead of silently deleting other scene prims.
    cleaned = re.sub(r'"(?:\\.|[^"\\])*"|#[^\n]*', '""', tail)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("template beam lacks a body")
    depth, end = 0, None
    for offset in range(start, len(cleaned)):
        depth += (cleaned[offset] == "{") - (cleaned[offset] == "}")
        if depth == 0:
            end = offset + 1
            break
    if end is None or cleaned[end:].strip() != "}":
        raise ValueError("beam must be the final prim in the shared single-root template")
    prefix = "\n".join(
        line for line in prefix.splitlines() if not line.startswith("#") or line.startswith("#usda")
    )
    if course_id is not None:
        if re.fullmatch(r"[a-zA-Z0-9_-]+", course_id) is None:
            raise ValueError("invalid course scene identifier")
        for key, value in (("sceneId", course_id), ("splitGroup", f"development_{course_id}")):
            prefix = re.sub(
                rf'(custom string g1Dataset:{key} = )"[^"]*"',
                lambda match: match[1] + f'"{value}"',
                prefix,
            )
    blocks = []
    for i, beam in enumerate(beams):
        x, y = beam["center_xy_m"]
        z = beam["underside_m"] + beam["thickness_m"] / 2
        length, width, thickness = (beam[k] for k in ("length_m", "width_m", "thickness_m"))
        blocks.append(f"""    def Cube "{beam_prim_name(i)}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsRigidBodyAPI",
                             "PhysxCollisionAPI", "PhysxContactReportAPI"]
    ) {{
        double size = 1
        bool physics:collisionEnabled = true
        bool physics:rigidBodyEnabled = true
        bool physics:kinematicEnabled = true
        float physxCollision:contactOffset = 0.002
        float physxCollision:restOffset = 0
        float physxContactReport:threshold = 0
        double3 xformOp:translate = ({x:.12g}, {y:.12g}, {z:.12g})
        double xformOp:rotateZ = {math.degrees(beam["yaw_rad"]):.12g}
        double3 xformOp:scale = ({length:.12g}, {width:.12g}, {thickness:.12g})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.65, 0.37, 0.15)]
    }}""")
    return (
        prefix
        + "\n# Fresh development course; no feasibility implied.\n"
        + "\n".join(blocks)
        + "\n}\n"
    )


def audit_imported_beams(shapes, beams):
    """Verify composed cube dimensions, orientation and pose for every beam."""
    validate_beams(beams)
    imported = {row["path"]: row for row in shapes}
    result = []
    for index, beam in enumerate(beams):
        path = f"/World/ground/terrain/{beam_prim_name(index)}"
        row = imported.get(path)
        if row is None or row["type"] != "Cube" or not row["collision"] or not row["rigid_body"]:
            raise ValueError("authored course beam missing from composed collision geometry")
        matrix = np.asarray(row["local_to_world_at_capture_start"], dtype=float)
        size = float(row["attributes"]["size"])
        yaw = beam["yaw_rad"]
        c, s = math.cos(yaw), math.sin(yaw)
        expected = np.eye(4)
        expected[:3, :3] = np.diag(
            [beam["length_m"], beam["width_m"], beam["thickness_m"]]
        ) @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        expected[3, :3] = [*beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2]
        actual = matrix.copy()
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not math.isfinite(size)
            or size <= 0
        ):
            raise ValueError("invalid composed beam transform")
        actual[:3, :3] *= size
        if not np.allclose(actual, expected, atol=1e-6, rtol=0):
            raise ValueError("composed beam pose or dimensions differ from course definition")
        result.append(
            {"path": path, "maximum_transform_error": float(np.abs(actual - expected).max())}
        )
    return result


def synchronize_course_forces(physics):
    """Retain the largest substep force vector per beam/body in each control tick."""
    steps = np.asarray(physics["physics_steps"])
    controls = np.asarray(physics["control_steps"])
    force = np.asarray(physics["physics_force_w"])
    sampled = np.asarray(physics["control_force_w"])
    if (
        force.ndim != 4
        or not all(force.shape[axis] > 0 for axis in (0, 1, 2))
        or force.shape[-1] != 3
        or sampled.shape != (len(controls), *force.shape[1:])
        or len(steps) != len(force)
        or steps.ndim != 1
        or controls.ndim != 1
        or not len(controls)
        or not np.isfinite(force).all()
        or not np.isfinite(sampled).all()
        or not np.isfinite(steps).all()
        or not np.isfinite(controls).all()
        or np.any(steps < 0)
        or np.any(controls < 0)
        or np.any(steps != np.floor(steps))
        or np.any(controls != np.floor(controls))
        or not np.all(np.diff(steps) == 1)
        or not np.all(np.diff(controls) == 4)
        or float(physics["physics_dt"]) != 0.005
    ):
        raise ValueError("requires complete finite 200 Hz multi-beam contact capture")
    expected = controls[:, None] - np.arange(3, -1, -1)
    indices = np.searchsorted(steps, expected)
    if np.any(indices >= len(steps)) or not np.array_equal(steps[indices], expected):
        raise ValueError("missing multi-beam physics substeps")
    blocks = force[indices]
    error = float(np.abs(blocks[:, -1] - sampled).max())
    if error > 1e-5:
        raise ValueError("additional-beam cached/direct contact disagreement")
    maxima = np.argmax(np.linalg.norm(blocks, axis=-1), axis=1)
    peak_vectors = np.take_along_axis(blocks, maxima[:, None, :, :, None], axis=1)[:, 0]
    return peak_vectors, error


def score_course(
    payload,
    forces,
    beams,
    *,
    commands_valid,
    timeout_s,
    reference_frames=None,
    final_reference_phase_s=None,
):
    """Score all constraints and contacts through common first-episode completion.

    ``forces`` is synchronized (control frame, beam, body, xyz) physics-window
    maxima. Every beam's contacts remain in scope until the course finishes,
    including contacts after the robot initially crossed an earlier constraint.
    """
    validate_beams(beams)
    force = np.asarray(forces)
    if (
        force.ndim != 4
        or force.shape[1] != len(beams)
        or not np.isfinite(force).all()
        or not isinstance(commands_valid, bool)
        or not np.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError("complete course forces and valid command/timeout audit required")
    zero = np.zeros_like(force[:, 0])
    crossings = [score_passage(payload, zero, beam) for beam in beams]
    ends = [row["passage_finish_frame_exclusive"] for row in crossings]
    completed = all(end is not None for end in ends)
    order_valid = completed and all(a <= b for a, b in zip(ends, ends[1:]))
    combined = crossings[-1]
    finish = max(ends) if order_valid else None
    horizon = finish if finish is not None else combined["first_episode_frames"]
    times = np.asarray(payload["motion_time_s"])
    root = np.asarray(payload["root_pos_w"])
    gravity = np.asarray(payload["projected_gravity_b"])
    no_fall = bool((root[:horizon, 2] >= 0.5).all() and (-gravity[:horizon, 2] >= 0.5).all())
    elapsed = float(times[horizon - 1] - times[0])
    max_forces = np.linalg.norm(force[:horizon], axis=-1)
    maximum = float(max_forces.max())
    contact = maximum > 1.0
    positions = np.asarray(payload["body_pos_w"])
    initially_upstream = []
    for beam in beams:
        normal = np.array([math.cos(beam["yaw_rad"]), math.sin(beam["yaw_rad"])])
        initially_upstream.append(
            bool(
                ((positions[0, :, :2] - beam["center_xy_m"]) @ normal).max() < -beam["length_m"] / 2
            )
        )
    exhausted = False
    if reference_frames is not None:
        if (
            type(reference_frames) is not int
            or reference_frames < 2
            or final_reference_phase_s is None
        ):
            raise ValueError("reference exhaustion needs the recorded finite phase limit")
        if not np.isfinite(final_reference_phase_s) or final_reference_phase_s < 0:
            raise ValueError("invalid final reference phase")
        if final_reference_phase_s > (reference_frames - 1) / payload["fps"] + 1e-8:
            raise ValueError("execution exceeded its qualified reference budget")
        exhausted = final_reference_phase_s >= (reference_frames - 1) / payload["fps"] - 1e-8
    failure_reasons = []
    for condition, reason in (
        (not order_valid, "not_all_constraints_completed_in_order"),
        (not all(initially_upstream), "invalid_initial_approach"),
        (contact, "course_beam_contact"),
        (not commands_valid, "invalid_command_execution"),
        (not no_fall, "fall_before_course_completion"),
        (elapsed > timeout_s, "course_timeout"),
        (finish is None and exhausted, "reference_horizon_reached_before_completion"),
        (finish is None and combined["reset_count"] > 0, "first_episode_reset_before_completion"),
    ):
        if condition:
            failure_reasons.append(reason)
    return {
        **combined,
        "pass": not failure_reasons,
        "passage_finish_frame_exclusive": finish,
        "incomplete_crossing": not completed,
        "stabilization_failed": not completed,
        "fall_observed": not no_fall,
        "fall_anywhere_in_first_episode": combined["fall_observed"],
        "observed_beam_contact": contact,
        "maximum_beam_normal_force_n_through_passage": maximum,
        "force_threshold_frame_counts": {
            str(t): int((max_forces.max(axis=(1, 2)) > t).sum()) for t in (0.1, 1, 10)
        },
        "failure_reasons": failure_reasons,
        "all_beams_crossed_in_order": bool(order_valid),
        "beam_finish_frames_exclusive": ends,
        "completed_beams": sum(end is not None for end in ends),
        "initially_upstream_by_beam": initially_upstream,
        "initially_upstream": all(initially_upstream),
        "invalid_initial_approach": not all(initially_upstream),
        "reference_horizon_reached": exhausted,
        "final_reference_phase_s": final_reference_phase_s,
        "commands_valid": commands_valid,
        "no_fall_through_course": no_fall,
        "within_timeout": elapsed <= timeout_s,
        "elapsed_through_scoring_horizon_s": elapsed,
        "beam_count": len(beams),
        "max_force_by_beam_n": max_forces.max(axis=(0, 2)).tolist(),
        "scope": "actual synchronized course execution; body-origin crossing and sampled contacts",
    }
