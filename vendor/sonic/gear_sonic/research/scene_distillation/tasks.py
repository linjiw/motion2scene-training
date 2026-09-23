"""Known-map navigation tasks and privileged trajectory labels with explicit availability."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gear_sonic.research.hindsight_training.observations import rotation_wxyz
from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.observations import navigation_observation

SCHEMA = "bfm_known_map_navigation_task_v1"

# Minimum horizontal distance from any obstacle footprint to the start pelvis XY and to the
# goal XY. The legacy 00399-corridor wall stood 0.23 m from both and produced a ~3.8 kN
# contact at the first control step.
MIN_START_GOAL_CLEARANCE_M = 0.35


def _convex_hull(points):
    """Counter-clockwise convex hull of 2-D points (monotone chain), without repeats."""
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64)})
    if len(pts) < 3:
        raise ValueError("Degenerate obstacle footprint")

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.asarray(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        raise ValueError("Degenerate obstacle footprint")
    return hull


def obstacle_footprint(obstacle):
    """Horizontal projection of one known-map primitive.

    Returns ("circle", center_xy, radius) for spheres and upright cylinders, otherwise
    ("polygon", ccw_vertices_xy) for the convex hull of the projected solid. Dimensions are
    full extents, as in write_collision_scene and scene_features.
    """
    shape = obstacle["shape"]
    center = np.asarray(obstacle["center_xyz"], dtype=np.float64)
    size = np.asarray(obstacle["full_dimensions_xyz"], dtype=np.float64)
    if center.shape != (3,) or size.shape != (3,) or not (size > 0).all():
        raise ValueError("Obstacle needs a 3-D center and positive full dimensions")
    if shape == "sphere":
        return ("circle", center[:2], size[0] / 2)
    rotation = rotation_wxyz(obstacle["quaternion_wxyz"])
    half = size / 2
    if shape in ("box", "beam"):
        local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    elif shape == "cylinder":
        if abs(rotation[2, 2]) >= 1 - 1e-9:
            return ("circle", center[:2], half[0])
        angle = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ring = np.stack([np.cos(angle) * half[0], np.sin(angle) * half[1]], -1)
        local = np.concatenate(
            [np.column_stack([ring, np.full(64, z)]) for z in (-half[2], half[2])]
        )
    else:
        raise ValueError(f"No footprint for obstacle shape {shape!r}")
    return ("polygon", _convex_hull((local @ rotation.T + center)[:, :2]))


def horizontal_clearance_m(obstacle, xy):
    """Horizontal distance from a point to an obstacle footprint; zero on or inside it."""
    p = np.asarray(xy, dtype=np.float64)[:2]
    footprint = obstacle_footprint(obstacle)
    if footprint[0] == "circle":
        return float(max(0.0, np.linalg.norm(p - footprint[1]) - footprint[2]))
    a = footprint[1]
    edge = np.roll(a, -1, axis=0) - a
    offset = p - a
    if (edge[:, 0] * offset[:, 1] - edge[:, 1] * offset[:, 0] >= 0).all():
        return 0.0
    t = np.clip((offset * edge).sum(1) / (edge * edge).sum(1), 0, 1)
    return float(np.linalg.norm(offset - t[:, None] * edge, axis=1).min())


def start_goal_clearance(obstacles, start_xyz, goal_xyz):
    """Per-obstacle horizontal clearance of the start pelvis and goal, in metres."""
    return [
        dict(
            obstacle_index=i,
            shape=obstacle["shape"],
            start_m=horizontal_clearance_m(obstacle, start_xyz),
            goal_m=horizontal_clearance_m(obstacle, goal_xyz),
        )
        for i, obstacle in enumerate(obstacles)
    ]


def require_start_goal_clearance(
    obstacles, start_xyz, goal_xyz, min_clearance_m=MIN_START_GOAL_CLEARANCE_M, label="task"
):
    """Fail loudly when any obstacle footprint crowds the start pelvis or the goal."""
    if not np.isfinite(min_clearance_m) or min_clearance_m < 0:
        raise ValueError("Clearance margin must be a finite nonnegative distance")
    rows = start_goal_clearance(obstacles, start_xyz, goal_xyz)
    bad = [
        f"obstacle {r['obstacle_index']} ({r['shape']}) {end} {r[end + '_m']:.3f} m"
        for r in rows
        for end in ("start", "goal")
        if r[end + "_m"] < min_clearance_m
    ]
    if bad:
        raise ValueError(
            f"{label}: obstacle footprint closer than {min_clearance_m} m horizontal to "
            "start pelvis/goal: " + "; ".join(bad)
        )
    return rows


def shift_to_clearance(
    obstacle,
    direction_xy,
    start_xyz,
    goal_xyz,
    min_clearance_m=MIN_START_GOAL_CLEARANCE_M,
    step_m=0.01,
    max_shift_m=1.0,
):
    """Smallest horizontal translation along direction_xy, in whole step_m increments, that
    restores start/goal clearance. Shape, size, height and orientation are kept.

    Returns (moved_obstacle, shift_m); shift_m is 0 when the obstacle already clears.
    """
    d = np.asarray(direction_xy, dtype=np.float64)[:2]
    if not np.isfinite(d).all() or np.linalg.norm(d) < 1e-9 or step_m <= 0:
        raise ValueError("Shift needs a nonzero direction and a positive step")
    d = d / np.linalg.norm(d)
    center = np.asarray(obstacle["center_xyz"], dtype=np.float64)
    for k in range(int(np.floor(max_shift_m / step_m + 1e-9)) + 1):
        shift = round(k * step_m, 9)
        xy = center[:2] + shift * d
        moved = dict(obstacle, center_xyz=[float(xy[0]), float(xy[1]), float(center[2])])
        rows = start_goal_clearance([moved], start_xyz, goal_xyz)
        if min(rows[0]["start_m"], rows[0]["goal_m"]) >= min_clearance_m:
            return moved, shift
    raise ValueError(f"No shift up to {max_shift_m} m restores {min_clearance_m} m clearance")


def navigation_request_sha256(task):
    """Identify a public request independently of its training-only motion/continuation ID."""
    keys = (
        "observation_profile",
        "world_frame",
        "anchor_body",
        "start_xyz",
        "start_wxyz",
        "goal_xyz",
        "goal_tolerance_m",
        "terminal_speed_mps",
        "hold_ticks",
        "deadline_ticks",
        "obstacles",
        "scene_usd_sha256",
    )
    value = {key: task[key] for key in keys}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_collision_scene(path, obstacles, start_xyz, goal_xyz):
    """Author exact known-map collision primitives and a z=0 support floor, without replay assets."""
    points = np.asarray([start_xyz, goal_xyz] + [o["center_xyz"] for o in obstacles])
    center = (points.min(0) + points.max(0)) / 2
    span = np.maximum(np.ptp(points, axis=0) + 20, 20)

    def vector(values):
        return "(" + ", ".join(str(float(v)) for v in values) + ")"

    text = [
        "#usda 1.0",
        '(defaultPrim = "World"; metersPerUnit = 1; upAxis = "Z")',
        'def Xform "World" {',
        'def Xform "Structure" {',
        'def Cube "Floor" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {',
        "double size = 1",
        "bool physics:collisionEnabled = true",
        f"double3 xformOp:translate = {vector([center[0], center[1], -0.05])}",
        f"double3 xformOp:scale = {vector([span[0], span[1], 0.1])}",
        'uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
        "}",
        "}",
        'def Xform "Obstacles" {',
    ]
    for i, obstacle in enumerate(obstacles, 1):
        shape = "Sphere" if obstacle["shape"] == "sphere" else "Cube"
        if obstacle["shape"] not in ("sphere", "box", "beam"):
            raise ValueError("Native scene author currently supports boxes/beams and spheres")
        text += [
            f'def {shape} "obstacle_{i}" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{',
            "bool physics:collisionEnabled = true",
            f'double3 xformOp:translate = {vector(obstacle["center_xyz"])}',
            f'quatd xformOp:orient = {vector(obstacle["quaternion_wxyz"])}',
        ]
        if shape == "Sphere":
            text += [
                f'double radius = {obstacle["full_dimensions_xyz"][0] / 2}',
                'uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
            ]
        else:
            text += [
                "double size = 1",
                f'double3 xformOp:scale = {vector(obstacle["full_dimensions_xyz"])}',
                'uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]',
            ]
        text += ["}"]
    text += ["}", "}"]
    with Path(path).open("x") as handle:
        handle.write("\n".join(text) + "\n")


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def validate_task(task, *, verify_files=True):
    """A task is a request; only a separate executed receipt qualifies its teacher."""
    if task.get("schema") != SCHEMA or task.get("split") not in ("train", "development"):
        raise ValueError("Unknown navigation task schema or split")
    if task.get("observation_profile") != "known_map_5_primitives_v1":
        raise ValueError("Unsupported navigation observation profile")
    if task.get("anchor_body") != "pelvis" or task.get("world_frame") != "scene_z_up_m_wxyz":
        raise ValueError("Task frame does not match native pelvis convention")
    if (
        task.get("goal_tolerance_m") != 0.25
        or task.get("terminal_speed_mps") != 0.1
        or task.get("hold_ticks") != 50
    ):
        raise ValueError("Task must use the declared navigation success profile")
    if not 1 <= task["deadline_ticks"] <= 3000:
        raise ValueError("Invalid task deadline")
    navigation_observation(
        proprio=np.zeros(930, np.float32),
        root_xyz=task["start_xyz"],
        root_wxyz=task["start_wxyz"],
        start_xyz=task["start_xyz"],
        goal_xyz=task["goal_xyz"],
        obstacles=task["obstacles"],
    )
    if verify_files:
        if sha(task["scene_usd_path"]) != task["scene_usd_sha256"]:
            raise ValueError("Task collision asset changed")
        for item in (task["reference"], task["source_scene"]):
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Task source/reference changed")
        if (
            task.get("native_motion")
            and sha(task["native_motion"]["path"]) != task["native_motion"]["sha256"]
        ):
            raise ValueError("Task native continuation changed")
    return task


def verify_native_motion(task, motion_lib):
    if task.get("native_motion"):
        loaded = Path(motion_lib.m_cfg.motion_file) / f"hindsight_{task['motion_id']}.pkl"
        if sha(loaded) != task["native_motion"]["sha256"]:
            raise ValueError("Loaded native continuation does not match task")


def trajectory_labels(time_s, xyz, now_s, root_xyz, root_wxyz, horizons=(0.2, 0.5, 1.0, 2.0)):
    """Training-only ordered future pelvis targets. Missing future stays masked, not repeated."""
    time_s = np.asarray(time_s, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    if time_s.ndim != 1 or len(time_s) < 2 or xyz.shape != (len(time_s), 3):
        raise ValueError("Invalid trajectory dimensions")
    if not np.isfinite(time_s).all() or not np.isfinite(xyz).all() or np.any(np.diff(time_s) <= 0):
        raise ValueError("Trajectory must be finite with strictly increasing timestamps")
    horizons = np.asarray(horizons, dtype=float)
    root_xyz = np.asarray(root_xyz, dtype=float)
    if (
        horizons.ndim != 1
        or not len(horizons)
        or not np.isfinite(horizons).all()
        or np.any(horizons <= 0)
        or np.any(np.diff(horizons) <= 0)
        or not np.isfinite(now_s)
        or root_xyz.shape != (3,)
        or not np.isfinite(root_xyz).all()
    ):
        raise ValueError("Invalid trajectory query time")
    requested = now_s + horizons
    mask = (requested >= time_s[0]) & (requested <= time_s[-1])
    positions = np.stack([np.interp(requested, time_s, xyz[:, j]) for j in range(3)], axis=-1)
    positions = (positions - np.asarray(root_xyz)) @ rotation_wxyz(root_wxyz)
    positions[~mask] = 0
    return {
        "label_future_pelvis_body": positions.astype(np.float32),
        "label_future_mask": mask,
        "label_horizon_s": np.asarray(horizons, np.float32),
    }


def export_tasks(dataset, output):
    """Materialize only the frozen training split's existing n3/n5 geometry proposals."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    catalog = json.loads((dataset / "catalog.json").read_text())
    train = sorted((r for r in catalog if r["split"] == "train"), key=lambda r: r["id"])
    if len(train) != 100:
        raise ValueError("Expected the frozen 100 training motions")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for motion in train:
        reference = dataset / "motions" / motion["id"] / "reference.npz"
        with np.load(reference, allow_pickle=False) as data:
            qpos = data["qpos"]
            duration = float(data["time_s"][-1])
        for count in (3, 5):
            scene_id = f"{motion['id']}-n{count}"
            source = dataset / "scenes" / f"{scene_id}.json"
            scene = json.loads(source.read_text())
            if scene["split"] != "train" or scene["motion_id"] != motion["id"]:
                raise ValueError("Scene and motion split mismatch")
            obstacles = []
            for draw in scene["sampling"]["draws"]:
                if draw["status"] != "sampled":
                    continue
                obstacle = draw["obstacle"]
                size = obstacle["dimensions_m"]
                if obstacle["shape"] == "sphere":
                    size = [2 * obstacle["radius_m"]] * 3
                obstacles.append(
                    {
                        "shape": obstacle["shape"],
                        "center_xyz": obstacle["position_m"],
                        "quaternion_wxyz": obstacle["quaternion_wxyz"],
                        "full_dimensions_xyz": size,
                    }
                )
            scene_path = output / f"{scene_id}.usda"
            write_collision_scene(scene_path, obstacles, qpos[0, :3], qpos[-1, :3])
            task = dict(
                schema=SCHEMA,
                task_id=scene_id,
                split="train",
                motion_id=motion["id"],
                source_group=motion["group"],
                category=motion["category"],
                observation_profile="known_map_5_primitives_v1",
                world_frame="scene_z_up_m_wxyz",
                anchor_body="pelvis",
                start_xyz=qpos[0, :3].tolist(),
                start_wxyz=qpos[0, 3:7].tolist(),
                goal_xyz=qpos[-1, :3].tolist(),
                goal_tolerance_m=0.25,
                terminal_speed_mps=0.1,
                hold_ticks=50,
                deadline_ticks=int(np.ceil(duration / 0.02)) + 50,
                obstacles=obstacles,
                scene_usd_path=str(scene_path),
                scene_usd_sha256=sha(scene_path),
                reference=binding(reference),
                source_scene=binding(source),
                reference_role="privileged_candidate_continuation_not_public_input",
                reward_profile="known_goal_progress_v1",
                timeouts_are_task_deadlines=True,
                qualification=None,
                state="proposed_requires_executed_scene_qualification",
                causal_pair_role="nested_scene_same_reference_control_not_route_choice_evidence",
            )
            validate_task(task)
            path = output / f"{scene_id}.json"
            write_new(path, task)
            records.append(
                dict(
                    binding(path), task_id=scene_id, motion_id=motion["id"], teacher_eligible=False
                )
            )
    write_new(
        output / "manifest.json",
        {
            "schema": "bfm_navigation_task_catalog_v1",
            "tasks": records,
            "training_motions": len(train),
            "qualified_tasks": 0,
            "route_choice_contrast_pairs": 0,
            "development_opened": False,
        },
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"tasks": len(export_tasks(args.dataset, args.output))}))
