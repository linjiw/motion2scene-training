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
