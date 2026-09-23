"""Build clear/corridor stopping tasks for chosen repaired training motions.

Mirrors prepare_repaired_tasks.py (stopping-tasks-v2 style): each motion gets a stationary
tail of --tail-seconds, a native motion file, a reference-with-hold.npz whose body_names
drive the scene contact sensors, and two known-map scenes. Tasks are proposals; only an
executed receipt qualifies them.

Every obstacle footprint must keep --min-clearance-m of horizontal clearance from the start
pelvis XY and from the goal XY, otherwise the build fails. --clearance-fix shift-out instead
moves each offending corridor wall outward (away from the path midpoint, along its own
lateral axis) by the smallest whole centimetre that restores the margin, and records the
shift in the proposal and the task (legacy-v1.1 rebuild of 00399-corridor).
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from gear_sonic.dataset_generation.kimodo_motion_adapter import (
    KIMODO_G1_JOINT_NAMES,
    qpos_to_sonic_motion_entry,
    save_sonic_motion_file,
)
from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.tasks import (
    MIN_START_GOAL_CLEARANCE_M,
    SCHEMA,
    binding,
    require_start_goal_clearance,
    shift_to_clearance,
    start_goal_clearance,
    validate_task,
    write_collision_scene,
)

CORRIDOR_OFFSET_M = 0.8
CLEARANCE_FIX_STEP_M = 0.01
CLEARANCE_FIX_MAX_M = 1.0


def corridor_walls(mid_xy, side, quat, start, goal, min_clearance_m, fix):
    """Two walls CORRIDOR_OFFSET_M either side of the path midpoint, parallel to start->goal.

    With fix == "shift-out" a wall that crowds the start or goal moves outward (sign * side)
    by the smallest CLEARANCE_FIX_STEP_M multiple that restores min_clearance_m. Returns the
    walls and one amendment record per moved wall.
    """
    walls, amendments = [], []
    for index, sign in enumerate((-1, 1)):
        wall = {
            "shape": "box",
            "center_xyz": [*(mid_xy + sign * CORRIDOR_OFFSET_M * side), 0.75],
            "quaternion_wxyz": quat,
            "full_dimensions_xyz": [1.0, 0.2, 1.5],
        }
        if fix == "shift-out":
            before = start_goal_clearance([wall], start, goal)[0]
            if min(before["start_m"], before["goal_m"]) < min_clearance_m:
                moved, shift = shift_to_clearance(
                    wall, sign * side, start, goal, min_clearance_m,
                    CLEARANCE_FIX_STEP_M, CLEARANCE_FIX_MAX_M,
                )
                after = start_goal_clearance([moved], start, goal)[0]
                amendments.append(
                    dict(
                        obstacle_index=index,
                        rule="shift_wall_outward_min_whole_cm_to_start_goal_clearance",
                        shift_m=shift,
                        direction_xy=[float(v) for v in sign * side],
                        original_center_xyz=[float(v) for v in wall["center_xyz"]],
                        center_xyz=moved["center_xyz"],
                        clearance_before_m=dict(start=before["start_m"], goal=before["goal_m"]),
                        clearance_after_m=dict(start=after["start_m"], goal=after["goal_m"]),
                    )
                )
                wall = moved
        walls.append(wall)
    return walls, amendments


def sonic_motion_entry_to_qpos(entry):
    """Inverse of qpos_to_sonic_motion_entry: [root xyz, root wxyz, 29 dof]; stored root_rot is xyzw.

    The caller round-trips the result through qpos_to_sonic_motion_entry and asserts equality
    with the stored entry, so a convention error cannot pass silently.
    """
    rot = np.asarray(entry["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]]
    return np.concatenate(
        [np.asarray(entry["root_trans_offset"], dtype=np.float64), rot,
         np.asarray(entry["dof"], dtype=np.float64)], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", type=Path, required=True, help="repaired train motion folder")
    parser.add_argument("--ledger", type=Path, required=True, help="teacher packet motion-ledger.json")
    parser.add_argument("--body-reference", type=Path, required=True, help="npz with robot body_names")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    parser.add_argument("--variants", nargs="+", choices=("clear", "corridor"),
                        default=["clear", "corridor"])
    parser.add_argument("--min-clearance-m", type=float, default=MIN_START_GOAL_CLEARANCE_M,
                        help="obstacle footprint to start pelvis / goal XY clearance")
    parser.add_argument("--clearance-fix", choices=("none", "shift-out"), default="none",
                        help="none: fail on a violation; shift-out: move offending walls outward")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = {r["id"]: r for r in json.loads(args.ledger.read_text())}
    body_names = np.load(args.body_reference, allow_pickle=False)["body_names"]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for id in args.ids:
        row = ledger[id]
        if row["split"] != "train":
            raise ValueError(f"{id} is not a training motion")
        k = "hindsight_" + id
        motion = args.motions / (k + ".pkl")
        if sha(motion) != row["pkl_sha256"]:
            raise ValueError(f"{id}: repaired motion differs from its ledger")
        d = joblib.load(motion)[k]
        folder = root / id
        folder.mkdir()
        fps = float(d["fps"])
        q = sonic_motion_entry_to_qpos(d)
        source_frames = len(q)
        q = np.concatenate([q, np.repeat(q[-1:], int(round(args.tail_seconds * fps)), axis=0)])
        ref = folder / "reference-with-hold.npz"
        np.savez_compressed(
            ref,
            qpos=q,
            time_s=np.arange(len(q)) / fps,
            fps=np.asarray(fps),
            body_names=body_names,
            joint_names=np.array(KIMODO_G1_JOINT_NAMES),
        )
        entry = qpos_to_sonic_motion_entry(q, source_fps=fps, canonicalize_horizontal_origin=False)
        for key in ("dof", "root_trans_offset", "root_rot", "pose_aa"):
            np.testing.assert_allclose(entry[key][: len(d[key])], d[key], atol=2e-7)
        motions = folder / "motions"
        motions.mkdir()
        native = motions / (k + ".pkl")
        save_sonic_motion_file(native, motion_key=k, motion_entry=entry)
        joblib.dump({k: {"length": len(q), "fps": fps}}, motions / "metadata.pkl", compress=3)
        start, goal = q[0, :3], q[-1, :3]
        direction = goal[:2] - start[:2]
        direction = direction / max(np.linalg.norm(direction), 1e-6)
        side = np.array([-direction[1], direction[0]])
        mid = q[(source_frames - 1) // 2, :3]
        yaw = float(np.arctan2(direction[1], direction[0]))
        quat = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
        for variant in args.variants:
            obstacles, amendments = [], []
            if variant == "corridor":
                obstacles, amendments = corridor_walls(
                    mid[:2], side, quat, start, goal, args.min_clearance_m, args.clearance_fix
                )
            clearance = require_start_goal_clearance(
                obstacles, start, goal, args.min_clearance_m, label=f"{id}-stop-{variant}"
            )
            scene = folder / (variant + ".usda")
            write_collision_scene(scene, obstacles, start, goal)
            source = folder / (variant + "-proposal.json")
            source.write_text(
                json.dumps(
                    {
                        "motion_id": id,
                        "variant": variant,
                        "obstacles": obstacles,
                        "tail_seconds": args.tail_seconds,
                        "source_frames": source_frames,
                        "source": "synthetic research geometry; not generator success evidence",
                        "min_start_goal_clearance_m": args.min_clearance_m,
                        "start_goal_clearance_m": clearance,
                        "clearance_amendments": amendments,
                    },
                    indent=2,
                )
            )
            task = {
                "schema": SCHEMA,
                "task_id": id + "-stop-" + variant,
                "split": "train",
                "motion_id": id,
                "source_group": row["group"],
                "category": row["category"],
                "observation_profile": "known_map_5_primitives_v1",
                "world_frame": "scene_z_up_m_wxyz",
                "anchor_body": "pelvis",
                "start_xyz": start.tolist(),
                "start_wxyz": q[0, 3:7].tolist(),
                "goal_xyz": goal.tolist(),
                "goal_tolerance_m": 0.25,
                "terminal_speed_mps": 0.1,
                "hold_ticks": 50,
                "deadline_ticks": int(np.ceil((len(q) - 1) / fps / 0.02)) + 1,
                "obstacles": obstacles,
                "scene_usd_path": str(scene),
                "scene_usd_sha256": sha(scene),
                "reference": binding(ref),
                "native_motion": binding(native),
                "source_scene": binding(source),
                "reference_role": "privileged_candidate_continuation_not_public_input",
                "reward_profile": "known_goal_progress_v1",
                "timeouts_are_task_deadlines": True,
                "qualification": None,
                "state": "proposed_requires_executed_scene_qualification",
                "causal_pair_role": "changed_geometry_requires_physical_requalification",
                "terminal_reference_amendment": (
                    f"append {args.tail_seconds} seconds of terminal pose; unqualified continuation"
                ),
                "min_start_goal_clearance_m": args.min_clearance_m,
            }
            if amendments:
                task["clearance_amendments"] = amendments
            validate_task(task)
            path = folder / (variant + ".json")
            path.write_text(json.dumps(task, indent=2))
            records.append(
                dict(binding(path), task_sha256=sha(path), task_id=task["task_id"], motion_id=id, teacher_eligible=False)
            )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "bfm_navigation_task_catalog_v1",
                "tasks": records,
                "qualified_tasks": 0,
                "training_motions": len(args.ids),
                "development_opened": False,
                "tail_seconds": args.tail_seconds,
                "variants": args.variants,
                "min_start_goal_clearance_m": args.min_clearance_m,
                "clearance_fix": args.clearance_fix,
                "warning": "Synthetic stopping candidates, not successful demonstrations.",
            },
            indent=2,
        )
    )
    print("Prepared tasks", len(records))


if __name__ == "__main__":
    main()
