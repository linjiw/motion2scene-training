"""Prepare repaired training-motion scene requests without reordering stored joints."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from gear_sonic.dataset_generation.kimodo_motion_adapter import (
    KIMODO_G1_JOINT_NAMES,
    qpos_to_sonic_motion_entry,
    save_sonic_motion_file,
    sonic_motion_entry_to_qpos,
)
from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.tasks import (
    MIN_START_GOAL_CLEARANCE_M,
    SCHEMA,
    binding,
    require_start_goal_clearance,
    validate_task,
    write_collision_scene,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--body-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-clearance-m", type=float, default=MIN_START_GOAL_CLEARANCE_M)
    args = parser.parse_args()
    p = args.packet.resolve()
    m = json.loads((p / "qualified-collection.json").read_text())
    ledger = {r["id"]: r for r in json.loads(args.ledger.read_text())}
    body_names = np.load(args.body_reference, allow_pickle=False)["body_names"]
    candidates = []
    for e in m["episodes"]:
        if not e["eligible"] or e["terminated"] or ledger[e["motion_id"]]["split"] != "train":
            raise ValueError("Scene candidates require qualified training episodes")
        if sha(e["path"]) != e["sha256"]:
            raise ValueError("Qualified episode changed")
        k = "hindsight_" + e["motion_id"]
        motion = p / "motions" / (k + ".pkl")
        if sha(motion) != ledger[e["motion_id"]]["pkl_sha256"]:
            raise ValueError("Repaired motion differs from its ledger")
        d = joblib.load(motion)[k]
        distance = float(
            np.linalg.norm(d["root_trans_offset"][-1, :2] - d["root_trans_offset"][0, :2])
        )
        candidates.append((distance, e["motion_id"], d))
    selected = sorted(candidates, key=lambda x: x[0], reverse=True)[:5]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for distance, id, d in selected:
        folder = root / id
        folder.mkdir()
        fps = float(d["fps"])
        q = sonic_motion_entry_to_qpos(d)
        q = np.concatenate([q, np.repeat(q[-1:], int(round(1.2 * fps)), axis=0)])
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
        native = motions / ("hindsight_" + id + ".pkl")
        save_sonic_motion_file(native, motion_key="hindsight_" + id, motion_entry=entry)
        joblib.dump(
            {"hindsight_" + id: {"length": len(q), "fps": fps}},
            motions / "metadata.pkl",
            compress=3,
        )
        start = q[0, :3]
        goal = q[-1, :3]
        direction = goal[:2] - start[:2]
        direction = direction / max(np.linalg.norm(direction), 1e-6)
        side = np.array([-direction[1], direction[0]])
        mid = q[(len(d["dof"]) - 1) // 2, :3]
        yaw = float(np.arctan2(direction[1], direction[0]))
        quat = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
        for variant in ["clear", "corridor", "low_beam", "changed_goal"]:
            obstacles = []
            target = goal.copy()
            if variant == "corridor":
                for sign in [-1, 1]:
                    obstacles.append(
                        {
                            "shape": "box",
                            "center_xyz": [*(mid[:2] + sign * 0.8 * side), 0.75],
                            "quaternion_wxyz": quat,
                            "full_dimensions_xyz": [1.0, 0.2, 1.5],
                        }
                    )
            if variant == "low_beam":
                obstacles = [
                    {
                        "shape": "beam",
                        "center_xyz": [*mid[:2], 1.1],
                        "quaternion_wxyz": quat,
                        "full_dimensions_xyz": [0.2, 1.4, 0.15],
                    }
                ]
            if variant == "changed_goal":
                target[:2] += side * 0.75
            # Fail before authoring a scene whose footprint crowds the start or goal.
            require_start_goal_clearance(
                obstacles, start, target, args.min_clearance_m, label=id + "-" + variant
            )
            scene = folder / (variant + ".usda")
            write_collision_scene(scene, obstacles, start, target)
            source = folder / (variant + "-proposal.json")
            source.write_text(
                json.dumps(
                    {
                        "motion_id": id,
                        "variant": variant,
                        "obstacles": obstacles,
                        "source": "new synthetic research geometry; not generator success evidence",
                        "goal_changed": variant == "changed_goal",
                    },
                    indent=2,
                )
            )
            task = {
                "schema": SCHEMA,
                "task_id": id + "-" + variant,
                "split": "train",
                "motion_id": id,
                "source_group": ledger[id]["group"],
                "category": ledger[id]["category"],
                "observation_profile": "known_map_5_primitives_v1",
                "world_frame": "scene_z_up_m_wxyz",
                "anchor_body": "pelvis",
                "start_xyz": start.tolist(),
                "start_wxyz": q[0, 3:7].tolist(),
                "goal_xyz": target.tolist(),
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
                "causal_pair_role": (
                    "changed_goal_requires_new_expert_continuation"
                    if variant == "changed_goal"
                    else "changed_geometry_requires_physical_requalification"
                ),
                "terminal_reference_amendment": (
                    "append 1.2 seconds of terminal pose; newly unqualified continuation"
                ),
            }
            validate_task(task)
            path = folder / (variant + ".json")
            path.write_text(json.dumps(task, indent=2))
            records.append(dict(binding(path), task_id=task["task_id"], teacher_eligible=False))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "bfm_navigation_task_catalog_v1",
                "tasks": records,
                "qualified_tasks": 0,
                "training_motions": len(selected),
                "development_opened": False,
                "warning": (
                    "Synthetic candidates, not successful traversal demonstrations. "
                    "Low beam may require command-interface expansion; "
                    "changed goal has no matching expert continuation."
                ),
            },
            indent=2,
        )
    )
    print("Prepared tasks", len(records))


if __name__ == "__main__":
    main()
