"""Phase 0.7 readout: how release SONIC executes SONIC-planner clips open-loop.

Reads PoseCaptureQualificationCallback outputs (``metrics/*.pose.npz``) from one or more
``run_eval.sh ... custom`` directories and reports, per clip and seed:

* completion (no native tracking termination);
* executed vs reference pelvis height and head-top height (minimum, and the steady-window
  median/max over t > ``--steady-after-s``, the pre-declared overhang statistic), from URDF FK of the recorded
  robot and reference joint states (visual surface for the head, collision shapes elsewhere);
* bodies whose collision surface comes within ``--floor-eps`` of the floor (floor-contact
  candidates), as a fraction of ticks;
* open-loop root drift: final XY distance between robot and reference root, and for goal
  clips the final XY distance to the planner's kinematic endpoint, plus final planar speed.

Kinematic geometry only: contact candidates are geometric proximity, not measured forces.
Usage: planner_tracking_readout.py --runs DIR [DIR ...] --manifest clips/manifest.json --out out.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.dataset_generation.kimodo_motion_adapter import KIMODO_G1_JOINT_NAMES
from gear_sonic.research.planner.g1_geometry import G1Geometry

FOOT_GROUPS = ("left_foot", "right_foot", "left_ankle_foot", "right_ankle_foot")


def to_qpos(root_pos, root_quat, joint_pos_isaac, isaac_names):
    """(T, 36) MuJoCo-order qpos from Isaac-order joints (root quat is wxyz)."""
    index = {n: i for i, n in enumerate(isaac_names)}
    joints = np.stack([joint_pos_isaac[:, index[n]] for n in KIMODO_G1_JOINT_NAMES], axis=1)
    return np.concatenate([root_pos, root_quat, joints], axis=1).astype(np.float64)


def heights(geometry, qpos):
    poses = geometry.link_poses(qpos)
    col = geometry.group_z_extents(poses, "collision")
    vis = geometry.group_z_extents(poses, "visual")
    head_top = vis["head"][1] if "head" in vis else col["head"][1]
    return poses, col, head_top


def clip_readout(path, geometry, floor_eps, manifest, steady_after_s=2.0):
    d = np.load(path, allow_pickle=True)
    names = [str(x) for x in d["joint_names_isaac"]]
    n = int(d["robot_root_pos"].shape[0])
    origin = d["env_origin"].astype(np.float64)
    robot = to_qpos(d["robot_root_pos"] - origin * [1, 1, 0], d["robot_root_quat"], d["robot_joint_pos"], names)
    ref = to_qpos(d["ref_root_pos"] - origin * [1, 1, 0], d["ref_root_quat"], d["ref_joint_pos"], names)
    _, col_r, head_r = heights(geometry, robot)
    _, _, head_ref = heights(geometry, ref)
    contact = {}
    for group, (low, _) in col_r.items():
        if group in FOOT_GROUPS:
            continue
        frac = float((low < floor_eps).mean())
        if frac > 0:
            contact[group] = round(frac, 3)
    key = str(d["motion_key"])
    info = manifest.get(key, {})
    dt = float(d["dt"])
    vel = np.linalg.norm(np.diff(robot[-6:, :2], axis=0), axis=1) / dt
    out = dict(
        clip=key,
        condition=info.get("condition") or info.get("source"),
        ticks=n,
        motion_steps=int(d["motion_num_steps"]),
        completed=bool(not d["native_terminated"] and float(d["native_progress"]) >= 1.0),
        progress=round(float(d["native_progress"]), 3),
        term_flags={k: int(d[k].any()) for k in d.files if k.startswith("term_") and d[k].any()},
        pelvis_min_z=dict(robot=round(float(robot[:, 2].min()), 3), ref=round(float(ref[:, 2].min()), 3)),
        head_top_min=dict(robot=round(float(head_r.min()), 3), ref=round(float(head_ref.min()), 3)),
        head_top_p10=dict(robot=round(float(np.quantile(head_r, 0.1)), 3), ref=round(float(np.quantile(head_ref, 0.1)), 3)),
        head_top_steady=dict(
            window_start_s=steady_after_s,
            robot_median=round(float(np.median(head_r[int(steady_after_s / float(d["dt"])):])), 3),
            robot_max=round(float(head_r[int(steady_after_s / float(d["dt"])):].max()), 3),
            ref_median=round(float(np.median(head_ref[int(steady_after_s / float(d["dt"])):])), 3),
            ref_max=round(float(head_ref[int(steady_after_s / float(d["dt"])):].max()), 3),
        ),
        floor_contact_candidates=contact,
        final_root_xy_drift_m=round(float(np.linalg.norm(robot[-1, :2] - ref[-1, :2])), 3),
        max_root_xy_drift_m=round(float(np.linalg.norm(robot[:, :2] - ref[:, :2], axis=1).max()), 3),
        final_planar_speed_mps=round(float(vel.mean()), 3),
    )
    if "goal_xy" in info:
        start = ref[0, :2]
        ref_disp = ref[-1, :2] - start
        out["goal_distance_m"] = round(float(np.linalg.norm(info["goal_xy"])), 3)
        out["final_xy_error_to_kinematic_endpoint_m"] = out["final_root_xy_drift_m"]
        out["kinematic_endpoint_vs_goal_m"] = round(float(np.linalg.norm(ref_disp - np.asarray(info["goal_xy"]))), 3)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--floor-eps", type=float, default=0.03)
    p.add_argument("--steady-after-s", type=float, default=2.0, help="start of the steady head-top window")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    manifest = {Path(c["file"]).stem: c for c in json.loads(a.manifest.read_text())["clips"]}
    geometry = G1Geometry()
    runs = {}
    for run in a.runs:
        clips = [clip_readout(f, geometry, a.floor_eps, manifest, a.steady_after_s) for f in sorted((run / "metrics").glob("*.pose.npz"))]
        runs[run.name] = clips
        done = sum(c["completed"] for c in clips)
        print(f"{run.name}: {done}/{len(clips)} completed")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(schema="planner_tracking_readout_v2", floor_eps=a.floor_eps, steady_after_s=a.steady_after_s, runs=runs), indent=1))


if __name__ == "__main__":
    main()
