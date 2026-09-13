#!/usr/bin/env python3
"""TEST ONLY: fabricate eval outputs (native npz + pose.npz + metrics_eval.json) for a few motions.

Robot motion is synthetic (delayed / perturbed reference, scripted failures). It exercises
check_run.py, render_review.py and summarize_review.py without launching Isaac Sim. Never
point it at the real review directory.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_common import ARMS, ASSETS, MJ_JOINTS, PACKET, TRACKED_BODIES, metrics_dir  # noqa: E402

M2I = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
       20, 27, 21, 28]
ISAAC_JOINTS = [MJ_JOINTS[i] for i in M2I]
TERMS = ["anchor_pos", "anchor_ori_full", "ee_body_pos", "time_out", "foot_pos_xyz"]
MOTIONS = {"development": ["hindsight_00047", "hindsight_00802"], "train": ["hindsight_00796"]}
# arm -> (fraction of motion survived or None for complete, failing term, delay frames, lateral drift m)
SCRIPT = {"release": (0.45, "anchor_ori_full", 6, 0.25), "trained500": (None, None, 2, 0.05),
          "previous8000": (0.8, "foot_pos_xyz", 3, 0.12)}


def track_50hz(key, split):
    (_, motion), = joblib.load(PACKET / "motions" / split / f"{key}.pkl").items()
    fps = float(motion["fps"])
    n_src = len(motion["dof"])
    times = np.arange(0, (n_src - 1) / fps, 0.02)
    phase = times * fps
    i0 = np.floor(phase).astype(int)
    i1 = np.minimum(i0 + 1, n_src - 1)
    w = (phase - i0)[:, None]
    pos = motion["root_trans_offset"][i0] * (1 - w) + motion["root_trans_offset"][i1] * w
    q0 = motion["root_rot"][i0][:, [3, 0, 1, 2]]
    q1 = motion["root_rot"][i1][:, [3, 0, 1, 2]]
    q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
    quat = q0 * (1 - w) + q1 * w
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    dof = motion["dof"][i0] * (1 - w) + motion["dof"][i1] * w
    return np.concatenate([pos, quat, dof], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    if "teacher-8192-500-review" in str(args.review.resolve()) and "scratchpad" not in str(args.review):
        sys.exit("refusing to write a synthetic fixture into the real review directory")
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "mjcf/g1_29dof_rev_1_0.xml"))
    data = mujoco.MjData(model)
    body_ids = [model.body(b).id for b in TRACKED_BODIES]

    def fk(qpos):
        out = np.zeros((len(qpos), len(body_ids), 3))
        for k, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            out[k] = data.xpos[body_ids]
        return out

    rng = np.random.default_rng(0)
    for split, keys in MOTIONS.items():
        for arm, (survive, term, delay, drift) in SCRIPT.items():
            folder = metrics_dir(args.review, arm, split)
            folder.mkdir(parents=True, exist_ok=True)
            rows = {"motion_keys": [], "terminated": [], "progress": [], "mpjpe_g": [], "mpjpe_l": []}
            for env_index, key in enumerate(keys):
                track = track_50hz(key, split)
                F = n = len(track)
                T = n - 1
                ref_frame = np.arange(1, T + 1)
                valid = T if survive is None else int(round(survive * n))
                if survive is not None:
                    ref_frame[valid:] = np.arange(1, T - valid + 1)
                if survive is not None and env_index == 0:
                    # failed motions can be recorded shorter than n-1 (loop ends with the longest
                    # surviving motion); emulate that
                    T = valid + 30
                    ref_frame = ref_frame[:T]
                robot = track[np.clip(ref_frame - delay, 0, F - 1)].copy()
                ramp = np.linspace(0, 1, T)[:, None]
                robot[:, 1:2] += drift * ramp
                robot[:, 7:] += rng.normal(0, 0.02, robot[:, 7:].shape)
                origin = np.array([12.0 * env_index, 0.0, 0.0])
                reference = fk(track[ref_frame]) + origin
                tracked = fk(robot) + origin
                np.savez_compressed(folder / f"{key}.npz", reference=reference.astype(np.float32),
                                    tracked=tracked.astype(np.float32),
                                    body_names=np.asarray(TRACKED_BODIES))
                flags = {name: np.zeros(T, dtype=bool) for name in TERMS}
                if survive is not None:
                    flags[term][valid] = True
                isaac = lambda q: q[:, 7:][:, M2I]  # noqa: E731
                ref_track_body = fk(track)
                np.savez_compressed(
                    folder / f"{key}.pose.npz",
                    robot_root_pos=robot[:, :3], robot_root_quat=robot[:, 3:7],
                    robot_joint_pos=isaac(robot),
                    ref_root_pos=track[ref_frame, :3], ref_root_quat=track[ref_frame, 3:7],
                    ref_joint_pos=isaac(track[ref_frame]), ref_frame=ref_frame,
                    motion_id=np.full(T, env_index),
                    **{f"term_{name}": value for name, value in flags.items()},
                    ref_track_root_pos=track[:, :3], ref_track_root_quat=track[:, 3:7],
                    ref_track_joint_pos=isaac(track), ref_track_body_pos=ref_track_body,
                    motion_num_frames=np.int64(F), env_index=np.int64(env_index),
                    env_origin=origin, motion_key=np.str_(key), motion_num_steps=np.int64(n),
                    native_progress=np.float64(1.0 if survive is None else valid / n),
                    native_terminated=np.bool_(survive is not None), dt=np.float64(0.02),
                    joint_names_isaac=np.asarray(ISAAC_JOINTS),
                    mujoco_to_isaaclab_dof=np.asarray(M2I), body_names=np.asarray(TRACKED_BODIES),
                    time_out_terms=np.asarray(["time_out"]),
                )
                err = np.linalg.norm(tracked - reference, axis=-1)
                local = np.linalg.norm((tracked - tracked[:, :1]) - (reference - reference[:, :1]), axis=-1)
                rows["motion_keys"].append(key)
                rows["terminated"].append(survive is not None)
                rows["progress"].append(1.0 if survive is None else valid / n)
                rows["mpjpe_g"].append(float(err.mean() * 1000))
                rows["mpjpe_l"].append(float(local.mean() * 1000))
            (folder / "metrics_eval.json").write_text(json.dumps({"eval/all_metrics_dict": rows}))
            (folder.parent / "receipt.json").write_text(json.dumps({"exit_code": 0, "wall_seconds": 1.0}))
    print("synthetic fixture written under", args.review / "eval", list(ARMS))


if __name__ == "__main__":
    main()
