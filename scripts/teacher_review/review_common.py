"""Shared constants and loaders for the teacher-8192-500 review packet."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _find_kit():
    """Checkout root: $M2S_KIT, else the nearest parent holding pyproject.toml and vendor/sonic."""
    if os.environ.get("M2S_KIT"):
        return Path(os.environ["M2S_KIT"]).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "vendor/sonic").is_dir():
            return parent
    raise RuntimeError("cannot locate the motion2scene-training checkout; set M2S_KIT")


KIT = _find_kit()
WORKSPACE = KIT / "workspace"
PACKET = WORKSPACE / "teacher-8192-500"
DEFAULT_REVIEW = WORKSPACE / "teacher-8192-500-review"
SONIC = KIT / "vendor/sonic"
ASSETS = SONIC / "gear_sonic/data/assets/robot_description"
URDF = ASSETS / "urdf/g1/main.urdf"
MESHDIR = ASSETS / "meshes/g1"
LEDGER = PACKET / "motion-ledger.json"
SEED = 91231

# Display order: starting point -> new teacher -> strongest earlier baseline.
ARMS = {
    "release": {
        "label": "release init (step 41,550)",
        "short": "release",
        "source": PACKET / "inputs/sonic_release.pt",
        "sha256": "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909",
        "global_step": 41550,
        "color": "#d97823",
    },
    "trained500": {
        "label": "8192-env teacher (step 500)",
        "short": "8192x500",
        "source": PACKET / "tracking-run-1/model_step_000500.pt",
        "sha256": "26ca1ed70c2287c1acb6243820649db4b8513a5d9b8b16abc115f890d87b6bff",
        "global_step": 500,
        "color": "#147dcc",
    },
    "previous8000": {
        "label": "previous teacher (step 8,000)",
        "short": "prev 8000",
        "source": WORKSPACE / "models/previous_teacher.pt",
        "sha256": "48b3a1c04cdbbcd9ffe8ad10b2591aff781c253c03c61ccb8683348d862c54ed",
        "global_step": 8000,
        "color": "#7b3fb0",
    },
}
SPLITS = {"development": 20, "train": 89}
LAUNCH_ORDER = [
    ("previous8000", "development"),  # canary: must reproduce the recorded 11/20 baseline
    ("trained500", "development"),
    ("release", "development"),
    ("trained500", "train"),
    ("release", "train"),
    ("previous8000", "train"),
]
PRIOR_PREVIOUS8000_DEV = (
    WORKSPACE / "m2s-repaired-teacher-interim-comparison-20260912/previous/metrics/metrics_eval.json"
)
HISTORY_DEV = {
    "previous 8000 (512 env), recorded": PRIOR_PREVIOUS8000_DEV,
    "repaired 128-env step 3400": WORKSPACE
    / "m2s-repaired-teacher-interim-comparison-20260912/current/metrics/metrics_eval.json",
    "repaired 128-env step 4700": WORKSPACE
    / "m2s-repaired-teacher-step4700-comparison-20260912/current/metrics/metrics_eval.json",
    "repaired 128-env step 6200": WORKSPACE
    / "m2s-repaired-teacher-step6200-comparison-20260912/current/metrics/metrics_eval.json",
    "repaired 128-env step 12500": KIT / "docs/teacher-status/step12500/metrics_eval.json",
}
ORIGINAL_DATA_QUALIFICATION = (
    WORKSPACE / "m2s-sonic-qualification-role-correction-20260912/per-motion-results.csv"
)

MJ_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint",
    "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint",
    "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_name(arm):
    return f"model_step_{ARMS[arm]['global_step']:06d}.pt"


def run_dir(review, arm, split):
    return Path(review) / "eval" / arm / split


def metrics_dir(review, arm, split):
    return run_dir(review, arm, split) / "metrics"


def split_keys(split):
    folder = PACKET / "motions" / split
    return sorted(p.stem for p in folder.glob("*.pkl") if p.name != "metadata.pkl")


def ledger_by_key():
    return {row["motion_key"]: row for row in json.loads(LEDGER.read_text())}


def load_metrics(path):
    return json.loads(Path(path).read_text())["eval/all_metrics_dict"]


def quat_yaw_wxyz(quat):
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Run:
    """One (arm, split, motion) evaluation record: native trajectories + captured pose."""

    def __init__(self, review, arm, split, key):
        self.arm, self.split, self.key = arm, split, key
        folder = metrics_dir(review, arm, split)
        native = np.load(folder / f"{key}.npz")
        pose = np.load(folder / f"{key}.pose.npz")
        self.pose = {name: pose[name] for name in pose.files}
        self.reference = native["reference"].astype(np.float64)
        self.tracked = native["tracked"].astype(np.float64)
        self.body_names = [str(b) for b in native["body_names"]]
        self.T = len(self.reference)
        self.n = int(self.pose["motion_num_steps"])
        self.dt = float(self.pose["dt"])
        self.origin = self.pose["env_origin"].astype(np.float64)
        self.completed = not bool(self.pose["native_terminated"])
        self.progress = float(self.pose["native_progress"])
        self.valid = self.T if self.completed else min(int(round(self.progress * self.n)), self.T)
        time_out = {str(t) for t in self.pose["time_out_terms"]}
        self.term_names = sorted(k[5:] for k in self.pose if k.startswith("term_"))
        failing = np.zeros(self.T, dtype=bool)
        timed_out = np.zeros(self.T, dtype=bool)
        for name in self.term_names:
            flags = self.pose[f"term_{name}"][: self.T].astype(bool)
            if name in time_out:
                timed_out |= flags
            else:
                failing |= flags
        failing &= ~timed_out
        hits = np.flatnonzero(failing)
        self.term_failure_frame = int(hits[0]) if len(hits) else None
        self.failure_terms = (
            [n for n in self.term_names if n not in time_out and self.pose[f"term_{n}"][hits[0]]]
            if len(hits) and not self.completed
            else []
        )
        diff = np.linalg.norm(self.tracked - self.reference, axis=-1)
        self.err_global_mm = diff.mean(axis=1) * 1000.0
        local = (self.tracked - self.tracked[:, :1]) - (self.reference - self.reference[:, :1])
        self.err_local_mm = np.linalg.norm(local, axis=-1).mean(axis=1) * 1000.0
        self.root_xy_m = np.linalg.norm(self.tracked[:, 0, :2] - self.reference[:, 0, :2], axis=-1)
        names = [str(j) for j in self.pose["joint_names_isaac"]]
        self.robot_perm = [names.index(j) for j in MJ_JOINTS]
        self.ref_perm = [int(i) for i in np.argsort(self.pose["mujoco_to_isaaclab_dof"])]
        self.joint_order_consistent = self.robot_perm == self.ref_perm

        def qpos(pos, quat, joints, perm):
            quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
            return np.concatenate([pos, quat, joints[:, perm]], axis=-1).astype(np.float64)

        p = self.pose
        self.robot_q = qpos(p["robot_root_pos"], p["robot_root_quat"], p["robot_joint_pos"],
                            self.robot_perm)
        self.track_q = qpos(p["ref_track_root_pos"], p["ref_track_root_quat"],
                            p["ref_track_joint_pos"], self.ref_perm)
        self.F = len(self.track_q)
        # Expected 1 (command time_steps is advanced inside env.step before capture);
        # taken from the capture itself so a different offset cannot misalign the ghost.
        self.ref_offset = int(self.pose["ref_frame"][0]) if self.T else 1

    def ghost_index(self, k):
        return int(min(max(k + self.ref_offset, 0), self.F - 1))

    def summary(self):
        v = self.valid
        return {
            "completed": self.completed,
            "native_progress": self.progress,
            "valid_frames": v,
            "recorded_frames": self.T,
            "motion_num_steps": self.n,
            "survival_s": v * self.dt,
            "failure_time_s": None if self.completed else v * self.dt,
            "failure_terms": "+".join(self.failure_terms),
            "prefix_mpjpe_g_mm": float(self.err_global_mm[:v].mean()) if v else None,
            "prefix_mpjpe_l_mm": float(self.err_local_mm[:v].mean()) if v else None,
            "root_xy_end_m": float(self.root_xy_m[v - 1]) if v else None,
            "root_xy_max_m": float(self.root_xy_m[:v].max()) if v else None,
        }
