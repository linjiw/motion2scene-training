"""Load planner clips with the SONIC CPU motion loader and cross-check them against URDF FK.

No physics and no Isaac Lab import (asserted). Run with the native interpreter:

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=vendor/sonic \
    /home/robotixx/motion2scene-training/.venv_native/bin/python \
        vendor/sonic/scripts/research/planner_validate_clips.py \
        /home/robotixx/motion2scene-training/workspace/phase0/planner/clips
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from omegaconf import OmegaConf
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.planner.g1_geometry import G1Geometry, find_robot_description  # noqa: E402
from gear_sonic.utils.config_utils import register_rl_resolvers  # noqa: E402
from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot  # noqa: E402

DEFAULT_CONFIG = "/home/robotixx/motion2scene-training/workspace/teacher-8192-500-review/eval/release/config.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", type=Path)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--robot-description", default=None)
    args = parser.parse_args()
    if any(name.startswith(("isaaclab", "isaacsim", "omni")) for name in sys.modules):
        raise RuntimeError("Isaac modules were imported; this check must stay CPU/file-only")
    torch.set_num_threads(2)
    register_rl_resolvers()
    config = OmegaConf.load(args.config)
    cfg = OmegaConf.create(
        OmegaConf.to_container(config.manager_env.commands.motion.motion_lib_cfg, resolve=True)
    )
    clips = args.clips.resolve()
    files = sorted(clips.glob("*.pkl"))
    root = find_robot_description(args.robot_description)
    cfg.motion_file = str(clips)
    cfg.override_num_motions_to_load = len(files)
    cfg.asset.assetRoot = str(root / "mjcf") + "/"
    cfg.sort_motion_keys = True
    library = MotionLibRobot(cfg, num_envs=len(files), device="cpu")
    library.load_motions_for_training()
    geometry = G1Geometry(root)
    body_names = list(library.mesh_parsers.body_names)
    if "mujoco_to_isaaclab_body" in cfg:
        body_names = [body_names[i] for i in cfg.mujoco_to_isaaclab_body]
    rows = []
    for index, key in enumerate(library.curr_motion_keys):
        entry = joblib.load(clips / f"{key}.pkl")[key]
        frames = len(entry["dof"])
        qpos = np.zeros((frames, 36))
        qpos[:, :3] = entry["root_trans_offset"]
        qpos[:, 3:7] = entry["root_rot"][:, [3, 0, 1, 2]]
        qpos[:, 7:] = entry["dof"]
        ticks = np.array([0, (frames - 1) // 2, frames - 1])
        state = library.get_motion_state(
            torch.full((len(ticks),), index), torch.tensor(ticks / 50.0, dtype=torch.float32)
        )
        poses = geometry.link_poses(qpos[ticks])
        body = state["body_pos_w"].numpy()
        error = max(
            float(np.abs(body[:, b] - poses[name][1]).max())
            for b, name in enumerate(body_names)
            if name in poses
        )
        length = float(library.get_motion_length(torch.tensor([index]))[0])
        rows.append(
            {
                "key": key,
                "frames": frames,
                "fps": int(entry["fps"]),
                "loaded_length_s": length,
                "expected_length_s": (frames - 1) / int(entry["fps"]),
                "max_body_position_error_m": error,
                "finite": bool(
                    all(
                        torch.isfinite(v).all()
                        for v in state.values()
                        if isinstance(v, torch.Tensor)
                    )
                ),
            }
        )
    result = {
        "loader": "gear_sonic.utils.motion_lib.motion_lib_robot.MotionLibRobot (CPU)",
        "config": str(args.config),
        "target_fps": library.target_fps,
        "clips": rows,
        "max_body_position_error_m": max(r["max_body_position_error_m"] for r in rows),
        "length_mismatches": sum(
            abs(r["loaded_length_s"] - r["expected_length_s"]) > 1e-6 for r in rows
        ),
        "isaac_imported": any(
            name.startswith(("isaaclab", "isaacsim", "omni")) for name in sys.modules
        ),
        "physics_steps": 0,
    }
    (clips / "loader_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "clips"}, indent=2))


if __name__ == "__main__":
    main()
