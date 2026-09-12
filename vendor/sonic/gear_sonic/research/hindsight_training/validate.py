"""Exercise the actual SONIC CPU motion loader without physics or learning."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from gear_sonic.research.hindsight_training.runtime import write_new
from gear_sonic.utils.config_utils import register_rl_resolvers
from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot


def main(packet):
    torch.set_num_threads(2)
    register_rl_resolvers()
    config = OmegaConf.load(packet / "composed-config.yaml")
    plan = json.loads((packet / "plan.json").read_text())
    reports = []
    for split, ids in (("train", plan["train_ids"]), ("development", plan["development_ids"])):
        cfg = OmegaConf.create(
            OmegaConf.to_container(config.manager_env.commands.motion.motion_lib_cfg, resolve=True)
        )
        cfg.motion_file = str(packet / "motions" / split)
        cfg.override_num_motions_to_load = len(ids)
        library = MotionLibRobot(cfg, num_envs=16, device="cpu")
        library.load_motions_for_training()
        assert set(library.curr_motion_keys) == {"hindsight_" + x for x in ids}
        indices = torch.arange(len(ids))
        lengths = library.get_motion_length(indices)
        checks = 0
        for fraction in (0.0, 0.5, 1.0):
            states = library.get_motion_state(indices, lengths * fraction)
            for key, value in states.items():
                if isinstance(value, torch.Tensor):
                    assert torch.isfinite(value).all(), (split, key, fraction)
                    checks += value.numel()
        reports.append(
            {
                "split": split,
                "loaded": len(ids),
                "finite_elements_checked": checks,
                "loaded_duration_seconds": float(lengths.sum()),
                "target_fps": library.target_fps,
            }
        )
    write_new(
        packet / "loader-validation.json",
        {"state": "passed", "splits": reports, "physics_steps": 0, "optimizer_updates": 0},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    main(parser.parse_args().packet.resolve())
