"""Prepare source-bound SONIC motion libraries and one bounded tracking fit."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from hydra import compose, initialize_config_dir
import joblib
import numpy as np
from omegaconf import OmegaConf
import torch

from gear_sonic.dataset_generation.kimodo_motion_adapter import (
    KIMODO_G1_JOINT_NAMES,
    qpos_to_sonic_motion_entry,
    save_sonic_motion_file,
)
from gear_sonic.research.hindsight_training.runtime import load_release_checkpoint, sha, write_new
from gear_sonic.utils.config_utils import register_rl_resolvers

ROOT = Path(__file__).resolve().parents[3]


def main(dataset, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    manifest = json.loads((dataset / "manifest.json").read_text())
    for row in manifest["files"]:
        assert sha(dataset / row["path"]) == row["sha256"], row["path"]
    catalog = json.loads((dataset / "catalog.json").read_text())
    rows = []
    for split in ("train", "development"):
        dest = output / "motions" / split
        dest.mkdir(parents=True)
        metadata = {}
        for motion in (r for r in catalog if r["split"] == split):
            path = dataset / "motions" / motion["id"] / "reference.npz"
            with np.load(path, allow_pickle=False) as data:
                q = data["qpos"]
                assert tuple(data["joint_names"]) == KIMODO_G1_JOINT_NAMES
                entry = qpos_to_sonic_motion_entry(
                    q, source_fps=motion["fps"], canonicalize_horizontal_origin=False
                )
            assert np.allclose(entry["root_trans_offset"], q[:, :3], atol=1e-6)
            assert np.allclose(entry["dof"], q[:, 7:], atol=1e-6)
            key = "hindsight_" + motion["id"]
            target = dest / (key + ".pkl")
            save_sonic_motion_file(target, motion_key=key, motion_entry=entry)
            loaded = joblib.load(target)
            assert list(loaded) == [key] and loaded[key]["fps"] == motion["fps"]
            metadata[key] = {"length": len(q), "fps": motion["fps"]}
            rows.append(
                {
                    **motion,
                    "motion_key": key,
                    "source_reference_sha256": sha(path),
                    "pkl": str(target),
                    "pkl_sha256": sha(target),
                    "conversion_status": "complete",
                    "tracking_success": None,
                    "ground_support_qualified": False,
                    "joints_clamped": False,
                }
            )
        joblib.dump(metadata, dest / "metadata.pkl", compress=3)
    assert len(rows) == 120
    train = [r["id"] for r in rows if r["split"] == "train"]
    dev = [r["id"] for r in rows if r["split"] == "development"]
    assert len(train) == 100 and len(dev) == 20
    assert not (
        {r["group"] for r in rows if r["split"] == "train"}
        & {r["group"] for r in rows if r["split"] == "development"}
    )
    write_new(output / "motion-ledger.json", rows)
    (output / "inputs").mkdir()
    original = ROOT / "sonic_release/last.pt"
    checkpoint = output / "inputs/sonic_release.pt"
    shutil.copy2(original, checkpoint)
    shutil.copy2(ROOT / "sonic_release/config.yaml", output / "inputs/original-release-config.yaml")
    source = load_release_checkpoint(checkpoint)
    policy = source["policy_state_dict"]
    audit = {
        "sha256": sha(checkpoint),
        "source": str(original),
        "policy_parameters": sum(t.numel() for t in policy.values()),
        "critic_state_elements": sum(t.numel() for t in source["value_state_dict"].values()),
        "g1_encoder_input": list(policy["actor_module.encoders.g1.module.0.weight"].shape),
        "dynamic_decoder_input": list(policy["actor_module.decoders.g1_dyn.module.0.weight"].shape),
        "dynamic_decoder_output": list(
            policy["actor_module.decoders.g1_dyn.module.12.weight"].shape
        ),
        "old_trl_compatibility": "Existing OnlineTrainerState mapping used; no tensor modification",
    }
    write_new(output / "checkpoint-audit.json", audit)
    run = output / "tracking-run-1"
    plan = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "New bounded engineering training amendment, all earlier proxy outcomes accessible",
        "dataset": str(dataset),
        "dataset_manifest_sha256": sha(dataset / "manifest.json"),
        "train_ids": train,
        "development_ids": dev,
        "tracking": {
            "iterations": 200,
            "num_envs": 16,
            "seed": 91143,
            "num_steps_per_env": 24,
            "physics_dt": 0.005,
            "decimation": 4,
            "maximum_rollout_env_transitions": 200 * 16 * 24,
            "maximum_rollout_env_physics_steps": 200 * 16 * 24 * 4,
            "initialization_cost": "Measured separately when observable; ceiling is not execution cost",
            "wall_time_cap_seconds": 3600,
            "cuda_min_free_bytes": 2 * 1024**3,
            "camera_enabled": False,
            "obstacles_enabled": False,
            "initialization": "Strict released actor and critic weights; fresh optimizer",
            "motion_sampling": "all 100 resident, uniform native sampler; no source clipping",
        },
        "checkpoint_sha256": audit["sha256"],
        "run_dir": str(run),
        "next_stages": {
            "scene_distillation": "Requires qualified executed teacher labels and a separate stage lock",
            "camera": "Requires sensor definition and measured GPU capacity; not enabled automatically",
        },
        "original_pilot_modified": False,
        "reserved_layouts_opened": False,
    }
    write_new(output / "plan.json", plan)
    overrides = [
        "+exp=manager/universal_token/all_modes/sonic_release",
        f"checkpoint={checkpoint}",
        "++resume=false",
        "num_envs=16",
        "seed=91143",
        "headless=true",
        "use_wandb=false",
        f"experiment_dir={run}",
        "project_name=hindsight_tracker_v1",
        "manager_env.config.terrain_type=plane",
        "manager_env.config.env_spacing=12.0",
        "manager_env.config.render_results=false",
        "manager_env.config.render_ego=false",
        "manager_env.commands.motion.debug_vis=false",
        f"manager_env.commands.motion.motion_lib_cfg.motion_file={output / 'motions/train'}",
        "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy",
        "manager_env.commands.motion.motion_lib_cfg.multi_thread=false",
        "++manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load=100",
        "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true",
        "manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=false",
        "manager_env.commands.motion.encoder_sample_probs.g1=1.0",
        "manager_env.commands.motion.encoder_sample_probs.teleop=0.0",
        "manager_env.commands.motion.encoder_sample_probs.smpl=0.0",
        "manager_env.commands.motion.cat_upper_body_poses=false",
        "manager_env.commands.motion.freeze_frame_aug=false",
        "algo.config.num_learning_iterations=200",
        "algo.config.num_steps_per_env=24",
        "algo.config.save_interval=200",
        "algo.config.actor_learning_rate=2e-5",
        "algo.config.actor.backbone.aux_loss_coef.g1_smpl_latent=0",
        "algo.config.actor.backbone.aux_loss_coef.g1_teleop_latent=0",
        "algo.config.actor.backbone.aux_loss_coef.teleop_smpl_latent=0",
        "algo.config.actor.backbone.aux_loss_coef.reencoded_smpl_g1_latent=0",
        "callbacks.model_save.save_frequency=200",
        "callbacks.model_save.max_disk_usage=null",
        "trainer._target_=gear_sonic.research.hindsight_training.tracker.HindsightTrackerTrainer",
        f"++algo.config.hindsight_run_dir={run}",
        "++callbacks.hindsight._target_=gear_sonic.research.hindsight_training.tracker.TrackingReceiptCallback",
        f"++callbacks.hindsight.packet={output / 'plan.json'}",
        f"++callbacks.hindsight.output={run}",
    ]
    register_rl_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "gear_sonic/config"), version_base="1.1"):
        config = compose(config_name="base", overrides=overrides)
    OmegaConf.save(config, output / "composed-config.yaml")
    assert config.manager_env.commands.motion.motion_lib_cfg.motion_file == str(
        output / "motions/train"
    )
    command = [str(ROOT / ".venv_isaaclab/bin/python"), "gear_sonic/train_agent_trl.py", *overrides]
    write_new(output / "command.json", {"cwd": str(ROOT), "argv": command})
    write_new(
        output / "preparation-receipt.json",
        {
            "state": "complete",
            "motions": 120,
            "training_motions": 100,
            "development_motions": 20,
            "physics_steps": 0,
            "optimizer_updates": 0,
            "wall_seconds": time.perf_counter() - started,
        },
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.dataset.resolve(), args.output.resolve())
