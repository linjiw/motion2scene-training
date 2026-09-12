"""Bind the user-requested 8,000-iteration continuation before execution."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.utils.config_utils import register_rl_resolvers


def main(parent, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((parent / "manifest.json").read_text())
    for row in manifest["files"]:
        assert sha(parent / row["path"]) == row["sha256"], row["path"]
    previous = json.loads((parent / "plan.json").read_text())
    previous_receipt = json.loads((Path(previous["run_dir"]) / "training-receipt.json").read_text())
    assert previous_receipt["state"] == "complete" and previous_receipt["iteration"] == 200
    output.mkdir(parents=True)
    shutil.copytree(parent / "motions", output / "motions")
    (output / "inputs").mkdir()
    checkpoint = output / "inputs/starting_checkpoint.pt"
    shutil.copy2(previous_receipt["checkpoint"]["path"], checkpoint)
    assert sha(checkpoint) == previous_receipt["checkpoint"]["sha256"]
    rows = json.loads((parent / "motion-ledger.json").read_text())
    for row in rows:
        row["pkl"] = str(output / Path(row["pkl"]).relative_to(parent))
        assert sha(row["pkl"]) == row["pkl_sha256"]
    write_new(output / "motion-ledger.json", rows)
    run = output / "tracking-run-1"
    tracking = {
        **previous["tracking"],
        "iterations": 8000,
        "num_envs": 512,
        "seed": 91145,
        "maximum_rollout_env_transitions": 8000 * 512 * 24,
        "maximum_rollout_env_physics_steps": 8000 * 512 * 24 * 4,
        "wall_time_cap_seconds": 48 * 3600,
        "cuda_min_free_bytes": 8 * 1024**3,
        "checkpoint_interval": 500,
        "initialization": (
            "Strict actor/critic and optimizer/scheduler history from completed "
            "200-iteration fit; fresh simulator state"
        ),
    }
    plan = {
        **previous,
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "User-authorized engineering continuation; first 200-iteration diagnostics already inspected",
        "parent": str(parent),
        "parent_manifest_sha256": sha(parent / "manifest.json"),
        "tracking": tracking,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "run_dir": str(run),
        "wandb": {
            "entity": "16726",
            "project": "hindsight-sonic-tracking",
            "id": uuid.uuid4().hex[:8],
            "group": "dataset-v1-teacher-8000",
            "name": "hindsight-v1-teacher-100motions-512envs-8000iter-s91145",
        },
        "launch_environment": {"WANDB_MODE": "online", "WANDB_DISABLE_CODE": "true"},
        "budget_interpretation": (
            "8000 additional PPO iterations, 24 rollout steps/environment/iteration; "
            "five optimization epochs per iteration"
        ),
        "quality_status": "No convergence or qualified-teacher claim before matched evaluation",
        "technical_retry_rule": (
            "No automatic restarts; preserve attempt, steps and failure receipt; diagnose before retry"
        ),
    }
    write_new(output / "plan.json", plan)
    source_command = json.loads((parent / "command.json").read_text())
    overrides = source_command["argv"][2:]
    replacements = {
        "checkpoint": str(checkpoint),
        "num_envs": "512",
        "seed": "91145",
        "use_wandb": "true",
        "experiment_dir": str(run),
        "project_name": plan["wandb"]["project"],
        "manager_env.commands.motion.motion_lib_cfg.motion_file": str(output / "motions/train"),
        "algo.config.num_learning_iterations": "8000",
        "algo.config.save_interval": "500",
        "callbacks.model_save.save_frequency": "500",
        "trainer._target_": "gear_sonic.research.hindsight_training.long_tracker.LongTrackingTrainer",
        "algo.config.hindsight_run_dir": str(run),
        "callbacks.hindsight._target_": (
            "gear_sonic.research.hindsight_training.long_tracker.OnlineTrackingReceipt"
        ),
        "callbacks.hindsight.packet": str(output / "plan.json"),
        "callbacks.hindsight.output": str(run),
    }
    overrides = [
        (
            arg.split("=")[0] + "=" + replacements[arg.split("=")[0].lstrip("+")]
            if arg.split("=")[0].lstrip("+") in replacements
            else arg
        )
        for arg in overrides
    ]
    overrides += [
        "callbacks.wandb._target_=gear_sonic.research.hindsight_training.long_tracker.TrackingWandbCallback",
        "++callbacks.model_save.save_last_frequency=100",
        "++wandb.wandb_entity='16726'",
        f"++wandb.wandb_id='{plan['wandb']['id']}'",
        f"++wandb.wandb_group={plan['wandb']['group']}",
        f"++wandb.wandb_dir={output / 'wandb'}",
    ]
    root = Path(source_command["cwd"])
    register_rl_resolvers()
    with initialize_config_dir(config_dir=str(root / "gear_sonic/config"), version_base="1.1"):
        config = compose(config_name="base", overrides=overrides)
    assert (
        config.num_envs == 512
        and config.use_wandb
        and config.algo.config.num_learning_iterations == 8000
    )
    OmegaConf.save(config, output / "composed-config.yaml")
    write_new(
        output / "command.json",
        {"cwd": str(root), "argv": [*source_command["argv"][:2], *overrides]},
    )
    write_new(
        output / "preparation-receipt.json",
        {
            "state": "complete",
            "physics_steps": 0,
            "training_motions": 100,
            "development_motions": 20,
            "source_iteration": 200,
        },
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.parent.resolve(), args.output.resolve())
