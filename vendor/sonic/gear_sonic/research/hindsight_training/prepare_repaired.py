"""Bind a memory-conscious new teacher fit on the screened repaired dataset."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from hydra import compose, initialize_config_dir
import joblib
from omegaconf import OmegaConf

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.utils.config_utils import register_rl_resolvers

ENV_COUNTS = (128, 256, 512, 1024, 2048, 4096, 8192)


def main(dataset, parent, output, num_envs=128, iterations=32000):
    if num_envs not in ENV_COUNTS or iterations <= 0:
        raise ValueError(f"Use a bounded teacher experiment with num_envs in {ENV_COUNTS}")
    manifest = json.loads((dataset / "manifest.json").read_text())
    for row in manifest["files"]:
        if sha(dataset / row["path"]) != row["sha256"]:
            raise ValueError(f"Repaired input changed: {row['path']}")
    ledger = json.loads((dataset / "motion-ledger.json").read_text())
    train = [r for r in ledger if r["split"] == "train" and r["route"] == "offline_qualified"]
    dev = [r for r in ledger if r["split"] == "development"]
    if (
        len(train) != 89
        or len(dev) != 20
        or {r["group"] for r in train} & {r["group"] for r in dev}
    ):
        raise ValueError("Unexpected repaired split or overlapping source groups")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for split, selected, source in [
        ("train", train, dataset / "screened/train"),
        ("development", dev, dataset / "matched/repaired/development"),
    ]:
        dest = output / "motions" / split
        dest.mkdir(parents=True)
        metadata = joblib.load(source / "metadata.pkl")
        if set(metadata) != {r["motion_key"] for r in selected}:
            raise ValueError("Metadata split mismatch")
        shutil.copy2(source / "metadata.pkl", dest / "metadata.pkl")
        for row in selected:
            src = source / (row["motion_key"] + ".pkl")
            if sha(src) != row["exports"]["repaired"]["sha256"]:
                raise ValueError("Screened clip differs from repaired export")
            target = dest / src.name
            shutil.copy2(src, target)
            rows.append(dict(row, pkl=str(target), pkl_sha256=sha(target)))
    write_new(output / "motion-ledger.json", rows)
    (output / "inputs").mkdir()
    checkpoint = output / "inputs/sonic_release.pt"
    shutil.copy2(parent / "inputs/sonic_release.pt", checkpoint)
    run = output / "tracking-run-1"
    previous = json.loads((parent / "plan.json").read_text())
    plan = dict(
        previous,
        utc=datetime.now(timezone.utc).isoformat(),
        dataset=str(dataset),
        dataset_manifest_sha256=sha(dataset / "manifest.json"),
        train_ids=[r["id"] for r in train],
        development_ids=[r["id"] for r in dev],
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),
        run_dir=str(run),
        status="New repaired-data teacher, released actor/critic initialization and fresh optimizer",
        excluded_train_ids=[
            r["id"] for r in ledger if r["split"] == "train" and r["route"] != "offline_qualified"
        ],
        evaluation_protocol=(
            "All 20 repaired development clips, including two unresolved offline screens; "
            "no screening of evaluation failures"
        ),
        quality_status="Offline reference repair only; policy tracking qualification pending",
        technical_retry_rule=(
            "No automatic restart; preserve all work and reconcile remaining budget "
            "before changing env count"
        ),
        launch_environment={"WANDB_MODE": "disabled"},
    )
    plan["tracking"].update(
        iterations=iterations,
        num_envs=num_envs,
        seed=91230,
        num_steps_per_env=24,
        maximum_rollout_env_transitions=iterations * num_envs * 24,
        maximum_rollout_env_physics_steps=iterations * num_envs * 24 * 4,
        wall_time_cap_seconds=48 * 3600,
        cuda_min_free_bytes=2 * 1024**3,
        initialization=(
            "Strict SONIC release actor/critic; fresh optimizer and simulator, "
            "no old repaired-data fit"
        ),
        motion_sampling="All 89 screened repaired clips resident, uniform native sampling",
        checkpoint_interval=500,
        num_ppo_epochs=5,
        num_mini_batches=8,
        budget_interpretation=(
            "More PPO iterations with smaller rollouts; "
            "five optimization epochs per rollout retained"
            if num_envs <= 256
            else "Larger rollouts per PPO iteration; five optimization epochs per rollout retained"
        ),
    )
    write_new(output / "plan.json", plan)
    command = json.loads((parent / "command.json").read_text())
    replacements = {
        "checkpoint": str(checkpoint),
        "num_envs": str(num_envs),
        "seed": "91230",
        "experiment_dir": str(run),
        "project_name": "sonic_repaired_v1_1_teacher",
        "manager_env.commands.motion.motion_lib_cfg.motion_file": str(output / "motions/train"),
        "manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load": "89",
        "algo.config.num_learning_iterations": str(iterations),
        "algo.config.save_interval": "500",
        "callbacks.model_save.save_frequency": "500",
        "trainer._target_": "gear_sonic.research.hindsight_training.repaired_tracker.RepairedTeacherTrainer",
        "algo.config.hindsight_run_dir": str(run),
        "callbacks.hindsight._target_": (
            "gear_sonic.research.hindsight_training.repaired_tracker." "RepairedTrackingReceipt"
        ),
        "callbacks.hindsight.packet": str(output / "plan.json"),
        "callbacks.hindsight.output": str(run),
    }
    overrides = [
        (
            x.split("=", 1)[0] + "=" + replacements[x.split("=", 1)[0].lstrip("+")]
            if x.split("=", 1)[0].lstrip("+") in replacements
            else x
        )
        for x in command["argv"][2:]
    ]
    overrides += [
        "algo.config.num_learning_epochs=5",
        "algo.config.num_mini_batches=8",
        "++callbacks.model_save.save_last_frequency=100",
    ]
    register_rl_resolvers()
    with initialize_config_dir(
        config_dir=str(Path(command["cwd"]) / "gear_sonic/config"), version_base="1.1"
    ):
        config = compose(config_name="base", overrides=overrides)
    assert config.num_envs == num_envs and config.algo.config.num_learning_iterations == iterations
    assert config.algo.config.num_mini_batches == 8 and not config.use_wandb
    OmegaConf.save(config, output / "composed-config.yaml")
    write_new(
        output / "command.json", dict(cwd=command["cwd"], argv=command["argv"][:2] + overrides)
    )
    write_new(
        output / "preparation-receipt.json",
        {
            "state": "complete",
            "training_motions": 89,
            "development_motions": 20,
            "source_files_verified": len(manifest["files"]),
            "physics_steps": 0,
            "optimizer_updates": 0,
        },
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=32000)
    args = parser.parse_args()
    main(
        args.dataset.resolve(),
        args.parent.resolve(),
        args.output.resolve(),
        args.num_envs,
        args.iterations,
    )
