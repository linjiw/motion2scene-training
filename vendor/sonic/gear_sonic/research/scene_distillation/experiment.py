"""Run a fixed 100-motion foundation experiment with two bounded DAgger rounds."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

import joblib
import torch

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.aggregate import aggregate
from gear_sonic.research.scene_distillation.train import load_episodes, make_decoder


def audit_labels(manifest, lock, output):
    """Check every retained same-state label before spending optimizer updates."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    episodes = load_episodes(
        manifest,
        lock["teacher_sha256"],
        set(lock["train_ids"]),
        "foundation",
        allow_exploratory_queries=True,
        allow_teacher_prefixes=True,
    )
    decoder = make_decoder(lock["teacher_checkpoint"], "cuda").eval()
    maximum = 0.0
    rows = 0
    passed = True
    with torch.no_grad():
        for episode in episodes:
            for start in range(0, len(episode["proprio"]), 128):
                p = episode["proprio"][start : start + 128].cuda()
                z = episode["teacher_tokens"][start : start + 128].cuda()
                target = episode["teacher_actions"][start : start + 128].cuda()
                actual = decoder(z, p)
                maximum = max(maximum, float((actual - target).abs().max()))
                passed = passed and torch.allclose(actual, target, atol=2e-4, rtol=2e-4)
                rows += len(p)
    write_new(
        output,
        {
            "passed": passed,
            "rows": rows,
            "episodes": len(episodes),
            "max_action_abs_error": maximum,
            "atol": 2e-4,
            "rtol": 2e-4,
        },
    )
    del decoder, episodes
    torch.cuda.empty_cache()
    if not passed:
        raise ValueError("All-label FP32 parity audit failed; no fitting launched")


def run(args):
    repo = Path.cwd().resolve()
    pilot = args.pilot.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    train = args.motion_parent.resolve() / "train"
    ids = sorted(p.stem.split("_")[-1] for p in train.glob("hindsight_*.pkl"))
    if len(ids) != 100:
        raise ValueError("Expected exactly the frozen 100 training motions")
    lock = json.loads((pilot / "collection-lock.json").read_text())
    lock.update(
        train_ids=ids,
        precision="fp32",
        minimum_prefix_rows=20,
        purpose="All 100 train motions, locally supported teacher prefixes and exploratory DAgger",
        max_student_updates=12000,
        wall_cap_seconds=7200,
        updates_per_stage=[6000, 3000, 3000],
        automatic_retries=0,
    )
    write_new(output / "experiment-lock.json", lock)
    sources = sorted(
        set(repo.glob("gear_sonic/research/scene_distillation/*.py"))
        | set(repo.glob("gear_sonic/research/hindsight_training/*.py"))
        | {
            repo / "gear_sonic/envs/wrapper/manager_env_wrapper.py",
            repo / "gear_sonic/envs/manager_env/mdp/commands.py",
            repo / "gear_sonic/trl/callbacks/im_eval_callback.py",
        }
    )
    write_new(output / "source-manifest.json", {str(p.relative_to(repo)): sha(p) for p in sources})
    with tarfile.open(output / "source-before-launch.tar.gz", "w:gz") as archive:
        for p in sources:
            archive.add(p, arcname=str(p.relative_to(repo)))
    (output / "motions").mkdir()
    for p in train.glob("*.pkl"):
        (output / "motions" / p.name).symlink_to(p.resolve())
    if len(joblib.load(output / "motions/metadata.pkl")) != 100:
        raise ValueError("Training metadata must contain exactly 100 motions")
    base = json.loads((pilot / "collection-command.json").read_text())
    base = [x for x in base if not x.startswith("++callbacks.im_eval.collection_lock=")]
    base = [
        x.replace("++num_envs=2", "++num_envs=100")
        .replace("override_num_motions_to_load=2", "override_num_motions_to_load=100")
        .replace(str(pilot / "motions"), str(output / "motions"))
        for x in base
    ]
    started = time.monotonic()
    ledger = []

    def launch(name, command, timeout=1200):
        for p in sources:
            if p.name in {
                "online.py",
                "scene_teacher.py",
                "observations.py",
                "tasks.py",
                "scene_env.py",
                "scene_qualification.py",
                "prepare_scene_probe.py",
            }:
                continue  # These scene collection modules are outside the foundation execution path.
            if (
                sha(p)
                != json.loads((output / "source-manifest.json").read_text())[
                    str(p.relative_to(repo))
                ]
            ):
                raise ValueError("Training source changed after experiment lock")
        remaining = lock["wall_cap_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Experiment total wall cap reached")
        write_new(output / f"{name}-command.json", command)
        before = time.monotonic()
        entry = {"stage": name, "state": "running"}
        ledger.append(entry)
        (output / "progress.json").write_text(json.dumps(ledger, indent=2))
        with (output / f"{name}.log").open("w") as log:
            try:
                result = subprocess.run(
                    command,
                    cwd=repo,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=min(timeout, remaining),
                    env=dict(os.environ, PYTHONPATH=str(repo)),
                )
                entry.update(
                    exit_code=result.returncode,
                    state="complete" if result.returncode == 0 else "failed",
                )
                if result.returncode:
                    raise RuntimeError(f"{name} failed; inspect its retained log")
            except Exception:
                entry["state"] = "failed"
                raise
            finally:
                entry["wall_seconds"] = time.monotonic() - before
                (output / "progress.json").write_text(json.dumps(ledger, indent=2))
        print(json.dumps(entry), flush=True)

    def native(name, callback, extras):
        command = []
        for x in base:
            if x.startswith("++eval_output_dir="):
                x = f"++eval_output_dir={output / name}"
            elif x.startswith("++eval_base_dir="):
                x = f"++eval_base_dir={output / (name + '-hydra')}"
            elif x.startswith("++callbacks.im_eval._target_="):
                x = (
                    "++callbacks.im_eval._target_=gear_sonic.research.scene_distillation."
                    + callback
                )
            command.append(x)
        launch(name, command + extras)

    initial = pilot / "foundation-round-1-v2/step-000200.pt"
    checkpoint = initial
    collections = []
    for round_id, updates in enumerate(lock["updates_per_stage"]):
        name = f"collection-{round_id}"
        collection_lock = dict(lock)
        if round_id:
            collection_lock.update(
                student_checkpoint=str(checkpoint),
                student_sha256=sha(checkpoint),
                command_profile="navigation" if round_id == 1 else "full",
            )
        write_new(output / f"{name}-lock.json", collection_lock)
        callback = (
            "collect.PrefixFoundationCollectionCallback"
            if round_id == 0
            else "collect.DaggerFoundationCollectionCallback"
        )
        native(
            name,
            callback,
            [f"++callbacks.im_eval.collection_lock={output / (name + '-lock.json')}"],
        )
        collections.append(output / name / "collection.json")
        manifest = output / f"aggregate-{round_id}.json"
        aggregate(collections, manifest)
        audit_labels(manifest, lock, output / f"parity-{round_id}.json")
        config = json.loads((pilot / "foundation-round-1-config.json").read_text())
        config.update(
            dataset_manifest=str(manifest),
            dataset_manifest_sha256=sha(manifest),
            train_ids=ids,
            updates=updates,
            batch_size=128,
            wall_cap_seconds=1800,
            save_interval=1000,
            seed=91210 + round_id,
            precision="fp32",
            decoder_atol=2e-4,
            initialize_checkpoint=str(checkpoint),
            initialize_sha256=sha(checkpoint),
            allow_teacher_prefixes=True,
            allow_exploratory_queries=True,
            purpose=lock["purpose"],
            claim="Training-set bounded scaling experiment; physical outcome measured separately",
        )
        config.pop("numerical_amendment", None)
        config_path = output / f"fit-{round_id}-config.json"
        write_new(config_path, config)
        launch(
            f"fit-{round_id}",
            [
                base[0],
                "-m",
                "gear_sonic.research.scene_distillation.train",
                "--config",
                str(config_path),
                "--output",
                str(output / f"fit-{round_id}"),
            ],
            1900,
        )
        checkpoint = output / f"fit-{round_id}" / f"step-{updates:06d}.pt"
        if not checkpoint.exists():
            raise ValueError("Bounded fit stopped before final checkpoint")
    for profile in ("full", "navigation"):
        native(
            f"final-eval-{profile}",
            "evaluate.FoundationEvaluationCallback",
            [
                f"++callbacks.im_eval.student_checkpoint={checkpoint}",
                f"++callbacks.im_eval.teacher_sha256={lock['teacher_sha256']}",
                f"++callbacks.im_eval.command_profile={profile}",
            ],
        )
    write_new(
        output / "complete.json",
        {
            "checkpoint": str(checkpoint),
            "sha256": sha(checkpoint),
            "new_optimizer_updates": 12000,
            "training_motion_denominator": 100,
            "wall_seconds": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--motion-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
