"""Reconcile the bounded tracking attempt and bind its actual outputs."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile

from gear_sonic.research.hindsight_training.runtime import sha, write_new


def main(packet):
    plan = json.loads((packet / "plan.json").read_text())
    run = Path(plan["run_dir"])
    receipt = json.loads((run / "training-receipt.json").read_text())
    exit_receipt = json.loads((packet / "attempt-1/exit.json").read_text())
    assert receipt["state"] == "complete" and exit_receipt["exit_code"] == 0
    assert receipt["iteration"] == plan["tracking"]["iterations"]
    progress = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in progress] == list(range(1, 201))
    assert receipt["observed_env_transitions"] == sum(
        receipt["motion_exposure_transitions"].values()
    )
    assert sha(receipt["checkpoint"]["path"]) == receipt["checkpoint"]["sha256"]
    assert (
        receipt["observed_env_transitions"] <= plan["tracking"]["maximum_rollout_env_transitions"]
    )
    assert (
        receipt["observed_env_physics_steps"]
        <= plan["tracking"]["maximum_rollout_env_physics_steps"]
    )
    with (packet / "motion-training-outcomes.csv").open("x") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "motion_id",
                "split",
                "loaded",
                "training_transitions",
                "tracking_success",
                "navigation_passage",
            ],
        )
        writer.writeheader()
        for split, ids in (("train", plan["train_ids"]), ("development", plan["development_ids"])):
            for motion in ids:
                writer.writerow(
                    {
                        "motion_id": motion,
                        "split": split,
                        "loaded": True,
                        "training_transitions": receipt["motion_exposure_transitions"].get(
                            "hindsight_" + motion, 0
                        ),
                        "tracking_success": "NA",
                        "navigation_passage": "NA",
                    }
                )
    # Preserve all printed per-update metrics; these are training diagnostics.
    log = (packet / "attempt-1/training.log").read_text()
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    parts = re.split(r"Learning iteration\s+(\d+)", log)
    metrics = []
    for index in range(1, len(parts), 2):
        values = {
            key.strip(): float(value)
            for key, value in re.findall(
                r"│\s*([^│\n:]+):\s*(-?\d+(?:\.\d+)?)\s*│", parts[index + 1]
            )
        }
        metrics.append({"iteration": int(parts[index]), **values})
    write_new(packet / "training-metrics.json", metrics)
    repo = Path(json.loads((packet / "command.json").read_text())["cwd"])
    original_sources = json.loads((packet / "attempt-1/code-input-manifest.json").read_text())[
        "files"
    ]
    changed = [row["path"] for row in original_sources if sha(row["path"]) != row["sha256"]]
    write_new(packet / "code-after-check.json", {"changed_since_launch": changed})
    with tarfile.open(packet / "pipeline-source.tar.gz", "w:gz") as archive:
        for path in sorted((repo / "gear_sonic/research/hindsight_training").glob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(repo)))
        archive.add(
            repo / "decoupled_wbc/tests/test_hindsight_training.py",
            arcname="test_hindsight_training.py",
        )
        archive.add(
            repo / "scripts/research/lflh_next/dataset/render_isaaclab.py",
            arcname="render_isaaclab.py",
        )
    versions = {}
    for package in ("torch", "isaacsim", "isaaclab", "numpy", "trl", "transformers", "hydra-core"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    versions.update(
        python=platform.python_version(),
        kernel=platform.release(),
        hostname=platform.node(),
        nvidia_kernel=Path("/proc/driver/nvidia/version").read_text(),
        isaaclab_commit=subprocess.check_output(
            ["git", "-C", "/home/linjiw/IsaacLab", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    write_new(packet / "runtime-identity.json", versions)
    dataset = Path(plan["dataset"])
    dataset_manifest = json.loads((dataset / "manifest.json").read_text())
    for row in dataset_manifest["files"]:
        assert sha(dataset / row["path"]) == row["sha256"]
    summary = {
        "state": "complete",
        "utc": datetime.now(timezone.utc).isoformat(),
        "training_receipt": receipt,
        "process_receipt": exit_receipt,
        "train_motions_seen": len(receipt["motion_exposure_transitions"]),
        "development_physics_evaluations": 0,
        "scene_teacher_label_shards": 0,
        "student_training_updates": 0,
        "frozen_dataset_hashes_verified": len(dataset_manifest["files"]),
        "tracking_quality": "UNTESTED on matched development evaluation",
        "navigation_utility": "BLOCKED/UNTESTED: scene teacher and physical evaluation absent",
        "gpu_launch": "SUPPORTED: actual CUDA/Isaac execution despite NVML telemetry failure",
        "next_action": "Lock matched release-versus-trained tracker evaluation before scene teaching",
    }
    write_new(packet / "SUMMARY.json", summary)
    shutil.copy2(repo / "gear_sonic/research/hindsight_training/README.md", packet / "PIPELINE.md")
    body = (
        f"# SONIC tracking receipt\n\nCompleted {receipt['iteration']} PPO updates "
        "with 16 Isaac Lab environments. "
        f"All {len(receipt['motion_exposure_transitions'])} training motions were sampled; 20 development motions "
        "remained outside training.\n\n"
        f"Observed {receipt['observed_env_transitions']:,} environment transitions and "
        f"{receipt['observed_env_physics_steps']:,} physics steps across environments "
        f"({receipt['observed_physics_step_events']:,} world-step events). "
        f"Training callback wall time: {receipt['wall_seconds']:.1f} seconds. "
        "Initialization physics was not observed.\n\n"
        f"Final checkpoint: `{receipt['checkpoint']['path']}`\n\n"
        f"SHA-256: `{receipt['checkpoint']['sha256']}`\n\n"
        "CUDA and Isaac Lab execution succeeded despite NVML's driver/library warning. "
        "No driver changes were needed. "
        "The frozen dataset and released checkpoint were preserved.\n\n"
        "The native trainer skipped its end callback at the normal stopping boundary. "
        "Completion was reconciled after exit from all 200 progress records and checkpoint step 200. "
        "The monitor's original incomplete receipt is preserved; no training was repeated. "
        "The callback fix for future runs and the original launched source are both archived.\n\n"
        "This completed a bounded tracking fit. It did not establish complete-motion tracking improvements, "
        "collision-free navigation, scene-aware teaching, or student performance. Training losses and termination "
        "diagnostics are retained in full. Per-motion passage and tracking-success labels remain NA.\n\n"
        "The student pipeline and interface tests are prepared. It requires new executed scene-teacher labels; "
        "none were fabricated from reference trajectories. Cameras were disabled. The next stage is a locked, "
        "matched development evaluation of the original release and this checkpoint.\n"
    )
    with (packet / "RESULTS.md").open("x") as handle:
        handle.write(body)
    files = sorted(
        path for path in packet.rglob("*") if path.is_file() and path.name != "manifest.json"
    )
    write_new(
        packet / "manifest.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "files": [
                {"path": str(path.relative_to(packet)), "sha256": sha(path)} for path in files
            ],
            "parent_dataset_manifest_sha256": sha(dataset / "manifest.json"),
            "hash_date_does_not_prove_prior_registration": True,
        },
    )
    print(
        json.dumps(
            {"state": "complete", "manifest_files": len(files), "checkpoint": receipt["checkpoint"]}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    main(parser.parse_args().packet.resolve())
