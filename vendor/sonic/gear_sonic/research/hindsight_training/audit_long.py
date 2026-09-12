"""Finish the long-run receipt after its detached monitor exits; never launch training."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from gear_sonic.research.hindsight_training.runtime import sha, write_new


def finish(packet):
    plan = json.loads((packet / "plan.json").read_text())
    run = Path(plan["run_dir"])
    exit_path = packet / "attempt-1/exit.json"
    deadline = time.monotonic() + plan["tracking"]["wall_time_cap_seconds"] + 120
    while not exit_path.exists() and time.monotonic() < deadline:
        time.sleep(30)
    if not exit_path.exists():
        write_new(
            packet / "audit-status.json",
            {"state": "BLOCKED", "reason": "Monitor exit receipt missing"},
        )
        return
    exit_receipt = json.loads(exit_path.read_text())
    final_path = run / "training-receipt.json"
    progress_path = run / "progress.jsonl"
    progress = []
    if progress_path.exists():
        progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
    receipt = (
        json.loads(final_path.read_text())
        if final_path.exists()
        else (progress[-1] if progress else {})
    )
    completed = (
        exit_receipt["exit_code"] == 0
        and receipt.get("state") == "complete"
        and receipt.get("iteration") == plan["tracking"]["iterations"]
    )
    if completed:
        assert [row["iteration"] for row in progress] == list(
            range(1, plan["tracking"]["iterations"] + 1)
        )
        assert sha(receipt["checkpoint"]["path"]) == receipt["checkpoint"]["sha256"]
        assert receipt["observed_env_transitions"] == sum(
            receipt["motion_exposure_transitions"].values()
        )
        assert (
            receipt["observed_env_transitions"]
            <= plan["tracking"]["maximum_rollout_env_transitions"]
        )
    exposures = receipt.get("motion_exposure_transitions", {})
    with (packet / "motion-training-outcomes.csv").open("x") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "motion_id",
                "split",
                "observed_training_transitions",
                "tracking_success",
                "navigation_passage",
            ]
        )
        for split, ids in (("train", plan["train_ids"]), ("development", plan["development_ids"])):
            for motion in ids:
                writer.writerow(
                    [motion, split, exposures.get("hindsight_" + motion, 0), "NA", "NA"]
                )
    write_new(
        packet / "FINAL-RECEIPT.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "state": "complete" if completed else "incomplete",
            "exit": exit_receipt,
            "training": receipt,
            "development_physics_evaluations": 0,
            "student_training_updates": 0,
            "teacher_quality": "UNTESTED by matched whole-motion evaluation",
            "next_action": "Freeze this checkpoint and run a separately locked matched tracker evaluation",
        },
    )
    files = sorted(
        path
        for path in packet.rglob("*")
        if path.is_file() and path.name not in ("FINAL-MANIFEST.json", "audit.log")
    )
    write_new(
        packet / "FINAL-MANIFEST.json",
        {"files": [{"path": str(path.relative_to(packet)), "sha256": sha(path)} for path in files]},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    finish(parser.parse_args().packet.resolve())
