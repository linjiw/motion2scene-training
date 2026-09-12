"""Monitor the repaired teacher fit, then run the predeclared matched development evaluation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from gear_sonic.research.hindsight_training.launch import main as launch
from gear_sonic.research.hindsight_training.runtime import sha, write_new


def main(packet):
    if launch(packet):
        return 1
    protocol = json.loads((packet / "evaluation-protocol.json").read_text())
    plan = json.loads((packet / "plan.json").read_text())
    run = Path(plan["run_dir"])
    final = json.loads((run / "training-receipt.json").read_text())["checkpoint"]
    if sha(final["path"]) != final["sha256"]:
        raise ValueError("Final teacher changed before evaluation")
    results = {}
    for arm in ("release", "repaired_teacher"):
        output = packet / "evaluation" / arm
        output.mkdir(parents=True, exist_ok=False)
        source = Path(final["path"]) if arm == "repaired_teacher" else Path(plan["checkpoint_path"])
        shutil.copy2(source, output / "checkpoint.pt")
        shutil.copy2(run / "config.yaml", output / "config.yaml")
        command = protocol["commands"][arm]
        with (output / "evaluation.log").open("x") as log:
            result = subprocess.run(
                command,
                cwd=protocol["cwd"],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
                env=dict(os.environ, PYTHONPATH=protocol["cwd"]),
            )
        write_new(
            output / "exit.json",
            {"exit_code": result.returncode, "checkpoint_sha256": sha(output / "checkpoint.pt")},
        )
        if result.returncode:
            return 1
        metrics = json.loads((output / "metrics/metrics_eval.json").read_text())[
            "eval/all_metrics_dict"
        ]
        results[arm] = {
            "motions": len(metrics["terminated"]),
            "completed": sum(not t for t in metrics["terminated"]),
        }
    write_new(packet / "evaluation-summary.json", results)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    sys.exit(main(parser.parse_args().packet.resolve()))
