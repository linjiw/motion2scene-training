"""Launch one immutable bounded attempt; CUDA execution gates launch, NVML does not."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gear_sonic.research.hindsight_training.runtime import sha, write_new


def main(packet):
    lock = (packet / "launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = packet / "attempt-1"
    attempt.mkdir(exist_ok=False)
    plan = json.loads((packet / "plan.json").read_text())
    command = json.loads((packet / "command.json").read_text())
    assert json.loads((packet / "loader-validation.json").read_text())["state"] == "passed"
    checkpoint = Path(plan.get("checkpoint_path", packet / "inputs/sonic_release.pt"))
    assert sha(checkpoint) == plan["checkpoint_sha256"]
    for row in json.loads((packet / "motion-ledger.json").read_text()):
        assert sha(row["pkl"]) == row["pkl_sha256"]
    assert not Path(plan["run_dir"]).exists(), "Run directory already exists; preserve attempts"
    repo = Path(command["cwd"])
    sources = list((repo / "gear_sonic").rglob("*.py")) + list(
        (repo / "gear_sonic/config").rglob("*.yaml")
    )
    sources += [packet / "plan.json", packet / "command.json", packet / "composed-config.yaml"]
    write_new(
        attempt / "code-input-manifest.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "files": [{"path": str(p), "sha256": sha(p)} for p in sorted(sources)],
        },
    )
    (attempt / "working-tree.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
    )
    (attempt / "git-status.txt").write_bytes(
        subprocess.check_output(["git", "status", "--short"], cwd=repo)
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from gear_sonic.research.hindsight_training.runtime import cuda_preflight; "
            f"print(json.dumps(cuda_preflight({plan['tracking']['cuda_min_free_bytes']})))",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
    )
    (attempt / "preflight-stdout.txt").write_text(probe.stdout)
    (attempt / "preflight-stderr.txt").write_text(probe.stderr)
    probe.check_returncode()
    preflight = json.loads(probe.stdout)
    write_new(attempt / "cuda-preflight.json", preflight)
    if not preflight["launch_allowed"]:
        write_new(
            attempt / "exit.json",
            {"state": "BLOCKED", "reason": "Actual CUDA execution or free memory gate"},
        )
        return 2
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        HYDRA_FULL_ERROR="1",
    )
    env.update(plan.get("launch_environment", {}))
    started = time.monotonic()
    with (attempt / "training.log").open("x") as log:
        proc = subprocess.Popen(
            command["argv"],
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_new(
            attempt / "process.json",
            {
                "pid": proc.pid,
                "monitor_pid": os.getpid(),
                "utc": datetime.now(timezone.utc).isoformat(),
                "hostname": os.uname().nodename,
                "command": command,
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip(),
            },
        )
        timed_out = False
        try:
            proc.wait(timeout=plan["tracking"]["wall_time_cap_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    receipt = Path(plan["run_dir"]) / "training-receipt.json"
    completed = receipt.is_file() and json.loads(receipt.read_text())["state"] == "complete"
    write_new(
        attempt / "exit.json",
        {
            "state": "complete" if completed else "incomplete",
            "exit_code": proc.returncode,
            "wall_seconds": time.monotonic() - started,
            "wall_cap_reached": timed_out,
            "training_receipt": str(receipt) if receipt.is_file() else None,
        },
    )
    return 0 if completed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    raise SystemExit(main(parser.parse_args().packet.resolve()))
