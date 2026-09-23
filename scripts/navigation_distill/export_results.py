"""Export every task-result.json and recovery.json under a packet into two CSVs.

Optionally copies each run's small JSON receipts (never traces, shards, checkpoints
or logs) for chosen stages, so a docs evidence folder can carry per-task results.

  --glob            task-result pattern relative to --packet (default: eval/collect
                    stages, */*/*/task/task-result.json). The stage column is the
                    path between the packet's top folder and the run, e.g.
                    'dag-approach-c2-91260' or, with 'dagger/*/c2-recovery/*/task/
                    task-result.json', 'dag-approach/c2-recovery'.
  --prefix          prefix for the two CSV names (default none).
  --copy-receipts   stage (as in the CSV) whose runs' receipts are copied to
                    <out>/<stage>/<run>/; repeatable.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

DEFAULT_GLOB = "*/*/*/task/task-result.json"
# Per-run receipts worth publishing: all small JSON. Traces (*.npz), recovery
# shards, checkpoints, native/hydra logs are deliberately left in the packet.
RECEIPTS = (
    "task/task-result.json",
    "task/recovery.json",
    "process-result.json",
    "config.json",
    "command.json",
)
RECEIPT_MAX_BYTES = 64_000


def collect(packet, pattern=DEFAULT_GLOB):
    tasks, recoveries, runs = [], [], []
    for f in sorted(packet.glob(pattern)):
        s = json.load(f.open())
        rel = f.relative_to(packet).parts
        stage, run = "/".join(rel[1:-3]), rel[-3]
        runs.append((stage, run, f.parents[1]))
        tasks.append(
            dict(
                stage=stage,
                run=run,
                success=s["navigation_success"],
                reached=s["goal_ever_reached"],
                hold=s["max_hold_ticks"],
                final_distance_m=round(s["final_goal_distance_m"], 3),
                max_force_n=round(s["max_undesired_force_n"], 2),
                fell=s["fell"],
                steps=s["control_steps"],
                stop=s["stop_reason"],
                actor_profile=s.get("actor_profile"),
                student=s.get("student_sha256"),
            )
        )
        r = f.parent / "recovery.json"
        if r.exists():
            d = json.load(r.open())
            sc = d.get("suffix_score") or {}
            recoveries.append(
                dict(
                    stage=stage,
                    run=run,
                    takeover_tick=d["takeover_tick"],
                    supported=d["supported"],
                    rows=d["rows"],
                    supported_rows=d["supported_rows"],
                    suffix_hold=sc.get("max_hold_ticks"),
                    suffix_final_distance_m=sc.get("final_goal_distance_m"),
                    behavior=d["behavior_checkpoint"]["sha256"],
                )
            )
    return tasks, recoveries, runs


def copy_receipts(runs, stages, out):
    """Copy RECEIPTS of every run in `stages` to out/<stage>/<run>/; return the copied paths."""
    missing = set(stages) - {stage for stage, _, _ in runs}
    if missing:
        raise SystemExit(f"no runs for stage(s): {sorted(missing)}")
    copied = []
    for stage, run, run_dir in runs:
        if stage not in stages:
            continue
        for name in RECEIPTS:
            src = run_dir / name
            if not src.is_file():
                continue
            if src.stat().st_size > RECEIPT_MAX_BYTES:
                raise SystemExit(
                    f"{src} is {src.stat().st_size} bytes; receipts are expected to be small"
                )
            dst = out / stage / run / Path(name).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied.append(dst)
    return copied


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--glob", default=DEFAULT_GLOB)
    p.add_argument("--prefix", default="")
    p.add_argument("--copy-receipts", action="append", default=[], metavar="STAGE")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    tasks, recoveries, runs = collect(a.packet, a.glob)
    for name, rows in (("task-results.csv", tasks), ("recovery-attempts.csv", recoveries)):
        if rows:
            with (a.out / f"{a.prefix}{name}").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    copied = copy_receipts(runs, a.copy_receipts, a.out) if a.copy_receipts else []
    print(
        json.dumps(
            dict(
                task_results=len(tasks),
                recovery_attempts=len(recoveries),
                receipts_copied=len(copied),
            )
        )
    )


if __name__ == "__main__":
    main()
