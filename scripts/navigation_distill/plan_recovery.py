"""Emit "task.json takeover_tick" lines from a reference stage's control_steps.

Ticks are fractions of the reference arm's completion tick (e.g. the teacher's first
completion), rounded down, so switches happen during approach/braking rather than
after the learner has already failed. Tasks the reference arm failed are skipped unless
--include-failed, in which case fractions apply to the deadline.
"""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--tasks", type=Path, required=True, help="tasks/manifest.json")
p.add_argument("--reference", type=Path, required=True, help="stage dir with task-result.json files")
p.add_argument("--fractions", type=float, nargs="+", default=[0.5])
p.add_argument("--include-failed", action="store_true")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
lines = []
for t in json.load(a.tasks.open())["tasks"]:
    task = json.load(open(t["path"]))
    f = a.reference / task["task_id"] / "task/task-result.json"
    if not f.exists():
        continue
    s = json.load(f.open())
    if not s["navigation_success"] and not a.include_failed:
        continue
    horizon = s["control_steps"] if s["navigation_success"] else task["deadline_ticks"]
    for frac in a.fractions:
        tick = int(horizon * frac)
        if 0 < tick < task["deadline_ticks"]:
            lines.append(f"{t['path']} {tick}")
a.out.write_text("\n".join(lines) + "\n")
print(len(lines), "planned attempts")
