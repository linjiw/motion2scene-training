"""Task-by-stage success table from task-result.json files. Usage: table.py STAGE_DIR..."""

import json
import sys
from pathlib import Path

stages = [Path(s) for s in sys.argv[1:]]
tasks = sorted({d.name for s in stages for d in s.iterdir() if (d / "task/task-result.json").exists()})
names = [s.name for s in stages]
w = max(len(t) for t in tasks) if tasks else 10
print(f"{'task':<{w}}  " + "  ".join(f"{n:>22}" for n in names))
totals = [0] * len(stages)
for t in tasks:
    cells = []
    for i, s in enumerate(stages):
        f = s / t / "task/task-result.json"
        if not f.exists():
            cells.append(f"{'-':>22}")
            continue
        r = json.loads(f.read_text())
        totals[i] += r["navigation_success"]
        tag = "PASS" if r["navigation_success"] else ("contact" if not r["collision_free"] else ("fall" if r["fell"] else "fail"))
        cells.append(f"{tag} h{r['max_hold_ticks']:>2} d{r['final_goal_distance_m']:.2f}".rjust(22))
    print(f"{t:<{w}}  " + "  ".join(cells))
print(f"{'TOTAL':<{w}}  " + "  ".join(f"{tot}/{sum(1 for t in tasks if (s / t / 'task/task-result.json').exists())}".rjust(22) for tot, s in zip(totals, stages)))
