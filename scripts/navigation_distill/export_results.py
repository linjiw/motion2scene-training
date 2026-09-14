"""Export every task-result.json and recovery.json under a packet into two CSVs."""

import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--packet", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
tasks, recoveries = [], []
for f in sorted(a.packet.glob("*/*/*/task/task-result.json")):
    s = json.load(f.open())
    stage, run = f.parts[-4], f.parts[-3]
    tasks.append(dict(stage=stage, run=run, success=s["navigation_success"], reached=s["goal_ever_reached"],
                      hold=s["max_hold_ticks"], final_distance_m=round(s["final_goal_distance_m"], 3),
                      max_force_n=round(s["max_undesired_force_n"], 2), fell=s["fell"], steps=s["control_steps"],
                      stop=s["stop_reason"], actor_profile=s.get("actor_profile"), student=s.get("student_sha256")))
    r = f.parent / "recovery.json"
    if r.exists():
        d = json.load(r.open()); sc = d.get("suffix_score") or {}
        recoveries.append(dict(stage=stage, run=run, takeover_tick=d["takeover_tick"], supported=d["supported"],
                               rows=d["rows"], supported_rows=d["supported_rows"], suffix_hold=sc.get("max_hold_ticks"),
                               suffix_final_distance_m=sc.get("final_goal_distance_m"), behavior=d["behavior_checkpoint"]["sha256"]))
for name, rows in (("task-results.csv", tasks), ("recovery-attempts.csv", recoveries)):
    if rows:
        with (a.out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(json.dumps(dict(task_results=len(tasks), recovery_attempts=len(recoveries))))
