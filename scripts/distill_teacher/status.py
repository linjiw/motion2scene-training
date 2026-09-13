"""Tabulate all evaluations under the current packet's eval/ directory."""
import json
from pathlib import Path
rows = []
for d in sorted(Path("eval").iterdir()):
    f = d / "metrics" / "metrics_eval.json"
    if not d.is_dir() or not f.exists():
        continue
    m = json.loads(f.read_text())["eval/all_metrics_dict"]
    n = len(m["terminated"]); done = sum(not t for t in m["terminated"])
    failed = [str(k)[-5:] for k, t in zip(m["motion_keys"], m["terminated"]) if t]
    rows.append((d.name, f"{done}/{n}", f"{sum(m['progress'])/n:.3f}", " ".join(failed)))
w = max(len(r[0]) for r in rows) if rows else 10
for r in rows:
    print(f"{r[0]:<{w}}  {r[1]:>6}  {r[2]}  {r[3]}")
