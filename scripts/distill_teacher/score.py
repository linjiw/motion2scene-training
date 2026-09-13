"""Print completion and mean progress for native metrics_eval.json files."""
import json, sys
from pathlib import Path
for p in sys.argv[1:]:
    f = Path(p) / "metrics" / "metrics_eval.json"
    if not f.exists():
        print(f"{p}: missing"); continue
    m = json.loads(f.read_text())["eval/all_metrics_dict"]
    n = len(m["terminated"]); done = sum(not t for t in m["terminated"])
    failed = [str(k)[-5:] for k, t in zip(m["motion_keys"], m["terminated"]) if t]
    print(f"{p}: {done}/{n} complete, progress {sum(m['progress'])/n:.3f}, failed {failed}")
