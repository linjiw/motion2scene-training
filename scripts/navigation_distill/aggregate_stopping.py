"""Bind successful StoppingTeacherCallback collections into one task-positive manifest.

Only collections with scene_task_success == true enter; failed tasks are listed separately
for cost accounting. The output matches what navigation_data.load_successful_tasks expects.
"""

import argparse
import json
from pathlib import Path

from gear_sonic.research.hindsight_training.runtime import sha, write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stages", nargs="+", type=Path, required=True, help="teacher stage dirs")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    parents, episodes, failed, teacher = [], [], [], None
    for stage in a.stages:
        for collection in sorted(stage.glob("*/task/collection.json")):
            m = json.loads(collection.read_text())
            teacher = teacher or m["teacher_sha256"]
            if m["teacher_sha256"] != teacher or m.get("context_schema") != "complete_known_map_goal_v1":
                raise ValueError(f"Mixed teacher or schema in {collection}")
            if not m.get("scene_task_success"):
                failed.append(str(collection))
                continue
            parents.append(dict(path=str(collection.resolve()), sha256=sha(collection)))
            for e in m["episodes"]:
                if sha(e["path"]) != e["sha256"]:
                    raise ValueError(f"Changed shard {e['path']}")
                episodes.append(dict(e, source=e.get("source", m.get("source"))))
    if not episodes:
        raise ValueError("No successful stopping episodes")
    write_new(
        a.output,
        dict(
            schema="bfm_executed_foundation_v1",
            context_schema="complete_known_map_goal_v1",
            teacher_sha256=teacher,
            source="executed_native_teacher_prefix",
            parents=parents,
            episodes=episodes,
            failed_teacher_tasks=failed,
            rows=sum(e["rows"] for e in episodes),
        ),
    )
    print(json.dumps(dict(successful=len(episodes), failed=len(failed), rows=sum(e["rows"] for e in episodes))))


if __name__ == "__main__":
    main()
