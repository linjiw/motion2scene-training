"""Bind multiple executed collection rounds without hiding failures or source roles."""

import argparse
import json
from pathlib import Path

from gear_sonic.research.hindsight_training.runtime import sha, write_new


def aggregate(inputs, output):
    parents = []
    episodes = []
    seen = set()
    teacher = None
    schema = None
    for path in inputs:
        manifest = json.loads(Path(path).read_text())
        if teacher is None:
            teacher = manifest["teacher_sha256"]
            schema = manifest["schema"]
        if teacher != manifest["teacher_sha256"] or schema != manifest["schema"]:
            raise ValueError("Collection teacher/stage mismatch")
        parents.append({"path": str(Path(path).resolve()), "sha256": sha(path)})
        for episode in manifest["episodes"]:
            key = str(Path(episode["path"]).resolve())
            if key in seen:
                raise ValueError("Duplicate episode in aggregation")
            if sha(key) != episode["sha256"]:
                raise ValueError("Collection episode changed")
            seen.add(key)
            episodes.append(dict(episode, source=episode.get("source", manifest.get("source"))))
    if not episodes:
        raise ValueError("No collection episodes supplied")
    write_new(
        output,
        {
            "schema": schema,
            "teacher_sha256": teacher,
            "source": "mixed",
            "parents": parents,
            "episodes": episodes,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.inputs, args.output)
