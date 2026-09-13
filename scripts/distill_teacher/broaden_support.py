"""Write a broad-support copy of a PrefixFoundationCollectionCallback collection.

Mirrors the recorded "native-completion support" tier: a teacher attempt that completed
without native termination keeps every recorded decision; a terminated attempt keeps its
original strict screened prefix. Original shards and collection.json stay untouched.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.research.hindsight_training.runtime import sha, write_new


def main(collection, output):
    manifest = json.loads(Path(collection).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    episodes, rows, strict_rows = [], 0, 0
    for episode in manifest["episodes"]:
        with np.load(episode["path"]) as data:
            arrays = {k: data[k].copy() for k in data.files}
        strict = arrays["query_mask"].copy()
        strict_rows += int(strict.sum())
        if not episode["terminated"]:
            arrays["query_mask"] = np.ones(episode["rows"], dtype=bool)
        path = output / Path(episode["path"]).name
        np.savez_compressed(path, **arrays)
        supported = int(arrays["query_mask"].sum())
        rows += supported
        episodes.append(
            dict(
                episode,
                path=str(path),
                sha256=sha(path),
                eligible=supported >= 2,
                supported_query_rows=supported,
                strict_supported_query_rows=int(strict.sum()),
                source="executed_native_teacher_prefix",
                support_tier="native_completion_broad" if not episode["terminated"] else "strict_prefix",
            )
        )
    write_new(
        output / "collection.json",
        dict(
            manifest,
            episodes=episodes,
            source="executed_native_teacher_prefix",
            parent_collection={"path": str(Path(collection).resolve()), "sha256": sha(collection)},
            query_qualification="broad: completed attempts keep all decisions; failed attempts keep strict prefix",
        ),
    )
    print(json.dumps(dict(episodes=len(episodes), broad_rows=rows, strict_rows=strict_rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    main(a.collection, a.output)
