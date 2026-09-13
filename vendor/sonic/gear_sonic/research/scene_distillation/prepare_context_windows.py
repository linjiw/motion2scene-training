"""Create exact contiguous training-motion windows for broader teacher coverage.

Window reset is a bounded reference-state initialization approximation. It does
not claim full-motion completion, randomized RSI, or scene-task qualification.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from gear_sonic.research.hindsight_training.runtime import sha, write_new


def prepare(catalog_path, output, fractions=(0.25, 0.5, 0.75)):
    catalog = json.loads(Path(catalog_path).read_text())
    if catalog["schema"] != "direct_context_catalog_v1" or catalog["terrain"] != "plane":
        raise ValueError("Window pilot requires paired plane tasks")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for fraction in fractions:
        if not 0 < fraction < 1:
            raise ValueError("Window start fraction must lie inside motion")
        folder = output / f"quarter-{int(fraction*4)}"
        motions = folder / "motions"
        motions.mkdir(parents=True)
        tasks = {}
        metadata = {}
        for ident, task in catalog["tasks"].items():
            if task["split"] != "train" or task["obstacles"]:
                raise ValueError("Development or obstacle task in training windows")
            source = task["native_motion"]["path"]
            if sha(source) != task["native_motion"]["sha256"]:
                raise ValueError("Original motion changed")
            key = "hindsight_" + ident
            original = joblib.load(source)[key]
            n = len(original["dof"])
            start = min(int(n * fraction), n - 2)
            entry = {
                k: (v[start:].copy() if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v)
                for k, v in original.items()
            }
            if len(entry["dof"]) < 2:
                raise ValueError("Empty reference window")
            for field in ["dof", "root_trans_offset", "root_rot", "pose_aa"]:
                np.testing.assert_array_equal(entry[field], original[field][start:])
            path = motions / (key + ".pkl")
            joblib.dump({key: entry}, path, compress=3)
            metadata[key] = {"length": len(entry["dof"]), "fps": float(entry["fps"])}
            tasks[ident] = dict(
                task,
                start_xyz=entry["root_trans_offset"][0].tolist(),
                native_motion={"path": str(path.resolve()), "sha256": sha(path)},
                source_native_motion=task["native_motion"],
                source_start_frame=start,
                source_total_frames=n,
                reference_window=True,
                provenance="contiguous source window; newly executed teacher prefixes only",
            )
        joblib.dump(metadata, motions / "metadata.pkl", compress=3)
        path = folder / "catalog.json"
        write_new(path, dict(catalog, tasks=tasks))
        records.append(
            {
                "fraction": fraction,
                "catalog": str(path.resolve()),
                "sha256": sha(path),
                "motions": str(motions.resolve()),
                "motion_count": len(tasks),
            }
        )
    write_new(
        output / "manifest.json",
        {
            "schema": "context_teacher_windows_v1",
            "windows": records,
            "source_catalog_sha256": sha(catalog_path),
        },
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.catalog, args.output))


if __name__ == "__main__":
    main()
