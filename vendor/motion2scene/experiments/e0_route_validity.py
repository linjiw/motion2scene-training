#!/usr/bin/env python3
"""Apply the frozen v2 route predicate descriptively to a generated reference corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from motion2scene.motion.route_semantics import RoutePolicy, classify_route


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()

    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    rows = []
    for csv_path in sorted(args.motions.glob("*.csv")):
        index = int(csv_path.stem.split("_", 1)[0])
        spec = taxonomy["specs"][index]
        qpos = np.loadtxt(csv_path, delimiter=",")
        result = classify_route(qpos[:, :2], spec["turn"], fps=30.0)
        rows.append({"index": index, "motion": csv_path.name, **result.to_dict()})

    counts = Counter(row["validity_class"] for row in rows)
    payload = {
        "schema_version": "motion2scene_e0_route_secondary_v1",
        "analysis_role": "descriptive_secondary_on_pilot_characterization_v1",
        "threshold_use": "frozen_for_shared_seed_v2_not_fitted_to_this_corpus",
        "policy": RoutePolicy().__dict__,
        "attempted": len(rows),
        "counts": dict(sorted(counts.items())),
        "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E0 v1 route-validity secondary analysis",
        "",
        (
            "This applies the route predicate frozen for the future shared-seed v2 design to "
            "the existing development corpus. It is descriptive and does not regrade the frozen "
            "Q3 selection."
        ),
        "",
        f"Full denominator: **{len(rows)}/{len(rows)} references measured**.",
        "",
        "| index | requested | class | net/path | heading (rad) | total turn (rad) | reversal |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['index']:03d} | `{row['expected_route']}` | "
            f"`{row['validity_class']}` | {row['net_to_path_ratio']:.3f} | "
            f"{row['signed_heading_change_rad']:.3f} | "
            f"{row['total_absolute_curvature_rad']:.3f} | {row['reversal']} |"
        )
    lines.extend(
        ["", "Counts: " + ", ".join(f"{key}={value}" for key, value in counts.items()), ""]
    )
    args.markdown_out.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
