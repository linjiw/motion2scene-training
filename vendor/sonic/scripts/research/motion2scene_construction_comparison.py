#!/usr/bin/env python3
"""Compare reference/executed construction on a common geometric candidate pool.

This measures geometric selection disagreement, not collision or traversal
success. Complete physical teacher rollouts supply the latter independently.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_prepare_acquisition_pools import read_bound  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)


def selected_support(rows, candidate_ids, option_ids, outer, counts, inner, margin, offsets):
    """Test exactly each proposal's chosen positive and negative schedules."""
    ci, oi = {v: i for i, v in enumerate(candidate_ids)}, {v: i for i, v in enumerate(option_ids)}
    reports = []
    for row in rows:
        i = ci[row["candidate_id"]]
        positive, negative = oi[row["positive_option_id"]], oi[row["negative_option_id"]]
        clear = bool(counts[i, positive] == offsets and outer[i, positive] >= margin)
        contrast = bool(inner[i, negative] <= -margin)
        reports.append(
            dict(
                candidate_id=row["candidate_id"],
                stratum=row["stratum"],
                positive_clear=clear,
                negative_intersects=contrast,
                contrast_retained=clear and contrast,
            )
        )
    return dict(
        candidates=len(rows),
        chosen_positive_clear=sum(r["positive_clear"] for r in reports),
        chosen_negative_intersects=sum(r["negative_intersects"] for r in reports),
        chosen_contrast_retained=sum(r["contrast_retained"] for r in reports),
        by_stratum={
            s: dict(
                candidates=sum(r["stratum"] == s for r in reports),
                chosen_contrast_retained=sum(
                    r["stratum"] == s and r["contrast_retained"] for r in reports
                ),
            )
            for s in sorted({r["stratum"] for r in reports})
        },
        rows=reports,
    )


def run(pools, references, out):
    source, reference = read_bound(artifact(pools / "result.json")), read_bound(
        artifact(references / "result.json")
    )
    registration = read_bound(artifact(pools / "registration.json"))
    by_seed = {p["seed"]: p for p in reference["pools"]}
    summaries = []
    for pool in source["pools"]:
        other = by_seed[pool["seed"]]
        executed_rows = read_bound(pool["arms"]["analytic_contrast"])["rows"]
        reference_rows = read_bound(other["queue"])["rows"]
        with np.load(checked(Path(pool["geometry"]["path"]), pool["geometry"]["sha256"])) as g:
            ids, options = g["candidate_ids"].tolist(), g["option_ids"].tolist()
            outer = np.nanmin(g["outer_clearance_by_offset_m"], axis=1)
            counts, inner = g["evaluated_offset_counts"], g["nominal_inner_clearance_m"]
            support = selected_support(
                reference_rows,
                ids,
                options,
                outer,
                counts,
                inner,
                registration["margin_m"],
                len(registration["offsets_world_xyz_yaw"]),
            )
        executed_ids, reference_ids = {r["candidate_id"] for r in executed_rows}, {
            r["candidate_id"] for r in reference_rows
        }
        summaries.append(
            dict(
                seed=pool["seed"],
                candidates=len(ids),
                executed_eligible=len(executed_rows),
                reference_eligible=len(reference_rows),
                shared_eligible=len(executed_ids & reference_ids),
                executed_strata=dict(Counter(r["stratum"] for r in executed_rows)),
                reference_strata=dict(Counter(r["stratum"] for r in reference_rows)),
                reference_proposals_under_executed_geometry=support,
            )
        )
    out.mkdir(parents=True, exist_ok=False)
    return write_new(
        out / "result.json",
        dict(
            pools=artifact(pools / "result.json"),
            reference=artifact(references / "result.json"),
            implementation=artifact(Path(__file__)),
            seeds=summaries,
            new_physics_steps=0,
            interpretation=(
                "common-pool geometric disagreement; physical acquisition yield "
                "and policy passage measured separately"
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pools", "references", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.pools, args.references, args.out)))
