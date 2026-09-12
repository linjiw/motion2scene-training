#!/usr/bin/env python3
"""Measure reference-side crouch intervals and sample diagnostic beam tokens.

This experiment deliberately keeps execution qualification separate. An interval may be
geometrically nonempty while its target motion is rejected by SONIC; such a row is reported but
marked ineligible for the hallucination training bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from motion2scene.inverse import solve_overhead_interval
from motion2scene.scene import (
    BeamNuisanceDistribution,
    sample_overhead_beams,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def q3_outcomes(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    run = json.loads(path.read_text())
    outcomes = {"nominal": "not_measured_in_this_run"}
    for cell_id, cell in run.get("cells", {}).items():
        label = next((value for value in ("d040", "d055") if value in cell_id), None)
        if label is not None:
            outcomes[label] = cell.get("scientific", {}).get("outcome", "not_measured")
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--q3-run", type=Path)
    parser.add_argument("--nominal-q3-outcome", default="not_measured")
    parser.add_argument("--margin-mm", type=float, default=10.0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=61000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.margin_mm < 0:
        raise SystemExit("--margin-mm must be nonnegative")
    source_repo = args.source_repo.resolve()
    sys.path.insert(0, str(source_repo))
    from gear_sonic.dataset_generation.local_adaptation import (
        _silhouette,
        active_frames,
    )
    from gear_sonic.dataset_generation.self_intersection import (
        DEFAULT_G1_MJCF,
    )

    candidates = json.loads(args.candidates.read_text())
    q3 = q3_outcomes(args.q3_run)
    q3["nominal"] = args.nominal_q3_outcome
    nuisance = BeamNuisanceDistribution()
    rows = []
    for pair in candidates["pairs"]:
        levels = [
            ("nominal", Path(pair["nominal_artifacts"]["csv"])),
            *((rung["label"], Path(rung["artifacts"]["csv"])) for rung in pair["rungs"]),
        ]
        loaded = {label: np.loadtxt(path, delimiter=",") for label, path in levels}
        silhouettes = {
            label: _silhouette(qpos, DEFAULT_G1_MJCF) for label, qpos in loaded.items()
        }
        for offset, ((weaker_id, weaker_path), (target_id, target_path)) in enumerate(
            pairwise(levels)
        ):
            mask = active_frames(loaded[weaker_id], loaded[target_id])
            weaker_reach = float(silhouettes[weaker_id][mask].max())
            target_reach = float(silhouettes[target_id][mask].max())
            interval = solve_overhead_interval(
                target_motion_id=f"{pair['pair_id']}__{target_id}",
                weaker_motion_id=f"{pair['pair_id']}__{weaker_id}",
                target_reach_m=target_reach,
                weaker_reach_m=weaker_reach,
                safety_margin_m=args.margin_mm / 1000.0,
                strike_margin_m=args.margin_mm / 1000.0,
            )
            target_outcome = q3.get(target_id, "not_measured")
            dataset_eligible = interval.nonempty and target_outcome == "accepted"
            beam_samples = (
                sample_overhead_beams(
                    interval,
                    nuisance,
                    args.samples,
                    seed=args.seed + offset,
                )
                if interval.nonempty
                else ()
            )
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "target_level": target_id,
                    "weaker_level": weaker_id,
                    "target_csv": str(target_path),
                    "target_csv_sha256": sha256(target_path),
                    "weaker_csv": str(weaker_path),
                    "weaker_csv_sha256": sha256(weaker_path),
                    "frames_in_station_window": int(mask.sum()),
                    "full_body_conservative_reach": True,
                    "interval": interval.to_dict(),
                    "beam_underside_samples_m": [
                        sample.beam_underside_m for sample in beam_samples
                    ],
                    "beam_samples": [asdict(sample) for sample in beam_samples],
                    "q3_target_outcome": target_outcome,
                    "dataset_eligible": dataset_eligible,
                    "ineligibility_reason": (
                        None
                        if dataset_eligible
                        else (
                            "empty_margin_robust_interval"
                            if not interval.nonempty
                            else "target_not_q3_accepted"
                        )
                    ),
                }
            )

    output = {
        "schema_version": "motion2scene_e1_reference_interval_v1",
        "evidence_tier": "Q0_Q1_reference_only_plus_linked_Q3_outcomes",
        "candidates": str(args.candidates.resolve()),
        "candidates_sha256": sha256(args.candidates),
        "q3_run": str(args.q3_run.resolve()) if args.q3_run else None,
        "q3_run_sha256": sha256(args.q3_run) if args.q3_run else None,
        "margin_mm_each_side": args.margin_mm,
        "samples_per_nonempty_interval": args.samples,
        "sampling": "uniform_stratified_within_margin_robust_critical_interval",
        "nuisance_distribution": asdict(nuisance),
        "attempted_intervals": len(rows),
        "nonempty_intervals": sum(row["interval"]["nonempty"] for row in rows),
        "dataset_eligible_intervals": sum(row["dataset_eligible"] for row in rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        f"{output['nonempty_intervals']}/{output['attempted_intervals']} nonempty; "
        f"{output['dataset_eligible_intervals']} dataset-eligible -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
