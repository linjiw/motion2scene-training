#!/usr/bin/env python3
"""Freeze the shared-seed E0/E1 corpus with hash-bound evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def collection(path: Path, pattern: str) -> dict:
    members = [
        {"path": str(member.relative_to(path)), "sha256": sha256(member)}
        for member in sorted(path.glob(pattern))
    ]
    digest = hashlib.sha256(
        json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"count": len(members), "collection_sha256": f"sha256:{digest}", "members": members}


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--q3-manifest", type=Path, required=True)
    parser.add_argument("--q3-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.corpus_root.resolve()
    motions = collection(root / "motions", "*.csv")
    sidecars = collection(root / "motions", "*.json")
    if motions["count"] != 144 or sidecars["count"] != 145:
        raise ValueError("frozen corpus requires 144 CSVs and 144 sidecars plus generation report")

    q0_path = root / "e0_q0_joint_reachability_v1.json"
    reference_eval_path = root / "e0_shared_seed_reference_eval_v2.json"
    candidates_path = root / "e1_controlled_duck_ladders_v1/candidates.json"
    q3_root = root / "e1_controlled_duck_ladders_v1/q3_v1"
    q3_run_path = q3_root / "run_record.json"
    retention_path = q3_root / "retention_v1.json"
    generation_path = root / "motions/generation_report.json"
    prompt_cache_path = root / "taxonomy/prompt_cache.npz"
    prompt_cache_manifest_path = root / "taxonomy/prompt_cache.manifest.json"

    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    reference_eval = json.loads(reference_eval_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    q3_run = json.loads(q3_run_path.read_text(encoding="utf-8"))
    retention = json.loads(retention_path.read_text(encoding="utf-8"))
    if generation["generated"] != 144 or generation["failed"] != 0:
        raise ValueError("generation denominator is incomplete")
    if reference_eval["observed"] != 144 or reference_eval["registered"] != 144:
        raise ValueError("reference evaluation denominator is incomplete")
    if q3_run["status"] != "completed" or retention["completed_cells"] != 24:
        raise ValueError("Q3 evidence is not complete")

    payload = {
        "schema_version": "motion2scene_corpus_manifest_v2_q3",
        "corpus_id": "cg-wbc-v2-shared-seed-confirmatory",
        "corpus_root": str(root),
        "status": "q3_complete_no_q4_candidates",
        "permanent_roles": {
            "kimodo_references": "confirmatory_shared_seed_motion_acquisition",
            "controlled_ladders": "reference_and_single_seed_execution_pilot",
            "allowed_for_hallucinator_training": False,
        },
        "experimental_units": {
            "prompt_experiment": "8 generation seeds",
            "controlled_ladder_experiment": "8 base-carrier seeds",
            "nuisance_or_ladder_levels_are_repeated_measurements": True,
        },
        "artifacts": {
            "design": artifact(args.design),
            "prompt_cache": artifact(prompt_cache_path),
            "prompt_cache_manifest": artifact(prompt_cache_manifest_path),
            "generation_report": artifact(generation_path),
            "motion_csv_collection": motions,
            "motion_sidecar_collection": sidecars,
            "q0_joint_reachability": artifact(q0_path),
            "reference_evaluation": artifact(reference_eval_path),
            "controlled_ladder_candidates": artifact(candidates_path),
            "q3_manifest": artifact(args.q3_manifest),
            "q3_predictions": artifact(args.q3_predictions),
            "q3_run_record": artifact(q3_run_path),
            "q3_retention": artifact(retention_path),
        },
        "accounting": {
            "references_generated": generation["generated"],
            "references_failed_generation": generation["failed"],
            "q0_passed": reference_eval["summary"]["q0"]["passed"],
            "q0_failed": reference_eval["summary"]["q0"]["failed"],
            "q1_passed": reference_eval["summary"]["q1"]["passed"],
            "duck_reference_s3": reference_eval["summary"]["semantics_by_mode"]["duck_under"].get(
                "route_aligned", 0
            ),
            "arm_tuck_reference_s3": reference_eval["summary"]["semantics_by_mode"]["arm_tuck"].get(
                "route_aligned", 0
            ),
            "controlled_ladders_attempted": candidates["attempted_ladder_groups"],
            "controlled_reference_candidates": candidates["reference_ladder_candidates"],
            "reference_intervals_attempted": candidates["attempted_intervals"],
            "reference_intervals_nonempty": candidates["nonempty_reference_intervals"],
            "q3_cells_registered": retention["registered_cells"],
            "q3_cells_completed": retention["completed_cells"],
            "q3_cells_accepted": retention["q3_outcomes"].get("accepted", 0),
            "q3_cells_rejected": retention["q3_outcomes"].get("rejected", 0),
            "adapted_s4": sum(item["retained"] for item in retention["s4_by_level"].values()),
            "q3_retained_three_level_prefixes": retention[
                "q3_three_level_retained_prefix_count"
            ],
            "q4_candidates": 0,
            "dataset_eligible_critical_intervals": 0,
        },
        "gate_decisions": {
            "start_q4": False,
            "start_analytic_obstacle_present_physics": False,
            "start_learned_hallucinator": False,
            "next_bottleneck": (
                "construct controller-retained same-carrier ladders or calibrate achieved-route "
                "retention without changing the frozen v2 result"
            ),
        },
        "invalidated_or_superseded": [
            str(root / "invalidated/e0_shared_seed_reference_eval_v1.json"),
            str(q3_root / "superseded/retention_partial.json"),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.out)
    print(f"froze {payload['corpus_id']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
