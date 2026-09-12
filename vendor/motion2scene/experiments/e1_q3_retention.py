#!/usr/bin/env python3
"""Audit Q3 tracker survival and paired controller-retained duck semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from motion2scene.dataset.schema import SemanticStatus
from motion2scene.motion.paired_semantics import (
    PairedSemanticPolicy,
    assess_controller_retention,
    assess_paired_reduction,
)
from motion2scene.motion.route_semantics import classify_route

STATIONS = np.linspace(0.0, 1.0, 101)


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()

    source_repo = args.source_repo.resolve()
    sys.path.insert(0, str(source_repo))
    from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints
    from gear_sonic.dataset_generation.hallucination.motion_envelope import extract_envelope
    from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    run = json.loads(args.run_record.read_text(encoding="utf-8"))
    manifest_cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    policy = PairedSemanticPolicy(minimum_effect=0.05)

    references = {
        (ladder["ladder_group_id"], level["label"]): level
        for ladder in candidates["ladders"]
        for level in ladder["levels"]
    }
    achieved: dict[tuple[str, str], dict] = {}
    rows = []
    for cell_id, cell in sorted(manifest_cells.items()):
        record = run["cells"].get(cell_id, {"status": "not_started"})
        if record["status"] != "completed":
            rows.append(
                {
                    "cell_id": cell_id,
                    "ladder_group_id": cell["ladder_group_id"],
                    "label": cell_id.rsplit("__", 1)[-1],
                    "run_status": record["status"],
                    "q3_tracker_survival": "not_measured",
                    "semantic_status": "not_measured",
                }
            )
            continue
        scientific = record["scientific"]
        trajectory = Path(scientific["artifacts"]["trajectory"])
        if sha256(trajectory) != scientific["artifacts"]["trajectory_sha256"]:
            raise ValueError(f"trajectory hash mismatch for {cell_id}")
        with trajectory.open("rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
        tracks = extract_keypoints(payload)
        envelope = extract_envelope(tracks, cell_id, fractions=STATIONS)
        route = classify_route(
            np.asarray(payload["root_pos_w"])[:, :2],
            "straight",
            fps=float(payload["fps"]),
        )
        label = cell_id.rsplit("__", 1)[-1]
        key = (cell["ladder_group_id"], label)
        achieved[key] = {
            "top": envelope.up_m,
            "route": route,
            "outcome": scientific["outcome"],
        }
        rows.append(
            {
                "cell_id": cell_id,
                "ladder_group_id": cell["ladder_group_id"],
                "base_carrier_id": cell["base_carrier_id"],
                "label": label,
                "run_status": "completed",
                "scientific_outcome": scientific["outcome"],
                "rejection_reasons": scientific["rejection_reasons"],
                "q3_tracker_survival": scientific["outcome"] == "accepted",
                "trajectory": str(trajectory),
                "trajectory_sha256": sha256(trajectory),
                "route_assessment": route.to_dict(),
                "semantic_status": "not_measured" if label != "neutral" else "not_applicable",
            }
        )

    row_by_key = {(row["ladder_group_id"], row["label"]): row for row in rows}
    for key, target in achieved.items():
        ladder_id, label = key
        if label == "neutral":
            continue
        neutral = achieved.get((ladder_id, "neutral"))
        if neutral is None:
            continue
        reference_target = references[(ladder_id, label)]
        reference_neutral = references[(ladder_id, "neutral")]
        reference_semantic = assess_paired_reduction(
            STATIONS,
            np.asarray(reference_target["whole_body_top_m"]),
            np.asarray(reference_neutral["whole_body_top_m"]),
            route_valid=True,
            policy=policy,
        )
        achieved_route_valid = target["route"].validity_class.startswith(
            "valid_"
        ) and neutral["route"].validity_class.startswith("valid_")
        achieved_semantic = assess_paired_reduction(
            STATIONS,
            target["top"],
            neutral["top"],
            route_valid=achieved_route_valid,
            policy=policy,
        )
        retained = assess_controller_retention(
            reference_semantic,
            achieved_semantic,
            policy=policy,
        )
        target_row = row_by_key[key]
        both_track = target["outcome"] == "accepted" and neutral["outcome"] == "accepted"
        controller_retained = (
            both_track and retained.semantic_status == SemanticStatus.CONTROLLER_RETAINED
        )
        target_row.update(
            {
                "reference_semantic": reference_semantic.to_dict(),
                "achieved_semantic": achieved_semantic.to_dict(),
                "controller_retention": retained.to_dict(),
                "semantic_status": (
                    SemanticStatus.CONTROLLER_RETAINED.value
                    if controller_retained
                    else retained.semantic_status.value
                ),
                "q3_controller_retained": controller_retained,
                "reference_to_achieved_peak_retention_ratio": (
                    achieved_semantic.peak_reduction
                    / max(reference_semantic.peak_reduction, 1e-12)
                ),
            }
        )

    completed_rows = [row for row in rows if row["run_status"] == "completed"]
    outcome_counts = Counter(row["scientific_outcome"] for row in completed_rows)
    adapted_completed = [row for row in completed_rows if row["label"] != "neutral"]
    s4_by_label = Counter(
        row["label"]
        for row in adapted_completed
        if row.get("q3_controller_retained") is True
    )
    measured_by_label = Counter(row["label"] for row in adapted_completed)
    complete_prefixes = []
    for ladder in candidates["ladders"]:
        ladder_id = ladder["ladder_group_id"]
        required = [row_by_key.get((ladder_id, label)) for label in ("neutral", "d040", "d055")]
        if (
            all(required)
            and all(row["run_status"] == "completed" for row in required)
            and required[0]["q3_tracker_survival"]
            and all(row.get("q3_controller_retained") is True for row in required[1:])
        ):
            complete_prefixes.append(ladder_id)

    output = {
        "schema_version": "motion2scene_e1_q3_retention_v1",
        "evidence_scope": "single_seed_obstacle_absent_Q3_with_paired_S4_audit",
        "run_record_status": run["status"],
        "registered_cells": len(manifest["cells"]),
        "completed_cells": len(completed_rows),
        "unstarted_or_dependency_skipped_cells": len(manifest["cells"]) - len(completed_rows),
        "q3_outcomes": dict(sorted(outcome_counts.items())),
        "s4_by_level": {
            label: {"retained": s4_by_label[label], "measured": measured_by_label[label]}
            for label in sorted(measured_by_label)
        },
        "q3_three_level_retained_prefixes": complete_prefixes,
        "q3_three_level_retained_prefix_count": len(complete_prefixes),
        "q4_admitted_ladders": 0,
        "candidates_sha256": sha256(args.candidates),
        "manifest_sha256": sha256(args.manifest),
        "run_record_sha256": sha256(args.run_record),
        "analysis_driver_sha256": sha256(Path(__file__)),
        "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E1 controlled-duck Q3 retention",
        "",
        f"Run status: `{run['status']}`.",
        "",
        (
            f"Full denominator: **{len(completed_rows)}/{len(manifest['cells'])} cells "
            f"completed**; accepted={outcome_counts['accepted']}, "
            f"rejected={outcome_counts['rejected']}."
        ),
        "",
        "| level | paired semantics measured | S4 controller-retained |",
        "|---|---:|---:|",
    ]
    for label in sorted(measured_by_label):
        lines.append(f"| `{label}` | {measured_by_label[label]} | {s4_by_label[label]} |")
    lines.extend(
        [
            "",
            f"Three-level Q3-retained prefixes: **{len(complete_prefixes)}**.",
            "",
            "No Q4 or dataset eligibility is inferred from this single-seed result.",
            "",
        ]
    )
    args.markdown_out.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"{len(completed_rows)}/{len(manifest['cells'])} complete; "
        f"{sum(s4_by_label.values())}/{len(adapted_completed)} adapted S4; "
        f"{len(complete_prefixes)} retained three-level prefixes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
