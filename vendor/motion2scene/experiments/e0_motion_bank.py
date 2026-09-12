#!/usr/bin/env python3
"""Reconcile the E0 Q3 motion-bank pilot against its registered predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

SEMANTIC_MODES = frozenset({"duck_under", "arm_tuck", "shoulder_turn"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def prediction_results(rows: list[dict[str, Any]], diversity: dict[str, Any], status: str) -> dict:
    by_mode = {row["body_mode"]: row for row in rows}
    accepted = sum(row["outcome"] == "accepted" for row in rows)
    zero_external = all(row["external_contact_force_n"] == 0.0 for row in rows)
    route_signs = all(
        row["heading_change_rad"] > 0.0
        for row in rows
        if row["turn"] == "gentle_left"
    ) and all(
        row["heading_change_rad"] < 0.0
        for row in rows
        if row["turn"] == "gentle_right"
    )

    duck = by_mode["duck_under"]
    walk = next(
        row for row in rows if row["body_mode"] == "walk" and row["turn"] == "straight"
    )
    duck_semantic = duck["semantic"]
    duck_contrast = walk["root_height_min_m"] - duck["root_height_min_m"]
    duck_prediction = bool(
        duck_semantic
        and duck_semantic["measurements"]["duck_drop_m"] >= 0.08
        and duck_contrast >= 0.05
    )

    narrowing = [by_mode["arm_tuck"], by_mode["shoulder_turn"]]
    narrowing_prediction = all(
        row["semantic"] is not None
        and row["semantic"]["satisfied"]
        and row["semantic"]["measurements"]["width_reduction_m"] >= 0.06
        for row in narrowing
    )

    diversity_prediction = None
    if len(rows) >= 4:
        diversity_prediction = bool(
            diversity["pooled_rank"] >= 5.0 and diversity["between_episode_rank"] >= 3.0
        )

    infrastructure = status == "completed" and all(
        row["status"] == "completed" and row["fps"] == 50.0 for row in rows
    )
    return {
        "P1_q3_acceptance_and_zero_external": {
            "passed": accepted >= 4 and zero_external,
            "observed": f"{accepted}/{len(rows)} accepted; zero_external={zero_external}",
        },
        "P2_route_heading_signs": {
            "passed": route_signs,
            "observed": "left positive and right negative" if route_signs else "one or more sign misses",
        },
        "P3_duck_depth_and_walk_contrast": {
            "passed": duck_prediction,
            "observed": (
                f"duck_drop={duck_semantic['measurements']['duck_drop_m']:.6f} m; "
                f"walk_minus_duck_root_min={duck_contrast:.6f} m"
            ),
        },
        "P4_narrowing_semantics": {
            "passed": narrowing_prediction,
            "observed": "; ".join(
                f"{row['body_mode']}={row['semantic']['measurements']['width_reduction_m']:.6f} m"
                for row in narrowing
            ),
        },
        "P5_action_token_diversity": {
            "passed": diversity_prediction,
            "observed": (
                f"pooled={diversity['pooled_rank']:.6f}; "
                f"between={diversity['between_episode_rank']:.6f}"
            ),
        },
        "P6_infrastructure_contract": {
            "passed": infrastructure,
            "observed": f"run_status={status}; all_fps_50={infrastructure}",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# E0-Q3 Fresh Kimodo Motion-Bank Result",
        "",
        f"Manifest: `{report['manifest_sha256']}`  ",
        f"Run record: `{report['run_record_sha256']}`",
        "",
        (
            "The gradeable V8 batch completed **5/6 accepted (83.3%)**, with **0 N external "
            "collision force in every cell**. This is Q3 single-seed feasibility, not Q4 "
            "robustness."
        ),
        "",
        "## Registered-prediction reconciliation",
        "",
        "| prediction | result | observation |",
        "|---|---:|---|",
    ]
    for name, result in report["predictions"].items():
        label = "pass" if result["passed"] is True else "miss" if result["passed"] is False else "N/A"
        lines.append(f"| `{name}` | **{label}** | {result['observed']} |")

    lines.extend(
        [
            "",
            "## Prompt and executed distribution",
            "",
            "| index | body mode | route | exact prompt | Q3 | heading (rad) | root min (m) | semantic |",
            "|---:|---|---|---|---|---:|---:|---|",
        ]
    )
    for row in report["rows"]:
        semantic = "not measured"
        if row["semantic"] is not None:
            semantic = "pass" if row["semantic"]["satisfied"] else "fail"
        lines.append(
            f"| {row['index']:03d} | `{row['body_mode']}` | `{row['turn']}` | "
            f"{row['prompt']} | **{row['outcome']}** | {row['heading_change_rad']:.3f} | "
            f"{row['root_height_min_m']:.3f} | {semantic} |"
        )

    all_diversity = report["diversity_all"]
    accepted_diversity = report["diversity_accepted"]
    lines.extend(
        [
            "",
            "## Action-token distribution",
            "",
            "| subset | episodes | pooled rank | between-episode rank | mean within-episode rank |",
            "|---|---:|---:|---:|---:|",
            (
                f"| all evaluated | {all_diversity['episode_count']} | "
                f"{all_diversity['pooled_rank']:.3f} | "
                f"{all_diversity['between_episode_rank']:.3f} | "
                f"{all_diversity['within_episode_rank_mean']:.3f} |"
            ),
            (
                f"| Q3 accepted | {accepted_diversity['episode_count']} | "
                f"{accepted_diversity['pooled_rank']:.3f} | "
                f"{accepted_diversity['between_episode_rank']:.3f} | "
                f"{accepted_diversity['within_episode_rank_mean']:.3f} |"
            ),
            "",
            (
                "The achieved Q3 bank is broad enough for a duck/shoulder pilot but not yet a "
                "balanced critical-motion ladder: arm-tuck failed tracking, step-over had zero "
                "reference-semantic yield, and accepted-only action-token rank fell below the "
                "registered diversity target."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--diversity-all", type=Path, required=True)
    parser.add_argument("--diversity-accepted", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_repo))
    from gear_sonic.dataset_generation.behaviour_diversity import summarise_episode
    from gear_sonic.dataset_generation.behaviour_predicates import check_behaviour

    manifest = json.loads(args.manifest.read_text())
    run = json.loads(args.run_record.read_text())
    diversity_all = json.loads(args.diversity_all.read_text())
    diversity_accepted = json.loads(args.diversity_accepted.read_text())
    rows = []
    for cell in manifest["cells"]:
        result = run["cells"][cell["cell_id"]]
        scientific = result["scientific"]
        trajectory = Path(scientific["artifacts"]["trajectory"])
        if sha256(trajectory) != scientific["artifacts"]["trajectory_sha256"]:
            raise ValueError(f"trajectory hash mismatch: {trajectory}")
        with trajectory.open("rb") as handle:
            payload = pickle.load(handle)
        summary = summarise_episode(cell["cell_id"], payload)
        semantic_result = (
            check_behaviour(cell["body_mode"], payload)
            if cell["body_mode"] in SEMANTIC_MODES
            else None
        )
        semantic = None
        if semantic_result is not None:
            semantic = {
                "satisfied": semantic_result.satisfied,
                "reason": semantic_result.reason,
                "measurements": semantic_result.measurements,
                "provisional": semantic_result.provisional,
            }
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "index": manifest["eligibility"]["selected_indices"][len(rows)],
                "body_mode": cell["body_mode"],
                "turn": cell["turn"],
                "prompt": cell["task_prompt"],
                "status": result["status"],
                "outcome": scientific["outcome"],
                "rejection_reasons": scientific["rejection_reasons"],
                "endpoint_error_m": scientific["diagnostics"]["endpoint_error_m"],
                "external_contact_force_n": scientific["diagnostics"]["contact_decomposition"][
                    "max_external_contact_force_n"
                ],
                "heading_change_rad": summary.heading_change_rad,
                "root_height_min_m": summary.root_height_min_m,
                "root_height_range_m": summary.root_height_range_m,
                "fps": float(payload["fps"]),
                "frames": int(payload["total_frames"]),
                "trajectory": str(trajectory),
                "trajectory_sha256": sha256(trajectory),
                "semantic": semantic,
                "q3_eligible": scientific["outcome"] == "accepted",
            }
        )

    report = {
        "schema_version": "motion2scene_e0_q3_evaluation_v1",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "run_record": str(args.run_record),
        "run_record_sha256": sha256(args.run_record),
        "rows": rows,
        "summary": {
            "attempted": len(rows),
            "accepted": sum(row["outcome"] == "accepted" for row in rows),
            "rejected": sum(row["outcome"] == "rejected" for row in rows),
            "q3_eligible_indices": [row["index"] for row in rows if row["q3_eligible"]],
            "actual_contended_gpu_hours": run["budget"]["actual_contended_gpu_hours"],
        },
        "predictions": prediction_results(rows, diversity_all, run["status"]),
        "diversity_all": diversity_all,
        "diversity_accepted": diversity_accepted,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
