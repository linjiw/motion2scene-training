#!/usr/bin/env python3
"""Evaluate the frozen 144-reference shared-seed Kimodo corpus.

This is a reference-level E0 analysis. It measures route semantics, Q1 self-intersection,
and exact path-aligned G1 collision envelopes. Duck and arm-tuck receive paired S0--S3
labels against the same-seed, same-route walk. Shoulder-turn, step-over, carry-walk and the
null walk remain ``not_measured`` until every preregistered component predicate exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from motion2scene.motion.paired_semantics import (
    PairedSemanticPolicy,
    assess_paired_reduction,
)
from motion2scene.motion.route_semantics import RoutePolicy, classify_route
from motion2scene.motion.shared_seed_corpus import (
    GeneratedReference,
    load_registered_references,
    matched_walks,
)

STATIONS = np.linspace(0.0, 1.0, 101)
MEASURED_MODES = {
    "duck_under": ("whole_body_top_m", 0.05),
    "arm_tuck": ("route_normal_width_m", 0.06),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_source_modules(source_repo: Path):
    sys.path.insert(0, str(source_repo))
    from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints
    from gear_sonic.dataset_generation.hallucination.motion_envelope import extract_envelope
    from gear_sonic.dataset_generation.reference_payload import payload_from_reference
    from gear_sonic.dataset_generation.self_intersection import (
        check_reference_self_intersection,
    )

    return (
        payload_from_reference,
        extract_keypoints,
        extract_envelope,
        check_reference_self_intersection,
    )


def _valid_route(validity_class: str) -> bool:
    return validity_class.startswith("valid_")


def _q0_lookup(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {str(item["csv"]): item for item in payload["records"]}
    if len(records) != int(payload["motions"]):
        raise ValueError("Q0 artifact has duplicate CSV identities")
    return records


def _reference_row(
    reference: GeneratedReference,
    *,
    q0: dict,
    route: dict,
    q1: dict,
    envelope: dict,
) -> dict:
    return {
        "motion_id": reference.motion_id,
        "csv": reference.csv_path.name,
        "csv_sha256": f"sha256:{sha256(reference.csv_path)}",
        "sidecar_sha256": f"sha256:{sha256(reference.sidecar_path)}",
        "prompt_index": reference.cell.prompt_index,
        "registered_prompt_cell_id": reference.cell.prompt_cell_id,
        "generated_prompt_cell_id": reference.generated_prompt_cell_id,
        "body_mode": reference.cell.body_mode,
        "route": reference.cell.route,
        "generation_seed": reference.generation_seed,
        "matched_seed_group_id": reference.matched_seed_group_id,
        "q0": q0,
        "q1": q1,
        "route_assessment": route,
        "envelope": envelope,
        "semantic_assessment": {
            "semantic_predicate_version": "paired_functional_reduction_v2",
            "semantic_status": "not_measured",
            "reason": "assigned after same-seed same-route pairing",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--q0", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    design, references = load_registered_references(args.motions, args.design)
    walks = matched_walks(references)
    q0_by_csv = _q0_lookup(args.q0)
    if set(q0_by_csv) != {reference.csv_path.name for reference in references}:
        raise ValueError("Q0 artifact and generated corpus do not have identical members")

    (
        payload_from_reference,
        extract_keypoints,
        extract_envelope,
        check_reference_self_intersection,
    ) = _load_source_modules(args.source_repo)

    rows: list[dict] = []
    for count, reference in enumerate(references, start=1):
        qpos = np.loadtxt(reference.csv_path, delimiter=",")
        route_result = classify_route(qpos[:, :2], reference.cell.route, fps=args.fps)
        q1_result = check_reference_self_intersection(qpos, mjcf_path=args.mjcf)
        payload = payload_from_reference(qpos, fps=args.fps, mjcf_path=args.mjcf)
        tracks = extract_keypoints(payload)
        envelope = extract_envelope(
            tracks,
            reference.motion_id,
            fractions=STATIONS,
        )
        rows.append(
            _reference_row(
                reference,
                q0=q0_by_csv[reference.csv_path.name],
                route=route_result.to_dict(),
                q1={
                    "status": "passed" if q1_result.passed else "failed",
                    "pelvis_hip_depth_m": q1_result.pelvis_hip_depth_m,
                    "max_self_intersection_depth_m": q1_result.max_depth_m,
                    "frames_checked": q1_result.frames_checked,
                    "reasons": list(q1_result.reasons),
                },
                envelope={
                    "station_progress": STATIONS.tolist(),
                    "whole_body_top_m": envelope.up_m.tolist(),
                    "route_normal_width_m": (envelope.left_m + envelope.right_m).tolist(),
                },
            )
        )
        if count % 24 == 0:
            print(f"evaluated {count}/{len(references)} references", flush=True)

    by_motion_id = {row["motion_id"]: row for row in rows}
    for reference in references:
        row = by_motion_id[reference.motion_id]
        mode = reference.cell.body_mode
        if mode not in MEASURED_MODES:
            reasons = {
                "walk": "null motion; route measured but no adaptation predicate applies",
                "shoulder_turn": (
                    "route-normal width measured, but the preregistered shoulder-axis yaw "
                    "component has not been implemented"
                ),
                "step_over": (
                    "foot apex alone is insufficient; support, landing, recovery and "
                    "bilateral-flight components are not measured"
                ),
                "carry_walk": "no frozen functional predicate exists",
            }
            row["semantic_assessment"]["reason"] = reasons[mode]
            continue

        field, minimum_effect = MEASURED_MODES[mode]
        walk_reference = walks[reference.pair_key]
        walk_row = by_motion_id[walk_reference.motion_id]
        route_valid = _valid_route(row["route_assessment"]["validity_class"]) and _valid_route(
            walk_row["route_assessment"]["validity_class"]
        )
        semantic = assess_paired_reduction(
            STATIONS,
            np.asarray(row["envelope"][field]),
            np.asarray(walk_row["envelope"][field]),
            route_valid=route_valid,
            policy=PairedSemanticPolicy(minimum_effect=minimum_effect),
        )
        row["semantic_assessment"] = {
            **semantic.to_dict(),
            "metric": field,
            "minimum_effect_m": minimum_effect,
            "matched_walk_motion_id": walk_reference.motion_id,
            "target_route_valid": _valid_route(row["route_assessment"]["validity_class"]),
            "matched_walk_route_valid": _valid_route(
                walk_row["route_assessment"]["validity_class"]
            ),
        }

    route_counts = Counter(row["route_assessment"]["validity_class"] for row in rows)
    q0_counts = Counter("passed" if row["q0"]["passed"] else "failed" for row in rows)
    q1_counts = Counter(row["q1"]["status"] for row in rows)
    semantic_counts: dict[str, Counter] = defaultdict(Counter)
    paired_effects: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        mode = row["body_mode"]
        status = row["semantic_assessment"]["semantic_status"]
        semantic_counts[mode][status] += 1
        if "peak_reduction" in row["semantic_assessment"]:
            paired_effects[mode].append(float(row["semantic_assessment"]["peak_reduction"]))

    source_files = {
        "reference_payload": args.source_repo
        / "gear_sonic/dataset_generation/reference_payload.py",
        "semantic_keypoints": args.source_repo
        / "gear_sonic/dataset_generation/hallucination/keypoints.py",
        "motion_envelope": args.source_repo
        / "gear_sonic/dataset_generation/hallucination/motion_envelope.py",
        "self_intersection": args.source_repo
        / "gear_sonic/dataset_generation/self_intersection.py",
    }
    payload = {
        "schema_version": "motion2scene_e0_shared_seed_reference_eval_v2",
        "corpus_id": design["corpus_id"],
        "analysis_scope": "reference_only_q0_q1_route_and_paired_s0_s3",
        "registered": int(design["design"]["registered_references"]),
        "observed": len(rows),
        "independent_experimental_units": len({row["generation_seed"] for row in rows}),
        "design_sha256": f"sha256:{sha256(args.design)}",
        "q0_artifact_sha256": f"sha256:{sha256(args.q0)}",
        "generation_source_commit": design["generator"]["commit"],
        "analysis_source_repo": str(args.source_repo.resolve()),
        "analysis_source_commit": git_head(args.source_repo),
        "analysis_source_files": {
            name: {"path": str(path), "sha256": f"sha256:{sha256(path)}"}
            for name, path in source_files.items()
        },
        "analysis_driver": {
            "path": str(Path(__file__).resolve()),
            "sha256": f"sha256:{sha256(Path(__file__))}",
        },
        "mjcf_sha256": f"sha256:{sha256(args.mjcf)}",
        "route_policy": RoutePolicy().__dict__,
        "station_progress": STATIONS.tolist(),
        "summary": {
            "q0": dict(sorted(q0_counts.items())),
            "q1": dict(sorted(q1_counts.items())),
            "route": dict(sorted(route_counts.items())),
            "semantics_by_mode": {
                mode: dict(sorted(counts.items()))
                for mode, counts in sorted(semantic_counts.items())
            },
            "paired_peak_reduction_m": {
                mode: {
                    "n": len(values),
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
                for mode, values in sorted(paired_effects.items())
            },
        },
        "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E0 shared-seed reference qualification",
        "",
        (
            f"Full denominator: **{len(rows)}/{payload['registered']} registered references** "
            f"across **{payload['independent_experimental_units']} independent seeds**."
        ),
        "",
        "This report is reference-level evidence only. S4 requires controller execution and is ",
        "not assigned here.",
        "",
        "## Qualification funnel",
        "",
        "| check | passed | failed |",
        "|---|---:|---:|",
        f"| Q0 joint reachability | {q0_counts['passed']} | {q0_counts['failed']} |",
        f"| Q1 self-intersection | {q1_counts['passed']} | {q1_counts['failed']} |",
        "",
        "## Route validity",
        "",
        "| class | references |",
        "|---|---:|",
    ]
    for name, value in sorted(route_counts.items()):
        lines.append(f"| `{name}` | {value} |")
    lines.extend(
        [
            "",
            "## Paired semantic evidence",
            "",
            "Every duck/arm-tuck target is compared with the same-seed, same-route walk.",
            "",
            "| body mode | n | S0 absent | S1 elicited | S2 localized | S3 route-aligned | not measured |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, counts in sorted(semantic_counts.items()):
        lines.append(
            f"| `{mode}` | {sum(counts.values())} | {counts['absent']} | "
            f"{counts['elicited']} | {counts['localized']} | {counts['route_aligned']} | "
            f"{counts['not_measured']} |"
        )
    lines.extend(
        [
            "",
            "Shoulder-turn is not promoted from width alone; step-over is not promoted from foot ",
            "height alone; carry-walk has no frozen predicate. These remain `not_measured`.",
            "",
        ]
    )
    args.markdown_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
