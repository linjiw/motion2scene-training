#!/usr/bin/env python3
"""Audit Q3a--Q3d without upgrading legacy self-baseline semantics to paired S0--S4."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from motion2scene.motion.route_semantics import classify_route


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _reference_csv(motions: Path, index: int) -> Path:
    matches = sorted(motions.glob(f"{index:03d}_*.csv"))
    if len(matches) != 1:
        raise ValueError(f"expected one reference CSV for index {index}, found {len(matches)}")
    return matches[0]


def _valid_class(route: str) -> str:
    return f"valid_{route}"


def _metric_retention(reference: dict | None, achieved: dict | None) -> dict | None:
    if not reference or not achieved:
        return None
    for key in ("duck_drop_m", "width_reduction_m"):
        if key in reference and key in achieved:
            ref = float(reference[key])
            got = float(achieved[key])
            return {
                "metric": key,
                "reference": ref,
                "achieved": got,
                "retention_ratio": got / ref if ref > 1e-12 else None,
                "loss": ref - got,
                "evidence_scope": "legacy_self_baseline_not_paired_envelope",
            }
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reference-semantics", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    reference_semantics = {
        int(row["index"]): row
        for row in json.loads(args.reference_semantics.read_text(encoding="utf-8"))
    }
    cell_specs = {cell["cell_id"]: cell for cell in manifest["cells"]}

    rows = []
    for evaluated in evaluation["rows"]:
        index = int(evaluated["index"])
        spec = cell_specs[evaluated["cell_id"]]
        reference_csv = _reference_csv(args.motions, index)
        reference_qpos = np.loadtxt(reference_csv, delimiter=",")
        reference_route = classify_route(reference_qpos[:, :2], spec["turn"], fps=30.0)

        trajectory = Path(evaluated["trajectory"])
        if sha256(trajectory) != evaluated["trajectory_sha256"]:
            raise ValueError(f"trajectory hash mismatch: {trajectory}")
        with trajectory.open("rb") as handle:
            achieved_payload = pickle.load(handle)
        achieved_xy = np.asarray(achieved_payload["root_pos_w"], dtype=np.float64)[:, :2]
        achieved_route = classify_route(
            achieved_xy,
            spec["turn"],
            fps=float(achieved_payload["fps"]),
        )

        reference_valid = reference_route.validity_class == _valid_class(spec["turn"])
        achieved_valid = achieved_route.validity_class == _valid_class(spec["turn"])
        if not reference_valid:
            route_retention = "reference_invalid"
        elif evaluated["outcome"] == "accepted" and achieved_valid:
            route_retention = "retained"
        else:
            route_retention = "lost"

        reference_semantic = reference_semantics[index]
        achieved_semantic = evaluated["semantic"]
        legacy_retained = (
            evaluated["outcome"] == "accepted"
            and reference_semantic["satisfied"] is True
            and achieved_semantic is not None
            and achieved_semantic["satisfied"] is True
        )
        rows.append(
            {
                "index": index,
                "cell_id": evaluated["cell_id"],
                "body_mode": spec["body_mode"],
                "requested_route": spec["turn"],
                "q3a_tracker_survival": evaluated["outcome"] == "accepted",
                "q3b_route_retention": route_retention,
                "reference_route": reference_route.to_dict(),
                "achieved_route": achieved_route.to_dict(),
                "route_delta": {
                    "net_displacement_m": (
                        achieved_route.net_displacement_m - reference_route.net_displacement_m
                    ),
                    "path_length_m": achieved_route.path_length_m - reference_route.path_length_m,
                    "signed_heading_change_rad": (
                        achieved_route.signed_heading_change_rad
                        - reference_route.signed_heading_change_rad
                    ),
                },
                "q3c_s0_s4_status": "not_measured",
                "q3c_reason": (
                    "pilot v1 has no same-seed, same-route walk null; legacy predicates cannot "
                    "be promoted to paired S0-S4 evidence"
                ),
                "legacy_reference_semantic_satisfied": reference_semantic["satisfied"],
                "legacy_achieved_semantic_satisfied": (
                    None if achieved_semantic is None else achieved_semantic["satisfied"]
                ),
                "legacy_controller_retained": legacy_retained,
                "q3d_envelope_retention": _metric_retention(
                    reference_semantic.get("measurements"),
                    None if achieved_semantic is None else achieved_semantic["measurements"],
                ),
            }
        )

    route_counts = Counter(row["q3b_route_retention"] for row in rows)
    semantic_measured = [
        row for row in rows if row["legacy_reference_semantic_satisfied"] is not None
    ]
    payload = {
        "schema_version": "motion2scene_e0_q3_retention_v1",
        "analysis_role": "postregistered_legacy_pilot_audit",
        "inputs": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "evaluation": str(args.evaluation),
            "evaluation_sha256": sha256(args.evaluation),
            "reference_semantics": str(args.reference_semantics),
            "reference_semantics_sha256": sha256(args.reference_semantics),
        },
        "summary": {
            "attempted": len(rows),
            "q3a_tracker_survived": sum(row["q3a_tracker_survival"] for row in rows),
            "q3b_route_retention_counts": dict(sorted(route_counts.items())),
            "legacy_semantic_measured": len(semantic_measured),
            "legacy_controller_retained": sum(
                row["legacy_controller_retained"] for row in semantic_measured
            ),
            "paired_s0_s4_measured": 0,
        },
        "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E0 Q3a--Q3d retention audit",
        "",
        (
            f"Full denominator: **{len(rows)}/{len(rows)} frozen V8 cells audited**; "
            f"Q3a tracker survival is **{payload['summary']['q3a_tracker_survived']}/{len(rows)}**."
        ),
        "",
        (
            "The v1 prompts use different generation seeds, so paired S0--S4 semantics are "
            "**0/6 measured**. Legacy self-baseline behavior predicates are reported below but "
            "are not relabeled as controller-retained S4 evidence."
        ),
        "",
        "| index | mode | Q3a | reference route | achieved route | Q3b | legacy semantic |",
        "|---:|---|---:|---|---|---|---|",
    ]
    for row in rows:
        legacy = "not measured"
        if row["legacy_reference_semantic_satisfied"] is not None:
            legacy = "retained" if row["legacy_controller_retained"] else "not retained"
        lines.append(
            f"| {row['index']:03d} | `{row['body_mode']}` | "
            f"{str(row['q3a_tracker_survival']).lower()} | "
            f"`{row['reference_route']['validity_class']}` | "
            f"`{row['achieved_route']['validity_class']}` | "
            f"`{row['q3b_route_retention']}` | {legacy} |"
        )
    lines.extend(
        [
            "",
            (
                "Q3d reductions use the older torso/link-origin self-baselines. Whole-body "
                "height, route-normal capsule width, and paired achieved-vs-walk envelopes remain "
                "to be measured under v2."
            ),
            "",
        ]
    )
    args.markdown_out.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
