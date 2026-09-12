#!/usr/bin/env python3
"""Audit reference-to-achieved route retention on frozen E1 Q3 neutral controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import statistics
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from motion2scene.motion.route_retention import compare_routes
from motion2scene.motion.route_semantics import DEFAULT_ROUTE_POLICY, classify_route

SMOOTHING_SECONDS = (0.25, 0.50, 0.75, 1.00)
STATION_SPACING_METRES = (0.25, 0.50, 0.75)


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _median(rows: list[dict], key: str) -> float:
    return float(statistics.median(row["retention"][key] for row in rows))


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
    from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    run = json.loads(args.run_record.read_text(encoding="utf-8"))
    reference_by_ladder = {
        ladder["ladder_group_id"]: next(
            level for level in ladder["levels"] if level["label"] == "neutral"
        )
        for ladder in candidates["ladders"]
    }
    neutral_cells = [
        cell for cell in manifest["cells"] if cell["cell_id"].endswith("__neutral")
    ]

    rows = []
    sensitivity_inputs = []
    for cell in sorted(neutral_cells, key=lambda item: item["cell_id"]):
        cell_id = cell["cell_id"]
        record = run["cells"].get(cell_id)
        if record is None or record["status"] != "completed":
            rows.append(
                {
                    "cell_id": cell_id,
                    "ladder_group_id": cell["ladder_group_id"],
                    "measurement_status": "not_measured",
                    "reason": "registered neutral cell did not complete",
                }
            )
            continue

        reference_record = reference_by_ladder[cell["ladder_group_id"]]
        reference_path = Path(reference_record["csv"])
        if sha256(reference_path) != reference_record["csv_sha256"]:
            raise ValueError(f"reference hash mismatch for {cell_id}")
        trajectory = Path(record["scientific"]["artifacts"]["trajectory"])
        if sha256(trajectory) != record["scientific"]["artifacts"]["trajectory_sha256"]:
            raise ValueError(f"trajectory hash mismatch for {cell_id}")

        reference_qpos = np.loadtxt(reference_path, delimiter=",")
        with trajectory.open("rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
        reference_xy = np.asarray(reference_qpos[:, :2], dtype=np.float64)
        achieved_xy = np.asarray(payload["root_pos_w"], dtype=np.float64)[:, :2]
        retention = compare_routes(
            reference_xy,
            achieved_xy,
            expected_route="straight",
            reference_fps=30.0,
            achieved_fps=float(payload["fps"]),
        )
        rows.append(
            {
                "cell_id": cell_id,
                "ladder_group_id": cell["ladder_group_id"],
                "base_carrier_id": cell["base_carrier_id"],
                "measurement_status": "measured",
                "scientific_outcome": record["scientific"]["outcome"],
                "reference_csv": str(reference_path),
                "reference_csv_sha256": sha256(reference_path),
                "achieved_trajectory": str(trajectory),
                "achieved_trajectory_sha256": sha256(trajectory),
                "retention": retention.to_dict(),
            }
        )
        sensitivity_inputs.append(
            (
                cell_id,
                reference_xy,
                achieved_xy,
                float(payload["fps"]),
            )
        )

    measured = [row for row in rows if row["measurement_status"] == "measured"]
    sensitivity = []
    for smoothing_s in SMOOTHING_SECONDS:
        for spacing_m in STATION_SPACING_METRES:
            policy = replace(
                DEFAULT_ROUTE_POLICY,
                centerline_smoothing_s=smoothing_s,
                centerline_station_spacing_m=spacing_m,
            )
            policy_rows = []
            for cell_id, reference_xy, achieved_xy, achieved_fps in sensitivity_inputs:
                reference = classify_route(reference_xy, "straight", fps=30.0, policy=policy)
                achieved = classify_route(
                    achieved_xy,
                    "straight",
                    fps=achieved_fps,
                    policy=policy,
                )
                policy_rows.append(
                    {
                        "cell_id": cell_id,
                        "reference_class": reference.validity_class,
                        "achieved_class": achieved.validity_class,
                        "reference_total_absolute_curvature_rad": (
                            reference.total_absolute_curvature_rad
                        ),
                        "achieved_total_absolute_curvature_rad": (
                            achieved.total_absolute_curvature_rad
                        ),
                    }
                )
            sensitivity.append(
                {
                    "centerline_smoothing_s": smoothing_s,
                    "centerline_station_spacing_m": spacing_m,
                    "reference_valid_count": sum(
                        row["reference_class"] == "valid_straight" for row in policy_rows
                    ),
                    "achieved_valid_count": sum(
                        row["achieved_class"] == "valid_straight" for row in policy_rows
                    ),
                    "achieved_class_counts": dict(
                        sorted(Counter(row["achieved_class"] for row in policy_rows).items())
                    ),
                    "median_achieved_total_absolute_curvature_rad": float(
                        statistics.median(
                            row["achieved_total_absolute_curvature_rad"] for row in policy_rows
                        )
                    ),
                    "rows": policy_rows,
                }
            )

    payload = {
        "schema_version": "motion2scene_e1_q3_route_retention_calibration_v1",
        "analysis_role": "descriptive_instrument_calibration_without_regrading",
        "independent_experimental_unit": "base_carrier_ladder_group",
        "registered_neutral_controls": len(neutral_cells),
        "measured_neutral_controls": len(measured),
        "not_measured_neutral_controls": len(neutral_cells) - len(measured),
        "promotion_decision": "not_authorized_from_calibration_data",
        "alignment": "initial_translation_only_no_rotation_scale_or_shape_warp",
        "sensitivity_grid": {
            "centerline_smoothing_seconds": list(SMOOTHING_SECONDS),
            "centerline_station_spacing_metres": list(STATION_SPACING_METRES),
            "selection_rule": "report_all_cells_no_post_hoc_parameter_selection",
        },
        "summary": {
            "frozen_reference_class_counts": dict(
                sorted(
                    Counter(
                        row["retention"]["reference_route"]["validity_class"]
                        for row in measured
                    ).items()
                )
            ),
            "frozen_achieved_class_counts": dict(
                sorted(
                    Counter(
                        row["retention"]["achieved_route"]["validity_class"]
                        for row in measured
                    ).items()
                )
            ),
            "median_path_length_ratio": _median(measured, "path_length_ratio"),
            "median_net_displacement_ratio": _median(measured, "net_displacement_ratio"),
            "median_endpoint_error_m": _median(measured, "endpoint_error_m"),
            "median_route_shape_rmse_m": _median(measured, "route_shape_rmse_m"),
            "median_cross_track_rmse_m": _median(measured, "cross_track_rmse_m"),
            "median_absolute_signed_heading_error_rad": float(
                statistics.median(
                    abs(row["retention"]["signed_heading_error_rad"]) for row in measured
                )
            ),
            "median_curvature_excess_rad": _median(
                measured, "total_absolute_curvature_excess_rad"
            ),
        },
        "candidates_sha256": sha256(args.candidates),
        "manifest_sha256": sha256(args.manifest),
        "run_record_sha256": sha256(args.run_record),
        "analysis_driver_sha256": sha256(Path(__file__)),
        "rows": rows,
        "sensitivity": sensitivity,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    summary = payload["summary"]
    lines = [
        "# E1 Q3 neutral-control route-retention calibration",
        "",
        (
            "This is descriptive instrument calibration on the eight frozen neutral controls. "
            "It does not alter the v2 route labels, admit Q4 candidates, or select new route "
            "parameters."
        ),
        "",
        (
            f"Full denominator: **{len(measured)}/{len(neutral_cells)} independent carrier "
            "groups measured**."
        ),
        "",
        "| carrier | ref class | achieved class | path ratio | endpoint (m) | shape RMSE (m) | cross RMSE (m) | heading error (rad) | curvature excess (rad) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in measured:
        retention = row["retention"]
        lines.append(
            f"| `{row['ladder_group_id']}` | "
            f"`{retention['reference_route']['validity_class']}` | "
            f"`{retention['achieved_route']['validity_class']}` | "
            f"{retention['path_length_ratio']:.3f} | "
            f"{retention['endpoint_error_m']:.3f} | "
            f"{retention['route_shape_rmse_m']:.3f} | "
            f"{retention['cross_track_rmse_m']:.3f} | "
            f"{retention['signed_heading_error_rad']:.3f} | "
            f"{retention['total_absolute_curvature_excess_rad']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Median reference-to-achieved metrics:",
            "",
            f"- path-length ratio: {summary['median_path_length_ratio']:.3f}",
            f"- net-displacement ratio: {summary['median_net_displacement_ratio']:.3f}",
            f"- endpoint error: {summary['median_endpoint_error_m']:.3f} m",
            f"- route-shape RMSE: {summary['median_route_shape_rmse_m']:.3f} m",
            f"- cross-track RMSE: {summary['median_cross_track_rmse_m']:.3f} m",
            (
                "- absolute signed-heading error: "
                f"{summary['median_absolute_signed_heading_error_rad']:.3f} rad"
            ),
            f"- curvature excess: {summary['median_curvature_excess_rad']:.3f} rad",
            "",
            "## Frozen sensitivity grid",
            "",
            "| smoothing (s) | station spacing (m) | reference valid | achieved valid | median achieved curvature (rad) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for item in sensitivity:
        lines.append(
            f"| {item['centerline_smoothing_s']:.2f} | "
            f"{item['centerline_station_spacing_m']:.2f} | "
            f"{item['reference_valid_count']}/{len(measured)} | "
            f"{item['achieved_valid_count']}/{len(measured)} | "
            f"{item['median_achieved_total_absolute_curvature_rad']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "No parameter combination is promoted from these calibration controls. A "
                "revised retention rule must be frozen and tested on new held-out neutral "
                "executions."
            ),
            "",
        ]
    )
    args.markdown_out.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"{len(measured)}/{len(neutral_cells)} neutral controls measured; "
        f"median endpoint={summary['median_endpoint_error_m']:.3f} m; "
        f"median cross-track RMSE={summary['median_cross_track_rmse_m']:.3f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
