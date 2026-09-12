#!/usr/bin/env python3
"""Screen all existing development layouts using qualified executed alternatives.

Cached native primitive subsets witness interference; enclosing geometry screens
positive clearance. Neither predicts a physical passage label or robustness.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance  # noqa: E402
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    CollisionCapsule,
    body_capsules_world,
)


def run(options, layouts, geometry, out):
    result = json.loads(options.read_text())
    manifest = json.loads(
        checked(Path(result["manifest"]["path"]), result["manifest"]["sha256"]).read_text()
    )
    layout_rows = json.loads(layouts.read_text())["independent_common_bank_layouts"]
    shape_rows = json.loads(geometry.read_text())
    for layer in shape_rows["layers"]:
        checked(Path(layer["path"]), layer["sha256"])
    shape_sets = {"inner": {}, "outer": {}}
    for shape in shape_rows["shapes"]:
        capsule = CollisionCapsule(tuple(shape["start"]), tuple(shape["end"]), shape["radius"])
        shape_sets["outer"].setdefault(shape["owner"], []).append(capsule)
        if shape["role"] == "native_primitive_subset":
            shape_sets["inner"].setdefault(shape["owner"], []).append(capsule)
    rows = []
    for row in result["rows"]:
        if not row["qualified"]:
            continue
        payload = load_reset_capture(
            checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
        )
        cell = next(c for c in manifest["cells"] if c["cell_id"] == row["cell_id"])
        motion = cell["motion"]
        entry = next(iter(joblib.load(checked(Path(motion["path"]), motion["sha256"])).values()))
        route = np.asarray(entry["root_trans_offset"])[:, :2]
        progress = np.r_[0.0, np.linalg.norm(np.diff(route, axis=0), axis=1).cumsum()]
        progress /= progress[-1]
        direction = route[-1] - route[0]
        yaw = float(np.arctan2(direction[1], direction[0]))
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        capsules = {
            kind: body_capsules_world(
                payload["body_pos_w"],
                payload["body_quat_w"],
                payload["body_names"],
                capsules=shapes,
            )
            for kind, shapes in shape_sets.items()
        }
        for layout in layout_rows:
            center = np.array(
                [
                    *[np.interp(layout["station"], progress, route[:, i]) for i in range(2)],
                    layout["underside_m"] + 0.05,
                ]
            )
            values = {}
            for kind, (starts, ends, radii, _) in capsules.items():
                values[kind] = float(
                    capsule_box_clearance(
                        (starts - center) @ rotation,
                        (ends - center) @ rotation,
                        radii,
                        [-0.05, -0.6, -0.05],
                        [0.05, 0.6, 0.05],
                    ).min()
                )
            rows.append(
                {
                    "source": row["source"],
                    "option_id": row["option"]["option_id"],
                    "layout_id": layout["id"],
                    "station": layout["station"],
                    "underside_m": layout["underside_m"],
                    "minimum_clearance_m": values,
                    "positive_screen": values["outer"] >= 0.01,
                    "interference_screen": values["inner"] <= -0.01,
                }
            )
    write_new(
        out,
        {
            "schema": "motion2scene_option_development_forecast_v1",
            "options": artifact(options),
            "layouts": artifact(layouts),
            "geometry": artifact(geometry),
            "implementation": artifact(Path(__file__)),
            "rows": rows,
            "physical_labels_created": 0,
            "whole_trajectory_clearance_queries": len(rows) * 2,
            "scope": "previously inspected development layouts; finite recorded-pose geometry screens only",
        },
    )
    for layout in layout_rows:
        chosen = [r for r in rows if r["layout_id"] == layout["id"]]
        print(
            layout["id"],
            {r["option_id"]: round(r["minimum_clearance_m"]["outer"], 4) for r in chosen},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--layouts", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.options, args.layouts, args.geometry, args.out)
