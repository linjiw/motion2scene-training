#!/usr/bin/env python3
"""Bind measured low-height onset and geometric encounter time before sensor audits."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance  # noqa: E402
from gear_sonic.dataset_generation.hallucination.motion2scene_observation_timing import (  # noqa: E402
    transition_timing_screen,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    CollisionCapsule,
    body_capsules_world,
)


def run(screen, qualification, out):
    screened = json.loads(screen.read_text())
    measured = json.loads(qualification.read_text())
    manifest = json.loads(
        checked(Path(measured["manifest"]["path"]), measured["manifest"]["sha256"]).read_text()
    )
    geometry_ref = manifest["geometry"]
    geometry = json.loads(checked(Path(geometry_ref["path"]), geometry_ref["sha256"]).read_text())
    shapes = {}
    for shape in geometry["shapes"]:
        if shape["role"] == "native_primitive_subset":
            shapes.setdefault(shape["owner"], []).append(
                CollisionCapsule(tuple(shape["start"]), tuple(shape["end"]), shape["radius"])
            )
    evidence = {}
    capsules = {}
    for row in measured["rows"]:
        ref = row["evidence"]
        e = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        evidence[row["cell_id"]] = e
        tr = e["artifacts"]["trajectory"]
        payload = load_reset_capture(checked(Path(tr["path"]), tr["sha256"]))
        capsules[row["cell_id"]] = body_capsules_world(
            payload["body_pos_w"], payload["body_quat_w"], payload["body_names"], capsules=shapes
        )
    out.mkdir(parents=True, exist_ok=False)
    snapshots = []
    for path in sorted(closure([Path(__file__)])):
        dest = out / "source_snapshot" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        snapshots.append({**artifact(path), "snapshot": artifact(dest)})
    write_new(
        out / "registration.json",
        dict(
            screen=artifact(screen),
            qualification=artifact(qualification),
            implementation=snapshots,
            stable_samples=5,
            clearance_margin_m=0.01,
            timing_margin_s=0.1,
            entry_tick=15,
            entry_capture_elapsed_s=0.28,
            scope=(
                "Geometry-based proposed deadlines from measured empty executions; "
                "actual sensor delivery remains required"
            ),
        ),
    )
    scenes = {}
    for ref in screened["scenes"]:
        scene = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        scenes[scene["provenance"]["grid_index"]] = scene
    rows = []
    for selected in screened["selected"]:
        scene = scenes[selected["grid_index"]]
        beam = scene["beam"]
        start, end, radii, _ = capsules[selected["negative"]]
        c, s = np.cos(beam["yaw_rad"]), np.sin(beam["yaw_rad"])
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        center = np.r_[beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2]
        half = np.array([beam[k] for k in ("length_m", "width_m", "thickness_m")]) / 2
        values = capsule_box_clearance(
            (start - center) @ rotation, (end - center) @ rotation, radii, -half, half
        ).min(axis=1)
        first = int(np.flatnonzero(values <= 0)[0])
        obstacle = first / 50
        e = evidence[selected["positive"]]
        # Synthetic visibility deliberately remains false; only extract measured
        # onset duration here. No proposal is declared observation eligible.
        timing = transition_timing_screen(
            e["executed_body_height_m"],
            np.arange(len(values)) / 50,
            0.28,
            beam["underside_m"],
            obstacle,
            {"surface": {"first_delivery_elapsed_s": None, "delivered_by_entry": False}},
        )
        duration = timing["transition_s"]
        rows.append(
            dict(
                scene_id=scene["scene_id"],
                positive=selected["positive"],
                negative=selected["negative"],
                geometric_first_interference_elapsed_s=obstacle,
                positive_timing=timing,
                latest_allowed_delivery_elapsed_s=(
                    None if duration is None else min(0.28, obstacle - duration - 0.1)
                ),
                observation_screen_status="pending actual native scene sensor capture",
                eligible=None,
            )
        )
    write_new(
        out / "result.json",
        dict(registration=artifact(out / "registration.json"), rows=rows, physics_steps=0),
    )
    print(json.dumps(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("screen", "qualification", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.screen, args.qualification, args.out)
