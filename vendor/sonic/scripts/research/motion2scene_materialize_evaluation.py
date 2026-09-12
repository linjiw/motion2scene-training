#!/usr/bin/env python3
"""Materialize reserved geometry without querying clearance, teachers or physics."""

import argparse
import copy
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
import joblib  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_course import (  # noqa: E402
    author_course,
    validate_beams,
)

LOCK_SHA256 = "sha256:7898862a0a02ab7d65a5e21f245c15a77b82a74c86245e88a611ad48ffd58dfc"


def place_beams(route, specifications):
    """Use locked arc-length station and lateral offset from the net route heading."""
    route = np.asarray(route, dtype=float)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2 or not np.isfinite(route).all():
        raise ValueError("finite XY reference route required")
    distance = np.r_[0.0, np.linalg.norm(np.diff(route, axis=0), axis=1).cumsum()]
    direction = route[-1] - route[0]
    if distance[-1] <= 0 or np.linalg.norm(direction) <= 1e-9:
        raise ValueError("nonzero route length and net heading required")
    progress = distance / distance[-1]
    # Repeated XY samples denote standing frames; retain the last identical station.
    keep = np.r_[np.diff(progress) > 0, True]
    progress, route = progress[keep], route[keep]
    yaw = float(np.arctan2(direction[1], direction[0]))
    lateral_axis = np.array([-np.sin(yaw), np.cos(yaw)])
    beams = []
    for spec in specifications:
        station = spec["route_progress_fraction"]
        if not np.isfinite(list(spec.values())).all() or not 0 < station < 1:
            raise ValueError("finite locked beam parameters and interior station required")
        center = np.array([np.interp(station, progress, route[:, i]) for i in range(2)])
        center += spec["lateral_offset_m"] * lateral_axis
        beams.append(
            {
                **spec,
                "route_progress": station,
                "center_xy_m": center.tolist(),
                "yaw_rad": yaw + spec["yaw_offset_rad"],
            }
        )
    validate_beams(beams)
    return beams


def reserved_scene_text(template, beams, layout_id):
    if re.fullmatch(r"locked_v2_(?:single|course)_\d{2}", layout_id) is None:
        raise ValueError("requires an exact reserved layout identifier")
    text = author_course(template, beams)
    for key, value in (
        ("sceneId", layout_id),
        ("splitGroup", "motion2scene_reserved_evaluation_v2"),
    ):
        text, count = re.subn(
            rf'(custom string g1Dataset:{key} = )"[^\"]*"',
            lambda match: match[1] + f'"{value}"',
            text,
        )
        if count != 1:
            raise ValueError("template must declare one scene and split identifier")
    return text.replace(
        "# Fresh development course; no feasibility implied.",
        "# Reserved evaluation geometry; no clearance, feasibility or physical outcome queried.",
    )


def materialize(lock_path, template_path, scene_template_path, out):
    lock = json.loads(checked(lock_path, LOCK_SHA256).read_text())
    template = json.loads(template_path.read_text())
    scene_template_ref = artifact(scene_template_path)
    shared_room_text = scene_template_path.read_text()
    layouts = lock["evaluation"]["single_layouts"] + lock["evaluation"]["two_beam_compositions"]
    if len(layouts) != 36 or len({layout["layout_id"] for layout in layouts}) != 36:
        raise ValueError("locked geometry requires all 36 distinct layouts")
    carriers = {}
    for source in lock["splits"]["development_sources"]:
        candidates = [c for c in template["cells"] if c["generation_seed"] == source]
        if not candidates:
            raise ValueError(f"template lacks source {source}")
        base = candidates[0]
        neutral = base["motion"]
        library = joblib.load(checked(Path(neutral["path"]), neutral["sha256"]))
        if len(library) != 1:
            raise ValueError("one registered neutral reference required per carrier")
        motion = next(iter(library.values()))
        route = np.asarray(motion["root_trans_offset"])[:, :2]
        carriers[source] = (base, route)
    out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__)])):
        snapshot = out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(snapshot)})
    write_new(
        out / "registration.json",
        {
            "schema": "motion2scene_reserved_geometry_materialization_v1",
            "lock": artifact(lock_path),
            "template": artifact(template_path),
            "scene_template": scene_template_ref,
            "carriers": {source: base["motion"] for source, (base, _) in carriers.items()},
            "implementation": sources,
            "coordinate_convention": "arc-length station; net neutral XY heading; positive lateral is left",
            "outcomes_queried": False,
            "clearance_queries": 0,
            "physics_steps": 0,
            "execution_authorized_by_this_registration": False,
            "scope": "complete geometry assets only; separate qualified runtime/protocol lock remains required",
        },
    )
    cells, definitions = [], []
    for layout in layouts:
        base, route = carriers[layout["source_id"]]
        beams = place_beams(route, layout["beams"])
        path = out / "scenes" / f"{layout['layout_id']}.usda"
        path.parent.mkdir(exist_ok=True)
        path.write_text(reserved_scene_text(shared_room_text, beams, layout["layout_id"]))
        definition = {
            "layout": layout,
            "beams": beams,
            "split": "reserved_evaluation",
            "neutral_motion": base["motion"],
            "scene": {**artifact(path), "scene_id": layout["layout_id"]},
            "physics_status": "not_run",
        }
        definitions.append(definition)
        cell = copy.deepcopy(base)
        cell.update(
            cell_id=layout["layout_id"],
            condition="present" if len(beams) == 1 else "reserved_course",
            scene=definition["scene"],
            beam=beams[0],
            evaluation_layout_id=layout["layout_id"],
            split="reserved_evaluation",
        )
        cell.pop("output", None)
        if len(beams) == 1:
            cells.append(cell)
    write_new(out / "layouts.json", definitions)
    write_new(
        out / "single_templates.json",
        {
            "implementation": template["implementation"],
            "cells": cells,
            "role": "reserved_geometry_only",
        },
    )
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "layouts": artifact(out / "layouts.json"),
            "single_templates": artifact(out / "single_templates.json"),
            "scene_files": [definition["scene"] for definition in definitions],
            "single_layouts": 24,
            "two_beam_layouts": 12,
            "clearance_queries": 0,
            "physics_steps": 0,
            "scope": "geometry materialized; native composition audit and runtime qualification separate",
        },
    )
    print(json.dumps({"layouts": len(definitions), "out": str(out), "physics_steps": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--scene-template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.lock, args.template, args.scene_template, args.out)
