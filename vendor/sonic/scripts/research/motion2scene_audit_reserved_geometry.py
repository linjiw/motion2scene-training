#!/usr/bin/env python3
"""CPU-only composed USD audit of reserved layouts; no clearance or physics query."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

LOCK_HASH = "7898862a0a02ab7d65a5e21f245c15a77b82a74c86245e88a611ad48ffd58dfc"


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def checked(ref):
    path = Path(ref["path"])
    if digest(path) != ref["sha256"]:
        raise ValueError(f"source hash mismatch: {path}")
    return path


def artifact(path):
    return {"path": str(path), "sha256": digest(path)}


def canonical(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    try:
        return [canonical(v) for v in value]
    except TypeError:
        return str(value)


def is_beam(prim):
    return prim.GetName() == "CounterfactualBeam" or prim.GetName().startswith(
        "CounterfactualBeam_"
    )


def room_snapshot(stage):
    rows = []
    for prim in stage.Traverse():
        if is_beam(prim):
            continue
        attrs = {}
        for attr in prim.GetAttributes():
            if str(prim.GetPath()) == "/World" and attr.GetName() in (
                "g1Dataset:sceneId",
                "g1Dataset:splitGroup",
            ):
                continue
            attrs[attr.GetName()] = {
                "type": str(attr.GetTypeName()),
                "default": canonical(attr.Get()),
                "metadata": canonical(attr.GetAllMetadata()),
                "samples": [[t, canonical(attr.Get(t))] for t in attr.GetTimeSamples()],
                "connections": [str(p) for p in attr.GetConnections()],
            }
        rows.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "metadata": canonical(prim.GetAllMetadata()),
                "attributes": attrs,
                "relationships": {
                    r.GetName(): [str(p) for p in r.GetTargets()] for r in prim.GetRelationships()
                },
                "world_matrix": np.asarray(
                    UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                ).tolist(),
            }
        )
    return {
        "prims": rows,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "metres_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "time_codes_per_second": stage.GetTimeCodesPerSecond(),
        "default_prim": str(stage.GetDefaultPrim().GetPath()),
    }


def independent_placement(route, spec):
    """Interpolate the original neutral polyline, without materializer helpers."""
    segment_lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
    total = float(segment_lengths.sum())
    if total <= 0:
        raise ValueError("zero neutral route length")
    requested = total * spec["route_progress_fraction"]
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    segment = min(int(np.searchsorted(cumulative, requested, side="right")) - 1, len(route) - 2)
    while segment_lengths[segment] == 0:
        segment += 1
    fraction = (requested - cumulative[segment]) / segment_lengths[segment]
    point = route[segment] + fraction * (route[segment + 1] - route[segment])
    heading = math.atan2(*(route[-1] - route[0])[::-1])
    point += spec["lateral_offset_m"] * np.array([-math.sin(heading), math.cos(heading)])
    return (
        np.r_[point, spec["underside_m"] + spec["thickness_m"] / 2],
        heading + spec["yaw_offset_rad"],
    )


def run(study):
    out = study / "native_geometry_audit.json"
    snapshot = study / "native_geometry_audit_driver.py"
    if out.exists() or snapshot.exists():
        raise FileExistsError(
            "preserve existing geometry audit; use a separate study copy for reruns"
        )
    result = json.loads((study / "result.json").read_text())
    registration = json.loads(checked(result["registration"]).read_text())
    definitions = json.loads(checked(result["layouts"]).read_text())
    lock_path = checked(registration["lock"])
    if digest(lock_path) != "sha256:" + LOCK_HASH:
        raise ValueError("unexpected frozen layout lock")
    lock = json.loads(lock_path.read_text())
    layouts = lock["evaluation"]["single_layouts"] + lock["evaluation"]["two_beam_compositions"]
    if [d["layout"] for d in definitions] != layouts or len(layouts) != 36:
        raise ValueError("materialized definitions do not preserve every original ordered layout")
    template_path = checked(registration["scene_template"])
    template_stage = Usd.Stage.Open(str(template_path))
    baseline = room_snapshot(template_stage)
    baseline_hash = (
        "sha256:" + hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
    )
    references, routes = {}, {}
    for source, ref in registration["carriers"].items():
        path = checked(ref)
        payload = joblib.load(path)
        if len(payload) != 1:
            raise ValueError("one original neutral motion required")
        routes[int(source)] = np.asarray(
            next(iter(payload.values()))["root_trans_offset"], dtype=float
        )[:, :2]
        references[source] = artifact(path)
    rows, maxima, source_hashes = (
        [],
        {},
        {str(template_path): digest(template_path), str(lock_path): digest(lock_path)},
    )
    for definition in definitions:
        layout, beams = definition["layout"], definition["beams"]
        source = layout["source_id"]
        if definition["neutral_motion"] != registration["carriers"][str(source)]:
            raise ValueError("layout carrier binding changed")
        if (
            definition["split"] != "reserved_evaluation"
            or definition["physics_status"] != "not_run"
        ):
            raise ValueError("reserved layout status changed")
        path = checked(definition["scene"])
        source_hashes[str(path)] = digest(path)
        stage = Usd.Stage.Open(str(path))
        if room_snapshot(stage) != baseline:
            raise ValueError(f"composed room changed: {layout['layout_id']}")
        world = stage.GetPrimAtPath("/World")
        metadata = {
            key: world.GetAttribute("g1Dataset:" + key).Get() for key in ("sceneId", "splitGroup")
        }
        if metadata != {
            "sceneId": layout["layout_id"],
            "splitGroup": "motion2scene_reserved_evaluation_v2",
        }:
            raise ValueError("composed reserved scene/split metadata changed")
        beam_prims = [p for p in stage.Traverse() if is_beam(p)]
        expected_names = [
            "CounterfactualBeam" if i == 0 else f"CounterfactualBeam_{i:02d}"
            for i in range(len(beams))
        ]
        if [p.GetName() for p in beam_prims] != expected_names or len(beams) != len(
            layout["beams"]
        ):
            raise ValueError("wrong composed beam count or order")
        beam_rows = []
        for prim, spec, authored in zip(beam_prims, layout["beams"], beams, strict=True):
            if any(authored[key] != value for key, value in spec.items()):
                raise ValueError("original locked parameter changed during materialization")
            center, yaw = independent_placement(routes[source], spec)
            dims = np.array([spec["length_m"], spec["width_m"], spec["thickness_m"]])
            cube = UsdGeom.Cube(prim)
            collision = UsdPhysics.CollisionAPI(prim)
            rigid = UsdPhysics.RigidBodyAPI(prim)
            if (
                not cube
                or not collision
                or not rigid
                or not collision.GetCollisionEnabledAttr().Get()
                or not rigid.GetRigidBodyEnabledAttr().Get()
                or not rigid.GetKinematicEnabledAttr().Get()
            ):
                raise ValueError("missing composed collision/kinematic beam settings")
            matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
            actual_dims = np.linalg.norm(matrix[:3, :3], axis=1) * cube.GetSizeAttr().Get()
            normalized = matrix[:3, :3] / np.linalg.norm(matrix[:3, :3], axis=1)[:, None]
            expected_rotation = np.array(
                [[math.cos(yaw), math.sin(yaw), 0], [-math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
            )
            actual_yaw = math.atan2(normalized[0, 1], normalized[0, 0])
            errors = {
                "center_m": float(np.abs(matrix[3, :3] - center).max()),
                "dimensions_m": float(np.abs(actual_dims - dims).max()),
                "rotation_matrix": float(np.abs(normalized - expected_rotation).max()),
                "yaw_rad": abs(math.atan2(math.sin(actual_yaw - yaw), math.cos(actual_yaw - yaw))),
                "underside_m": abs(float(matrix[3, 2] - actual_dims[2] / 2) - spec["underside_m"]),
                "materialized_center_vs_independent_route_m": float(
                    np.abs(np.array(authored["center_xy_m"]) - center[:2]).max()
                ),
            }
            if max(errors.values()) > 1e-8:
                raise ValueError(
                    f"composed beam differs from frozen geometry: {layout['layout_id']} {errors}"
                )
            for key, value in errors.items():
                maxima[key] = max(maxima.get(key, 0), value)
            beam_rows.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "world_center_m": matrix[3, :3].tolist(),
                    "world_dimensions_m": actual_dims.tolist(),
                    "yaw_rad": actual_yaw,
                    "collision_enabled": True,
                    "rigid_body_enabled": True,
                    "kinematic_enabled": True,
                    "errors": errors,
                }
            )
        rows.append(
            {
                "layout_id": layout["layout_id"],
                "source_id": source,
                "scene": artifact(path),
                "scene_metadata": metadata,
                "room_fingerprint": baseline_hash,
                "beams": beam_rows,
                "used_layers": [
                    artifact(Path(layer.realPath))
                    for layer in stage.GetUsedLayers()
                    if layer.realPath and Path(layer.realPath).is_file()
                ],
            }
        )
    for path, expected in source_hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError("source asset changed during read-only native audit")
    snapshot.write_bytes(Path(__file__).read_bytes())
    report = {
        "schema": "motion2scene_reserved_native_geometry_audit_v1",
        "status": "PASS",
        "registration": result["registration"],
        "materialized_layouts": result["layouts"],
        "original_lock": registration["lock"],
        "template_scene": registration["scene_template"],
        "neutral_motion_references": references,
        "driver": artifact(snapshot),
        "python_executable": sys.executable,
        "usd_version": list(Usd.GetVersion()),
        "layouts": 36,
        "single_layouts": 24,
        "two_beam_layouts": 12,
        "beams": sum(len(r["beams"]) for r in rows),
        "maximum_errors": maxima,
        "room_prims_unchanged": len(baseline["prims"]),
        "room_snapshot": baseline,
        "room_fingerprint": baseline_hash,
        "rows": rows,
        "physics_steps": 0,
        "robot_clearance_queries": 0,
        "outcomes_queried": False,
        "source_assets_unchanged": True,
        "scope": (
            "CPU composed USD geometry only; no PhysX cooking, contacts, feasibility, "
            "policy execution or implementation amendment adoption"
        ),
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {
        "audit": artifact(out),
        "status": report["status"],
        "layouts": 36,
        "beams": report["beams"],
        "room_prims_unchanged": report["room_prims_unchanged"],
        "maximum_errors": maxima,
        "physics_steps": 0,
        "robot_clearance_queries": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.study), indent=2))
