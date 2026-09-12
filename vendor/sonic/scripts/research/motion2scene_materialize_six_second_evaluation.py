#!/usr/bin/env python3
"""Materialize/audit every frozen V3 world layout without robot or outcome queries."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path("/home/linjiw/groot-wbc-sonic-sim-trackb")
LOCK_SHA = "509817600075888ef5cf681c1033c9dcd6796d7a08561e30ee4157049aff5181"
V2_SHA = "7898862a0a02ab7d65a5e21f245c15a77b82a74c86245e88a611ad48ffd58dfc"


def artifact(path):
    path = Path(path)
    return {
        "path": str(path),
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def checked(ref):
    actual = artifact(ref["path"])
    if actual["sha256"].removeprefix("sha256:") != ref["sha256"].removeprefix("sha256:"):
        raise ValueError("Registered source changed: " + ref["path"])
    return Path(ref["path"])


def write(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def locked_world_beams(beams):
    """Validate redundant authority fields, preserving exact world coordinates."""
    for beam in beams:
        center = np.asarray(beam["center_xyz_m"], dtype=float)
        dims = np.asarray(beam["full_dimensions_xyz_m"], dtype=float)
        if (
            center.shape != (3,)
            or dims.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite(dims).all()
        ):
            raise ValueError("Finite world XYZ and full dimensions required")
        if np.any(dims <= 0) or not np.isfinite(beam["yaw_rad"]):
            raise ValueError("Positive dimensions and finite world yaw required")
        redundant_center = np.r_[beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2]
        redundant_dims = np.array([beam["length_m"], beam["width_m"], beam["thickness_m"]])
        if not np.array_equal(center, redundant_center) or not np.array_equal(dims, redundant_dims):
            raise ValueError("Redundant locked world geometry disagrees; never relocate or repair")
        if (
            beam["collision_enabled"] is not True
            or beam["placement_authority"]
            != "these fixed world coordinates; never relocate from another reference"
        ):
            raise ValueError("Reserved world placement/collision authority changed")
    if len(beams) not in (1, 2):
        raise ValueError("V3 requires one or two beams per variant")
    return beams


def load_lock(path):
    checked({"path": str(path), "sha256": LOCK_SHA})
    lock = json.loads(path.read_text())
    layouts = lock["evaluation"]["layouts"]
    if len(layouts) != 18 or len({v["layout_id"] for v in layouts}) != 18:
        raise ValueError("All 18 distinct locked layouts required")
    if sum(len(v["nominal_beams"]) for v in layouts) != 24:
        raise ValueError("Expected 24 nominal beams")
    for layout in layouts:
        if (
            not re.fullmatch(r"locked_v3_(single|course)_\d{2}", layout["layout_id"])
            or layout["split"] != "reserved_evaluation_v3"
        ):
            raise ValueError("Expected original reserved identifiers")
        variants = layout["fixed_world_variants"]
        if len(variants) != 9 or len({v["offset_id"] for v in variants}) != 9:
            raise ValueError("Every nominal and eight stress variants required")
        if variants[0]["offset_id"] != "nominal" or variants[0]["beams"] != layout["nominal_beams"]:
            raise ValueError("Nominal authority changed")
        for variant in variants:
            if not re.fullmatch("[a-z_]+", variant["offset_id"]):
                raise ValueError("Unsafe variant identifier")
            locked_world_beams(variant["beams"])
    return lock


def reserved_text(template, beams, identifier):
    from gear_sonic.dataset_generation.hallucination.motion2scene_course import author_course

    text = author_course(template, locked_world_beams(beams))
    for key, value in [
        ("sceneId", identifier),
        ("splitGroup", "motion2scene_reserved_evaluation_v3"),
    ]:
        text, count = re.subn(
            rf'(custom string g1Dataset:{key} = )"[^\"]*"',
            lambda match: match[1] + f'"{value}"',
            text,
        )
        if count != 1:
            raise ValueError("Template requires one scene and split identifier")
    return text.replace(
        "# Fresh development course; no feasibility implied.",
        "# Reserved V3 fixed world geometry; no robot clearance or outcomes queried.",
    )


def register(lock_path, template_path, out):
    from bundle_motion2scene_sources import closure

    lock = load_lock(lock_path)
    v2 = ROOT / "docs/motion2scene/TRAVERSAL_EVALUATION_LOCK_V2.json"
    checked({"path": str(v2), "sha256": V2_SHA})
    checked(lock["carrier"]["motion"])
    out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__).resolve()])):
        target = out / "source_snapshot/repository" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        sources.append({"source": artifact(path), "snapshot": artifact(target)})
    driver = out / "source_snapshot/repository/scripts/research" / Path(__file__).name
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONPATH": str(out / "source_snapshot/repository"),
    }
    registration = {
        "schema": "motion2scene_fixed_world_reserved_materialization_v3",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "lock": artifact(lock_path),
        "untouched_v2_lock": artifact(v2),
        "scene_template": artifact(template_path),
        "carrier_provenance_only": lock["carrier"]["motion"],
        "implementation": sources,
        "driver": artifact(driver),
        "environment": environment,
        "command": [
            str(ROOT / ".venv_isaaclab/bin/python"),
            str(driver),
            "materialize",
            "--out",
            str(out),
        ],
        "operation": (
            "Author every fixed world XYZ/yaw/full-dimension variant; "
            "no reference loading or route placement"
        ),
        "counts": {"base_layouts": 18, "variants_per_layout": 9, "scene_files": 162, "beams": 216},
        "audit": (
            "Native composed USD matrices, cube sizes, collision/kinematic flags, "
            "scene/split and unchanged room"
        ),
        "maximum_geometric_error_tolerance": 1e-8,
        "outcomes_queried": False,
        "robot_clearance_queries": 0,
        "physics_steps": 0,
        "execution_authorized": False,
        "implementation_amendment_adopted": False,
        "terminal_scoring_policy_adopted": False,
    }
    write(out / "registration.json", registration)
    (out / "registration.sha256").write_text(artifact(out / "registration.json")["sha256"] + "\n")
    print(
        json.dumps(
            {
                "registration": artifact(out / "registration.json"),
                "command": registration["command"],
            }
        )
    )


def materialize(out):
    from motion2scene_audit_reserved_geometry import is_beam, room_snapshot
    from pxr import Usd, UsdGeom, UsdPhysics

    registration = json.loads((out / "registration.json").read_text())
    if (out / "materialization_started.json").exists():
        raise ValueError("Preserve the original single materialization attempt")
    for key, value in registration["environment"].items():
        if os.environ.get(key) != value:
            raise ValueError("Registered environment differs: " + key)
    for source in registration["implementation"]:
        checked(source["snapshot"])
    checked(registration["driver"])
    checked(registration["untouched_v2_lock"])
    lock = load_lock(checked(registration["lock"]))
    template_path = checked(registration["scene_template"])
    template_text = template_path.read_text()
    write(
        out / "materialization_started.json",
        {
            "registration": artifact(out / "registration.json"),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    baseline = room_snapshot(Usd.Stage.Open(str(template_path)))
    room_hash = (
        "sha256:" + hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
    )
    definitions = []
    (out / "scenes").mkdir()
    for layout in lock["evaluation"]["layouts"]:
        for variant in layout["fixed_world_variants"]:
            identifier = layout["layout_id"] + "__" + variant["offset_id"]
            path = out / "scenes" / (identifier + ".usda")
            with path.open("x") as handle:
                handle.write(reserved_text(template_text, variant["beams"], identifier))
            definitions.append(
                {
                    "layout_id": layout["layout_id"],
                    "source_id": layout["source_id"],
                    "family": layout["family"],
                    "variant": variant,
                    "scene_id": identifier,
                    "split": layout["split"],
                    "scene": artifact(path),
                    "physics_status": "not_run",
                }
            )
    write(out / "layouts.json", definitions)
    # Independent native USD extraction checks against the original lock, not a route or generated scene table.
    rows, maxima = [], {}
    for layout in lock["evaluation"]["layouts"]:
        for variant in layout["fixed_world_variants"]:
            identifier = layout["layout_id"] + "__" + variant["offset_id"]
            path = out / "scenes" / (identifier + ".usda")
            stage = Usd.Stage.Open(str(path))
            if room_snapshot(stage) != baseline:
                raise ValueError("Shared room changed: " + identifier)
            world = stage.GetPrimAtPath("/World")
            if (
                world.GetAttribute("g1Dataset:sceneId").Get() != identifier
                or world.GetAttribute("g1Dataset:splitGroup").Get()
                != "motion2scene_reserved_evaluation_v3"
            ):
                raise ValueError("Reserved scene metadata changed")
            prims = [p for p in stage.Traverse() if is_beam(p)]
            expected_names = [
                "CounterfactualBeam" if i == 0 else f"CounterfactualBeam_{i:02d}"
                for i in range(len(variant["beams"]))
            ]
            if [p.GetName() for p in prims] != expected_names:
                raise ValueError("Beam count/order changed")
            beam_rows = []
            for prim, spec in zip(prims, variant["beams"], strict=True):
                cube, collision, rigid = (
                    UsdGeom.Cube(prim),
                    UsdPhysics.CollisionAPI(prim),
                    UsdPhysics.RigidBodyAPI(prim),
                )
                if (
                    not cube
                    or not collision
                    or not rigid
                    or not collision.GetCollisionEnabledAttr().Get()
                    or not rigid.GetRigidBodyEnabledAttr().Get()
                    or not rigid.GetKinematicEnabledAttr().Get()
                ):
                    raise ValueError("Invalid collision/kinematic cube flags")
                matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
                rownorm = np.linalg.norm(matrix[:3, :3], axis=1)
                dimensions = rownorm * cube.GetSizeAttr().Get()
                rotation = matrix[:3, :3] / rownorm[:, None]
                yaw = spec["yaw_rad"]
                expected_rotation = np.array(
                    [
                        [math.cos(yaw), math.sin(yaw), 0],
                        [-math.sin(yaw), math.cos(yaw), 0],
                        [0, 0, 1],
                    ]
                )
                actual_yaw = math.atan2(rotation[0, 1], rotation[0, 0])
                errors = {
                    "center_m": float(abs(matrix[3, :3] - np.asarray(spec["center_xyz_m"])).max()),
                    "full_dimensions_m": float(
                        abs(dimensions - np.asarray(spec["full_dimensions_xyz_m"])).max()
                    ),
                    "rotation_matrix": float(abs(rotation - expected_rotation).max()),
                    "yaw_rad": abs(
                        math.atan2(math.sin(actual_yaw - yaw), math.cos(actual_yaw - yaw))
                    ),
                    "underside_m": abs(
                        float(matrix[3, 2] - dimensions[2] / 2) - spec["underside_m"]
                    ),
                }
                if max(errors.values()) > registration["maximum_geometric_error_tolerance"]:
                    raise ValueError(
                        "Composed geometry differs from fixed world lock: " + identifier
                    )
                for key, value in errors.items():
                    maxima[key] = max(maxima.get(key, 0), value)
                beam_rows.append(
                    {
                        "path": str(prim.GetPath()),
                        "composed_center_xyz_m": matrix[3, :3].tolist(),
                        "composed_full_dimensions_xyz_m": dimensions.tolist(),
                        "composed_yaw_rad": actual_yaw,
                        "collision_enabled": True,
                        "kinematic_enabled": True,
                        "errors": errors,
                    }
                )
            rows.append(
                {
                    "scene": artifact(path),
                    "scene_id": identifier,
                    "layout_id": layout["layout_id"],
                    "offset_id": variant["offset_id"],
                    "beams": beam_rows,
                    "room_fingerprint": room_hash,
                }
            )
    if len(rows) != 162 or sum(len(row["beams"]) for row in rows) != 216:
        raise ValueError("Incomplete native geometry audit")
    checked(registration["lock"])
    checked(registration["untouched_v2_lock"])
    checked(registration["scene_template"])
    checked(registration["carrier_provenance_only"])
    audit = {
        "schema": "motion2scene_fixed_world_native_geometry_audit_v3",
        "status": "PASS",
        "registration": artifact(out / "registration.json"),
        "lock": registration["lock"],
        "driver": registration["driver"],
        "usd_version": list(Usd.GetVersion()),
        "python": sys.executable,
        "base_layouts": 18,
        "scene_files": 162,
        "single_variants": 108,
        "two_beam_variants": 54,
        "beams": 216,
        "maximum_errors": maxima,
        "room_prims_unchanged": len(baseline["prims"]),
        "room_snapshot": baseline,
        "room_fingerprint": room_hash,
        "rows": rows,
        "source_assets_unchanged": True,
        "physics_steps": 0,
        "robot_clearance_queries": 0,
        "outcomes_queried": False,
        "implementation_amendment_adopted": False,
        "scope": (
            "Native USD composition only; no simulation, clearance, policy, outcome, "
            "PhysX cooking or protocol adoption"
        ),
    }
    write(out / "native_geometry_audit.json", audit)
    result = {
        "registration": artifact(out / "registration.json"),
        "layouts": artifact(out / "layouts.json"),
        "native_geometry_audit": artifact(out / "native_geometry_audit.json"),
        "scene_files": [row["scene"] for row in rows],
        "base_layouts": 18,
        "scene_variants": 162,
        "beams": 216,
        "physics_steps": 0,
        "robot_clearance_queries": 0,
        "outcomes_queried": False,
        "scope": "All frozen V3 world geometry materialized; common implementation and scoring lock still pending",
    }
    write(out / "result.json", result)
    print(
        json.dumps(
            {
                "result": artifact(out / "result.json"),
                "audit": result["native_geometry_audit"],
                "maximum_errors": maxima,
                "scene_files": 162,
                "beams": 216,
                "room_prims_unchanged": len(baseline["prims"]),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["register", "materialize"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--scene-template", type=Path)
    args = parser.parse_args()
    if args.mode == "register":
        if args.lock is None or args.scene_template is None:
            parser.error("registration requires --lock and --scene-template")
        register(args.lock.resolve(), args.scene_template.resolve(), args.out.resolve())
    else:
        materialize(args.out.resolve())


if __name__ == "__main__":
    main()
