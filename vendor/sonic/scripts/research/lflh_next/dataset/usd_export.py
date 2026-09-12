"""Author portable USD reference animations and exact static obstacle primitives.

Run outside Kit with usd-core and numpy. This authors scene files, not Isaac images.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt


def stage_file(path, root_name, fps=30.0, frames=1):
    if path.exists():
        raise FileExistsError(path)
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/" + root_name)
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(frames - 1)
    stage.GetRootLayer().customLayerData = {
        "evidence": "Kinematic reference visualization only; zero executed physics"
    }
    return stage


def quat(q):
    return Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:])))


def main(out):
    start = time.perf_counter()
    meshes = json.loads((out / "assets/visual-meshes.json").read_text())
    model_path = out / "assets/g1_visuals.usdc"
    stage = stage_file(model_path, "Robot")
    for mesh in meshes:
        prim = UsdGeom.Mesh.Define(stage, f"/Robot/geom_{mesh['id']}")
        prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(mesh["v"], dtype=np.float32)))
        prim.CreateFaceVertexCountsAttr([3] * len(mesh["f"]))
        prim.CreateFaceVertexIndicesAttr(np.array(mesh["f"]).flatten().tolist())
        prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        prim.CreateDoubleSidedAttr(True)
        prim.CreateDisplayColorAttr([Gf.Vec3f(*mesh["color"])])
    stage.GetRootLayer().Save()
    catalog = json.loads((out / "catalog.json").read_text())
    transforms = 0
    for motion in catalog:
        id = motion["id"]
        reference = np.load(out / "motions" / id / "reference.npz", allow_pickle=False)
        positions = reference["geom_position_m"]
        quaternions = reference["geom_quaternion_wxyz"]
        stage = stage_file(
            out / "motions" / id / "replay.usdc", "Robot", motion["fps"], motion["frames"]
        )
        stage.GetDefaultPrim().GetReferences().AddReference(
            "../../assets/g1_visuals.usdc", "/Robot"
        )
        for j, mesh in enumerate(meshes):
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(f"/Robot/geom_{mesh['id']}"))
            translate = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            orient = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            for t in range(motion["frames"]):
                translate.Set(Gf.Vec3d(*positions[t, j]), t)
                orient.Set(quat(quaternions[t, j]), t)
                transforms += 1
        stage.GetRootLayer().Save()
    by_id = {m["id"]: m for m in catalog}
    assignments = json.loads((out / "assignments.json").read_text())
    obstacle_count = 0
    for row in assignments:
        scene = json.loads((out / "scenes" / (row["scene_id"] + ".json")).read_text())
        motion = by_id[row["motion_id"]]
        reference = np.load(
            out / "motions" / row["motion_id"] / "reference.npz", allow_pickle=False
        )
        stage = stage_file(
            out / "scenes" / (row["scene_id"] + ".usdc"), "World", motion["fps"], motion["frames"]
        )
        robot = UsdGeom.Xform.Define(stage, "/World/Robot")
        robot.GetPrim().GetReferences().AddReference(
            f"../motions/{row['motion_id']}/replay.usdc", "/Robot"
        )
        for draw in scene["sampling"]["draws"]:
            obstacle = draw["obstacle"]
            if obstacle is None:
                continue
            path = f"/World/Obstacles/obstacle_{draw['draw']}"
            if obstacle["shape"] == "sphere":
                prim = UsdGeom.Sphere.Define(stage, path)
                prim.CreateRadiusAttr(obstacle["radius_m"])
            else:
                prim = UsdGeom.Cube.Define(stage, path)
                prim.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(prim)
            xf.AddTranslateOp().Set(Gf.Vec3d(*obstacle["position_m"]))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat(obstacle["quaternion_wxyz"]))
            if obstacle["shape"] != "sphere":
                xf.AddScaleOp().Set(Gf.Vec3f(*obstacle["dimensions_m"]))
            prim.CreateDisplayColorAttr([Gf.Vec3f(0.82, 0.52, 0.20)])
            # Exact static collision primitives for future import. Robot is visual only.
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
            prim.GetPrim().SetCustomData(
                {
                    "candidate_id": int(draw["cell"]),
                    "draw_probability": float(draw["conditionalQ"]),
                    "reference_clearance_lower_m": float(obstacle["reference_lower_bound_m"]),
                    "physical_label_available": False,
                }
            )
            obstacle_count += 1
        trajectory = reference["qpos"][:, :3]
        line = UsdGeom.BasisCurves.Define(stage, "/World/ReferenceRoute")
        line.CreateTypeAttr(UsdGeom.Tokens.linear)
        line.CreateCurveVertexCountsAttr([len(trajectory)])
        line.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(trajectory.astype(np.float32)))
        line.CreateWidthsAttr([0.018])
        line.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        line.CreateDisplayColorAttr([Gf.Vec3f(0.14, 0.8, 0.68)])
        center = (trajectory.min(0) + trajectory.max(0)) / 2
        span = max(float(np.ptp(trajectory[:, :2], axis=0).max()), 3.0)
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(800.0)
        ground = UsdGeom.Cube.Define(stage, "/World/DisplayFloor")
        ground.CreateSizeAttr(1.0)
        ground.CreateDisplayColorAttr([Gf.Vec3f(0.10, 0.14, 0.18)])
        ground.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), -0.05))
        ground.AddScaleOp().Set(Gf.Vec3f(span + 6, span + 6, 0.05))
        # Display floor has no collision API and is not a qualified support surface.
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*(center + [span * 0.75, -span * 0.95, span * 0.6 + 1])),
            Gf.Vec3d(*center),
            Gf.Vec3d(0, 0, 1),
        )
        camera.AddTransformOp().Set(view.GetInverse())
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000))
        camera.CreateFocalLengthAttr(28.0)
        stage.GetRootLayer().Save()
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "state": "complete",
        "usd_version": list(Usd.GetVersion()),
        "motion_animations": len(catalog),
        "scene_files": len(assignments),
        "obstacle_primitives": obstacle_count,
        "authored_geom_frame_transforms": transforms,
        "physics_steps": 0,
        "Isaac_rendered_frames": 0,
        "measured_export_wall_seconds": time.perf_counter() - start,
        "robot": "Native visual meshes with sampled reference transforms, no articulation or dynamics",
        "scope": "USD file authoring; Isaac rendering blocked by GPU preflight",
    }
    with (out / "usd-export-receipt.json").open("x") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
