"""Check every packaged scene's sampling and composed USD reference geometry."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def main(root):
    started = time.perf_counter()
    catalog = json.loads((root / "catalog.json").read_text())
    rows = json.loads((root / "assignments.json").read_text())
    assert len(catalog) == 120 and len(rows) == 240
    assert len({r["id"] for r in catalog}) == 120
    assert not (
        {r["group"] for r in catalog if r["split"] == "train"}
        & {r["group"] for r in catalog if r["split"] == "development"}
    )
    scene_records = [
        json.loads((root / "scenes" / (r["scene_id"] + ".json")).read_text()) for r in rows
    ]
    code = """const s=require(process.argv[1]),fs=require('fs'),assert=require('assert');
const a=JSON.parse(fs.readFileSync(0,'utf8'));let count=0;
for(const {record,candidates} of a){const x=record.sampling;
const y=s.sampleSet({weights:x.base,mask:candidates.map(c=>c.reference_clear),count:x.count,seed:x.seed,avoidOverlap:true,specs:candidates.map(c=>c.spec_xyzkindyaw)});
for(let k=0;k<x.count;k++){const d=x.draws[k],e=y.draws[k];assert.equal(d.cell,e.cell);assert.equal(d.u,e.u);
for(let j=0;j<225;j++)assert(Math.abs(d.q[j]-e.q[j])<1e-12);
if(d.cell!==null){assert(d.obstacle.reference_clear);assert.equal(d.obstacle.cell,d.cell);assert(Math.abs(d.conditionalQ-d.q[d.cell])<1e-12);}
count++;}
const obs=x.draws.filter(d=>d.obstacle);for(let i=0;i<obs.length;i++)for(let j=0;j<i;j++)assert(!s.overlaps(obs[i].obstacle.spec_xyzkindyaw,obs[j].obstacle.spec_xyzkindyaw));}
process.stdout.write(JSON.stringify({scene_sets:a.length,draws:count,status:'passed'}));"""
    records = [
        {
            "record": r,
            "candidates": json.loads(
                (root / "motions" / r["motion_id"] / "candidates.json").read_text()
            ),
        }
        for r in scene_records
    ]
    p = subprocess.run(
        ["node", "-e", code, str(root / "code/sampling.js")],
        input=json.dumps(records),
        text=True,
        capture_output=True,
        check=True,
    )
    sampling = json.loads(p.stdout)
    positions_checked, obstacles_checked, maximum_position_error = 0, 0, 0.0
    meshes = json.loads((root / "assets/visual-meshes.json").read_text())
    for m in catalog:
        with np.load(root / "motions" / m["id"] / "reference.npz", allow_pickle=False) as arrays:
            positions, quaternions = arrays["geom_position_m"], arrays["geom_quaternion_wxyz"]
            assert arrays["qpos"].shape == (m["frames"], 36)
            assert arrays["joint_names"].shape == (29,)
            assert np.isfinite(arrays["qpos"]).all()
        stage = Usd.Stage.Open(str(root / "motions" / m["id"] / "replay.usdc"))
        assert stage.GetTimeCodesPerSecond() == m["fps"]
        for t in sorted({0, m["frames"] // 2, m["frames"] - 1}):
            cache = UsdGeom.XformCache(t)
            for j, mesh in enumerate(meshes):
                prim = stage.GetPrimAtPath(f"/Robot/geom_{mesh['id']}")
                matrix = cache.GetLocalToWorldTransform(prim)
                error = float(
                    np.max(np.abs(np.array(matrix.ExtractTranslation()) - positions[t, j]))
                )
                maximum_position_error = max(maximum_position_error, error)
                assert error < 1e-9
                q = quaternions[t, j]
                expected = Gf.Rotation(Gf.Quatd(float(q[0]), Gf.Vec3d(*q[1:])))
                point = Gf.Vec3d(*mesh["v"][0])
                assert (
                    np.max(
                        np.abs(
                            np.array(matrix.Transform(point))
                            - np.array(expected.TransformDir(point))
                            - positions[t, j]
                        )
                    )
                    < 1e-8
                )
                positions_checked += 1
    for scene in scene_records:
        assert scene["teacher_eligible"] is False
        assert all(v is None for v in scene["physical_labels"].values())
        stage = Usd.Stage.Open(str(root / "scenes" / (scene["scene_id"] + ".usdc")))
        assert stage.GetPrimAtPath("/World/Camera")
        assert stage.GetPrimAtPath(f"/World/Robot/geom_{meshes[0]['id']}")
        cache = UsdGeom.XformCache(0)
        for draw in scene["sampling"]["draws"]:
            o = draw["obstacle"]
            if o is None:
                continue
            prim = stage.GetPrimAtPath(f"/World/Obstacles/obstacle_{draw['draw']}")
            assert prim.HasAPI(UsdPhysics.CollisionAPI)
            matrix = cache.GetLocalToWorldTransform(prim)
            assert np.max(np.abs(np.array(matrix.ExtractTranslation()) - o["position_m"])) < 1e-9
            q = o["quaternion_wxyz"]
            expected = Gf.Rotation(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
            if o["shape"] == "sphere":
                assert abs(UsdGeom.Sphere(prim).GetRadiusAttr().Get() - o["radius_m"]) < 1e-9
            else:
                assert UsdGeom.Cube(prim).GetSizeAttr().Get() == 1.0
                point = Gf.Vec3d(0.5, 0.5, 0.5)
                v = expected.TransformDir(Gf.Vec3d(*[d / 2 for d in o["dimensions_m"]]))
                assert (
                    np.max(
                        np.abs(np.array(matrix.Transform(point)) - np.array(v) - o["position_m"])
                    )
                    < 1e-7
                )
            obstacles_checked += 1
        assert not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse())
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "state": "passed",
        "sampling": sampling,
        "native_visual_pose_checks": positions_checked,
        "maximum_translation_error_m": maximum_position_error,
        "USD_obstacle_checks": obstacles_checked,
        "USD_scene_checks": len(scene_records),
        "zero_physical_labels_checked": len(scene_records),
        "physics_steps": 0,
        "measured_validation_wall_seconds": time.perf_counter() - started,
    }
    with (root / "validation.json").open("x") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    main(parser.parse_args().dataset.resolve())
