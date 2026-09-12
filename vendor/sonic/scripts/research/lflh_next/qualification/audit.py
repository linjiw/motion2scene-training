"""Read-only motion mapping and temporal-proxy audit. Kinematics, never dynamics."""

import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from scripts.research.lflh_next.multimotion.run import candidates, gaps

PREVIOUS = Path("/home/linjiw/research-data/m2s-multimotion-shapes-20260911")
OUT = Path("/home/linjiw/research-data/m2s-motion-qualification-20260911")
MODELS = {
    "conversion_candidate": Path(
        "/home/linjiw/climb-feasibility-first/mjlab-1.6.0/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
    ),
    "kimodo_primitives": Path("/home/linjiw/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"),
    "sonic_deploy": Path(
        "/home/linjiw/groot-wbc-sonic-sim-trackb/gear_sonic_deploy/g1/g1_29dof.xml"
    ),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def orientation_error(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return 2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1))


def surface_witness(data, model, centers):
    """Finite primitive surface samples; positive gap is a noncontainment witness."""
    points = []
    for i in range(model.ngeom):
        if not (model.geom_contype[i] or model.geom_conaffinity[i]):
            continue
        kind = int(model.geom_type[i])
        if kind not in [2, 3]:
            continue
        center = data.geom_xpos[i]
        axis = data.geom_xmat[i].reshape(3, 3)[:, 2]
        ends = (
            [center]
            if kind == 2
            else [center - axis * model.geom_size[i, 1], center + axis * model.geom_size[i, 1]]
        )
        for end in ends:
            for direction in np.concatenate([np.eye(3), -np.eye(3)]):
                points.append(end + model.geom_size[i, 0] * direction)
    points = np.array(points)
    return float((np.linalg.norm(points[:, None] - centers[None], axis=-1).min(-1) - 0.1).max())


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    design = json.loads((PREVIOUS / "design.json").read_text())
    for ref in design["source_bindings"]:
        assert sha(ref["path"]) == ref["sha256"]
    write(
        "design.json",
        dict(
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            scope="All existing 80 files; no replacement; read-only numerical compatibility and aliasing audit",
            outcome_access="Earlier geometric outcomes inspected; this audit is exploratory, not original registration",
            models={k: {"path": str(p), "sha256": sha(p)} for k, p in MODELS.items()},
            prior_design_sha256=sha(PREVIOUS / "design.json"),
            code_sha256=sha(Path(__file__)),
            kinematic_tolerance={"position_m": 1e-5, "orientation_rad": 1e-3},
            budget={"CPU_seconds": 180, "physics_steps": 0, "training_updates": 0},
            tests="All frames against three FK models; compare 32-frame proxy fields to all-frame original fields; primitive noncontainment witness at 32 frames",
            limits="Numerical model agreement is not authentic historical converter provenance or native Isaac containment",
        ),
    )
    start = time.monotonic()
    models = {k: mujoco.MjModel.from_xml_path(str(p)) for k, p in MODELS.items()}
    datas = {k: mujoco.MjData(m) for k, m in models.items()}
    for m in models.values():
        assert m.nbody == 31 and m.nq == 36
    names = {
        k: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)]
        for k, m in models.items()
    }
    joint_names = {
        k: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
        for k, m in models.items()
    }
    write("model-mapping.json", dict(body_names=names, joint_names=joint_names))
    old = torch.load(PREVIOUS / "geometry.pt", weights_only=True)
    rows = []
    calls = 0
    for index, ref in enumerate(design["source_bindings"]):
        with np.load(ref["path"], allow_pickle=False) as a:
            pos = a["body_pos_w"].astype(np.float64)
            quat = a["body_quat_w"].astype(np.float64)
            joint = a["joint_pos"].astype(np.float64)
            fps = float(a["fps"][0])
        assert (
            np.isfinite(pos).all()
            and np.isfinite(quat).all()
            and np.isfinite(joint).all()
            and (np.linalg.norm(quat, axis=-1) > 0).all()
        )
        idx = np.linspace(0, len(pos) - 1, 32).round().astype(int)
        stats = {}
        witness = []
        for key, m in models.items():
            d = datas[key]
            maxp = 0.0
            maxa = 0.0
            for frame in range(len(pos)):
                d.qpos[:] = np.concatenate([pos[frame, 0], quat[frame, 0], joint[frame]])
                mujoco.mj_kinematics(m, d)
                calls += 1
                maxp = max(maxp, float(np.linalg.norm(d.xpos[1:] - pos[frame], axis=-1).max()))
                maxa = max(maxa, float(orientation_error(d.xquat[1:], quat[frame]).max()))
                if key == "kimodo_primitives" and frame in idx:
                    witness.append(surface_witness(d, m, pos[frame]))
            stats[key] = dict(
                max_position_error_m=maxp,
                max_orientation_error_rad=maxa,
                numerically_compatible=maxp <= 1e-5 and maxa <= 1e-3,
            )
        # Replay the ORIGINAL normalization and 32-index choice in original float32 arithmetic.
        x = torch.tensor(pos, dtype=torch.float32)
        x[:, :, :2] -= x[0, 0, :2].clone()
        x[:, :, 2] -= x[:, :, 2].amin() - 0.1
        ti = torch.linspace(0, len(x) - 1, 32).round().long()
        sparse = gaps(x[ti].reshape(-1, 3), candidates())
        dense = old["gaps"][index]
        rows.append(
            dict(
                file=ref["path"],
                sha256=ref["sha256"],
                split=ref["split"],
                frames=len(pos),
                fps=fps,
                sample_max_interval_s=float(np.diff(ti.numpy()).max() / fps),
                kinematics=stats,
                primitive_surface_max_outside_spheres_m=max(witness),
                sparse_clear_dense_penetrating=int(((sparse >= 0.02) & (dense < 0)).sum()),
                sparse_clear_dense_not_clear=int(((sparse >= 0.02) & (dense < 0.02)).sum()),
                max_sparse_gap_overestimate_m=float((sparse - dense).max()),
            )
        )
        write("partial-ledger.json", rows)
        if time.monotonic() - start > 180:
            raise TimeoutError("audit cap; no extension")
    write("ledger.json", rows)
    summary = dict(
        files=len(rows),
        frames=sum(r["frames"] for r in rows),
        kinematics_calls=calls,
        physics_steps=0,
        training_updates=0,
        wall_seconds=time.monotonic() - start,
        compatible_files={
            k: sum(r["kinematics"][k]["numerically_compatible"] for r in rows) for k in models
        },
        worst_position_m={
            k: max(r["kinematics"][k]["max_position_error_m"] for r in rows) for k in models
        },
        sphere_noncontainment_witness_files=sum(
            r["primitive_surface_max_outside_spheres_m"] > 1e-5 for r in rows
        ),
        worst_sphere_noncontainment_m=max(
            r["primitive_surface_max_outside_spheres_m"] for r in rows
        ),
        sparse_clear_dense_penetrating=sum(r["sparse_clear_dense_penetrating"] for r in rows),
        sparse_clear_dense_not_clear=sum(r["sparse_clear_dense_not_clear"] for r in rows),
        files_with_sparse_false_clear=sum(r["sparse_clear_dense_penetrating"] > 0 for r in rows),
        max_input_interval_s=max(r["sample_max_interval_s"] for r in rows),
    )
    write("summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if OUT.exists():
            write("failure.json", dict(error=repr(e), retry="No automatic retry"))
        raise
