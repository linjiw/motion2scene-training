"""Rescore frozen outputs on compatible model primitives; no dynamics or retraining."""

import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from gear_sonic.dataset_generation.swept_volume import CollisionCapsule, body_capsules_world
from scripts.research.lflh_next.geometry import box_sdf, capsule_samples
from scripts.research.lflh_next.qualification.audit import MODELS, PREVIOUS, sha

OUT = Path("/home/linjiw/research-data/m2s-primitive-rescore-20260911")


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    start = time.monotonic()
    design = json.loads((PREVIOUS / "design.json").read_text())
    refs = design["source_bindings"][64:]
    audit = Path("/home/linjiw/research-data/m2s-motion-qualification-20260911/summary.json")
    assert json.loads(audit.read_text())["compatible_files"]["kimodo_primitives"] == 80
    save(
        "design.json",
        dict(
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            status="Outcome-motivated exploratory rescore after geometry audit",
            scope="All 16 previously evaluated motions; all six frozen fits plus uniform; no replacement or tuning",
            access="Original model scores, sphere labels, FK mismatch and containment audit already inspected",
            geometry="Exact enabled sphere/capsule primitives of checked Kimodo MJCF; 5 axis samples with covering lower bound; every recorded frame",
            labels="Clear lower>=.02m; penetration witness upper<0; bounded near lower>=.02 and upper<=.12; neither native Isaac nor physical outcome",
            budget={"physics_steps": 0, "updates": 0, "wall_cap_s": 180},
            code_sha256=sha(Path(__file__)),
            model_sha256=sha(MODELS["kimodo_primitives"]),
            audit_sha256=sha(audit),
            probability_sha256=sha(PREVIOUS / "probabilities.pt"),
            inputs=refs,
        ),
    )
    m = mujoco.MjModel.from_xml_path(str(MODELS["kimodo_primitives"]))
    shapes = {}
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)]
    for i in range(m.ngeom):
        if not (m.geom_contype[i] or m.geom_conaffinity[i]) or m.geom_bodyid[i] == 0:
            continue
        kind = int(m.geom_type[i])
        assert kind in [2, 3]
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[i])
        axis = rot.reshape(3, 3)[:, 2]
        half = 0 if kind == 2 else m.geom_size[i, 1]
        shape = CollisionCapsule(
            tuple(m.geom_pos[i] - axis * half),
            tuple(m.geom_pos[i] + axis * half),
            float(m.geom_size[i, 0]),
        )
        owner = names[m.geom_bodyid[i] - 1]
        shapes.setdefault(owner, []).append(shape)
    d = torch.load(PREVIOUS / "geometry.pt", weights_only=True)
    spec = d["specs"]
    L = []
    U = []
    for ref in refs:
        assert sha(ref["path"]) == ref["sha256"]
        with np.load(ref["path"], allow_pickle=False) as ar:
            pos = ar["body_pos_w"].copy()
            quat = ar["body_quat_w"].copy()
        # Match the original translation exactly, rather than repairing ground alignment mid-comparison.
        pos[:, :, :2] -= pos[0, 0, :2].copy()
        pos[:, :, 2] -= pos[:, :, 2].min() - 0.1
        a, b, r, _ = body_capsules_world(pos, quat, names, capsules=shapes)
        pts, rad, cover = capsule_samples(
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(b, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
            5,
        )
        pts = pts.reshape(-1, 3)
        rad = rad.flatten()
        cover = cover.flatten()
        lo = []
        up = []
        for row in spec:
            if row[3] == 2:
                sdf = (pts - row[:3]).norm(dim=-1) - 0.3
            else:
                sdf = box_sdf(
                    pts,
                    row[:3],
                    torch.tensor([0.15, 0.8, 0.1] if row[3] == 0 else [0.3, 0.3, 0.4]),
                    torch.tensor(0.0),
                )
            up.append((sdf - rad).amin())
            lo.append((sdf - rad - cover).amin())
        L.append(torch.stack(lo))
        U.append(torch.stack(up))
        if time.monotonic() - start > 180:
            raise TimeoutError("fixed rescore cap")
    lower = torch.stack(L)
    upper = torch.stack(U)
    torch.save({"lower": lower, "upper": upper}, OUT / "bounds.pt")
    probs = torch.load(PREVIOUS / "probabilities.pt", weights_only=True)
    probs["uniform"] = torch.ones(16, 225) / 225
    rows = []
    for key, q in probs.items():
        for i in range(16):
            rows.append(
                dict(
                    model_seed=key,
                    target=Path(refs[i]["path"]).name,
                    primitive_clear_mass=float((q[i] * (lower[i] >= 0.02)).sum()),
                    primitive_penetration_witness_mass=float((q[i] * (upper[i] < 0)).sum()),
                    bounded_near_mass=float(
                        (q[i] * ((lower[i] >= 0.02) & (upper[i] <= 0.12))).sum()
                    ),
                    sphere_clear_primitive_penetrating_cells=int(
                        ((d["gaps"][64 + i] >= 0.02) & (upper[i] < 0)).sum()
                    ),
                )
            )
    save("outcomes.json", rows)
    summary = []
    for kind in ["uniform", "unconditional", "mlp", "transformer"]:
        rs = [r for r in rows if r["model_seed"].split("-")[0] == kind]
        summary.append(
            dict(
                model=kind,
                **{
                    k: float(np.mean([r[k] for r in rs]))
                    for k in [
                        "primitive_clear_mass",
                        "primitive_penetration_witness_mass",
                        "bounded_near_mass",
                    ]
                }
            )
        )
    save("summary.json", summary)
    save(
        "receipt.json",
        dict(
            state="complete",
            wall_seconds=time.monotonic() - start,
            primitives=sum(map(len, shapes.values())),
            motion_candidate_pairs=16 * 225,
            rows=len(rows),
            physics_steps=0,
            updates=0,
            sphere_clear_primitive_penetrating_cells=int(
                ((d["gaps"][64:] >= 0.02) & (upper < 0)).sum()
            ),
            files_with_disagreement=int(
                (((d["gaps"][64:] >= 0.02) & (upper < 0)).sum(-1) > 0).sum()
            ),
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if OUT.exists():
            save("failure.json", {"error": repr(e), "retry": "No automatic retry"})
        raise
