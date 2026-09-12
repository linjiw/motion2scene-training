"""Bind frozen native assets, verify reference frames, and label development geometry."""

import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from gear_sonic.dataset_generation.swept_volume import CollisionCapsule, body_capsules_world
from scripts.research.lflh_next.geometry import box_sdf, capsule_samples
from scripts.research.lflh_next.qualification.audit import PREVIOUS, orientation_error, sha

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT.parent / "research-data/groot-wbc"
OUT = ROOT.parent / "research-data/m2s-native-development-labels-20260911"
ASSETS = DATA / "m2s-native-runtime-freeze-v6/runtime_assets.json"
GEOMETRY = DATA / "m2s-native-beam-audit-v1/geometry.json"
MJCF = ROOT / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bound_roles(sampled_sdf, radius, cover, inner):
    """All enclosing geometry supports clearance; only inner primitives witness penetration."""
    lower = (sampled_sdf - radius - cover).amin()
    upper = (sampled_sdf[inner] - radius[inner]).amin()
    return lower, upper


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    start = time.monotonic()
    old = json.loads((PREVIOUS / "design.json").read_text())
    assets = json.loads(ASSETS.read_text())
    geo = json.loads(GEOMETRY.read_text())
    audit_refs = json.loads(
        (
            ROOT / "docs/motion2scene/audit/20260911-pilot-boundary/input-artifact-manifest.json"
        ).read_text()
    )
    selected = [
        r
        for r in audit_refs
        if r["path"]
        in [
            str(ASSETS),
            str(MJCF),
            str(GEOMETRY),
            str(ROOT / "gear_sonic/envs/manager_env/robots/g1.py"),
        ]
    ]
    assert any(r["path"] == str(ASSETS) for r in selected)
    for r in selected:
        assert sha(r["path"]) == r["sha256"].removeprefix("sha256:")
    assert any(r["path"] == str(MJCF) for r in assets)
    for r in assets + geo["layers"] + old["source_bindings"]:
        assert sha(r["path"]) == r["sha256"].removeprefix("sha256:")
    save(
        "design.json",
        dict(
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            status="Exploratory development relabeling; all prior outcome panels already inspected",
            scope="All 80 original references, all recorded frames, all 225 previous shape-position candidates. No replacements, no training.",
            evidence="Native cached geometry on numerically verified reference frames; not obstacle-present dynamics or source transfer",
            positive_rule="lower bound using native primitives plus enclosing mesh spheres >= .02m",
            negative_rule="sampled upper bound using native primitive SUBSET only <0m; outer mesh overlaps never negative labels",
            near_rule="lower>=.02m and inner upper<=.12m is sufficient near membership only; not decision criticality",
            cap_seconds=180,
            physics_steps=0,
            optimizer_updates=0,
            bindings={
                "assets": sha(ASSETS),
                "geometry": sha(GEOMETRY),
                "mjcf": sha(MJCF),
                "code": sha(Path(__file__)),
            },
            historical_audit_bindings=selected,
            source_bindings=old["source_bindings"],
        ),
    )
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    simdata = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    shapes = {}
    roles = {}
    for s in geo["shapes"]:
        assert s["role"] in ["native_primitive_subset", "outer_mesh_sphere"]
        shapes.setdefault(s["owner"], []).append(
            CollisionCapsule(tuple(s["start"]), tuple(s["end"]), s["radius"])
        )
        roles.setdefault(s["owner"], []).append(s["role"] == "native_primitive_subset")
    inner_caps = torch.tensor([v for owner in shapes for v in roles[owner]])
    specs = torch.load(PREVIOUS / "geometry.pt", weights_only=True)["specs"]
    low = []
    upp = []
    ledger = []
    full = []
    calls = 0
    for ref in old["source_bindings"]:
        with np.load(ref["path"], allow_pickle=False) as p:
            pos = p["body_pos_w"].copy()
            quat = p["body_quat_w"].astype(np.float64)
            joints = p["joint_pos"]
            fps = float(p["fps"][0])
        quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
        mp = 0.0
        ma = 0.0
        for t in range(len(pos)):
            simdata.qpos[:] = np.r_[pos[t, 0], quat[t, 0], joints[t]]
            mujoco.mj_kinematics(model, simdata)
            calls += 1
            mp = max(mp, float(np.linalg.norm(simdata.xpos[1:] - pos[t], axis=-1).max()))
            ma = max(ma, float(orientation_error(simdata.xquat[1:], quat[t]).max()))
        assert (
            mp <= 1e-5 and ma <= 1e-3
        ), "reference/model mismatch; refuse correction or replacement"
        pos[:, :, :2] -= pos[0, 0, :2].copy()
        pos[:, :, 2] -= pos[:, :, 2].min() - 0.1
        a, b, r, owners = body_capsules_world(pos, quat, names, capsules=shapes)
        pts, rad, cover = capsule_samples(
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(b, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
            5,
        )
        inner = inner_caps[None, :, None].expand(pts.shape[:-1]).flatten()
        pts = pts.reshape(-1, 3)
        rad = rad.flatten()
        cover = cover.flatten()
        lo = []
        up = []
        for spec in specs:
            sdf = (
                (pts - spec[:3]).norm(dim=-1) - 0.3
                if spec[3] == 2
                else box_sdf(
                    pts,
                    spec[:3],
                    torch.tensor([0.15, 0.8, 0.1] if spec[3] == 0 else [0.3, 0.3, 0.4]),
                    torch.tensor(0.0),
                )
            )
            lower_bound, upper_bound = bound_roles(sdf, rad, cover, inner)
            lo.append(lower_bound)
            up.append(upper_bound)
        low.append(torch.stack(lo))
        upp.append(torch.stack(up))
        full.append(
            {
                "positions": torch.tensor(pos),
                "times_s": torch.arange(len(pos), dtype=torch.float64) / fps,
                "duration_s": (len(pos) - 1) / fps,
            }
        )
        ledger.append(
            dict(
                path=ref["path"],
                sha256=ref["sha256"],
                original_split=ref["split"],
                current_role="development",
                frames=len(pos),
                fps=fps,
                max_fk_position_m=mp,
                max_fk_orientation_rad=ma,
                status="native_reference_labels_complete",
                source_ancestry="unverified",
                physical_qualification="unverified",
            )
        )
        save("partial-ledger.json", ledger)
        if time.monotonic() - start > 180:
            raise TimeoutError("fixed cap")
    lower = torch.stack(low)
    upper = torch.stack(upp)
    assert (lower <= upper + 1e-5).all()
    torch.save(
        dict(
            lower=lower,
            inner_upper=upper,
            specs=specs,
            full_resolution_inputs=full,
            body_names=names,
        ),
        OUT / "labels.pt",
    )
    save("ledger.json", ledger)
    probs = torch.load(PREVIOUS / "probabilities.pt", weights_only=True)
    probs["uniform"] = torch.ones(16, 225) / 225
    rows = []
    for key, q in probs.items():
        for i in range(16):
            clear = lower[64 + i] >= 0.02
            hit = upper[64 + i] < 0
            near = clear & (upper[64 + i] <= 0.12)
            rows.append(
                dict(
                    model_seed=key,
                    target=i,
                    clear_mass=float((q[i] * clear).sum()),
                    penetration_witness_mass=float((q[i] * hit).sum()),
                    unknown_mass=float((q[i] * (~clear & ~hit)).sum()),
                    near_sufficient_mass=float((q[i] * near).sum()),
                )
            )
    save("frozen-model-outcomes.json", rows)
    summary = []
    for kind in ["uniform", "unconditional", "mlp", "transformer"]:
        rs = [r for r in rows if r["model_seed"].split("-")[0] == kind]
        summary.append(
            dict(
                model=kind,
                **{
                    k: float(np.mean([r[k] for r in rs]))
                    for k in [
                        "clear_mass",
                        "penetration_witness_mass",
                        "unknown_mass",
                        "near_sufficient_mass",
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
            asset_hashes_verified=len(assets),
            cached_layers_verified=len(geo["layers"]),
            references=len(ledger),
            kinematics_calls=calls,
            physics_steps=0,
            updates=0,
            primitive_shapes=int(inner_caps.sum()),
            outer_mesh_spheres=int((~inner_caps).sum()),
            motion_candidate_pairs=len(ledger) * len(specs),
            rows=len(rows),
            max_fk_position_m=max(r["max_fk_position_m"] for r in ledger),
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
