"""Bounded, private Kimodo reference-geometry development study; no dynamics."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from torch import nn

from gear_sonic.dataset_generation.swept_volume import CollisionCapsule, body_capsules_world
from scripts.research.lflh_next.constrained import coverage_acceptance_loss, project_clearance
from scripts.research.lflh_next.geometry import box_sdf, capsule_samples
from scripts.research.lflh_next.navigation.route_geometry import route_candidates
from scripts.research.lflh_next.qualification.native_labels import GEOMETRY, MJCF, bound_roles
from scripts.research.lflh_next.qualification.split_safe_learning import Generator, sha

ROOT = Path(__file__).resolve().parents[4]
BANK = Path("/home/linjiw/kimodo/data/indoor_nav_1k")
OUT = ROOT.parent / "research-data/m2s-kimodo-route-100-20260911"
KIMODO = Path("/home/linjiw/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml")
_CREATED_THIS_ATTEMPT = False


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def prompt_groups(rows):
    """Transitive grouping: any shared normalized prompt joins source records."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        if not row.get("prompts"):
            raise ValueError("missing prompt ancestry")
        for p in row["prompts"]:
            key = " ".join(p.lower().split())
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    members = {}
    for i, row in enumerate(rows):
        members.setdefault(find(i), []).append(row["id"])
    keys = {k: hashlib.sha256("|".join(sorted(v)).encode()).hexdigest() for k, v in members.items()}
    return {r["id"]: keys[find(i)] for i, r in enumerate(rows)}


def main():
    global _CREATED_THIS_ATTEMPT
    _CREATED_THIS_ATTEMPT = False
    OUT.mkdir(exist_ok=False)
    _CREATED_THIS_ATTEMPT = True
    started = time.monotonic()
    torch.set_num_threads(2)
    rows = [json.loads(x) for x in (BANK / "metadata.jsonl").read_text().splitlines()]
    assert len(rows) == len({r["id"] for r in rows}) == 1000
    groups = prompt_groups(rows)
    keys = sorted(set(groups.values()))
    # Entire prompt components stay together. New DEVELOPMENT partition only.
    development_groups = set(keys[::5])
    ordered = sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    fit = [r for r in ordered if groups[r["id"]] not in development_groups][:100]
    dev = [r for r in ordered if groups[r["id"]] in development_groups][:20]
    assert len(fit) == 100 and len(dev) == 20
    assert not ({groups[r["id"]] for r in fit} & {groups[r["id"]] for r in dev})
    write(
        "design.json",
        dict(
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            status="New exploratory reference-geometry amendment; not original physical pilot",
            prior_access="Previous AMASS and seven-schedule outcomes accessible; Kimodo aggregate reports inspected; all prior fixed-grid Kimodo placement outcomes inspected; route repair is exploratory",
            scope="100 generated references fit, 20 prompt-component-separated development references; 225 NEW route-relative recipes: 5 arc-length stations x 5 lateral offsets x 3 heights x 3 shapes; tangent-aligned box yaw",
            model="Identical 30401-parameter architecture, unconditional versus conditioned",
            objective="coverage_acceptance_loss, acceptance_weight=2; sufficient near-boundary is NOT decision criticality",
            seeds=[701, 702],
            updates_each=200,
            batch_size=16,
            lr=0.001,
            cap=dict(fits=4, updates=800, label_references=120, wall_seconds=300, physics_steps=0),
            normalization="Initial root xy removed; all-frame minimum body-origin z shifted to .1 m; inherited reference convention, not ground or SONIC qualification; heading unchanged",
            exclusions="No filtering by near mass, joint-limit excess, path tracking, category, or physical outcome; malformed data halts without replacement",
            partition="Connected components sharing any normalized exact prompt; sorted component hash every fifth development; select ID-hash first 100/20 before labels",
            fit_ids=[r["id"] for r in fit],
            development_ids=[r["id"] for r in dev],
            bindings={
                str(p): sha(p)
                for p in [
                    BANK / "metadata.jsonl",
                    BANK / "spec.jsonl",
                    KIMODO,
                    MJCF,
                    GEOMETRY,
                    Path(__file__),
                    Path(__file__).with_name("route_geometry.py"),
                ]
            },
            larger_stages="1000 and 10000 distinct TRAINING references require additional verified, partition-compatible sources; no duplication to fill targets",
            analysis="All development rows, per-seed effects; p/interval NA: no registered population estimand or defensible exchangeability scheme; no model selection",
        ),
    )
    native = mujoco.MjModel.from_xml_path(str(MJCF))
    source = mujoco.MjModel.from_xml_path(str(KIMODO))
    dn, ds = mujoco.MjData(native), mujoco.MjData(source)

    def joint_names(m):
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]

    assert joint_names(native) == joint_names(source)
    names = [mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, native.nbody)]
    assert names == [
        mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, source.nbody)
    ]
    shapes, roles = {}, {}
    for s in json.loads(GEOMETRY.read_text())["shapes"]:
        shapes.setdefault(s["owner"], []).append(
            CollisionCapsule(tuple(s["start"]), tuple(s["end"]), s["radius"])
        )
        roles.setdefault(s["owner"], []).append(s["role"] == "native_primitive_subset")
    inner_caps = torch.tensor([v for owner in shapes for v in roles[owner]])
    specs = torch.load(
        ROOT.parent / "research-data/m2s-native-development-labels-20260911/labels.pt",
        weights_only=True,
    )["specs"]
    inventory = []
    # Verify all 1000 CSV receipts, but inspect no physics outcomes or reserved layouts.
    for r in rows:
        path = BANK / r["csv"]
        q = np.loadtxt(path, delimiter=",")
        assert q.shape == tuple(r["qpos_shape"]) and q.shape[1] == 36 and np.isfinite(q).all()
        err = float(np.abs(np.linalg.norm(q[:, 3:7], axis=-1) - 1).max())
        assert err < 1e-4
        excess = np.maximum(
            native.jnt_range[1:, 0] - q[:, 7:], q[:, 7:] - native.jnt_range[1:, 1]
        ).clip(0)
        inventory.append(
            dict(
                id=r["id"],
                path=str(path),
                sha256=sha(path),
                frames=len(q),
                fps=r["fps"],
                category=r["category"],
                prompts=r["prompts"],
                seed=r["seed"],
                group=groups[r["id"]],
                source_native_npz_exists=(BANK / r["npz"]).exists(),
                joint_limit_excess_rad=float(excess.max()),
                quaternion_norm_error=err,
                physical_qualification="UNTESTED",
            )
        )
    assert len({r["sha256"] for r in inventory}) == 1000
    write("inventory.json", inventory)
    lows, upps, features, display, world_specs = [], [], [], [], []
    fk_calls = 0
    label_start = time.monotonic()
    for index, r in enumerate(fit + dev):
        q = np.loadtxt(BANK / r["csv"], delimiter=",")
        q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=-1, keepdims=True)
        pos, quat = [], []
        max_fk = 0.0
        for frame in q:
            dn.qpos[:], ds.qpos[:] = frame, frame
            mujoco.mj_kinematics(native, dn)
            mujoco.mj_kinematics(source, ds)
            fk_calls += 2
            max_fk = max(max_fk, float(np.linalg.norm(dn.xpos[1:] - ds.xpos[1:], axis=-1).max()))
            pos.append(dn.xpos[1:].copy())
            quat.append(dn.xquat[1:].copy())
        assert max_fk < 1e-5
        pos, quat = np.array(pos), np.array(quat)
        shift = np.r_[pos[0, 0, :2], pos[:, :, 2].min() - 0.1]
        pos -= shift
        q[:, :3] -= shift
        w, xx, yy, zz = q[0, 3:7]
        heading = np.arctan2(2 * (w * zz + xx * yy), 1 - 2 * (yy * yy + zz * zz))
        candidate_array, recipe_array, anchors = route_candidates(pos[:, 0, :2], heading)
        specs = torch.tensor(candidate_array, dtype=torch.float32)
        recipes = torch.tensor(recipe_array, dtype=torch.float32)
        world_specs.append(specs)
        a, b, rad, _ = body_capsules_world(pos, quat, names, capsules=shapes)
        pts, rad, cover = capsule_samples(
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(b, dtype=torch.float32),
            torch.tensor(rad, dtype=torch.float32),
            5,
        )
        inner = inner_caps[None, :, None].expand(pts.shape[:-1]).flatten()
        pts, rad, cover = pts.reshape(-1, 3), rad.flatten(), cover.flatten()
        lo, up = [], []
        for spec in specs:
            sdf = (
                (pts - spec[:3]).norm(dim=-1) - 0.3
                if spec[3] == 2
                else box_sdf(
                    pts,
                    spec[:3],
                    torch.tensor([0.15, 0.8, 0.1] if spec[3] == 0 else [0.3, 0.3, 0.4]),
                    spec[4],
                )
            )
            lower, upper = bound_roles(sdf, rad, cover, inner)
            lo.append(lower)
            up.append(upper)
        lows.append(torch.stack(lo))
        upps.append(torch.stack(up))
        x = torch.tensor(pos, dtype=torch.float32).flatten(1)
        features.append(
            torch.cat(
                [
                    x.mean(0),
                    x.amin(0),
                    x.amax(0),
                    torch.tensor([(len(q) - 1) / r["fps"] / 50, r["fps"] / 50]),
                ]
            )
        )
        if index >= 100:
            ids = np.linspace(0, len(q) - 1, min(60, len(q))).astype(int)
            display.append(
                dict(
                    id=r["id"],
                    category=r["category"],
                    prompts=r["prompts"],
                    fps=r["fps"],
                    frames=len(q),
                    times=(ids / r["fps"]).tolist(),
                    qpos=q[ids].tolist(),
                    positions=pos[ids].tolist(),
                    fk_residual_m=max_fk,
                    anchors=anchors.tolist(),
                )
            )
        write("progress.json", dict(completed_labels=index + 1, fk_calls=fk_calls, physics_steps=0))
        if time.monotonic() - started > 300:
            raise TimeoutError("fixed study cap")
    label_seconds = time.monotonic() - label_start
    lower, upper = torch.stack(lows), torch.stack(upps)
    assert (lower <= upper + 1e-5).all()
    X = torch.stack(features)
    mean = X[:100].mean(0)
    scale = X[:100].std(0).clamp_min(0.1)
    X = (X - mean) / scale
    clear = lower >= 0.02
    near = clear & (upper <= 0.12)
    encoded = torch.cat([recipes[:, :3] / 3, nn.functional.one_hot(recipes[:, 3].long(), 3)], -1)
    torch.save(
        dict(
            lower=lower,
            inner_upper=upper,
            specs=torch.stack(world_specs),
            recipes=recipes,
            features=X,
            mean=mean,
            scale=scale,
        ),
        OUT / "labels.pt",
    )
    write("display.json", display)
    fits = []
    train_start = time.monotonic()
    for conditioned in [False, True]:
        for seed in [701, 702]:
            torch.manual_seed(seed)
            model = Generator(conditioned)
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            t = time.monotonic()
            for step in range(200):
                ix = torch.randperm(100)[:16]
                loss = coverage_acceptance_loss(
                    model(X[ix], encoded), near[ix], clear[ix], acceptance_weight=2
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                if time.monotonic() - started > 300:
                    raise TimeoutError("fixed study cap")
            key = f"{'conditioned' if conditioned else 'unconditional'}-{seed}"
            torch.save(dict(state=model.state_dict(), mean=mean, scale=scale), OUT / f"{key}.pt")
            fits.append(
                dict(
                    key=key,
                    seed=seed,
                    conditioned=conditioned,
                    updates=200,
                    wall_seconds=time.monotonic() - t,
                    sha256=sha(OUT / f"{key}.pt"),
                )
            )
    write("model-lock.json", fits)
    train_seconds = time.monotonic() - train_start
    probs = {"uniform": torch.ones(20, len(specs)) / len(specs)}
    with torch.no_grad():
        for f in fits:
            model = Generator(f["conditioned"])
            model.load_state_dict(torch.load(OUT / f"{f['key']}.pt", weights_only=True)["state"])
            model.eval()
            probs[f["key"]] = model(X[100:], encoded).softmax(-1)
    torch.save(probs, OUT / "probabilities.pt")
    outcomes = []
    for key, q in probs.items():
        proj, z = project_clearance(q, lower[100:], 0.02)
        for mode, dist in [("raw", q), ("projected", proj)]:
            for j, r in enumerate(dev):
                c = clear[100 + j]
                hit = upper[100 + j] < 0
                outcomes.append(
                    dict(
                        model=key,
                        mode=mode,
                        id=r["id"],
                        category=r["category"],
                        near=float((dist[j] * near[100 + j]).sum()),
                        clear=float((dist[j] * c).sum()),
                        penetration=float((dist[j] * hit).sum()),
                        unknown=float((dist[j] * (~c & ~hit)).sum()),
                        retained=float(z[j]),
                        abstained=bool(dist[j].sum() == 0),
                        near_cells=int(near[100 + j].sum()),
                    )
                )
    write("outcomes.json", outcomes)
    with (OUT / "outcomes.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(outcomes[0]))
        w.writeheader()
        w.writerows(outcomes)
    summary = []
    for key in probs:
        for mode in ["raw", "projected"]:
            rr = [r for r in outcomes if r["model"] == key and r["mode"] == mode]
            summary.append(
                dict(
                    model=key,
                    mode=mode,
                    **{
                        k: float(np.mean([r[k] for r in rr]))
                        for k in ["near", "clear", "penetration", "unknown", "retained"]
                    },
                    abstentions=sum(r["abstained"] for r in rr),
                )
            )
    write("summary.json", summary)
    write(
        "receipt.json",
        dict(
            state="complete",
            verified_csv_files=1000,
            unique_csv_hashes=1000,
            categories=dict(Counter(r["category"] for r in inventory)),
            prompt_components=len(keys),
            fit=100,
            development=20,
            fits=4,
            updates=800,
            physics_steps=0,
            fk_calls=fk_calls,
            geometry_queries=120 * 225,
            label_seconds=label_seconds,
            training_seconds=train_seconds,
            wall_seconds=time.monotonic() - started,
            fit_zero_near=int((near[:100].sum(-1) == 0).sum()),
            dev_zero_near=int((near[100:].sum(-1) == 0).sum()),
            joint_limit_excess_files=sum(r["joint_limit_excess_rad"] > 0 for r in inventory),
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if _CREATED_THIS_ATTEMPT:
            write("failure.json", dict(error=repr(e), retry="No automatic retry"))
        raise
