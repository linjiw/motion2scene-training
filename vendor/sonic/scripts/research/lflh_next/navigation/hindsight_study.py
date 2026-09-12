"""Bounded motion-event and counterfactual-geometry study, never dynamics."""

import csv
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from torch import nn

from gear_sonic.dataset_generation.swept_volume import CollisionCapsule, body_capsules_world
from scripts.research.lflh_next.constrained import project_clearance
from scripts.research.lflh_next.geometry import box_sdf, capsule_samples
from scripts.research.lflh_next.navigation.motion_events import (
    chord_root,
    event_anchors,
    event_candidates,
    observations,
)
from scripts.research.lflh_next.navigation.route_study import GEOMETRY, MJCF, sha

ROOT = Path(__file__).resolve().parents[4]
PREV = ROOT.parent / "research-data/m2s-kimodo-route-100-20260911"
OUT = ROOT.parent / "research-data/m2s-hindsight-events-100-20260911"
_CREATED = False


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class LocalGenerator(nn.Module):
    def __init__(self, conditioned=True):
        super().__init__()
        self.conditioned = conditioned
        self.motion = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.shape = nn.Sequential(nn.Linear(6, 64), nn.SiLU())
        self.score = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x, recipe):
        if not self.conditioned:
            x = torch.zeros_like(x)
        return self.score(torch.cat([self.motion(x), self.shape(recipe)], -1)).squeeze(-1)


def loss_fn(logits, target, clear):
    logq = logits.log_softmax(-1)
    present = target.sum(-1) > 0
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    coverage = -(target * logq).sum(-1)
    logz = torch.logsumexp(logq.masked_fill(~clear, -torch.inf), -1)
    valid = clear.any(-1)
    # Generalizes previous lambda=2 loss: conditional coverage + 2*(-log Z).
    return (
        (coverage[valid] - (2 - present[valid].float()) * logz[valid]).mean()
        if valid.any()
        else logits.sum() * 0
    )


def main():
    global _CREATED
    _CREATED = False
    OUT.mkdir(exist_ok=False)
    _CREATED = True
    start = time.monotonic()
    torch.set_num_threads(2)
    for r in json.loads((PREV / "manifest.json").read_text()):
        assert sha(r["path"]) == r["sha256"]
    prior = json.loads((PREV / "design.json").read_text())
    inventory = {r["id"]: r for r in json.loads((PREV / "inventory.json").read_text())}
    ids = prior["fit_ids"] + prior["development_ids"]
    write(
        "design.json",
        dict(
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            status="New exploratory amendment motivated by inspected route-study outcomes and user hypothesis; not original physical registration",
            fit_ids=ids[:100],
            development_ids=ids[100:],
            scope="Same 100 fit / 20 development; 225 event-relative recipes; all arms share new geometry and labels",
            observations="64 full-record channels from COM, hip, head mesh center, hand mesh centers and foot bodies; 7-frame cubic Savitzky-Golay smoothing then derivatives at stored fps; offline hindsight uses future frames",
            candidate_anchors="Endpoints plus three peaks with 12% duration separation. Saliency=.4*turn + .3*head-relative vertical speed + .3*COM acceleration; each clipped to training 95th-percentile scale at 3. This anchor detector is explicit, not learned.",
            features="Per-anchor local 0.5-second-half-window mean and std of 64 channels = 128, normalized using fit references only",
            alternatives="Kinematic root chord or entry-pose hold within +/-0.5 seconds of each anchor; not dynamically qualified continuations. Pose hold preserves actual root orientation/translation but fixes body-local pose at window entry.",
            target="Baseline sufficient-near uniform target. Hindsight weighted target: .1*clear + 1*near + 3*(clear AND any counterfactual inner penetration). Explicit analytic target baseline also reported. Heuristic event concentration alone cannot establish utility.",
            variants=["geometry", "hindsight", "hindsight_no_motion"],
            seeds=[801, 802],
            updates_each=200,
            lr=0.001,
            batch_size=16,
            loss="conditional weighted coverage + 2*(-log retained clear mass)",
            cap=dict(
                fits=6,
                updates=1200,
                reference_labels=27000,
                counterfactual_labels=54000,
                wall_seconds=300,
                CPU_threads=2,
                physics_steps=0,
            ),
            analysis="All 20 known development references, per seed and variant, raw/projected mass. No model selection, p/CI NA: exploratory dependent generated corpus, no registered exchangeability scheme.",
            bindings={
                str(p): sha(p)
                for p in [
                    PREV / "design.json",
                    PREV / "manifest.json",
                    MJCF,
                    GEOMETRY,
                    Path(__file__),
                    Path(__file__).with_name("motion_events.py"),
                ]
            },
        ),
    )
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)]

    def bodyid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

    hip = bodyid("pelvis")
    feet = [bodyid("left_ankle_roll_link"), bodyid("right_ankle_roll_link")]
    mesh_ids = {}
    for g in range(m.ngeom):
        if m.geom_type[g] == 7 and m.geom_group[g] == 1:
            mesh_ids[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g]))] = g
    head = mesh_ids["head_link"]
    hands = [mesh_ids["left_rubber_hand"], mesh_ids["right_rubber_hand"]]
    owner_ids = [hip, int(m.geom_bodyid[head]), *[int(m.geom_bodyid[g]) for g in hands], *feet]
    records = []
    fk_calls = 0
    for id in ids:
        row = inventory[id]
        assert sha(row["path"]) == row["sha256"]
        q = np.loadtxt(row["path"], delimiter=",")
        q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=-1, keepdims=True)
        pos = []
        quat = []
        points = []
        rots = []
        for frame in q:
            d.qpos[:] = frame
            mujoco.mj_kinematics(m, d)
            fk_calls += 1
            body_com = d.xpos + np.einsum("bij,bj->bi", d.xmat.reshape(-1, 3, 3), m.body_ipos)
            com = (body_com * m.body_mass[:, None]).sum(0) / m.body_mass.sum()
            points.append(
                np.stack(
                    [
                        com,
                        d.xpos[hip],
                        d.geom_xpos[head],
                        *[d.geom_xpos[g] for g in hands],
                        *[d.xpos[b] for b in feet],
                    ]
                )
            )
            rots.append(d.xmat[owner_ids].reshape(6, 3, 3).copy())
            pos.append(d.xpos[1:].copy())
            quat.append(d.xquat[1:].copy())
        pos = np.array(pos)
        quat = np.array(quat)
        points = np.array(points)
        rots = np.array(rots)
        shift = np.r_[pos[0, 0, :2], pos[:, :, 2].min() - 0.1]
        pos -= shift
        points -= shift
        q[:, :3] -= shift
        x, event, traces = observations(points, rots, row["fps"])
        records.append(
            dict(
                id=id, row=row, q=q, pos=pos, quat=quat, x=x, event=event, traces=traces, rots=rots
            )
        )
    scales = np.percentile(np.concatenate([r["event"] for r in records[:100]]), 95, axis=0).clip(
        0.01
    )
    write(
        "observation-definition.json",
        dict(
            points=[
                "native mass-weighted COM",
                "pelvis body origin",
                "head visual mesh center",
                "left rubber hand mesh center",
                "right rubber hand mesh center",
                "left ankle roll body origin",
                "right ankle roll body origin",
            ],
            mass_kg=float(m.body_mass.sum()),
            model="Native MJCF inertial masses and local inertial centers; no hardware COM measurement",
            channels="7 point heights; 7 speed norms; 7 acceleration norms; 21 hip-relative point coordinates; 6 body angular-speed norms; COM velocity xyz; COM acceleration xyz; first two columns of pelvis rotation; pelvis yaw rate; route turn rate; 0.5-second chord deviation; hip xy speed",
            event_scales_95=scales.tolist(),
            fit_only=True,
        ),
    )
    shapes = {}
    innerroles = {}
    for s in json.loads(GEOMETRY.read_text())["shapes"]:
        shapes.setdefault(s["owner"], []).append(
            CollisionCapsule(tuple(s["start"]), tuple(s["end"]), s["radius"])
        )
        innerroles.setdefault(s["owner"], []).append(s["role"] == "native_primitive_subset")
    flags = torch.tensor([v for owner in shapes for v in innerroles[owner]])
    all_lo = []
    all_up = []
    all_cf = []
    all_specs = []
    all_features = []
    all_recipes = []
    all_event = []
    display = []
    analyses = []
    label_start = time.monotonic()
    for ix, r in enumerate(records):
        event = (np.minimum(r["event"] / scales, 3) * [0.4, 0.3, 0.3]).sum(-1)
        anchor_ids = event_anchors(event)
        fps = r["row"]["fps"]
        half = max(1, round(0.5 * fps))
        yaw = r["traces"]["root_yaw"]
        # Candidate cross-sections use actual path direction when moving, root yaw otherwise.
        v = r["traces"]["velocity"][:, 1]
        direction = np.where(r["traces"]["root_speed"] >= 0.05, np.arctan2(v[:, 1], v[:, 0]), yaw)
        specs, recipes, anchors = event_candidates(r["pos"][:, 0], direction, anchor_ids)
        a, b, rad, _ = body_capsules_world(r["pos"], r["quat"], names, capsules=shapes)
        pts, rad, cover = capsule_samples(
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(b, dtype=torch.float32),
            torch.tensor(rad, dtype=torch.float32),
            5,
        )
        mask = flags[None, :, None].expand(pts.shape[:-1])
        rflat = rad.flatten()
        cflat = cover.flatten()
        pflat = pts.reshape(-1, 3)
        inner = mask.flatten()
        lo = []
        up = []
        cf = []
        features = []
        events = []
        for station, t in enumerate(anchor_ids):
            root = r["pos"][:, 0]
            shortcut, st, en = chord_root(root, t, half)
            delta = torch.tensor(shortcut[st : en + 1] - root[st : en + 1], dtype=torch.float32)
            chord_points = pts[st : en + 1] + delta[:, None, None, :]
            # Move entry-pose capsule samples with the actual pelvis pose. No physical execution.
            hiprot = r["rots"][:, 0]
            entry = (pts[st].numpy() - root[st]) @ hiprot[st]
            held = (
                np.einsum("cnj,tij->tcni", entry, hiprot[st : en + 1])
                + root[st : en + 1, None, None, :]
            )
            held = torch.tensor(held, dtype=torch.float32)
            local = r["x"][st : en + 1]
            feat = np.r_[local.mean(0), local.std(0)]
            for spec in specs[station * 45 : (station + 1) * 45]:

                def sdf(p):
                    center = torch.tensor(spec[:3], dtype=torch.float32)
                    if spec[3] == 2:
                        return (p - center).norm(dim=-1) - 0.3
                    return box_sdf(
                        p,
                        center,
                        torch.tensor([0.15, 0.8, 0.1] if spec[3] == 0 else [0.3, 0.3, 0.4]),
                        torch.tensor(float(spec[4])),
                    )

                values = sdf(pflat)
                lo.append((values - rflat - cflat).amin())
                up.append((values[inner] - rflat[inner]).amin())
                cf.append(
                    torch.stack(
                        [
                            (sdf(c)[mask[st : en + 1]] - rad[st : en + 1][mask[st : en + 1]]).amin()
                            for c in [chord_points, held]
                        ]
                    )
                )
                features.append(feat)
                events.append(float(event[t]))
        all_lo.append(torch.stack(lo))
        all_up.append(torch.stack(up))
        all_cf.append(torch.stack(cf))
        all_specs.append(specs)
        all_recipes.append(recipes)
        all_features.append(features)
        all_event.append(events)
        sampled = np.linspace(0, len(r["q"]) - 1, min(60, len(r["q"]))).astype(int)
        metric = dict(
            id=r["id"],
            fps=fps,
            frames=len(r["q"]),
            anchor_frames=anchor_ids.tolist(),
            anchor_times=(anchor_ids / fps).tolist(),
            anchor_saliency=event[anchor_ids].tolist(),
            max_head_vertical_speed=float(np.abs(r["traces"]["velocity"][:, 2, 2]).max()),
            max_com_acceleration=float(
                np.linalg.norm(r["traces"]["acceleration"][:, 0], axis=-1).max()
            ),
            max_abs_turn_rate=float(np.abs(r["traces"]["turn_rate"]).max()),
        )
        analyses.append(metric)
        if ix >= 100:
            display.append(
                dict(
                    id=r["id"],
                    category=r["row"]["category"],
                    prompts=r["row"]["prompts"],
                    fps=fps,
                    frames=len(r["q"]),
                    times=(sampled / fps).tolist(),
                    qpos=r["q"][sampled].tolist(),
                    positions=r["pos"][sampled].tolist(),
                    anchors=anchors.tolist(),
                    events=metric,
                    motion_analysis=dict(
                        times=(np.arange(len(r["q"])) / fps).tolist(),
                        saliency=event.tolist(),
                        turn_rate=r["traces"]["turn_rate"].tolist(),
                        root_yaw=r["traces"]["root_yaw"].tolist(),
                        chord_deviation=r["traces"]["chord_deviation"].tolist(),
                        points=r["traces"]["points"].tolist(),
                        speed=np.linalg.norm(r["traces"]["velocity"], axis=-1).tolist(),
                        acceleration=np.linalg.norm(r["traces"]["acceleration"], axis=-1).tolist(),
                        angular_speed=r["traces"]["angular_speed"].tolist(),
                    ),
                )
            )
        write("progress.json", dict(labels=ix + 1, physics_steps=0))
        if time.monotonic() - start > 300:
            raise TimeoutError("fixed cap")
    label_seconds = time.monotonic() - label_start
    lower = torch.stack(all_lo)
    upper = torch.stack(all_up)
    cf = torch.stack(all_cf)
    clear = lower >= 0.02
    near = clear & (upper <= 0.12)
    contrast = clear & (cf.amin(-1) < 0)
    assert (lower <= upper + 1e-5).all()
    X = torch.tensor(np.asarray(all_features), dtype=torch.float32)
    mean = X[:100].mean((0, 1))
    scale = X[:100].std((0, 1)).clamp_min(0.1)
    X = (X - mean) / scale
    recipes = torch.tensor(np.asarray(all_recipes), dtype=torch.float32)
    encoded = torch.cat(
        [recipes[:, :, :3] / 3, nn.functional.one_hot(recipes[:, :, 3].long(), 3)], -1
    )
    event = torch.tensor(all_event)
    hindsight = 0.1 * clear + near.float() + 3 * contrast
    torch.save(
        dict(
            lower=lower,
            inner_upper=upper,
            cf_upper=cf,
            specs=torch.tensor(np.array(all_specs), dtype=torch.float32),
            features=X,
            recipes=encoded,
            event=event,
            mean=mean,
            scale=scale,
        ),
        OUT / "labels.pt",
    )
    write("display.json", display)
    write("motion-analysis.json", analyses)
    fits = []
    train_start = time.monotonic()
    for variant in ["geometry", "hindsight", "hindsight_no_motion"]:
        for seed in [801, 802]:
            torch.manual_seed(seed)
            model = LocalGenerator(variant != "hindsight_no_motion")
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            ts = time.monotonic()
            for step in range(200):
                batch = torch.randperm(100)[:16]
                target = near[batch].float() if variant == "geometry" else hindsight[batch]
                loss = loss_fn(model(X[batch], encoded[batch]), target, clear[batch])
                opt.zero_grad()
                loss.backward()
                opt.step()
                if time.monotonic() - start > 300:
                    raise TimeoutError("fixed cap")
            key = f"{variant}-{seed}"
            torch.save(dict(state=model.state_dict(), mean=mean, scale=scale), OUT / f"{key}.pt")
            fits.append(
                dict(
                    key=key,
                    variant=variant,
                    seed=seed,
                    updates=200,
                    parameters=sum(p.numel() for p in model.parameters()),
                    wall_seconds=time.monotonic() - ts,
                    sha256=sha(OUT / f"{key}.pt"),
                )
            )
    write("model-lock.json", fits)
    train_seconds = time.monotonic() - train_start
    prob = {
        "uniform": torch.ones(20, 225) / 225,
        "explicit_event": (0.1 + event[100:]) / (0.1 + event[100:]).sum(-1, keepdim=True),
        "analytic_target": hindsight[100:] / hindsight[100:].sum(-1, keepdim=True).clamp_min(1e-12),
    }
    with torch.no_grad():
        for f in fits:
            m = LocalGenerator(f["variant"] != "hindsight_no_motion")
            m.load_state_dict(torch.load(OUT / f"{f['key']}.pt", weights_only=True)["state"])
            m.eval()
            prob[f["key"]] = m(X[100:], encoded[100:]).softmax(-1)
    torch.save(prob, OUT / "probabilities.pt")
    outcomes = []
    for key, q in prob.items():
        proj, z = project_clearance(q, lower[100:], 0.02)
        for mode, dist in [("raw", q), ("projected", proj)]:
            for j, id in enumerate(ids[100:]):
                i = 100 + j
                c = clear[i]
                hit = upper[i] < 0
                outcomes.append(
                    dict(
                        model=key,
                        mode=mode,
                        id=id,
                        near=float((dist[j] * near[i]).sum()),
                        contrast=float((dist[j] * contrast[i]).sum()),
                        shortcut=float((dist[j] * (c & (cf[i, :, 0] < 0))).sum()),
                        posehold=float((dist[j] * (c & (cf[i, :, 1] < 0))).sum()),
                        clear=float((dist[j] * c).sum()),
                        penetration=float((dist[j] * hit).sum()),
                        unknown=float((dist[j] * (~c & ~hit)).sum()),
                        retained=float(z[j]),
                        abstained=bool(dist[j].sum() == 0),
                        contrast_cells=int(contrast[i].sum()),
                    )
                )
    write("outcomes.json", outcomes)
    with (OUT / "outcomes.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(outcomes[0]))
        w.writeheader()
        w.writerows(outcomes)
    summary = []
    for prefix in [
        "uniform",
        "explicit_event",
        "analytic_target",
        "geometry",
        "hindsight",
        "hindsight_no_motion",
    ]:
        for mode in ["raw", "projected"]:
            rr = [
                r
                for r in outcomes
                if r["mode"] == mode
                and (r["model"] == prefix or r["model"].rsplit("-", 1)[0] == prefix)
            ]
            summary.append(
                dict(
                    model=prefix,
                    mode=mode,
                    **{
                        k: float(np.mean([r[k] for r in rr]))
                        for k in [
                            "near",
                            "contrast",
                            "shortcut",
                            "posehold",
                            "clear",
                            "penetration",
                            "unknown",
                            "retained",
                        ]
                    },
                    abstentions=sum(r["abstained"] for r in rr),
                )
            )
    write("summary.json", summary)
    write(
        "receipt.json",
        dict(
            state="complete",
            fits=6,
            updates=1200,
            physics_steps=0,
            fk_calls=fk_calls,
            reference_geometry_queries=27000,
            counterfactual_geometry_queries=54000,
            label_seconds=label_seconds,
            training_seconds=train_seconds,
            wall_seconds=time.monotonic() - start,
            fit_zero_contrast=int((contrast[:100].sum(-1) == 0).sum()),
            dev_zero_contrast=int((contrast[100:].sum(-1) == 0).sum()),
            fit_zero_near=int((near[:100].sum(-1) == 0).sum()),
            dev_zero_near=int((near[100:].sum(-1) == 0).sum()),
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if _CREATED:
            write("failure.json", dict(error=repr(e), retry="No automatic retry"))
        raise
