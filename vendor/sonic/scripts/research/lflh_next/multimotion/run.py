"""Bounded reference-geometry study; no executed-motion or native-collision claims."""

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

BANK = Path("/home/linjiw/dataset/amass-licensed/phase-g-bank-rebuilt-fmt8")
OUT = Path("/home/linjiw/research-data/m2s-multimotion-shapes-20260911")


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def candidates():
    rows = []
    for x in np.linspace(-1, 3, 5):
        for y in np.linspace(-1.5, 1.5, 5):
            for z in [0.4, 1.0, 1.6]:
                for kind in range(3):
                    rows.append([x, y, z, kind])
    return torch.tensor(rows, dtype=torch.float32)


def gaps(points, specs):
    delta = points[None] - specs[:, None, :3]
    values = []
    for i, row in enumerate(specs):
        d = delta[i]
        if row[3] == 2:
            sdf = d.norm(dim=-1) - 0.3
        else:
            half = torch.tensor([0.15, 0.8, 0.1] if row[3] == 0 else [0.3, 0.3, 0.4])
            q = d.abs() - half
            sdf = q.clamp_min(0).norm(dim=-1) + q.amax(-1).clamp_max(0)
        values.append((sdf - 0.1).amin())
    return torch.stack(values)


class Model(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.embed = nn.Linear(90, 64)
        self.position = nn.Parameter(torch.zeros(1, 32, 64))
        if kind == "transformer":
            layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True)
            self.temporal = nn.TransformerEncoder(layer, 2)
        else:
            self.temporal = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        self.query = nn.Sequential(nn.Linear(7, 64), nn.SiLU(), nn.Linear(64, 64))
        self.score = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x, spec):
        if self.kind == "unconditional":
            x = torch.zeros_like(x)
        h = self.temporal(self.embed(x) + self.position).mean(1)
        c = self.query(spec)
        return self.score(
            torch.cat([h[:, None].expand(-1, len(c), -1), c[None].expand(len(h), -1, -1)], -1)
        ).squeeze(-1)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    train = sorted(BANK.glob("KIT_*.npz"), key=lambda p: digest_name(p.name))[:64]
    test = sorted(BANK.glob("CMU_*.npz"), key=lambda p: digest_name(p.name))[:16]
    assert len(train) == 64 and len(test) == 16
    paths = train + test
    save(
        "design.json",
        dict(
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            status="New exploratory offline reference-geometry design; previous seven-motion outcomes accessible",
            scope=(
                "64 KIT training files;16 CMU evaluation files; collection-separated within "
                "this study, original ancestry and other-project holdouts not certified"
            ),
            geometry=(
                "Union of 0.10m spheres at 30 converted body origins at "
                "every supplied frame; not native robot capsules or continuous-time collision"
            ),
            target=(
                "near-boundary CLEAR proxy: minimum sphere gap between "
                ".02 and .12m inclusive; not executed-contrast criticality"
            ),
            normalization=(
                "Translate first body initial xy to zero; shift minimum "
                "body z to .10m; no rotation or articulation edits"
            ),
            shapes={
                "beam_half": [0.15, 0.8, 0.1],
                "box_half": [0.3, 0.3, 0.4],
                "sphere_radius": 0.3,
            },
            budget={
                "fits": 6,
                "updates_each": 200,
                "seeds": [501, 502],
                "CPU_threads": 2,
                "wall_cap_s": 600,
                "physics_steps": 0,
            },
            models=["unconditional", "mlp", "transformer"],
            selection=(
                "SHA256 filename order, predetermined collection counts, no "
                "outcome selection; fail without replacement on malformed data"
            ),
            privacy=(
                "All motion-derived arrays and models remain outside "
                "repository and are not served or published"
            ),
            source_bindings=[
                dict(path=str(p), sha256=digest(p), split="train" if i < 64 else "test")
                for i, p in enumerate(paths)
            ],
            code_sha256=digest(Path(__file__)),
        ),
    )
    start = time.monotonic()
    spec = candidates()
    features = []
    fields = []
    audit = []
    for i, p in enumerate(paths):
        with np.load(p, allow_pickle=False) as a:
            x = torch.tensor(a["body_pos_w"].copy())
            fps = float(a["fps"][0])
        assert (
            x.ndim == 3
            and x.shape[1:] == (30, 3)
            and torch.isfinite(x).all()
            and len(x) >= 2
            and fps > 0
        )
        x[:, :, :2] -= x[0, 0, :2].clone()
        x[:, :, 2] -= x[:, :, 2].amin() - 0.1
        idx = torch.linspace(0, len(x) - 1, 32).round().long()
        features.append(x[idx].flatten(1))
        fields.append(gaps(x.reshape(-1, 3), spec))
        audit.append(
            dict(
                name=p.name,
                split="train" if i < 64 else "test",
                frames=len(x),
                fps=fps,
                status="geometry_complete",
            )
        )
        if time.monotonic() - start > 600:
            raise TimeoutError("fixed study wall cap")
    X = torch.stack(features)
    G = torch.stack(fields)
    near = (G >= 0.02) & (G <= 0.12)
    clear = G >= 0.02
    mean = X[:64].mean((0, 1), keepdim=True)
    scale = X[:64].std((0, 1), keepdim=True).clamp_min(0.1)
    X = (X - mean) / scale
    S = torch.cat(
        [
            spec[:, :3] / 3,
            torch.nn.functional.one_hot(spec[:, 3].long(), 3),
            torch.ones(len(spec), 1),
        ],
        1,
    )
    torch.save(
        dict(features=X, gaps=G, specs=spec, near=near, mean=mean, scale=scale), OUT / "geometry.pt"
    )
    save("motion-ledger.json", audit)
    save(
        "geometry-receipt.json",
        dict(
            wall_seconds=time.monotonic() - start,
            files=80,
            frames=sum(r["frames"] for r in audit),
            motion_shape_pairs=80 * len(spec),
        ),
    )
    fits = []
    for kind in ["unconditional", "mlp", "transformer"]:
        for seed in [501, 502]:
            t = time.monotonic()
            torch.manual_seed(seed)
            model = Model(kind)
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            for step in range(200):
                indices = torch.randperm(64)[:16]
                logq = model(X[indices], S).log_softmax(-1)
                labels = near[indices]
                counts = labels.sum(-1)
                supported = counts > 0
                cover = -(logq * labels).sum(-1) / counts.clamp_min(1)
                logz = logq.masked_fill(~clear[indices], -torch.inf).logsumexp(-1)
                valid = clear[indices].any(-1)
                loss = cover[supported].mean() if supported.any() else logq.sum() * 0
                if valid.any():
                    loss = loss - logz[valid].mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if time.monotonic() - start > 600:
                    raise TimeoutError("fixed study wall cap")
            path = OUT / f"{kind}-{seed}.pt"
            torch.save(model.state_dict(), path)
            fits.append(
                dict(
                    kind=kind,
                    seed=seed,
                    updates=200,
                    parameters=sum(p.numel() for p in model.parameters()),
                    wall_seconds=time.monotonic() - t,
                    sha256=digest(path),
                )
            )
    save("model-lock.json", fits)
    rows = []
    prob = {}
    with torch.no_grad():
        for fit in fits:
            model = Model(fit["kind"])
            model.load_state_dict(
                torch.load(OUT / f"{fit['kind']}-{fit['seed']}.pt", weights_only=True)
            )
            model.eval()
            q = model(X[64:], S).softmax(-1)
            prob[f"{fit['kind']}-{fit['seed']}"] = q
            for j in range(16):
                for mode in ["raw", "projected"]:
                    z = (q[j] * clear[64 + j]).sum()
                    p = q[j] if mode == "raw" else q[j] * clear[64 + j] / z.clamp_min(1e-30)
                    rows.append(
                        dict(
                            model=fit["kind"],
                            seed=fit["seed"],
                            target=paths[64 + j].name,
                            mode=mode,
                            near_mass=float((p * near[64 + j]).sum()),
                            clear_mass=float((p * clear[64 + j]).sum()),
                            penetration_mass=float((p * (G[64 + j] < 0)).sum()),
                            retained=float(z),
                            near_cells=int(near[64 + j].sum()),
                            abstained=bool(p.sum() == 0),
                        )
                    )
        for j in range(16):
            q = torch.ones(len(spec)) / len(spec)
            for mode in ["raw", "projected"]:
                z = (q * clear[64 + j]).sum()
                p = q if mode == "raw" else q * clear[64 + j] / z.clamp_min(1e-30)
                rows.append(
                    dict(
                        model="uniform",
                        seed="NA",
                        target=paths[64 + j].name,
                        mode=mode,
                        near_mass=float((p * near[64 + j]).sum()),
                        clear_mass=float((p * clear[64 + j]).sum()),
                        penetration_mass=float((p * (G[64 + j] < 0)).sum()),
                        retained=float(z),
                        near_cells=int(near[64 + j].sum()),
                        abstained=bool(p.sum() == 0),
                    )
                )
    torch.save(prob, OUT / "probabilities.pt")
    with (OUT / "outcomes.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for kind in ["uniform", "unconditional", "mlp", "transformer"]:
        for mode in ["raw", "projected"]:
            r = [r for r in rows if r["model"] == kind and r["mode"] == mode]
            summary.append(
                dict(
                    model=kind,
                    mode=mode,
                    **{
                        k: float(np.mean([v[k] for v in r]))
                        for k in ["near_mass", "clear_mass", "penetration_mass", "retained"]
                    },
                    abstentions=sum(v["abstained"] for v in r),
                )
            )
    save("summary.json", summary)
    save(
        "receipt.json",
        dict(
            state="complete",
            wall_seconds=time.monotonic() - start,
            updates=1200,
            fits=6,
            physics_steps=0,
            rows=len(rows),
            train_no_near=int((near[:64].sum(-1) == 0).sum()),
            test_no_near=int((near[64:].sum(-1) == 0).sum()),
        ),
    )
    print(json.dumps(summary, indent=2))


def digest_name(s):
    return hashlib.sha256(s.encode()).hexdigest()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if OUT.exists():
            save("failure.json", dict(error=repr(e), retry="No automatic retry or replacement"))
        raise
