"""Fresh development fits using source-training references and native geometric bounds."""

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from scripts.research.lflh_next.constrained import coverage_acceptance_loss, project_clearance

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT.parent / "research-data"
PREVIOUS = DATA / "m2s-native-development-labels-20260911"
OUT = DATA / "m2s-split-safe-development-20260911"
RECEIPT = Path("/home/linjiw/dataset/amass-licensed/provenance/phase_g_fleaven_source_receipt.tsv")
ANCESTRY = DATA / "m2s-motion-qualification-20260911/ancestry-ledger.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def eligible_rows(ledger, receipt):
    """Use explicit source training membership; absent/unknown roles fail closed."""
    ids = {}
    for row in receipt:
        key = row["climb_g1_stem"]
        if key in ids:
            raise ValueError("duplicate source receipt identity")
        ids[key] = row
    result = []
    for i, row in enumerate(ledger):
        record = ids.get(Path(row["path"]).stem)
        if record is None:
            raise ValueError("source identity absent")
        result.append(
            dict(
                index=i,
                path=row["path"],
                prior_role=row["original_split"],
                source_role=record["split"],
                eligible=record["split"] == "training",
                new_role=(
                    ("fit" if row["original_split"] == "train" else "development_score")
                    if record["split"] == "training"
                    else "excluded_reserved"
                ),
                expected_npz=record["expected_g1_sha256"],
                source_sha256=record["source_sha256"],
                source_path=record["hf_path"],
            )
        )
    return result


class Generator(nn.Module):
    def __init__(self, conditioned):
        super().__init__()
        self.conditioned = conditioned
        self.motion = nn.Sequential(nn.Linear(272, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.shape = nn.Sequential(nn.Linear(6, 64), nn.SiLU())
        self.score = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x, specs):
        if not self.conditioned:
            x = torch.zeros_like(x)
        h = self.motion(x)
        s = self.shape(specs)
        return self.score(
            torch.cat([h[:, None].expand(-1, len(s), -1), s[None].expand(len(h), -1, -1)], -1)
        ).squeeze(-1)


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    ledger = json.loads((PREVIOUS / "ledger.json").read_text())
    with RECEIPT.open() as f:
        receipt = list(csv.DictReader(f, delimiter="\t"))
    rows = eligible_rows(ledger, receipt)
    eligible = [r for r in rows if r["eligible"]]
    ancestry = {Path(r["npz"]).stem: r for r in json.loads(ANCESTRY.read_text())}
    for row in eligible:
        source = ancestry[Path(row["path"]).stem]
        assert sha(row["path"]) == row["expected_npz"]
        assert sha(source["retarget_path"]) == row["source_sha256"]
        row["verified_source_array"] = source["retarget_path"]
    fit = [r["index"] for r in rows if r["new_role"] == "fit"]
    dev = [r["index"] for r in rows if r["new_role"] == "development_score"]
    assert len(fit) == 56 and len(dev) == 14
    write("split-ledger.json", rows)
    write(
        "design.json",
        dict(
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            scope="Explicit development restart; native geometric labels; exclude ALL 10 source-evaluation references",
            correction="Eight earlier fit inputs and two earlier evaluation inputs were source-reserved. Original fits preserved and not reused.",
            prior_access="All prior panels inspected. This is not fresh held-out inference or retroactive repair of prior split contamination.",
            remaining_limits="Raw AMASS ancestry/subject identity and historical converter command not fully verified; no confirmatory or physical claim",
            models=["unconditional", "conditioned"],
            acceptance_weights=[1.0, 2.0],
            seeds=[601, 602],
            updates_each=200,
            learning_rate=0.001,
            batch_size=16,
            fit_files=56,
            score_files=14,
            features="Full-record body-coordinate mean/min/max plus stored duration and fps; 272 features; no 32-frame subsampling; order invariant, not temporal reasoning",
            metrics="Raw/projection near-sufficient, clear, penetration-witness, unknown probability; all target rows; zero-near rows retained",
            limits={
                "fits": 8,
                "optimizer_updates": 1600,
                "CPU_threads": 2,
                "wall_cap_s": 180,
                "physics_steps": 0,
            },
            bindings={
                "receipt": sha(RECEIPT),
                "native_labels": sha(PREVIOUS / "labels.pt"),
                "native_ledger": sha(PREVIOUS / "ledger.json"),
                "ancestry": sha(ANCESTRY),
                "source_code": sha(Path(__file__)),
            },
            fit_indices=fit,
            development_indices=dev,
        ),
    )
    start = time.monotonic()
    data = torch.load(PREVIOUS / "labels.pt", weights_only=True)
    # Reserved rows are excluded before constructing or normalizing any model input.
    indices = fit + dev
    features = []
    for i in indices:
        x = data["full_resolution_inputs"][i]["positions"].flatten(1).float()
        duration = data["full_resolution_inputs"][i]["duration_s"]
        features.append(
            torch.cat(
                [
                    x.mean(0),
                    x.amin(0),
                    x.amax(0),
                    torch.tensor([duration / 50, ledger[i]["fps"] / 50]),
                ]
            )
        )
    X = torch.stack(features)
    mean = X[:56].mean(0)
    scale = X[:56].std(0).clamp_min(0.1)
    X = (X - mean) / scale
    lower = data["lower"][indices]
    upper = data["inner_upper"][indices]
    clear = lower >= 0.02
    near = clear & (upper <= 0.12)
    specs = torch.cat(
        [data["specs"][:, :3] / 3, nn.functional.one_hot(data["specs"][:, 3].long(), 3)], -1
    )
    fits = []
    for conditioned in [False, True]:
        for weight in [1.0, 2.0]:
            for seed in [601, 602]:
                t = time.monotonic()
                torch.manual_seed(seed)
                model = Generator(conditioned)
                opt = torch.optim.Adam(model.parameters(), lr=0.001)
                for step in range(200):
                    ix = torch.randperm(56)[:16]
                    logits = model(X[ix], specs)
                    loss = coverage_acceptance_loss(
                        logits, near[ix], clear[ix], acceptance_weight=weight
                    )
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if time.monotonic() - start > 180:
                        raise TimeoutError("fixed development cap")
                key = f"{'conditioned' if conditioned else 'unconditional'}-w{int(weight)}-{seed}"
                path = OUT / f"{key}.pt"
                torch.save(dict(state=model.state_dict(), mean=mean, scale=scale), path)
                fits.append(
                    dict(
                        key=key,
                        conditioned=conditioned,
                        weight=weight,
                        seed=seed,
                        parameters=sum(p.numel() for p in model.parameters()),
                        updates=200,
                        wall_seconds=time.monotonic() - t,
                        sha256=sha(path),
                    )
                )
    write("model-lock.json", fits)
    outcomes = []
    prob = {}
    with torch.no_grad():
        for f in fits:
            model = Generator(f["conditioned"])
            model.load_state_dict(torch.load(OUT / f"{f['key']}.pt", weights_only=True)["state"])
            model.eval()
            prob[f["key"]] = model(X[56:], specs).softmax(-1)
        prob["uniform"] = torch.ones(14, len(specs)) / len(specs)
        for key, q in prob.items():
            projected, z = project_clearance(q, lower[56:], 0.02)
            for mode, dist in [("raw", q), ("projected", projected)]:
                for j in range(14):
                    c = clear[56 + j]
                    hit = upper[56 + j] < 0
                    outcomes.append(
                        dict(
                            model=key,
                            mode=mode,
                            target=dev[j],
                            near_mass=float((dist[j] * near[56 + j]).sum()),
                            clear_mass=float((dist[j] * c).sum()),
                            penetration_mass=float((dist[j] * hit).sum()),
                            unknown_mass=float((dist[j] * (~c & ~hit)).sum()),
                            retained_mass=float(z[j]),
                            near_cells=int(near[56 + j].sum()),
                            abstained=bool(dist[j].sum() == 0),
                        )
                    )
    write("outcomes.json", outcomes)
    torch.save(prob, OUT / "probabilities.pt")
    with (OUT / "outcomes.csv").open("w") as handle:
        w = csv.DictWriter(handle, fieldnames=list(outcomes[0]))
        w.writeheader()
        w.writerows(outcomes)
    summaries = []
    for key in [
        "uniform",
        "unconditional-w1",
        "unconditional-w2",
        "conditioned-w1",
        "conditioned-w2",
    ]:
        for mode in ["raw", "projected"]:
            rs = [
                r
                for r in outcomes
                if (r["model"] == key or r["model"].startswith(key + "-")) and r["mode"] == mode
            ]
            summaries.append(
                dict(
                    model=key,
                    mode=mode,
                    rows=len(rs),
                    **{
                        k: float(np.mean([r[k] for r in rs]))
                        for k in [
                            "near_mass",
                            "clear_mass",
                            "penetration_mass",
                            "unknown_mass",
                            "retained_mass",
                        ]
                    },
                    abstentions=sum(r["abstained"] for r in rs),
                )
            )
    write("summary.json", summaries)
    write(
        "receipt.json",
        dict(
            state="complete",
            wall_seconds=time.monotonic() - start,
            updates=1600,
            fits=8,
            outcome_rows=len(outcomes),
            physics_steps=0,
            fit_zero_near=int((near[:56].sum(-1) == 0).sum()),
            dev_zero_near=int((near[56:].sum(-1) == 0).sum()),
            excluded_source_evaluation=10,
        ),
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if OUT.exists():
            write("failure.json", {"error": repr(e), "retry": "No automatic retry"})
        raise
