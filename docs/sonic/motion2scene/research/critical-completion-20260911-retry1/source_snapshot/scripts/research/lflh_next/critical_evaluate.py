"""Score frozen critical-location models once on withheld coordinate blocks.

Geometry-only interpolation. Oracle baselines have additional query access.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.lflh_next.benchmark import Encoder, bind  # noqa: E402
from scripts.research.lflh_next.critical import (  # noqa: E402
    CONFIG,
    PREVIOUS,
    CriticalNet,
    critical_mask,
    write,
)


def summarize(logits, lower, upper, critical):
    # A fixed subset of coordinates defines the queried domain, before seeing labels.
    q = logits.softmax(-1)
    clear = lower >= CONFIG["margin_m"]
    blocked = upper <= -CONFIG["margin_m"]
    unknown = ~clear & ~blocked
    return q, {
        "critical_mass": float((q * critical).sum()),
        "clear_mass": float((q * clear).sum()),
        "penetration_witness_mass": float((q * blocked).sum()),
        "uncertain_mass": float((q * unknown).sum()),
        "effective_grid_cells": float(torch.exp(-(q * q.clamp_min(1e-30).log()).sum())),
    }


def run(out):
    start = time.monotonic()
    torch.set_num_threads(2)
    train_result = json.loads((out / "training-results.json").read_text())
    if train_result["state"] != "complete" or len(train_result["fits"]) != 10:
        raise ValueError("all intended policies must be complete before evaluation")
    design = json.loads((out / "design.json").read_text())
    for ref in design["input_bindings"]:
        bind(ref["path"], ref["sha256"])
    bind(
        out / "fields.pt", json.loads((out / "geometry-receipt.json").read_text())["fields_sha256"]
    )
    models = [out / f"{r['variant']}-{r['seed']}" / "model.pt" for r in train_result["fits"]]
    write(
        out / "evaluation-lock.json",
        {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": [
                {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in models
            ],
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "status": "locked before evaluator reads withheld fields",
        },
    )
    d = torch.load(out / "fields.pt", weights_only=True)
    lower, upper, test = d["lower"], d["upper"], d["test"]
    critical = critical_mask(lower, upper, CONFIG["margin_m"])
    xy = d["coordinates"].T.reshape(1, 2, CONFIG["nx"], CONFIG["nz"])
    xy = (xy - torch.tensor([2.75, 1.175])[None, :, None, None]) / torch.tensor([1.75, 0.475])[
        None, :, None, None
    ]
    empty = torch.zeros(7, 2, CONFIG["nx"], CONFIG["nz"])
    mask0 = torch.zeros_like(empty[:, :1])
    records, probability_arrays = [], {}

    def record(method, seed, condition, energies):
        probs = []
        for i, name in enumerate(d["names"]):
            q, metrics = summarize(
                energies[i, test], lower[i, test], upper[i, test], critical[i, test]
            )
            row = {
                "method": method,
                "seed": seed,
                "condition": condition,
                "target": name,
                "test_cells": int(test.sum()),
                "critical_cells": int(critical[i, test].sum()),
                "training_critical_cells": int(critical[i, d["train"]].sum()),
                "abstained": False,
                **metrics,
            }
            records.append(row)
            probs.append(q)
        probability_arrays[f"{method}-{seed}-{condition}"] = torch.stack(probs)

    with torch.no_grad():
        for fit in train_result["fits"]:
            saved = torch.load(
                out / f"{fit['variant']}-{fit['seed']}" / "model.pt", weights_only=True
            )
            model = CriticalNet()
            model.load_state_dict(saved["state_dict"])
            model.eval()
            profile = (d["profile"] - saved["mean"]) / saved["scale"]
            if fit["variant"] == "unconditional":
                profile.zero_()
            for condition, x in [("conditioned", profile), ("input_rotated", profile.roll(1, 0))]:
                e, _ = model(x, xy, empty, mask0)
                record(fit["variant"], fit["seed"], condition, e)
        record("uniform_grid", "NA", "conditioned", torch.zeros_like(lower))
        for method, eligible in [
            ("geometry_screened_uniform", lower >= 0.01),
            ("critical_oracle", critical),
        ]:
            for i, name in enumerate(d["names"]):
                allowed = eligible[i, test]
                if allowed.any():
                    _q, metrics = summarize(
                        torch.zeros_like(lower[i, test]).masked_fill(~allowed, -torch.inf),
                        lower[i, test],
                        upper[i, test],
                        critical[i, test],
                    )
                else:
                    metrics = {
                        "critical_mass": 0.0,
                        "clear_mass": 0.0,
                        "penetration_witness_mass": 0.0,
                        "uncertain_mass": 0.0,
                        "effective_grid_cells": 0.0,
                    }
                records.append(
                    {
                        "method": method,
                        "seed": "NA",
                        "condition": "conditioned",
                        "target": name,
                        "test_cells": int(test.sum()),
                        "critical_cells": int(critical[i, test].sum()),
                        "training_critical_cells": int(critical[i, d["train"]].sum()),
                        "abstained": not bool(allowed.any()),
                        **metrics,
                    }
                )
        # Comparison to previous checkpoint is a package comparison, not a matched loss ablation.
        for seed in [101, 102]:
            saved = torch.load(PREVIOUS / f"conv-{seed}" / "model.pt", weights_only=True)
            model = Encoder("conv")
            model.load_state_dict(saved["state_dict"])
            model.eval()
            profile = (d["profile"] - saved["mean"]) / saved["scale"]
            mean, log_sigma = model(profile)
            u = (d["coordinates"] - torch.tensor([1.0, 0.7])) / torch.tensor([3.5, 0.95])
            latent = torch.logit(u)
            logpdf = (
                -(latent[None] - mean[:, None]).square() / (2 * (2 * log_sigma[:, None]).exp())
                - log_sigma[:, None]
            )
            # Change of variables to bounded scene coordinates; grid-cell area is constant.
            logpdf = logpdf.sum(-1) - (u.log() + torch.log1p(-u)).sum(-1)[None]
            record("old_gaussian_cnn", seed, "conditioned", logpdf.reshape_as(lower))
    with (out / "per-target.csv").open("x") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    torch.save(probability_arrays, out / "probabilities.pt")
    summary = []
    for key in dict.fromkeys((r["method"], r["seed"], r["condition"]) for r in records):
        rows = [r for r in records if (r["method"], r["seed"], r["condition"]) == key]
        summary.append(
            {
                "method": key[0],
                "seed": key[1],
                "condition": key[2],
                "targets": len(rows),
                "critical_mass": sum(r["critical_mass"] for r in rows) / len(rows),
                "clear_mass": sum(r["clear_mass"] for r in rows) / len(rows),
                "penetration_witness_mass": sum(r["penetration_witness_mass"] for r in rows)
                / len(rows),
                "abstentions": sum(r["abstained"] for r in rows),
            }
        )
    write(
        out / "evaluation-results.json",
        {
            "rows": summary,
            "wall_seconds": time.monotonic() - start,
            "withheld_critical_counts": critical[:, test].sum(-1).tolist(),
            "physics_steps": 0,
            "p_value": None,
            "reason": (
                "One-bank coordinate interpolation, no source-level exchangeability or physical "
                "labels."
            ),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
