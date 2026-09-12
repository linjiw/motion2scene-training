"""Export trusted bank motions and frozen distributions to an offline 3D viewer.

Run from repository root with python -m scripts.research.lflh_next.build_viewer.
No simulator, retraining, outcome collection, or network requests.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from plotly.offline import get_plotlyjs
import torch

from scripts.research.lflh_next.benchmark import (
    BANK,
    BINDINGS,
    CollisionCapsule,
    bind,
    body_capsules_world,
    load_reset_capture,
    read,
)
from scripts.research.lflh_next.constrained import project_clearance

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "docs/motion2scene/research/critical-completion-20260911-retry1"


def write(path, obj):
    path.write_text(json.dumps(obj, allow_nan=False, separators=(",", ":")))


def build(out):
    start = time.monotonic()
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=False)
    previous = json.loads((SOURCE / "manifest.json").read_text())
    for ref in previous["artifacts"]:
        bind(ref["path"], ref["sha256"])
    write(
        out / "design.json",
        {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "Outcome-motivated exploratory sampler amendment, not original pilot registration",
            "outcome_access": (
                "All previous geometric evaluation and "
                "reported physical development outcomes accessible"
            ),
            "intervention": (
                "Normalize frozen q on lower capsule clearance "
                ">= 0.01 m; abstain on zero retained mass"
            ),
            "comparators": "Every five-arm checkpoint, both seeds, plus raw and screened uniform",
            "domain": "Same previously inspected 384 coordinate cells and all seven motions; no new holdout claim",
            "budget": "0 training updates, 0 physics steps, one finite cached-field evaluation",
            "primary": (
                "Report critical probability together with "
                "retained probability, abstention and penetration probability"
            ),
            "inference": (
                "Descriptive finite-domain probabilities; p=NA, no "
                "exchangeability scheme or independent source sample"
            ),
            "inputs": list(BINDINGS.values()),
        },
    )
    d = torch.load(SOURCE / "fields.pt", weights_only=True)
    probabilities = torch.load(SOURCE / "probabilities.pt", weights_only=True)
    test = d["test"]
    lower, upper = d["lower"][:, test], d["upper"][:, test]
    critical = (lower >= 0.01) & (upper <= -0.01).any(0)
    coordinates = d["coordinates"][test.flatten()].tolist()
    distributions, rows = {}, []
    for method in ["uniform_grid", "unconditional", "cover", "masked", "contrast", "combined"]:
        seeds = ["NA"] if method == "uniform_grid" else [301, 302]
        for seed in seeds:
            key = f"{method}-{seed}"
            q = probabilities[key + "-conditioned"].double()
            projected, retained = project_clearance(q, lower)
            distributions[key] = {
                "raw": q.tolist(),
                "constrained": projected.tolist(),
                "retained": retained.tolist(),
            }
            for mode, dist in [("raw", q), ("constrained", projected)]:
                for i, name in enumerate(d["names"]):
                    rows.append(
                        dict(
                            method=method,
                            seed=seed,
                            mode=mode,
                            target=name,
                            critical_mass=float((dist[i] * critical[i]).sum()),
                            clear_mass=float((dist[i] * (lower[i] >= 0.01)).sum()),
                            penetration_mass=float((dist[i] * (upper[i] <= -0.01)).sum()),
                            retained_mass=float(retained[i]),
                            abstained=bool(dist[i].sum() == 0),
                            p=None,
                        )
                    )
    with (out / "outcomes.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for method in ["uniform_grid", "unconditional", "cover", "masked", "contrast", "combined"]:
        for mode in ["raw", "constrained"]:
            subset = [r for r in rows if r["method"] == method and r["mode"] == mode]
            summary.append(
                dict(
                    method=method,
                    mode=mode,
                    rows=len(subset),
                    **{
                        k: float(np.mean([r[k] for r in subset]))
                        for k in [
                            "critical_mass",
                            "clear_mass",
                            "penetration_mass",
                            "retained_mass",
                        ]
                    },
                    abstentions=sum(r["abstained"] for r in subset),
                )
            )
    write(out / "summary.json", summary)
    source = json.loads(bind(BANK).read_text())
    manifest = read(source["manifest"])
    geometry = read(manifest["geometry"])
    shapes = {}
    for shape in geometry["shapes"]:
        shapes.setdefault(shape["owner"], []).append(
            CollisionCapsule(tuple(shape["start"]), tuple(shape["end"]), shape["radius"])
        )
    motions = []
    for i, row in enumerate(source["rows"]):
        assert row["qualified"] and row["cell_id"] == d["names"][i]
        evidence = read(row["evidence"])
        ref = evidence["artifacts"]["trajectory"]
        payload = load_reset_capture(bind(ref["path"], ref["sha256"]))
        a, b, r, _ = body_capsules_world(
            payload["body_pos_w"], payload["body_quat_w"], payload["body_names"], capsules=shapes
        )
        motions.append(
            dict(
                name=row["cell_id"],
                a=np.round(a, 5).tolist(),
                b=np.round(b, 5).tolist(),
                radius=np.round(r, 5).tolist(),
                root=np.round(payload["root_pos_w"], 5).tolist(),
                fps=float(payload["fps"]),
                frames=len(a),
            )
        )
    data = dict(
        motions=motions,
        coordinates=coordinates,
        distributions=distributions,
        lower=lower.tolist(),
        upper=upper.tolist(),
        critical=critical.tolist(),
        half_extents=[0.15, 1.0, 0.1],
        margin=0.01,
        scope=(
            "Previously inspected 384 coordinate cells; 2D placement family "
            "rendered in 3D. Geometric capsules, not physical passage labels."
        ),
    )
    write(out / "data.json", data)
    template = Path(__file__).with_name("viewer") / "index.html"
    js = Path(__file__).with_name("viewer") / "app.js"
    html = (
        template.read_text()
        .replace("/*PLOTLY*/", get_plotlyjs())
        .replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
        .replace("/*APP*/", js.read_text())
    )
    (out / "index.html").write_text(html)
    write(
        out / "receipt.json",
        dict(
            state="complete",
            wall_seconds=time.monotonic() - start,
            new_physics_steps=0,
            new_training_updates=0,
            cached_clearance_pairs=int(lower.numel()),
            new_clearance_queries=0,
            conceptual_generation_queries_per_motion=len(coordinates),
            motion_frames=[m["frames"] for m in motions],
            outcome_rows=len(rows),
            note=(
                "Cached fields reused. Production projection requires geometry "
                "queries; this is not free neural inference."
            ),
            inputs=list(BINDINGS.values()),
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out.resolve())
