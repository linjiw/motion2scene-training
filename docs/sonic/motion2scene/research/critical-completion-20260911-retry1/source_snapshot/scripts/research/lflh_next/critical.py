"""Exploratory, CPU-only critical-location learning and masked-field completion.

All labels are geometry-derived. No dynamics, pilot contexts, or protected layouts.
A withheld coordinate-block split is interpolation within one existing motion bank.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.swept_volume import (
    CollisionCapsule,
    body_capsules_world,
)  # noqa: E402
from scripts.research.lflh_next.benchmark import BANK, BINDINGS, bind, read  # noqa: E402
from scripts.research.lflh_next.geometry import box_sdf  # noqa: E402

PREVIOUS = ROOT / "docs/motion2scene/research/lflh-next-20260911/benchmark"
CONFIG = {
    "nx": 32,
    "nz": 48,
    "block": 4,
    "margin_m": 0.01,
    "seeds": [301, 302],
    "updates": 200,
    "threads": 2,
    "lr": 0.002,
    "variants": ["unconditional", "cover", "masked", "contrast", "combined"],
    "completion_weight": 0.3,
    "contrast_weight": 0.3,
    "visible_block_probability": 0.25,
    "geometry_wall_cap_s": 300,
    "training_wall_cap_s": 600,
    "half_extents": [0.15, 1.0, 0.1],
    "x_range": [1.0, 4.5],
    "underside_range": [0.7, 1.65],
}


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def partition(nx, nz, block):
    i, j = torch.meshgrid(torch.arange(nx), torch.arange(nz), indexing="ij")
    test = ((i // block + j // block) % 4) == 0
    return ~test, test


def critical_mask(lower, upper, margin):
    clear = lower >= margin
    penetrating = upper <= -margin
    # Clear and penetrating for the SAME motion are mutually exclusive for valid bounds.
    if (clear & penetrating).any():
        raise ValueError("inconsistent clearance bounds")
    return clear & penetrating.any(dim=0, keepdim=True)


def observations(fields, requested, train):
    allowed = requested & train[None]
    values = fields.masked_fill(~allowed[:, None], 0)
    return values, allowed[:, None].to(fields.dtype)


def completion_fields(lower, upper, dtype=torch.float32):
    """Normalize geometry targets into the explicitly declared network dtype."""
    return (torch.stack((lower, upper), 1).clamp(-0.2, 0.2) / 0.2).to(dtype)


def cover_loss(energy, positives, train):
    logits = energy[:, train]
    positive = positives[:, train]
    count = positive.sum(-1)
    supported = count > 0
    if not supported.any():
        return energy.sum() * 0
    target = positive[supported].to(energy.dtype) / count[supported, None]
    return -(target * logits[supported].log_softmax(-1)).sum(-1).mean()


def pair_loss(energy, lower, upper, train, margin):
    clear = (lower >= margin) & train
    blocked = (upper <= -margin) & train
    valid = clear[:, None] & blocked[None]
    if not valid.any():
        return energy.sum() * 0
    difference = energy[None] - energy[:, None]  # negative - positive
    return nn.functional.softplus(difference[valid]).mean()


class CriticalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion = nn.Sequential(nn.Flatten(), nn.Linear(96, 32), nn.SiLU())
        self.spatial = nn.Sequential(
            nn.Conv2d(37, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 3, 1),
        )

    def forward(self, profile, xy, observed, mask):
        context = self.motion(profile)[:, :, None, None].expand(-1, -1, *xy.shape[-2:])
        image = torch.cat((context, xy.expand(len(profile), -1, -1, -1), observed, mask), 1)
        result = self.spatial(image)
        return result[:, 0], result[:, 1:]


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(CONFIG["threads"])
    old = json.loads((PREVIOUS / "design.json").read_text())
    for ref in old["inputs"]:
        bind(ref["path"], ref["sha256"])
    for path in [
        Path(__file__),
        Path(__file__).with_name("geometry.py"),
        Path(__file__).with_name("critical_evaluate.py"),
    ]:
        bind(path)
    write(
        out / "design.json",
        {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": CONFIG,
            "input_bindings": list(BINDINGS.values()),
            "outcome_access": (
                "Prior geometric benchmark and development/partial pilot summaries inspected. "
                "New study exploratory."
            ),
            "primary": (
                "Mean probability mass on geometric critical cells across all seven targets on "
                "withheld coordinate blocks, no map observations at inference."
            ),
            "scope": (
                "One existing bank, no source holdout, geometry labels only, no physical "
                "utility inference."
            ),
            "split": (
                "4x4 coordinate blocks, (block_x+block_z)%4==0 withheld. No withheld field "
                "value used for training, masks, selection, or normalization."
            ),
            "comparators": [
                "uniform grid",
                "geometry-screened uniform with oracle query access",
                "critical-cell oracle",
                "old CNN Gaussian projected to grid",
            ],
            "budget": (
                "1536 new geometry placements x 7 existing motions; 10 fits x 200 updates. Zero "
                "simulator steps. No tuning or retries."
            ),
            "analysis": (
                "Descriptive exact finite-grid mass, per-target denominators; p=NA. Geometry "
                "and architecture changes confound comparison to old Gaussian."
            ),
        },
    )
    data = torch.load(PREVIOUS / "bank.pt", weights_only=True)
    bank = json.loads(BANK.read_text())
    geometry = read(read(bank["manifest"])["geometry"])
    shapes = {}
    for s in geometry["shapes"]:
        shapes.setdefault(s["owner"], []).append(
            CollisionCapsule(tuple(s["start"]), tuple(s["end"]), s["radius"])
        )
    radius = []
    for row in bank["rows"]:
        ref = read(row["evidence"])["artifacts"]["trajectory"]
        p = load_reset_capture(bind(ref["path"], ref["sha256"]))
        a, _, r, _ = body_capsules_world(
            p["body_pos_w"], p["body_quat_w"], p["body_names"], capsules=shapes
        )
        radius.append(
            torch.tensor(np.broadcast_to(r[None, :, None], (len(a), len(r), 5)).copy()).flatten()
        )
    radius = torch.stack(radius)
    x = 1 + (torch.arange(CONFIG["nx"]) + 0.5) / CONFIG["nx"] * 3.5
    z = 0.7 + (torch.arange(CONFIG["nz"]) + 0.5) / CONFIG["nz"] * 0.95
    xx, zz = torch.meshgrid(x, z, indexing="ij")
    coordinates = torch.stack((xx, zz), -1).reshape(-1, 2)
    lower, upper = [], []
    start = time.monotonic()
    for i in range(0, len(coordinates), 8):
        if time.monotonic() - start > CONFIG["geometry_wall_cap_s"]:
            write(
                out / "geometry-stopped.json",
                {"completed_placements": i, "wall_s": time.monotonic() - start},
            )
            raise RuntimeError("geometry cap reached; retained design, no retry")
        c = coordinates[i : i + 8]
        centre = torch.stack((c[:, 0], c[:, 0] * 0, c[:, 1] + 0.1), -1)
        sdf = box_sdf(
            data["dense_points"][None],
            centre[:, None, None],
            torch.tensor(CONFIG["half_extents"]),
            torch.tensor(0.0),
        )
        lower.append((sdf - data["dense_radius"][None]).amin(-1))
        upper.append((sdf - radius[None]).amin(-1))
        if i % 256 == 0:
            print(f"geometry placements {i+len(c)}/{len(coordinates)}", flush=True)
    lower = torch.cat(lower).T.reshape(7, CONFIG["nx"], CONFIG["nz"])
    upper = torch.cat(upper).T.reshape_as(lower)
    train, test = partition(CONFIG["nx"], CONFIG["nz"], CONFIG["block"])
    torch.save(
        {
            "lower": lower,
            "upper": upper,
            "train": train,
            "test": test,
            "coordinates": coordinates,
            "profile": data["profiles"],
            "names": old["names"],
        },
        out / "fields.pt",
    )
    critical = critical_mask(lower, upper, CONFIG["margin_m"])
    write(
        out / "geometry-receipt.json",
        {
            "geometry_placements": len(coordinates),
            "motion_placement_pairs": len(coordinates) * 7,
            "wall_seconds": time.monotonic() - start,
            "physics_steps": 0,
            "training_cells": int(train.sum()),
            "withheld_cells": int(test.sum()),
            "training_critical_counts": critical[:, train].sum(-1).tolist(),
            "fields_sha256": hashlib.sha256((out / "fields.pt").read_bytes()).hexdigest(),
            "note": (
                "Dense at 298 recorded frames, spatial capsule bounds; no temporal or dynamics "
                "certificate."
            ),
        },
    )


def fit(out):
    design = json.loads((out / "design.json").read_text())
    assert design["config"] == CONFIG
    for ref in design["input_bindings"]:
        bind(ref["path"], ref["sha256"])
    geometry_receipt = json.loads((out / "geometry-receipt.json").read_text())
    bind(out / "fields.pt", geometry_receipt["fields_sha256"])
    torch.set_num_threads(CONFIG["threads"])
    torch.use_deterministic_algorithms(True)
    d = torch.load(out / "fields.pt", weights_only=True)
    train = d["train"]
    # Explicitly blank withheld labels BEFORE constructing any training target or observation.
    lower = d["lower"].masked_fill(~train, 0)
    upper = d["upper"].masked_fill(~train, 0)
    target = critical_mask(lower, upper, CONFIG["margin_m"])
    field = completion_fields(lower, upper)
    profile = d["profile"]
    mu, sd = profile.mean((0, 1)), profile.std((0, 1)).clamp_min(0.05)
    profile = (profile - mu) / sd
    xy = d["coordinates"].T.reshape(1, 2, CONFIG["nx"], CONFIG["nz"])
    xy = (xy - torch.tensor([2.75, 1.175])[None, :, None, None]) / torch.tensor([1.75, 0.475])[
        None, :, None, None
    ]
    empty, mask0 = torch.zeros_like(field), torch.zeros_like(field[:, :1])
    write(
        out / "training-started.json",
        {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": "cpu",
            "torch": torch.__version__,
            "normalization_mean": mu.tolist(),
            "normalization_scale": sd.tolist(),
        },
    )
    start = time.monotonic()
    receipts = []
    for seed in CONFIG["seeds"]:
        for variant in CONFIG["variants"]:
            folder = out / f"{variant}-{seed}"
            folder.mkdir(exist_ok=False)
            torch.manual_seed(seed)
            model = CriticalNet()
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
            rng = torch.Generator().manual_seed(seed + 1000)
            x = torch.zeros_like(profile) if variant == "unconditional" else profile
            history, updates = [], 0
            state = "complete"
            t0 = time.monotonic()
            for step in range(CONFIG["updates"]):
                if time.monotonic() - start > CONFIG["training_wall_cap_s"]:
                    state = "wall_cap"
                    break
                energy, _ = model(x, xy, empty, mask0)
                cover = cover_loss(energy, target, train)
                contrast = (
                    pair_loss(energy, lower, upper, train, CONFIG["margin_m"])
                    if variant in ("contrast", "combined")
                    else energy.sum() * 0
                )
                completion = energy.sum() * 0
                if variant in ("masked", "combined"):
                    block = (
                        torch.rand(7, 1, CONFIG["nx"] // 4, CONFIG["nz"] // 4, generator=rng)
                        < CONFIG["visible_block_probability"]
                    )
                    requested = block.repeat_interleave(4, 2).repeat_interleave(4, 3)[:, 0]
                    observed, mask = observations(field, requested, train)
                    _, recovered = model(x, xy, observed, mask)
                    missing = (train[None] & ~requested)[:, None].expand_as(field)
                    completion = nn.functional.smooth_l1_loss(recovered[missing], field[missing])
                loss = (
                    cover
                    + CONFIG["contrast_weight"] * contrast
                    + CONFIG["completion_weight"] * completion
                )
                if not torch.isfinite(loss):
                    state = "nonfinite"
                    break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                updates += 1
                if step % 25 == 0 or step == CONFIG["updates"] - 1:
                    history.append(
                        {
                            "update": updates,
                            "cover": float(cover.detach()),
                            "contrast": float(contrast.detach()),
                            "completion": float(completion.detach()),
                        }
                    )
            torch.save(
                {"state_dict": model.state_dict(), "variant": variant, "mean": mu, "scale": sd},
                folder / "model.pt",
            )
            record = {
                "variant": variant,
                "seed": seed,
                "updates": updates,
                "state": state,
                "parameters": sum(p.numel() for p in model.parameters()),
                "wall_seconds": time.monotonic() - t0,
                "history": history,
            }
            write(folder / "receipt.json", record)
            receipts.append({k: v for k, v in record.items() if k != "history"})
            print(json.dumps(receipts[-1]), flush=True)
            if state != "complete":
                write(
                    out / "training-results.json",
                    {
                        "state": "stopped",
                        "fits": receipts,
                        "wall_seconds": time.monotonic() - start,
                    },
                )
                return
    write(
        out / "training-results.json",
        {
            "state": "complete",
            "fits": receipts,
            "wall_seconds": time.monotonic() - start,
            "physics_steps": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "fit"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else fit)(args.out)
