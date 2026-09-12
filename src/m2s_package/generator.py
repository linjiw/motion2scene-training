"""Bounded replay of learned obstacle scoring using preserved geometry labels."""

import time

from m2s_package.cli import digest, save


def fit(labels_path, output, updates):
    import torch
    from scripts.research.lflh_next.navigation.hindsight_study import (
        LocalGenerator,
        loss_fn,
    )

    if not 1 <= updates <= 1000:
        raise ValueError("Use 1..1000 bounded updates")
    torch.set_num_threads(2)
    torch.manual_seed(801)
    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    clear = labels["lower"] >= 0.02
    near = clear & (labels["inner_upper"] <= 0.12)
    contrast = clear & (labels["cf_upper"].amin(-1) < 0)
    target = 0.1 * clear + near.float() + 3 * contrast
    x, recipes = labels["features"], labels["recipes"]
    if len(x) != 120:
        raise ValueError("Expected frozen 100-train/20-development tensor corpus")
    output.mkdir(parents=True, exist_ok=False)
    model = LocalGenerator(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    started = time.monotonic()
    for step in range(updates):
        if time.monotonic() - started > 300:
            raise TimeoutError("Bounded generator training exceeded 300 seconds")
        batch = torch.randperm(100)[:16]
        loss = loss_fn(model(x[batch], recipes[batch]), target[batch], clear[batch])
        if not torch.isfinite(loss):
            raise ValueError("Non-finite generator loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(
        {"state": model.state_dict(), "mean": labels["mean"], "scale": labels["scale"]},
        output / "generator.pt",
    )
    save(
        output / "receipt.json",
        {
            "state": "complete",
            "updates": updates,
            "seed": 801,
            "labels_sha256": digest(labels_path),
            "checkpoint_sha256": digest(output / "generator.pt"),
            "last_training_loss": float(loss.detach()),
            "physics_steps": 0,
            "wall_seconds": time.monotonic() - started,
            "claim": "Offline obstacle-scoring training only; no physical scene qualification",
        },
    )
