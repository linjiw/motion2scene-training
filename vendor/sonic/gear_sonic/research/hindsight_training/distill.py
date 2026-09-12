"""Bounded offline student fit; requires an executed teacher shard, not reference replay."""

import argparse
import json
from pathlib import Path
import time

import torch

from gear_sonic.research.hindsight_training.records import load_teacher_shard
from gear_sonic.research.hindsight_training.runtime import load_release_checkpoint, sha, write_new
from gear_sonic.research.hindsight_training.student import (
    FrozenSonicDecoder,
    SceneTokenStudent,
    distillation_loss,
)


def main(args):
    if not 1 <= args.steps <= 200:
        raise ValueError("This prototype is bounded to 1–200 offline updates")
    plan = json.loads((args.packet / "plan.json").read_text())
    arrays = load_teacher_shard(
        args.shard,
        args.metadata,
        allowed_motion_ids=plan["train_ids"],
        decoder_sha256=sha(args.checkpoint),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    model = SceneTokenStudent(camera=False)
    decoder = FrozenSonicDecoder(load_release_checkpoint(args.checkpoint)["policy_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    fields = (
        "proprio",
        "start_goal_body",
        "obstacles_body",
        "obstacle_mask",
        "route_body",
        "route_mask",
    )
    tensors = {
        key: torch.from_numpy(arrays[key])
        for key in (*fields, "teacher_tokens", "teacher_action_mean")
    }
    write_new(
        args.output / "design.json",
        {
            "seed": args.seed,
            "updates": args.steps,
            "batch_size": 32,
            "checkpoint_sha256": sha(args.checkpoint),
            "shard_sha256": sha(args.shard),
            "loss": "token MSE + frozen decoder action-mean MSE",
            "camera": False,
            "evidence": "Offline imitation only; no physical navigation evaluation",
        },
    )
    started = time.perf_counter()
    with (args.output / "loss.jsonl").open("x") as log:
        for step in range(args.steps):
            indices = torch.randint(len(arrays["proprio"]), (32,))
            batch = {key: tensors[key][indices] for key in fields}
            prediction = model(**batch)
            losses = distillation_loss(
                prediction,
                tensors["teacher_tokens"][indices],
                tensors["teacher_action_mean"][indices],
                batch["proprio"],
                decoder,
            )
            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            log.write(
                json.dumps(
                    {
                        "step": step + 1,
                        **{key: float(value.detach()) for key, value in losses.items()},
                    }
                )
                + "\n"
            )
    torch.save(
        {"student": model.state_dict(), "decoder_sha256": sha(args.checkpoint)},
        args.output / "student.pt",
    )
    write_new(
        args.output / "receipt.json",
        {
            "updates": args.steps,
            "wall_seconds": time.perf_counter() - started,
            "checkpoint_sha256": sha(args.output / "student.pt"),
            "physics_steps": 0,
            "physical_passage": None,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "shard", "metadata", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=91144)
    main(parser.parse_args())
