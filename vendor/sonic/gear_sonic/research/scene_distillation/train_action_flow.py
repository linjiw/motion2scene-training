"""Bounded direct-action/flow experiment, sharing the validated teacher data loader."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.action_flow import CONTEXT_SHAPES, ActionStudent
from gear_sonic.research.scene_distillation.commands import PROFILES
from gear_sonic.research.scene_distillation.train import load_episodes


def profile_mask(available, profile):
    if profile == "context":
        return torch.zeros_like(available)
    if profile not in PROFILES:
        raise ValueError("Unknown experimental command profile")
    mask = torch.zeros_like(available)
    mask[:, list(PROFILES[profile])] = True
    return mask & available


def initialize_student(model, saved, *, encoder_only=False):
    if encoder_only:
        if saved["config"].get("foundation_architecture") != "transformer":
            raise ValueError("Expected transformer encoder initialization")
        return {"copied_encoder_keys": model.condition.initialize_motion_encoder(saved["model"])}
    own, source = model.state_dict(), saved["model"]
    unexpected = set(source) - set(own)
    missing = set(own) - set(source)
    allowed = ("condition.goal_", "condition.obstacle_", "condition.context_", "residual.")
    if unexpected or any(not k.startswith(allowed) for k in missing):
        raise ValueError("Unapproved architecture migration")
    if any(own[k].shape != v.shape for k, v in source.items()):
        raise ValueError("Initialization tensor shape mismatch")
    model.load_state_dict({**own, **source}, strict=True)
    return {"copied_tensors": len(source), "new_tensors": sorted(missing)}


def fit(config, output):
    if not 1 <= config["updates"] <= 100000 or not 0 < config["wall_cap_seconds"] <= 21600:
        raise ValueError("Invalid bounded fit budget")
    if not 0 < config["learning_rate"] <= 0.01 or not 1 <= config["batch_size"] <= 4096:
        raise ValueError("Invalid optimization configuration")
    if sha(config["teacher_checkpoint"]) != config["teacher_sha256"]:
        raise ValueError("Teacher hash mismatch")
    if sha(config["dataset_manifest"]) != config["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest hash mismatch")
    episodes = load_episodes(
        config["dataset_manifest"],
        config["teacher_sha256"],
        set(config["train_ids"]),
        "foundation",
        allow_exploratory_queries=config.get("allow_exploratory_queries", False),
        allow_teacher_prefixes=config.get("allow_teacher_prefixes", False),
    )
    context_enabled = config["model"].get("context", False)
    if context_enabled:
        manifest = json.loads(Path(config["dataset_manifest"]).read_text())
        if manifest.get("context_schema") != "complete_known_map_goal_v1":
            raise ValueError("Missing explicit context provenance/schema")
        for episode in episodes:
            for key, shape in CONTEXT_SHAPES.items():
                if episode[key].shape != (len(episode["proprio"]), *shape):
                    raise ValueError("Invalid context collection shape")
    profiles, weights = zip(*config["profile_weights"].items())
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Profile weights must be a probability distribution")
    if "context" in profiles and not context_enabled:
        raise ValueError("Context-only training requires context inputs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "config.json", config)
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(config["seed"])
    device = torch.device(config.get("device", "cuda"))
    model = ActionStudent(**config["model"]).to(device)
    model.set_normalization(torch.cat([e["teacher_actions"] for e in episodes]).to(device))
    saved = None
    if config.get("initialize_checkpoint"):
        if sha(config["initialize_checkpoint"]) != config["initialize_sha256"]:
            raise ValueError("Initialization hash mismatch")
        saved = torch.load(config["initialize_checkpoint"], map_location=device, weights_only=False)
        if saved["teacher_sha256"] != config["teacher_sha256"]:
            raise ValueError("Initialization teacher mismatch")
        migration = initialize_student(model, saved, encoder_only=config.get("encoder_only", False))
        write_new(output / "initialization.json", migration)
    residual_only = config.get("residual_only", False)
    if residual_only:
        if model.residual is None or saved is None:
            raise ValueError("Residual experiment needs an initialized base and a residual head")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("residual."))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"]
    )
    if config.get("continue_optimizer", False):
        if saved is None or saved["config"]["model"] != config["model"]:
            raise ValueError("Optimizer continuation requires identical architecture")
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            if group["lr"] != config["learning_rate"]:
                raise ValueError("Optimizer continuation changed learning rate")
    groups = [
        [e for e in episodes if bool(e["_exploratory_source"][0]) == flag] for flag in [False, True]
    ]
    groups = [g for g in groups if g]
    fields = ["proprio", "controls", "control_mask", "teacher_actions"]
    if context_enabled:
        fields += list(CONTEXT_SHAPES)
    started, step, state = time.monotonic(), 0, "running"
    try:
        with (output / "metrics.jsonl").open("x") as log:
            for step in range(1, config["updates"] + 1):
                if time.monotonic() - started > config["wall_cap_seconds"]:
                    raise TimeoutError("Bounded fit wall cap reached")
                # One mode per minibatch of independently sampled episode rows.
                profile = str(rng.choice(profiles, p=weights))
                samples = []
                for _ in range(config["batch_size"]):
                    group = groups[int(rng.integers(len(groups)))]
                    episode = group[int(rng.integers(len(group)))]
                    index = int(rng.integers(len(episode["proprio"])))
                    samples.append({k: episode[k][index] for k in fields})
                batch = {k: torch.stack([s[k] for s in samples]).to(device) for k in fields}
                mask = profile_mask(batch["control_mask"], profile)
                context = {k: batch[k] for k in CONTEXT_SHAPES} if context_enabled else None
                losses = model.imitation_loss(
                    batch, mask, context=context, residual_only=residual_only
                )
                if not torch.isfinite(losses["loss"]):
                    raise ValueError("Nonfinite training loss")
                optimizer.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if step % 10 == 0 or step == 1:
                    log.write(
                        json.dumps(
                            {
                                "step": step,
                                "profile": profile,
                                **{k: float(v.detach()) for k, v in losses.items()},
                            }
                        )
                        + "\n"
                    )
                    log.flush()
                if step % config.get("save_interval", 1000) == 0 or step == config["updates"]:
                    torch.save(
                        {
                            "stage": "action_distillation",
                            "config": config,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": step,
                            "teacher_sha256": config["teacher_sha256"],
                            "rng_torch": torch.get_rng_state(),
                            "rng_numpy": rng.bit_generator.state,
                            "rng_cuda": (
                                torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                            ),
                            "qualification": "unqualified_research_student",
                        },
                        output / f"step-{step:06d}.pt",
                    )
            state = "complete"
    finally:
        write_new(
            output / "receipt.json",
            {
                "state": state if state == "complete" else "failed",
                "updates": step,
                "wall_seconds": time.monotonic() - started,
                "parameters": sum(p.numel() for p in model.parameters()),
                "eligible_episodes": len(episodes),
                "supported_rows": sum(len(e["proprio"]) for e in episodes),
                "physics_steps": 0,
                "physical_success": None,
            },
        )
    return output / f"step-{step:06d}.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(fit(json.loads(args.config.read_text()), args.output))


if __name__ == "__main__":
    main()
