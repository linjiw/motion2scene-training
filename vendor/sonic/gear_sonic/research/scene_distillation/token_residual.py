"""Bounded supervised action residual around an unchanged, hash-bound SONIC BFM.

This diagnostic is action imitation, not residual PPO or a sim-to-real claim.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.hindsight_training.student import mlp
from gear_sonic.research.scene_distillation.collect import native_commands
from gear_sonic.research.scene_distillation.commands import build_foundation
from gear_sonic.research.scene_distillation.evaluate_commands import CommandEvaluationCallback
from gear_sonic.research.scene_distillation.train import load_episodes, make_decoder
from gear_sonic.research.scene_distillation.train_action_flow import profile_mask


class TokenActionResidual(nn.Module):
    def __init__(self, foundation, decoder, limit=0.05):
        super().__init__()
        if not np.isfinite(limit) or not 0 < limit <= 0.2:
            raise ValueError("Invalid action residual bound")
        self.foundation = foundation.eval()
        self.decoder = decoder.eval()
        self.limit = limit
        for module in (foundation, decoder):
            for p in module.parameters():
                p.requires_grad_(False)
        self.head = mlp([930 + 158 + 29, 512, 256, 29])
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def actions(self, proprio, controls, mask, *, residual_enabled=True):
        with torch.no_grad():
            tokens = self.foundation.prior_step(proprio, controls, mask)["tokens"]
            base = self.decoder(tokens, proprio)
        clean = torch.where(mask, controls, 0.0)
        if not torch.isfinite(clean).all():
            raise ValueError("Nonfinite available commands")
        residual = (
            self.limit * self.head(torch.cat([proprio, clean, mask.float(), base], -1)).tanh()
        )
        return {
            "actions": base + residual if residual_enabled else base,
            "base_actions": base,
            "residual": residual,
        }


def build_residual(config, device):
    for field in ["teacher", "foundation"]:
        if sha(config[field + "_checkpoint"]) != config[field + "_sha256"]:
            raise ValueError(field + " checkpoint changed")
    saved = torch.load(config["foundation_checkpoint"], map_location=device, weights_only=False)
    if saved["teacher_sha256"] != config["teacher_sha256"] or saved["stage"] != "foundation":
        raise ValueError("Foundation provenance mismatch")
    foundation = build_foundation(saved["config"]).to(device)
    foundation.load_state_dict(saved["model"], strict=True)
    return TokenActionResidual(
        foundation, make_decoder(config["teacher_checkpoint"], device), config["residual_limit"]
    ).to(device)


class TokenResidualEvaluationCallback(CommandEvaluationCallback):
    def _get_inference_policy(self, device=None):
        device = self.env.device if device is None else device
        saved = torch.load(self.student_checkpoint, map_location=device, weights_only=False)
        if (
            saved["stage"] != "bfm_action_residual"
            or saved["teacher_sha256"] != self.teacher_sha256
        ):
            raise ValueError("Residual checkpoint mismatch")
        student = build_residual(saved["config"], device).eval()
        student.head.load_state_dict(saved["residual"], strict=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        def policy(obs_dict, **kwargs):
            controls, mask = native_commands(
                self.env.motion_command, self.env.env.scene.env_origins
            )
            proprio = obs_dict["actor_obs"]
            if self.model.policy.running_mean_std is not None:
                proprio = self.model.policy.running_mean_std(proprio)
            return student.actions(proprio, controls, profile_mask(mask, self.command_profile))[
                "actions"
            ]

        return policy


def fit(config, output):
    if not 1 <= config["updates"] <= 10000 or not 0 < config["wall_cap_seconds"] <= 3600:
        raise ValueError("Invalid residual experiment budget")
    if sha(config["dataset_manifest"]) != config["dataset_manifest_sha256"]:
        raise ValueError("Dataset changed")
    episodes = load_episodes(
        config["dataset_manifest"],
        config["teacher_sha256"],
        set(config["train_ids"]),
        "foundation",
        allow_exploratory_queries=True,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "config.json", config)
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_residual(config, "cuda")
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config["learning_rate"])
    rng = np.random.default_rng(config["seed"])
    started = time.monotonic()
    groups = [
        [e for e in episodes if bool(e["_exploratory_source"][0]) == flag] for flag in [False, True]
    ]
    groups = [g for g in groups if g]
    step = 0
    state = "failed"
    try:
        with (output / "metrics.jsonl").open("x") as log:
            for step in range(1, config["updates"] + 1):
                if time.monotonic() - started > config["wall_cap_seconds"]:
                    raise TimeoutError("Residual wall cap")
                samples = []
                for _ in range(config["batch_size"]):
                    g = groups[int(rng.integers(len(groups)))]
                    e = g[int(rng.integers(len(g)))]
                    i = int(rng.integers(len(e["proprio"])))
                    samples.append(
                        {
                            k: e[k][i]
                            for k in ["proprio", "controls", "control_mask", "teacher_actions"]
                        }
                    )
                b = {k: torch.stack([s[k] for s in samples]).cuda() for k in samples[0]}
                profile = str(rng.choice(["full", "root", "navigation"], p=[0.4, 0.2, 0.4]))
                result = model.actions(
                    b["proprio"], b["controls"], profile_mask(b["control_mask"], profile)
                )
                mse = (result["actions"] - b["teacher_actions"]).square().mean()
                magnitude = result["residual"].square().mean()
                loss = mse + 0.01 * magnitude
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.head.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                if step % 10 == 0:
                    log.write(
                        json.dumps(
                            {
                                "step": step,
                                "action_mse": float(mse.detach()),
                                "residual_mse": float(magnitude.detach()),
                                "base_action_mse": float(
                                    (result["base_actions"] - b["teacher_actions"]).square().mean()
                                ),
                            }
                        )
                        + "\n"
                    )
                    log.flush()
        torch.save(
            {
                "stage": "bfm_action_residual",
                "teacher_sha256": config["teacher_sha256"],
                "config": config,
                "residual": model.head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "updates": step,
            },
            output / f"step-{step:06d}.pt",
        )
        state = "complete"
    finally:
        write_new(
            output / "receipt.json",
            {
                "state": state,
                "updates": step,
                "wall_seconds": time.monotonic() - started,
                "base_frozen": True,
                "residual_limit": config["residual_limit"],
                "physical_success": None,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
