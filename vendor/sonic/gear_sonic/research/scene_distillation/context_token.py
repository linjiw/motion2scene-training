"""Direct context-conditioned BFM preserving the pretrained SONIC token motor path."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.action_flow import (
    CONTEXT_SHAPES,
    PublicConditionEncoder,
)
from gear_sonic.research.scene_distillation.cvae import diagonal_gaussian_kl
from gear_sonic.research.scene_distillation.losses import variational_imitation_loss
from gear_sonic.research.scene_distillation.train import load_episodes, make_decoder
from gear_sonic.research.scene_distillation.train_action_flow import profile_mask
from gear_sonic.research.scene_distillation.transformer import TransformerMotionFoundation


class ContextTokenFoundation(TransformerMotionFoundation):
    def __init__(self, width=256, layers=4, heads=8, latent_dim=64):
        super().__init__(width, layers, heads, latent_dim)
        del (
            self.history_projection,
            self.command_projection,
            self.position,
            self.encoder,
            self.readout,
        )
        self.condition = PublicConditionEncoder(width, layers, heads, context=True)

    def initialize_foundation(self, state):
        copied = self.condition.initialize_motion_encoder(state)
        for name in ["prior", "posterior", "token_adapter"]:
            module = getattr(self, name)
            module.load_state_dict(
                {k[len(name) + 1 :]: v for k, v in state.items() if k.startswith(name + ".")},
                strict=True,
            )
        return copied

    def context_step(
        self, proprio, controls, mask, context, *, privileged_state=None, future_reference=None
    ):
        c = self.condition(proprio, controls, mask, context)
        mean, logvar = self.prior(c).chunk(2, -1)
        logvar = logvar.clamp(-8, 4)
        output = {"prior_mean": mean, "prior_logvar": logvar}
        if privileged_state is None and future_reference is None:
            output["tokens"] = self.tokens(proprio, mean)
            return output
        if privileged_state is None or future_reference is None:
            raise ValueError("Posterior needs both privileged inputs")
        residual, q_logvar = self.posterior(
            torch.cat([c, privileged_state.detach(), future_reference.detach()], -1)
        ).chunk(2, -1)
        q_mean, q_logvar = mean + residual, q_logvar.clamp(-8, 4)
        latent = self.sample(q_mean, q_logvar, torch.randn_like(mean))
        output.update(
            tokens=self.tokens(proprio, latent),
            posterior_mean=q_mean,
            posterior_logvar=q_logvar,
            kl=diagonal_gaussian_kl(q_mean, q_logvar, mean, logvar),
        )
        return output


class ContextTokenPolicy(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        if sha(config["teacher_checkpoint"]) != config["teacher_sha256"]:
            raise ValueError("Teacher changed")
        self.foundation = ContextTokenFoundation(**config["transformer"]).to(device)
        self.decoder = make_decoder(config["teacher_checkpoint"], device)

    @property
    def condition(self):
        return self.foundation.condition

    def actions(
        self,
        proprio,
        controls,
        mask,
        *,
        context=None,
        noise=None,
        steps=None,
        residual_enabled=True,
    ):
        result = self.foundation.context_step(proprio, controls, mask, context)
        actions = self.decoder(result["tokens"], proprio)
        return {"actions": actions, "base_actions": actions, "residual": torch.zeros_like(actions)}


def fit(config, output):
    if not 1 <= config["updates"] <= 20000 or not 0 < config["wall_cap_seconds"] <= 3600:
        raise ValueError("Invalid context BFM fit budget")
    if sha(config["dataset_manifest"]) != config["dataset_manifest_sha256"]:
        raise ValueError("Context dataset changed")
    manifest = json.loads(Path(config["dataset_manifest"]).read_text())
    if manifest.get("context_schema") != "complete_known_map_goal_v1":
        raise ValueError("Context provenance missing")
    episodes = load_episodes(
        config["dataset_manifest"],
        config["teacher_sha256"],
        set(config["train_ids"]),
        "foundation",
        allow_teacher_prefixes=True,
        allow_exploratory_queries=True,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "config.json", config)
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = ContextTokenPolicy(config, "cuda")
    if sha(config["foundation_checkpoint"]) != config["foundation_sha256"]:
        raise ValueError("Foundation changed")
    saved = torch.load(config["foundation_checkpoint"], map_location="cuda", weights_only=False)
    if saved["teacher_sha256"] != config["teacher_sha256"] or saved["stage"] != "foundation":
        raise ValueError("Foundation provenance mismatch")
    policy.foundation.initialize_foundation(saved["model"])
    optimizer = torch.optim.AdamW(policy.foundation.parameters(), lr=config["learning_rate"])
    rng = np.random.default_rng(config["seed"])
    started = time.monotonic()
    state = "failed"
    step = 0
    fields = [
        "proprio",
        "controls",
        "control_mask",
        "teacher_actions",
        "teacher_tokens",
        "privileged_state",
        "future_reference",
    ] + list(CONTEXT_SHAPES)
    try:
        with (output / "metrics.jsonl").open("x") as log:
            for step in range(1, config["updates"] + 1):
                if time.monotonic() - started > config["wall_cap_seconds"]:
                    raise TimeoutError("Context BFM wall cap")
                samples = []
                for _ in range(config["batch_size"]):
                    e = episodes[int(rng.integers(len(episodes)))]
                    i = int(rng.integers(len(e["proprio"])))
                    samples.append({k: e[k][i] for k in fields})
                b = {k: torch.stack([s[k] for s in samples]).cuda() for k in fields}
                profile = str(rng.choice(["full", "root", "context"], p=[0.3, 0.2, 0.5]))
                mask = profile_mask(b["control_mask"], profile)
                context = {k: b[k] for k in CONTEXT_SHAPES}
                posterior = policy.foundation.context_step(
                    b["proprio"],
                    b["controls"],
                    mask,
                    context,
                    privileged_state=b["privileged_state"],
                    future_reference=b["future_reference"],
                )
                result = variational_imitation_loss(
                    posterior,
                    b["proprio"],
                    b["teacher_tokens"],
                    b["teacher_actions"],
                    policy.decoder,
                    beta=0.001,
                    decoder_atol=2e-4,
                )
                public = policy.foundation.context_step(b["proprio"], b["controls"], mask, context)
                mse = (
                    (policy.decoder(public["tokens"], b["proprio"]) - b["teacher_actions"])
                    .square()
                    .mean()
                )
                token = (
                    (public["tokens"] - b["teacher_tokens"]).square().mean()
                    if profile == "full"
                    else mse * 0
                )
                loss = result["loss"] + mse + 0.1 * token
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.foundation.parameters(), 1, error_if_nonfinite=True
                )
                optimizer.step()
                if step % 10 == 0:
                    log.write(
                        json.dumps(
                            {
                                "step": step,
                                "profile": profile,
                                "loss": float(loss.detach()),
                                "prior_action_mse": float(mse.detach()),
                            }
                        )
                        + "\n"
                    )
                    log.flush()
        torch.save(
            {
                "stage": "context_token",
                "config": config,
                "teacher_sha256": config["teacher_sha256"],
                "model": policy.foundation.state_dict(),
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
                "physical_success": None,
                "decoder_frozen": True,
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
