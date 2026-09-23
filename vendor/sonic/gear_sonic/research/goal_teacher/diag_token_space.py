"""Offline token-space diagnostics for a policy acting in SONIC's FSQ token space.

Roadmap Phase 0.8. Uses recorded teacher collection shards (proprio, future_reference,
teacher_tokens, teacher_actions) and evaluates frozen SONIC modules on CPU:

* parity of the bound decoder/encoder with the recorded teacher outputs;
* action sensitivity to token perturbations in lattice units (bin = 1/16);
* per-dimension sensitivity, token dynamics and level occupancy;
* blending and held-token behaviour;
* the g1_kin cycle residual (token -> kinematic reference -> encoder -> token), an
  off-manifold detector for learned token policies.

No physics is run: every number is a property of the frozen networks on recorded
states, not closed-loop behaviour.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from vector_quantize_pytorch import FSQ

from gear_sonic.research.hindsight_training.runtime import load_release_checkpoint, sha
from gear_sonic.research.hindsight_training.student import mlp

BIN = 1.0 / 16.0  # spacing of the 32-level FSQ lattice in code space
LEVEL_MIN, LEVEL_MAX = -16, 15  # lattice values are k / 16


class FrozenSonic(torch.nn.Module):
    """G1 encoder, FSQ, g1_dyn decoder and g1_kin decoder from one checkpoint."""

    def __init__(self, policy_state):
        super().__init__()

        def load(prefix, dims):
            net = mlp(dims)
            net.load_state_dict(
                {k[len(prefix) :]: v for k, v in policy_state.items() if k.startswith(prefix)},
                strict=True,
            )
            return net

        self.encoder = load("actor_module.encoders.g1.module.", [640, 2048, 1024, 512, 512, 64])
        self.dyn = load(
            "actor_module.decoders.g1_dyn.module.", [994, 2048, 2048, 1024, 1024, 512, 512, 29]
        )
        self.kin = load("actor_module.decoders.g1_kin.module.", [64, 2048, 1024, 512, 512, 640])
        self.quantizer = FSQ([32] * 32)
        self.requires_grad_(False)
        self.eval()

    def encode(self, reference):
        latent = self.encoder(reference.reshape(-1, 640)).reshape(-1, 2, 32)
        tokens, _ = self.quantizer(latent)
        return tokens.reshape(-1, 64)

    def act(self, tokens, proprio):
        return self.dyn(torch.cat([tokens, proprio], -1))

    def cycle(self, tokens):
        return self.encode(self.kin(tokens))


def to_levels(tokens):
    return torch.round(tokens / BIN)


def snap(tokens):
    return to_levels(tokens).clamp(LEVEL_MIN, LEVEL_MAX) * BIN


def rms(x):
    """Per-sample RMS over the last dimension, then mean over samples."""
    return float(x.pow(2).mean(-1).sqrt().mean())


def load_shards(paths):
    keys = ("proprio", "future_reference", "teacher_tokens", "teacher_actions")
    data = {k: [] for k in keys}
    episode = []
    for index, path in enumerate(paths):
        with np.load(path) as shard:
            for k in keys:
                data[k].append(shard[k])
            episode.append(np.full(len(shard["proprio"]), index))
    out = {k: torch.from_numpy(np.concatenate(v)).float() for k, v in data.items()}
    out["episode"] = torch.from_numpy(np.concatenate(episode))
    return out


def lag_pairs(episode, lag):
    """Indices (i, i+lag) that stay within one episode."""
    i = torch.arange(len(episode) - lag)
    keep = episode[i] == episode[i + lag]
    return i[keep], i[keep] + lag


def perturbation_study(model, tokens, proprio, base_action, generator):
    out = {}
    for k in (0.5, 1.0, 2.0):
        noise = torch.randn(tokens.shape, generator=generator) * k * BIN
        continuous = model.act(tokens + noise, proprio)
        lattice_shift = torch.randint(-int(np.ceil(k)), int(np.ceil(k)) + 1, tokens.shape, generator=generator)
        if k < 1:  # half-bin: each dimension moves by one bin with probability 1/2
            lattice_shift = lattice_shift * (torch.rand(tokens.shape, generator=generator) < 0.5)
        shifted = (to_levels(tokens) + lattice_shift).clamp(LEVEL_MIN, LEVEL_MAX) * BIN
        on_lattice = model.act(shifted, proprio)
        out[f"{k:g}_bin"] = dict(
            gaussian_offlattice_action_rms=rms(continuous - base_action),
            uniform_lattice_action_rms=rms(on_lattice - base_action),
            uniform_lattice_mean_abs_shift_bins=float(lattice_shift.abs().float().mean()),
        )
    per_dim = []
    for d in range(64):
        up = tokens.clone()
        up[:, d] = snap(up[:, d] + BIN)
        per_dim.append(rms(model.act(up, proprio) - base_action))
    out["single_dim_plus1_bin_action_rms"] = dict(
        mean=float(np.mean(per_dim)), max=float(np.max(per_dim)), min=float(np.min(per_dim)),
        top8_dims=[int(i) for i in np.argsort(per_dim)[::-1][:8]],
    )
    return out


def token_dynamics(tokens, episode):
    levels = to_levels(tokens)
    i, j = lag_pairs(episode, 1)
    step = (levels[j] - levels[i]).abs()
    occupancy = [int(torch.unique(levels[:, d]).numel()) for d in range(64)]
    entropy = []
    for d in range(64):
        _, counts = torch.unique(levels[:, d], return_counts=True)
        p = counts.float() / counts.sum()
        entropy.append(float(-(p * p.log2()).sum()))
    return dict(
        per_step_mean_abs_change_bins=float(step.mean()),
        per_step_fraction_dims_changed=float((step > 0).float().mean()),
        per_step_p95_abs_change_bins=float(step.flatten().quantile(0.95)),
        per_step_max_abs_change_bins=float(step.max()),
        levels_used_per_dim=dict(mean=float(np.mean(occupancy)), min=int(min(occupancy)), max=int(max(occupancy))),
        level_entropy_bits_per_dim=dict(mean=float(np.mean(entropy)), min=float(min(entropy))),
        level_range=[int(levels.min()), int(levels.max())],
    )


def blend_study(model, tokens, proprio, episode):
    out = {}
    for lag in (10, 25, 50):
        i, j = lag_pairs(episode, lag)
        a_i = model.act(tokens[i], proprio[i])
        a_j = model.act(tokens[j], proprio[i])
        mid = 0.5 * (tokens[i] + tokens[j])
        a_mid = model.act(mid, proprio[i])
        a_mid_snap = model.act(snap(mid), proprio[i])
        out[f"lag_{lag}"] = dict(
            endpoint_action_rms=rms(a_j - a_i),
            midpoint_nonlinearity_rms=rms(a_mid - 0.5 * (a_i + a_j)),
            snapped_vs_continuous_midpoint_rms=rms(a_mid_snap - a_mid),
        )
    return out


def held_token_study(model, tokens, proprio, actions, episode):
    out = {}
    for lag in (1, 2, 5, 10, 25):
        i, j = lag_pairs(episode, lag)
        held = model.act(tokens[i], proprio[j])
        out[f"lag_{lag}"] = dict(
            held_vs_teacher_action_rms=rms(held - actions[j]),
            teacher_action_change_rms=rms(actions[j] - actions[i]),
        )
    return out


def cycle_study(model, tokens, generator):
    def residual(t):
        diff = (to_levels(model.cycle(t)) - to_levels(t)).abs()
        per_sample = diff.mean(-1)
        return dict(
            mean_abs_bins=float(diff.mean()),
            fraction_dims_changed=float((diff > 0).float().mean()),
            per_sample_mean_bins_p50=float(per_sample.quantile(0.5)),
            per_sample_mean_bins_p99=float(per_sample.quantile(0.99)),
        ), per_sample

    base, base_per_sample = residual(tokens)
    threshold = float(base_per_sample.quantile(0.99))
    out = dict(recorded_tokens=base, corpus_p99_threshold_bins=threshold)
    for k in (1, 2, 4):
        shift = torch.randint(-k, k + 1, tokens.shape, generator=generator)
        stats, per_sample = residual((to_levels(tokens) + shift).clamp(LEVEL_MIN, LEVEL_MAX) * BIN)
        stats["fraction_above_corpus_p99"] = float((per_sample > threshold).float().mean())
        out[f"uniform_pm{k}_bins"] = stats
    random_tokens = torch.randint(LEVEL_MIN, LEVEL_MAX + 1, tokens.shape, generator=generator) * BIN
    stats, per_sample = residual(random_tokens.float())
    stats["fraction_above_corpus_p99"] = float((per_sample > threshold).float().mean())
    out["random_lattice_tokens"] = stats
    return out


def evaluate(name, checkpoint, data, recorded_teacher_sha, seed):
    state = load_release_checkpoint(checkpoint)["policy_state_dict"]
    model = FrozenSonic(state)
    generator = torch.Generator().manual_seed(seed)
    proprio, reference = data["proprio"], data["future_reference"]
    with torch.no_grad():
        tokens = model.encode(reference)
        actions = model.act(tokens, proprio)
        std = state["std"].float()
        result = dict(checkpoint=str(checkpoint), sha256=sha(checkpoint), policy_std=dict(mean=float(std.mean()), min=float(std.min()), max=float(std.max())))
        if sha(checkpoint) == recorded_teacher_sha:
            result["parity"] = dict(
                token_max_abs_diff=float((tokens - data["teacher_tokens"]).abs().max()),
                action_max_abs_diff=float((actions - data["teacher_actions"]).abs().max()),
            )
        i, j = lag_pairs(data["episode"], 1)
        result["natural_per_step_action_change_rms"] = rms(actions[j] - actions[i])
        result["perturbation"] = perturbation_study(model, tokens, proprio, actions, generator)
        result["token_dynamics"] = token_dynamics(tokens, data["episode"])
        result["blend"] = blend_study(model, tokens, proprio, data["episode"])
        result["held_token"] = held_token_study(model, tokens, proprio, actions, data["episode"])
        result["cycle"] = cycle_study(model, tokens, generator)
        recon = model.kin(tokens)
        result["kin_reconstruction_rms"] = rms(recon - reference)
        result["reference_rms"] = rms(reference)
    return result, tokens, actions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", type=Path, nargs="+", required=True, help="episode-*.npz collection shards")
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--teacher", type=Path, help="checkpoint that produced the shards (parity check)")
    p.add_argument("--teacher-sha256", default="")
    p.add_argument("--max-episodes", type=int, default=0, help="evenly spaced subset of shards (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    shards = sorted(a.shards)
    if a.max_episodes and len(shards) > a.max_episodes:
        shards = [shards[i] for i in np.linspace(0, len(shards) - 1, a.max_episodes).round().astype(int)]
    data = load_shards(shards)
    report = dict(
        schema="token_space_diagnostics_v1",
        bin_width=BIN,
        rows=int(len(data["proprio"])),
        episodes=int(data["episode"].unique().numel()),
        shards=[str(x) for x in shards],
        note="Offline properties of frozen networks on recorded states; no physics.",
    )
    arms = {"release": a.release}
    if a.teacher:
        arms["teacher"] = a.teacher
    actions = {}
    for name, ckpt in arms.items():
        report[name], _, actions[name] = evaluate(name, ckpt, data, a.teacher_sha256, a.seed)
    if len(actions) == 2:
        report["release_vs_teacher_action_rms"] = rms(actions["release"] - actions["teacher"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
