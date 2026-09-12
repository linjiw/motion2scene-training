"""Bounded action-residual PPO for physical reward improvement after imitation.

Freeze the command navigator and foundation during this stage. Features/base
outputs in a rollout must come from that exact frozen checkpoint. Rewards must
come from actual task execution; offline imitation errors are not RL rewards.
"""

import math

import torch
from torch import nn
from torch.distributions import Normal

from gear_sonic.research.hindsight_training.student import mlp


class BoundedResidualActorCritic(nn.Module):
    """Tanh-Gaussian residual; its mean starts at zero, stochastic actions stay bounded."""

    def __init__(self, limits, initial_log_std=-2.0):
        super().__init__()
        limits = torch.as_tensor(limits, dtype=torch.float32)
        if limits.shape != (29,) or not torch.isfinite(limits).all() or not (limits > 0).all():
            raise ValueError("PPO residual requires 29 positive finite bounds")
        if not math.isfinite(initial_log_std):
            raise ValueError("Non-finite residual exploration")
        self.actor = mlp([128, 128, 29])
        self.critic = mlp([128, 128, 1])
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((29,), initial_log_std))
        self.register_buffer("limits", limits.clone())

    def distribution(self, features):
        if features.ndim != 2 or features.shape[-1] != 128 or not torch.isfinite(features).all():
            raise ValueError("Residual policy requires finite public B,128 features")
        return Normal(self.actor(features), self.log_std.clamp(-5, 0).exp())

    def evaluate(self, features, pre_tanh):
        distribution = self.distribution(features)
        # Exact transformed-action likelihood; softplus form remains stable in tails.
        log_jacobian = 2 * (math.log(2) - pre_tanh - torch.nn.functional.softplus(-2 * pre_tanh))
        log_prob = (distribution.log_prob(pre_tanh) - self.limits.log() - log_jacobian).sum(-1)
        return log_prob, self.critic(features).squeeze(-1)

    def act(self, features, base_actions, *, deterministic=False):
        distribution = self.distribution(features)
        u = distribution.mean if deterministic else distribution.sample()
        residual = self.limits * u.tanh()
        log_prob, value = self.evaluate(features, u)
        return {
            "actions": base_actions + residual,
            "residual": residual,
            "pre_tanh": u,
            "log_prob": log_prob,
            "value": value,
        }


def generalized_advantage(
    rewards, values, next_values, terminated, episode_end, *, gamma=0.99, lam=0.95
):
    """Bootstrap time-limit truncations but stop GAE recursion across every reset."""
    if not (
        rewards.shape == values.shape == next_values.shape == terminated.shape == episode_end.shape
    ):
        raise ValueError("GAE rollout fields must agree")
    if terminated.dtype != torch.bool or episode_end.dtype != torch.bool or rewards.ndim != 2:
        raise ValueError("GAE requires T,B rewards and boolean boundary masks")
    if (terminated & ~episode_end).any() or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("Invalid GAE boundaries or discounts")
    advantage = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        running = delta + gamma * lam * (~episode_end[t]) * running
        advantage[t] = running
    return advantage, advantage + values


def optimize_residual_ppo(
    policy,
    optimizer,
    batch,
    *,
    epochs=4,
    clip=0.2,
    value_weight=0.5,
    residual_weight=0.01,
    max_kl=0.02,
):
    """Optimize only residual/value parameters on a detached, on-policy physical rollout."""
    if not 1 <= epochs <= 10 or not 0 < clip < 1 or min(value_weight, residual_weight, max_kl) < 0:
        raise ValueError("Invalid bounded PPO configuration")
    required = ("features", "pre_tanh", "old_log_prob", "advantages", "returns")
    if any(not torch.isfinite(batch[k]).all() for k in required):
        raise ValueError("Non-finite PPO rollout")
    data = {k: batch[k].detach() for k in required}
    advantage = data["advantages"]
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    history = []
    for epoch in range(epochs):
        logp, value = policy.evaluate(data["features"], data["pre_tanh"])
        log_ratio = logp - data["old_log_prob"]
        ratio = log_ratio.exp()
        approximate_kl = ((ratio - 1) - log_ratio).mean()
        if approximate_kl > max_kl:
            break
        surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
        residual = policy.limits * policy.actor(data["features"]).tanh()
        loss = (
            -surrogate.mean()
            + value_weight * (value - data["returns"]).square().mean()
            + residual_weight * residual.square().mean()
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.detach()),
                "approx_kl": float(approximate_kl.detach()),
            }
        )
    return history


def train_live_residual(
    runtime, navigator, policy, output, *, iterations, horizon, learning_rate, reward_profile
):
    """Collect fresh physical rollouts and optimize the residual, preserving the motor policy.

    Requires an explicit reward profile matching the initialized runtime. This is
    not invoked by offline imitation. Timeouts lacking terminal observations fail
    closed; reset-state values must never bootstrap the preceding episode.
    """
    import json
    from pathlib import Path

    from gear_sonic.research.hindsight_training.runtime import write_new

    if not 1 <= iterations <= 1000 or not 1 <= horizon <= 256:
        raise ValueError("Invalid residual rollout budget")
    if runtime.task.get("reward_profile") != reward_profile or reward_profile not in (
        "native_tracking_v1",
        "known_goal_progress_v1",
    ):
        raise ValueError("An explicit runtime reward profile is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    navigator.requires_grad_(False).eval()
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    observation = runtime.reset()
    hidden = None
    steps = 0
    status = "running"
    write_new(
        output / "lock.json",
        {
            "iterations": iterations,
            "horizon": horizon,
            "learning_rate": learning_rate,
            "reward_profile": reward_profile,
            "teacher_sha256": runtime.teacher_sha256,
            "maximum_environment_transitions": iterations * horizon,
        },
    )
    try:
        with (output / "metrics.jsonl").open("x") as log:
            for iteration in range(iterations):
                rows = []
                for _ in range(horizon):
                    with torch.no_grad():
                        base = navigator.forward_step(observation, hidden)
                        hidden = base["hidden"]
                        action = policy.act(hidden, base["base_actions"])
                        next_observation, reward, done, extras = runtime.step(action["actions"])
                        steps += 1
                        timeout = bool(extras["time_outs"].any())
                        if timeout and not runtime.task.get("timeouts_are_task_deadlines", False):
                            raise ValueError(
                                "Timeout terminal observation unavailable; refusing reset-state bootstrap"
                            )
                        next_base = navigator.forward_step(next_observation, hidden)
                        next_value = policy.critic(next_base["hidden"]).squeeze(-1)
                        rows.append(
                            {
                                "features": hidden,
                                "pre_tanh": action["pre_tanh"],
                                "old_log_prob": action["log_prob"],
                                "values": action["value"],
                                "next_values": next_value,
                                "rewards": reward.reshape(-1),
                                "terminated": done.reshape(-1).bool(),
                                "episode_end": done.reshape(-1).bool(),
                            }
                        )
                        observation = next_observation
                        if done.any():
                            observation = runtime.reset()
                            hidden = None
                batch = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
                advantage, returns = generalized_advantage(
                    batch["rewards"],
                    batch["values"],
                    batch["next_values"],
                    batch["terminated"],
                    batch["episode_end"],
                )
                flat = {k: v.flatten(0, 1) for k, v in batch.items()}
                flat.update(advantages=advantage.flatten(), returns=returns.flatten())
                updates = optimize_residual_ppo(policy, optimizer, flat)
                log.write(
                    json.dumps({"iteration": iteration + 1, "steps": steps, "updates": updates})
                    + "\n"
                )
                log.flush()
                torch.save(
                    {
                        "policy": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iteration": iteration + 1,
                        "teacher_sha256": runtime.teacher_sha256,
                        "reward_profile": reward_profile,
                    },
                    output / f"residual-{iteration+1:06d}.pt",
                )
        status = "complete"
    finally:
        write_new(
            output / "receipt.json",
            {
                "state": status if status == "complete" else "failed",
                "environment_transitions": steps,
                "reward_profile": reward_profile,
                "navigation_qualification": None,
            },
        )
