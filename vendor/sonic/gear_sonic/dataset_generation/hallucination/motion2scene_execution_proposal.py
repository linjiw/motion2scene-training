"""Conditional learned scene proposals from executed alternatives and sensor context.

The model amortizes accepted analytic examples. Samples still require the same
geometric, observation, and physical checks as analytic proposals. Likelihood is
an optimization diagnostic, never evidence of downstream traversal performance.
"""

import numpy as np
import torch
from torch import nn

from .motion2scene_inverse import mixture_log_prob, sample_latents


def execution_summary(capsules, bins=16):
    """Temporal whole-body bounding envelopes from achieved capsule trajectories."""
    starts, ends = np.asarray(capsules["starts"]), np.asarray(capsules["ends"])
    radii = np.broadcast_to(capsules["radii"], starts.shape[:-1])
    if (
        starts.ndim != 3
        or starts.shape != ends.shape
        or starts.shape[-1] != 3
        or len(starts) < bins
        or bins < 1
        or not all(np.isfinite(a).all() for a in (starts, ends, radii))
        or (radii < 0).any()
    ):
        raise ValueError("finite achieved trajectories and nonnegative capsule radii required")
    lower = np.minimum(starts, ends) - radii[..., None]
    upper = np.maximum(starts, ends) + radii[..., None]
    # Extremes over each time bin retain entry, maintenance, and recovery phases.
    return np.concatenate(
        [
            np.r_[lower[idx].min(axis=(0, 1)), upper[idx].max(axis=(0, 1))]
            for idx in np.array_split(np.arange(len(starts)), bins)
        ]
    )


def proposal_condition(positive, negative, approach_history, sensor_configuration, bins=16):
    """Explicit e+, e-, x/h, sensor conditioning; no reference pose substitute."""
    history = np.asarray(approach_history, dtype=float)
    sensor = np.asarray(sensor_configuration, dtype=float)
    if (
        not history.size
        or not sensor.size
        or not all(np.isfinite(v).all() for v in (history, sensor))
    ):
        raise ValueError("finite approach history and sensor configuration required")
    return np.r_[
        execution_summary(positive, bins),
        execution_summary(negative, bins),
        history.ravel(),
        sensor.ravel(),
    ]


class ExecutionConditionedProposal(nn.Module):
    """Gaussian mixture over scene logits, conditioned on both achieved options."""

    def __init__(self, condition_dim, components=4):
        super().__init__()
        if condition_dim < 1 or components < 1:
            raise ValueError("positive model dimensions required")
        self.components = components
        self.network = nn.Sequential(
            nn.Linear(condition_dim, 64), nn.Tanh(), nn.Linear(64, components * 6)
        )
        self.register_buffer("condition_mean", torch.zeros(condition_dim))
        self.register_buffer("condition_scale", torch.ones(condition_dim))

    def forward(self, condition):
        if (
            condition.ndim != 1
            or condition.shape != self.condition_mean.shape
            or not torch.isfinite(condition).all()
        ):
            raise ValueError("one complete execution/history/sensor condition required")
        raw = self.network((condition - self.condition_mean) / self.condition_scale).reshape(
            self.components, 6
        )
        scales = torch.nn.functional.softplus(raw[:, 3:5]) + 0.03
        correlation = 0.95 * raw[:, 5].tanh()
        factor = torch.stack(
            (
                scales[:, 0],
                torch.zeros_like(correlation),
                correlation * scales[:, 1],
                (1 - correlation.square()).sqrt() * scales[:, 1],
            ),
            dim=-1,
        ).reshape(-1, 2, 2)
        return raw[:, 0], raw[:, 1:3], factor

    def sample(self, condition, count, low, high, generator):
        low, high = condition.new_tensor(low), condition.new_tensor(high)
        if (
            low.shape != (2,)
            or high.shape != (2,)
            or not torch.isfinite(low).all()
            or not torch.isfinite(high).all()
            or not torch.all(high > low)
            or not isinstance(count, int)
            or count < 1
        ):
            raise ValueError("two finite scene parameter bounds required")
        latent = sample_latents(self(condition), count, generator)
        return low + (high - low) * latent.sigmoid()


def fit_proposal(conditions, accepted_scenes, low, high, *, steps=400, seed=0):
    """Equal-condition likelihood fit to screened analytic exemplars, on CPU.

    Each condition's exemplar set has equal weight irrespective of its size.
    Split by source/context before calling this function for a transfer study.
    """
    conditions = torch.as_tensor(np.asarray(conditions), dtype=torch.float64)
    if conditions.ndim != 2 or len(conditions) != len(accepted_scenes) or not len(conditions):
        raise ValueError("aligned nonempty conditions and accepted exemplar sets required")
    low, high = conditions.new_tensor(low), conditions.new_tensor(high)
    if (
        low.shape != (2,)
        or high.shape != (2,)
        or not torch.isfinite(conditions).all()
        or not torch.isfinite(low).all()
        or not torch.isfinite(high).all()
        or not torch.all(high > low)
        or steps < 1
    ):
        raise ValueError("finite conditions, positive bounds and training steps required")
    targets = []
    for scenes in accepted_scenes:
        values = (conditions.new_tensor(np.asarray(scenes)) - low) / (high - low)
        if (
            values.ndim != 2
            or values.shape[1] != 2
            or not len(values)
            or not torch.isfinite(values).all()
            or not torch.all((values > 0) & (values < 1))
        ):
            raise ValueError("accepted scene exemplars must be strictly inside proposal domain")
        targets.append(torch.logit(values))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ExecutionConditionedProposal(conditions.shape[1]).double()
        model.condition_mean.copy_(conditions.mean(dim=0))
        model.condition_scale.copy_(conditions.std(dim=0, unbiased=False).clamp(min=0.05))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        history = []
        for step in range(steps):
            optimizer.zero_grad()
            loss = torch.stack(
                [-mixture_log_prob(y, *model(x)).mean() for x, y in zip(conditions, targets)]
            ).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite conditional proposal likelihood")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            if step % 50 == 0 or step == steps - 1:
                history.append({"step": step, "negative_log_likelihood": float(loss.detach())})
    return model.eval(), history
