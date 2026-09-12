"""Finite-grid clearance projection; geometry access is explicit, not learned."""

import math

import torch


def project_clearance(q, lower, margin=0.01):
    """KL-nearest distribution on certified sampled-frame capsule-clear cells.

    Returns zero probability and retained mass zero when support is empty.
    Unknown cells are rejected. No claim is made about between-frame dynamics.
    """
    if q.shape != lower.shape or q.ndim < 1:
        raise ValueError("matching probability and clearance arrays required")
    if not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError("invalid probabilities")
    if not torch.isfinite(lower).all() or not math.isfinite(margin) or margin < 0:
        raise ValueError("invalid clearance or margin")
    if not torch.allclose(q.sum(-1), torch.ones_like(q.sum(-1)), atol=1e-6):
        raise ValueError("input distributions must sum to one")
    kept = q * (lower >= margin)
    mass = kept.sum(-1, keepdim=True)
    return kept / mass.masked_fill(mass == 0, 1), mass.squeeze(-1)


def coverage_acceptance_loss(logits, critical, clear, acceptance_weight=2.0):
    """Proposed, untrained objective on TRAINING cells only.

    KL(uniform critical || q conditioned on clear) - weight * log P_q(clear).
    Constants independent of logits are omitted. With weight=1 and nonempty
    critical support this equals ordinary uniform-critical cross entropy.
    No-clear rows are excluded; callers must retain their abstention counts.
    Rows with clear but no critical cells receive the acceptance term only.
    """
    if logits.ndim != 2 or logits.shape != critical.shape or logits.shape != clear.shape:
        raise ValueError("matching batched cell arrays required")
    if critical.dtype != torch.bool or clear.dtype != torch.bool:
        raise ValueError("boolean geometric support required")
    if (critical & ~clear).any():
        raise ValueError("critical support must be clear")
    if not math.isfinite(acceptance_weight) or acceptance_weight < 0:
        raise ValueError("nonnegative finite acceptance weight required")
    valid = clear.any(-1)
    if not valid.any():
        return logits.sum() * 0
    logq = logits[valid].log_softmax(-1)
    labels = critical[valid]
    logz = logq.masked_fill(~clear[valid], -torch.inf).logsumexp(-1)
    count = labels.sum(-1)
    conditional_ce = -(logq * labels).sum(-1) / count.clamp_min(1) + logz
    conditional_ce = conditional_ce.masked_fill(count == 0, 0)
    return (conditional_ce - acceptance_weight * logz).mean()
