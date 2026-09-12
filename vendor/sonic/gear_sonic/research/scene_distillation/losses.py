"""Admissible-mode imitation; geometric predictions cannot replace physics labels."""

import math

import torch
from torch.nn import functional as F


def admissible_imitation_loss(
    student_tokens,
    proprio,
    teacher_tokens,
    teacher_actions,
    valid_modes,
    decoder,
    *,
    token_weight=0.1,
    decoder_atol=2e-4,
):
    """Imitate one of K admissible continuations, without averaging left/right modes.

    Validity must come from the bound teacher/qualification records. This function
    checks numerical consistency; it does not establish physical admissibility.
    """
    batch = len(student_tokens)
    if (
        teacher_tokens.ndim != 3
        or teacher_tokens.shape[0] != batch
        or teacher_tokens.shape[-1] != 64
    ):
        raise ValueError("Teacher tokens must be B,K,64")
    modes = teacher_tokens.shape[1]
    if (
        student_tokens.shape != (batch, 64)
        or proprio.shape != (batch, 930)
        or teacher_actions.shape != (batch, modes, 29)
        or valid_modes.shape != (batch, modes)
        or valid_modes.dtype != torch.bool
    ):
        raise ValueError("Incompatible distillation dimensions")
    if modes < 1 or not valid_modes.any(-1).all():
        raise ValueError("Every supervised state needs at least one qualified teacher mode")
    if not math.isfinite(token_weight) or token_weight < 0:
        raise ValueError("Token weight must be nonnegative")
    if not math.isfinite(decoder_atol) or not 0 <= decoder_atol <= 0.005:
        raise ValueError("Decoder parity tolerance must be finite and at most 0.005 native units")
    if any(p.requires_grad for p in decoder.parameters()):
        raise ValueError("The selected teacher decoder must be frozen for distillation")
    tokens = torch.where(valid_modes[..., None], teacher_tokens, 0.0)
    targets = torch.where(valid_modes[..., None], teacher_actions, 0.0)
    if not all(torch.isfinite(x).all() for x in (student_tokens, proprio, tokens, targets)):
        raise ValueError("Non-finite imitation value")
    for value in (student_tokens, tokens):
        if (
            (value < -1).any()
            or (value > 15 / 16).any()
            or not torch.allclose(value * 16, (value * 16).round(), atol=1e-5, rtol=0)
        ):
            raise ValueError("Imitation expects post-FSQ tokens on the native SONIC lattice")
    with torch.no_grad():
        expanded = proprio[:, None].expand(-1, modes, -1).reshape(-1, 930)
        expected = decoder(tokens.reshape(-1, 64), expanded).reshape(batch, modes, 29)
        if not torch.allclose(
            expected[valid_modes], targets[valid_modes], atol=decoder_atol, rtol=2e-4
        ):
            raise ValueError("Teacher query is inconsistent with the frozen decoder and same state")
    actions = decoder(student_tokens, proprio)
    action_cost = (actions[:, None] - targets.detach()).square().mean(-1)
    token_cost = (student_tokens[:, None] - tokens.detach()).square().mean(-1)
    costs = action_cost + token_weight * token_cost
    costs = costs.masked_fill(~valid_modes, torch.inf)
    selected = costs.argmin(-1)
    indices = torch.arange(batch, device=selected.device)
    return {
        "loss": costs[indices, selected].mean(),
        "selected_mode": selected,
        "action_mse": action_cost[indices, selected].mean(),
        "token_mse": token_cost[indices, selected].mean(),
    }


def endpoint_auxiliary_loss(predicted_displacement, actual_displacement, observed_mask):
    """Only complete observed horizons contribute; resets/timeouts are not zero targets."""
    if (
        predicted_displacement.shape != actual_displacement.shape
        or observed_mask.shape != predicted_displacement.shape[:-1]
        or observed_mask.dtype != torch.bool
    ):
        raise ValueError("Invalid future-displacement target mask")
    if not observed_mask.any():
        return predicted_displacement.sum() * 0
    if not torch.isfinite(actual_displacement[observed_mask]).all():
        raise ValueError("Non-finite observed displacement label")
    return F.smooth_l1_loss(
        predicted_displacement[observed_mask], actual_displacement[observed_mask]
    )


def variational_imitation_loss(
    posterior_output,
    proprio,
    teacher_tokens,
    teacher_actions,
    decoder,
    *,
    beta,
    token_weight=0.0,
    decoder_atol=2e-4,
):
    """Single queried expert per posterior: action reconstruction plus KL(q || p).

    beta is an explicit design parameter, not an inferred paper hyperparameter.
    Deploy/collect with prior_step; low posterior loss alone is not student success.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("KL weight must be finite and nonnegative")
    tokens, kl = posterior_output["tokens"], posterior_output["kl"]
    if kl.shape != tokens.shape[:-1] or not torch.isfinite(kl).all():
        raise ValueError("Invalid posterior KL vector")
    result = admissible_imitation_loss(
        tokens,
        proprio,
        teacher_tokens[:, None],
        teacher_actions[:, None],
        torch.ones(len(tokens), 1, device=tokens.device, dtype=torch.bool),
        decoder,
        token_weight=token_weight,
        decoder_atol=decoder_atol,
    )
    result["reconstruction"] = result["loss"]
    result["kl"] = kl.mean()
    result["loss"] = result["reconstruction"] + beta * result["kl"]
    return result
