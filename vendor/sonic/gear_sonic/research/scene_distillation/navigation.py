"""Learn goal/scene command selection while preserving a frozen motion foundation.

This prototype does not include a qualified foundation checkpoint or scene expert.
Do not launch physical navigation from its randomly initialized command director.
"""

import math

import torch
from torch import nn

from gear_sonic.research.hindsight_training.student import mlp
from gear_sonic.research.scene_distillation.cvae import diagonal_gaussian_kl
from gear_sonic.research.scene_distillation.policy import PublicNavigationEncoder

# Compact profile only; these commands must be trained and qualified in the foundation.
NAVIGATION_CONTROLS = ("velocity_body_x", "velocity_body_y", "yaw_rate", "base_height")
FOUNDATION_CONTROL_INDICES = (2, 3, 6, 5)


class FrozenFoundationNavigator(nn.Module):
    """Public navigation head -> bounded commands -> frozen prior and decoder.

    At residual_scale=0 (primary proposal), the foundation prior is used exactly.
    Positive residual_scale enables an explicit latent-residual ablation measured
    in prior standard deviations. This is not BFM's action-residual experiment.
    Bounds are required from the intended command profile; numerical bounding is
    not physical qualification. This module assumes ownership of foundation/decoder.
    """

    def __init__(
        self,
        foundation,
        decoder,
        *,
        command_lower,
        command_upper,
        residual_scale=0.0,
        action_residual_limit=None,
    ):
        super().__init__()
        lower, upper = torch.as_tensor(command_lower, dtype=torch.float32), torch.as_tensor(
            command_upper, dtype=torch.float32
        )
        if (
            lower.shape != (4,)
            or upper.shape != (4,)
            or not torch.isfinite(lower).all()
            or not torch.isfinite(upper).all()
            or not (lower < upper).all()
            or lower[3] <= 0
        ):
            raise ValueError("Four finite ordered command bounds and positive height are required")
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("Residual scale must be finite and nonnegative")
        self.foundation = foundation.requires_grad_(False).eval()
        self.decoder = decoder.requires_grad_(False).eval()
        self.encoder = PublicNavigationEncoder()
        self.command_head = mlp([128, 128, 4])
        self.residual_scale = float(residual_scale)
        self.residual_head = None
        if self.residual_scale:
            self.residual_head = mlp([128, 128, foundation.latent_dim])
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)
        self.register_buffer("command_lower", lower.clone())
        self.register_buffer("command_upper", upper.clone())
        limit = (
            torch.zeros(29)
            if action_residual_limit is None
            else torch.as_tensor(action_residual_limit, dtype=torch.float32)
        )
        if limit.shape != (29,) or not torch.isfinite(limit).all() or (limit < 0).any():
            raise ValueError("Action residual requires 29 finite nonnegative native-action bounds")
        self.register_buffer("action_residual_limit", limit.clone())
        self.action_residual_head = None
        if (limit > 0).any():
            self.action_residual_head = mlp([128, 128, 29])
            nn.init.zeros_(self.action_residual_head[-1].weight)
            nn.init.zeros_(self.action_residual_head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.foundation.eval()
        self.decoder.eval()
        return self

    def forward_step(self, observation, hidden=None, *, episode_start=None, epsilon=None):
        # Keep autograd through frozen modules: commands must receive action-loss gradients.
        hidden, _, attention = self.encoder.encode_step(
            observation, hidden, episode_start=episode_start
        )
        middle = (self.command_lower + self.command_upper) / 2
        radius = (self.command_upper - self.command_lower) / 2
        command = middle + radius * self.command_head(hidden).tanh()
        controls = hidden.new_zeros(len(hidden), getattr(self.foundation, "command_dim", 8))
        controls[:, list(FOUNDATION_CONTROL_INDICES)] = command
        control_mask = torch.zeros_like(controls, dtype=torch.bool)
        control_mask[:, list(FOUNDATION_CONTROL_INDICES)] = True
        proprio = observation["proprio"]
        base_mean, logvar = self.foundation.prior_stats(proprio, controls, control_mask)
        standardized_delta = torch.zeros_like(base_mean)
        if self.residual_head is not None:
            standardized_delta = self.residual_scale * self.residual_head(hidden).tanh()
        mean = base_mean + (0.5 * logvar).exp() * standardized_delta
        latent = self.foundation.sample(mean, logvar, epsilon)
        tokens = self.foundation.tokens(proprio, latent)
        base_actions = self.decoder(tokens, proprio)
        action_residual = torch.zeros_like(base_actions)
        if self.action_residual_head is not None:
            action_residual = self.action_residual_limit * self.action_residual_head(hidden).tanh()
        return {
            "actions": base_actions + action_residual,
            "base_actions": base_actions,
            "action_residual": action_residual,
            "tokens": tokens,
            "hidden": hidden,
            "controls": controls,
            "control_mask": control_mask,
            "base_mean": base_mean,
            "navigation_mean": mean,
            "logvar": logvar,
            "standardized_residual": standardized_delta,
            "prior_deviation_kl": diagonal_gaussian_kl(mean, logvar, base_mean, logvar),
            "obstacle_attention": attention,
        }


def navigation_distillation_loss(
    output,
    teacher_actions,
    qualified_mask,
    *,
    prior_weight=0.0,
    residual_weight=0.0,
    smoothness_weight=0.0,
    previous_residual=None,
    continuation_mask=None,
):
    """Primary action imitation on same-state qualified queries; optional residual KL.

    State/provenance checks belong to the collector. Missing expert queries are
    excluded, never zero-filled. All-missing batches fail instead of faking progress.
    """
    actions = output["actions"]
    if (
        actions.ndim != 2
        or actions.shape[1] != 29
        or teacher_actions.shape != actions.shape
        or qualified_mask.shape != (len(actions),)
        or qualified_mask.dtype != torch.bool
        or qualified_mask.device != actions.device
    ):
        raise ValueError("Invalid navigation imitation dimensions")
    if not qualified_mask.any():
        raise ValueError("No qualified same-state teacher labels in this batch")
    if not math.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError("Prior regularization weight must be finite and nonnegative")
    kl = output["prior_deviation_kl"]
    if (
        kl.shape != (len(actions),)
        or not torch.isfinite(kl).all()
        or not torch.isfinite(actions).all()
        or not torch.isfinite(teacher_actions[qualified_mask]).all()
    ):
        raise ValueError("Non-finite navigation imitation value")
    imitation = (actions[qualified_mask] - teacher_actions[qualified_mask].detach()).square().mean()
    prior = kl[qualified_mask].mean()
    for weight in (residual_weight, smoothness_weight):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Residual penalties must be finite and nonnegative")
    residual = output.get("action_residual", torch.zeros_like(actions))
    magnitude = residual[qualified_mask].square().mean()
    smoothness = actions.sum() * 0
    if smoothness_weight:
        if (
            previous_residual is None
            or previous_residual.shape != residual.shape
            or continuation_mask is None
            or continuation_mask.shape != qualified_mask.shape
            or continuation_mask.dtype != torch.bool
        ):
            raise ValueError(
                "Residual smoothness needs previous residual and episode continuation mask"
            )
        valid = continuation_mask & qualified_mask
        if valid.any():
            smoothness = (residual[valid] - previous_residual[valid]).square().mean()
    return {
        "loss": imitation
        + prior_weight * prior
        + residual_weight * magnitude
        + smoothness_weight * smoothness,
        "action_mse": imitation,
        "prior_kl": prior,
        "residual_mse": magnitude,
        "residual_smoothness": smoothness,
        "supervised_rows": int(qualified_mask.sum()),
    }
