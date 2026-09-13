"""Experimental direct-action regression/flow students with public scene context.

Conditional flow matching uses noise-to-data time: x(t)=(1-t)*noise+t*action,
velocity=action-noise. This is the reverse of OmniXtreme's time convention.
No privileged reference, phase, or motion ID is accepted by the actor API.
"""

import math

import torch
from torch import nn

from gear_sonic.research.hindsight_training.student import mlp
from gear_sonic.research.scene_distillation.transformer import TransformerMotionFoundation

CONTEXT_SHAPES = {
    "navigation_context": (10,),
    "obstacles_body": (5, 15),
    "obstacle_mask": (5,),
}


def flow_path(target, noise, time):
    if target.shape != noise.shape or time.shape != (len(target), 1):
        raise ValueError("Invalid flow path shapes")
    if not all(torch.isfinite(x).all() for x in (target, noise, time)):
        raise ValueError("Nonfinite flow path")
    if (time < 0).any() or (time > 1).any():
        raise ValueError("Flow time must lie in [0,1]")
    return (1 - time) * noise + time * target, target - noise


def euler_flow(velocity, noise, steps):
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 64:
        raise ValueError("Flow integration requires 1..64 steps")
    value = noise
    for i in range(steps):
        time = value.new_full((len(value), 1), i / steps)
        value = value + velocity(value, time) / steps
    return value


class PublicConditionEncoder(TransformerMotionFoundation):
    """Existing motion encoder plus trainable scene/goal cross-attention.

    A zero-initialized context output preserves the original encoder exactly at
    migration. Goal token is always present, even when the obstacle set is empty.
    This first interface is an explicitly complete known map of <=5 primitives.
    Camera/partial-map observations must use a different schema.
    """

    def __init__(self, width=256, layers=4, heads=8, context=False):
        super().__init__(width=width, layers=layers, heads=heads, latent_dim=64)
        del self.prior, self.posterior, self.token_adapter
        self.context_enabled = context
        if context:
            self.goal_projection = mlp([10, width, width])
            self.obstacle_projection = mlp([15, width, width])
            self.context_query = nn.Linear(192, width)
            self.context_attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.context_output = nn.Linear(width, 192)
            nn.init.zeros_(self.context_output.weight)
            nn.init.zeros_(self.context_output.bias)

    def forward(self, proprio, controls, mask, context=None):
        if proprio.dtype != torch.float32 or mask.dtype != torch.bool:
            raise ValueError("Expected FP32 history and boolean command availability")
        condition = self._condition(proprio, controls, mask)
        if not self.context_enabled:
            if context is not None:
                raise ValueError("Context supplied to motion-only model")
            return condition
        if context is None or set(context) != set(CONTEXT_SHAPES):
            raise ValueError("Explicit complete known-map context is required")
        for key, shape in CONTEXT_SHAPES.items():
            if context[key].shape != (len(proprio), *shape):
                raise ValueError(f"Invalid context dimensions: {key}")
            if context[key].device != proprio.device:
                raise ValueError("Context device mismatch")
        nav = context["navigation_context"]
        valid = context["obstacle_mask"]
        if valid.dtype != torch.bool or nav.dtype != proprio.dtype:
            raise ValueError("Invalid context dtypes")
        if not torch.isfinite(nav).all() or not (nav[:, -1] == 1).all():
            raise ValueError("This pilot requires an explicitly complete known map")
        geometry = torch.where(valid[..., None], context["obstacles_body"], 0.0)
        if geometry.dtype != proprio.dtype or not torch.isfinite(geometry).all():
            raise ValueError("Invalid available geometry")
        tokens = torch.cat(
            [self.goal_projection(nav)[:, None], self.obstacle_projection(geometry)], 1
        )
        padding = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], 1)
        attended, _ = self.context_attention(
            self.context_query(condition)[:, None],
            tokens,
            tokens,
            key_padding_mask=padding,
            need_weights=False,
        )
        return condition + self.context_output(attended[:, 0])

    def initialize_motion_encoder(self, source):
        own = self.state_dict()
        keys = [k for k in own if not k.startswith(("goal_", "obstacle_", "context_"))]
        if any(k not in source or source[k].shape != own[k].shape for k in keys):
            raise ValueError("Motion encoder migration mismatch")
        self.load_state_dict({**own, **{k: source[k] for k in keys}}, strict=True)
        return keys


class ActionStudent(nn.Module):
    """Direct normalized-action head; flow sampler caches the observation encoder."""

    def __init__(
        self,
        kind="regression",
        width=256,
        layers=4,
        heads=8,
        context=False,
        flow_steps=8,
        residual_limit=0.0,
    ):
        super().__init__()
        if kind not in ("regression", "flow"):
            raise ValueError("Unknown action-student kind")
        if not math.isfinite(residual_limit) or not 0 <= residual_limit <= 0.2:
            raise ValueError("Residual limit outside bounded experiment range")
        if not isinstance(flow_steps, int) or not 1 <= flow_steps <= 64:
            raise ValueError("Invalid flow steps")
        self.kind, self.flow_steps = kind, flow_steps
        self.condition = PublicConditionEncoder(width, layers, heads, context)
        self.head = mlp([192 + (29 + 32 if kind == "flow" else 0), width * 2, width * 2, 29])
        self.register_buffer("action_mean", torch.zeros(29))
        self.register_buffer("action_std", torch.ones(29))
        self.register_buffer("frequencies", 2 ** torch.arange(16, dtype=torch.float32) * math.pi)
        self.residual_limit = residual_limit
        self.residual = None
        if residual_limit:
            self.residual = mlp([192 + 29, 128, 128, 29])
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def set_normalization(self, actions):
        if actions.ndim != 2 or actions.shape[1] != 29 or not torch.isfinite(actions).all():
            raise ValueError("Invalid training actions")
        self.action_mean.copy_(actions.mean(0))
        self.action_std.copy_(actions.std(0, unbiased=False).clamp_min(0.05))

    def velocity(self, value, time, condition):
        angles = time * self.frequencies
        return self.head(torch.cat([condition, value, angles.sin(), angles.cos()], -1))

    def base_action(self, condition, *, noise=None, steps=None):
        if self.kind == "regression":
            normalized = self.head(condition)
        else:
            if (
                noise is None
                or noise.shape != (len(condition), 29)
                or not torch.isfinite(noise).all()
            ):
                raise ValueError("Flow inference requires explicit finite B,29 noise")
            normalized = euler_flow(
                lambda value, time: self.velocity(value, time, condition),
                noise,
                self.flow_steps if steps is None else steps,
            )
        return normalized * self.action_std + self.action_mean

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
        condition = self.condition(proprio, controls, mask, context)
        base = self.base_action(condition, noise=noise, steps=steps)
        delta = torch.zeros_like(base)
        if self.residual is not None and residual_enabled:
            delta = self.residual_limit * self.residual(torch.cat([condition, base], -1)).tanh()
        return {"actions": base + delta, "base_actions": base, "residual": delta}

    def imitation_loss(self, batch, mask, *, context=None, residual_only=False):
        condition = self.condition(batch["proprio"], batch["controls"], mask, context)
        target = batch["teacher_actions"]
        normalized = (target - self.action_mean) / self.action_std
        if residual_only:
            if self.residual is None:
                raise ValueError("Residual-only fitting requires a residual head")
            base = self.base_action(condition, noise=torch.zeros_like(target)).detach()
            delta = (
                self.residual_limit
                * self.residual(torch.cat([condition.detach(), base], -1)).tanh()
            )
            mse = (base + delta - target).square().mean()
            return {
                "loss": mse + 0.01 * delta.square().mean(),
                "action_mse": mse,
                "residual_mse": delta.square().mean(),
            }
        if self.kind == "regression":
            predicted = self.head(condition)
            loss = (predicted - normalized).square().mean()
            mse = ((predicted - normalized) * self.action_std).square().mean()
            return {"loss": loss, "action_mse": mse}
        noise = torch.randn_like(normalized)
        time = torch.rand(len(target), 1, device=target.device)
        value, target_velocity = flow_path(normalized, noise, time)
        predicted = self.velocity(value, time, condition)
        loss = (predicted - target_velocity).square().mean()
        # Endpoint estimate is a denoising diagnostic, not an integrated rollout action.
        endpoint = value + (1 - time) * predicted
        endpoint_mse = ((endpoint - normalized) * self.action_std).square().mean()
        return {"loss": loss, "flow_velocity_mse": loss, "endpoint_action_mse": endpoint_mse}
