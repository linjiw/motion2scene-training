"""BFM-inspired privileged posterior/public prior, adapted to frozen SONIC tokens.

This is a new, untrained navigation candidate, not the authors' BFM implementation.
The 32D Gaussian behavior latent is distinct from SONIC's 64D quantized code.
No function in this module executes physics or establishes obstacle clearance.
"""

import torch

from gear_sonic.research.hindsight_training.student import mlp
from gear_sonic.research.scene_distillation.policy import PublicNavigationEncoder

CONTROL_FIELDS = (
    "goal_yaw_sin",
    "goal_yaw_cos",
    "velocity_body_x",
    "velocity_body_y",
    "velocity_body_z",
    "base_height",
    "yaw_rate",
    "arrival_speed",
)


def masked_controls(proprio, controls=None, control_mask=None):
    """Absence has its own bit; absent controls are not a command to become zero.

    Start/goal and known obstacle geometry are mandatory public observations.
    This mask concerns optional task commands, never obstacle visibility.
    """
    batch = len(proprio)
    if controls is None and control_mask is None:
        return proprio.new_zeros(batch, 16)
    if controls is None or control_mask is None:
        raise ValueError("Optional controls need both values and explicit availability bits")
    if (
        controls.shape != (batch, 8)
        or control_mask.shape != (batch, 8)
        or controls.dtype != proprio.dtype
        or control_mask.dtype != torch.bool
        or controls.device != proprio.device
        or control_mask.device != proprio.device
    ):
        raise ValueError("Invalid optional-control shape, dtype or device")
    if not torch.equal(control_mask[:, 0], control_mask[:, 1]):
        raise ValueError("Yaw sine/cosine are one command and must be masked together")
    clean = torch.where(control_mask, controls, 0.0)
    if not torch.isfinite(clean).all():
        raise ValueError("Non-finite available control")
    yaw = control_mask[:, 0]
    if yaw.any() and not torch.allclose(
        clean[yaw, :2].square().sum(-1), proprio.new_ones(int(yaw.sum())), atol=1e-4, rtol=1e-4
    ):
        raise ValueError("Available yaw must have unit sine/cosine norm")
    return torch.cat([clean, control_mask.float()], dim=-1)


def diagonal_gaussian_kl(q_mean, q_logvar, p_mean, p_logvar):
    """KL(q || p), summed over behavior dimensions, one value per sample."""
    if not (q_mean.shape == q_logvar.shape == p_mean.shape == p_logvar.shape):
        raise ValueError("Gaussian parameter shapes must agree")
    if not all(torch.isfinite(x).all() for x in (q_mean, q_logvar, p_mean, p_logvar)):
        raise ValueError("Non-finite Gaussian parameters")
    return 0.5 * (
        p_logvar
        - q_logvar
        + (q_logvar - p_logvar).exp()
        + (q_mean - p_mean).square() * (-p_logvar).exp()
        - 1
    ).sum(-1)


class VariationalNavigationStudent(PublicNavigationEncoder):
    """Prior-only acting API; separate training-only posterior API.

    Noise is explicit to make rollout sampling reproducible. None uses the prior
    mean; a collector may persist a sampled epsilon across a registered chunk.
    A Gaussian latent alone does not guarantee consistent route choices.
    """

    def __init__(self, hidden_dim=128, latent_dim=32):
        super().__init__(hidden_dim)
        self.latent_dim = latent_dim
        self.prior = mlp([hidden_dim + 16, 128, 2 * latent_dim])
        # Native critic1645 and target encoder640 are privileged, training only.
        self.posterior = mlp([hidden_dim + 16 + 1645 + 640, 256, 128, 2 * latent_dim])
        # No direct scene/goal bypass around the Gaussian behavior latent.
        self.token_adapter = mlp([128 + latent_dim, 128, 64])

    def _public(self, observation, hidden, episode_start, controls, control_mask):
        hidden, body, attention = self.encode_step(observation, hidden, episode_start=episode_start)
        commands = masked_controls(observation["proprio"], controls, control_mask)
        mean, logvar = self.prior(torch.cat([hidden, commands], -1)).chunk(2, -1)
        return hidden, body, attention, commands, mean, logvar.clamp(-8, 4)

    def _tokens(self, body, mean, logvar, epsilon):
        if epsilon is None:
            epsilon = torch.zeros_like(mean)
        if (
            epsilon.shape != mean.shape
            or epsilon.dtype != mean.dtype
            or epsilon.device != mean.device
            or not torch.isfinite(epsilon).all()
        ):
            raise ValueError("Latent noise must match Gaussian mean and be finite")
        latent = mean + (0.5 * logvar).exp() * epsilon
        continuous = self.token_adapter(torch.cat([body, latent], -1)).tanh() * (31 / 32) - 1 / 32
        quantized = (continuous * 16).round() / 16
        return continuous + (quantized - continuous).detach(), latent

    def prior_step(
        self,
        observation,
        hidden=None,
        *,
        episode_start=None,
        controls=None,
        control_mask=None,
        epsilon=None
    ):
        """Deployment entry point: no privileged state or future reference argument."""
        hidden, body, attention, _, mean, logvar = self._public(
            observation, hidden, episode_start, controls, control_mask
        )
        tokens, latent = self._tokens(body, mean, logvar, epsilon)
        return {
            "tokens": tokens,
            "latent": latent,
            "hidden": hidden,
            "prior_mean": mean,
            "prior_logvar": logvar,
            "obstacle_attention": attention,
        }

    def posterior_step(
        self,
        observation,
        privileged_state,
        future_reference,
        hidden=None,
        *,
        episode_start=None,
        controls=None,
        control_mask=None,
        epsilon=None
    ):
        """Training-only inference; labels/reference must be bound to the queried state."""
        hidden, body, attention, commands, p_mean, p_logvar = self._public(
            observation, hidden, episode_start, controls, control_mask
        )
        batch = len(hidden)
        for value, width in ((privileged_state, 1645), (future_reference, 640)):
            if (
                value.shape != (batch, width)
                or value.device != hidden.device
                or value.dtype != hidden.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Invalid privileged posterior input")
        residual, q_logvar = self.posterior(
            torch.cat([hidden, commands, privileged_state.detach(), future_reference.detach()], -1)
        ).chunk(2, -1)
        q_mean, q_logvar = p_mean + residual, q_logvar.clamp(-8, 4)
        tokens, latent = self._tokens(body, q_mean, q_logvar, epsilon)
        return {
            "tokens": tokens,
            "latent": latent,
            "hidden": hidden,
            "prior_mean": p_mean,
            "prior_logvar": p_logvar,
            "posterior_mean": q_mean,
            "posterior_logvar": q_logvar,
            "kl": diagonal_gaussian_kl(q_mean, q_logvar, p_mean, p_logvar),
            "obstacle_attention": attention,
        }
