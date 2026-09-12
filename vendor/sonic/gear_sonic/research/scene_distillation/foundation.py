"""Scene-independent BFM-style behavior foundation for a SONIC decoder.

Constructors initialize untrained weights. No trained BFM prior is supplied here.
The compact eight-command profile is an adaptation, not BFM's complete interface.
"""

import torch
from torch import nn

from gear_sonic.research.hindsight_training.student import mlp
from gear_sonic.research.scene_distillation.cvae import diagonal_gaussian_kl, masked_controls


class MotionBehaviorFoundation(nn.Module):
    """Distill from a tracking teacher first; freeze before navigation adaptation.

    Public prior: measured proprioception and explicitly available motion commands.
    Training posterior: same inputs plus native privileged state/future target.
    Obstacles, scene IDs and global navigation goals are not foundation inputs.
    """

    def __init__(self, latent_dim=32):
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        self.latent_dim = latent_dim
        self.proprio = mlp([930, 256, 128])
        self.command = mlp([16, 64, 64])
        self.prior = mlp([192, 128, 2 * latent_dim])
        self.posterior = mlp([192 + 1645 + 640, 256, 128, 2 * latent_dim])
        self.token_adapter = mlp([128 + latent_dim, 128, 64])

    def _body(self, proprio):
        if (
            proprio.ndim != 2
            or proprio.shape[1] != 930
            or proprio.dtype != torch.float32
            or not torch.isfinite(proprio).all()
        ):
            raise ValueError("Foundation requires finite B,930 measured proprioception")
        return self.proprio(proprio)

    def _condition(self, proprio, controls, control_mask):
        body = self._body(proprio)
        commands = self.command(masked_controls(proprio, controls, control_mask))
        return torch.cat([body, commands], -1)

    def prior_stats(self, proprio, controls=None, control_mask=None):
        condition = self._condition(proprio, controls, control_mask)
        mean, logvar = self.prior(condition).chunk(2, -1)
        return mean, logvar.clamp(-8, 4)

    def tokens(self, proprio, latent):
        body = self._body(proprio)
        if (
            latent.shape != (len(proprio), self.latent_dim)
            or latent.dtype != proprio.dtype
            or latent.device != proprio.device
            or not torch.isfinite(latent).all()
        ):
            raise ValueError("Invalid behavior latent")
        continuous = self.token_adapter(torch.cat([body, latent], -1)).tanh() * (31 / 32) - 1 / 32
        quantized = (continuous * 16).round() / 16
        return continuous + (quantized - continuous).detach()

    def sample(self, mean, logvar, epsilon=None):
        if epsilon is None:
            epsilon = torch.zeros_like(mean)
        if (
            epsilon.shape != mean.shape
            or epsilon.dtype != mean.dtype
            or epsilon.device != mean.device
            or not torch.isfinite(epsilon).all()
        ):
            raise ValueError("Explicit latent noise must match the prior")
        return mean + (0.5 * logvar).exp() * epsilon

    def prior_step(self, proprio, controls=None, control_mask=None, *, epsilon=None):
        mean, logvar = self.prior_stats(proprio, controls, control_mask)
        latent = self.sample(mean, logvar, epsilon)
        return {
            "tokens": self.tokens(proprio, latent),
            "latent": latent,
            "prior_mean": mean,
            "prior_logvar": logvar,
        }

    def posterior_step(
        self,
        proprio,
        privileged_state,
        future_reference,
        controls=None,
        control_mask=None,
        *,
        epsilon=None
    ):
        condition = self._condition(proprio, controls, control_mask)
        p_mean, p_logvar = self.prior(condition).chunk(2, -1)
        p_logvar = p_logvar.clamp(-8, 4)
        for value, width in ((privileged_state, 1645), (future_reference, 640)):
            if (
                value.shape != (len(proprio), width)
                or value.dtype != proprio.dtype
                or value.device != proprio.device
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Invalid training-only foundation input")
        residual, q_logvar = self.posterior(
            torch.cat([condition, privileged_state.detach(), future_reference.detach()], -1)
        ).chunk(2, -1)
        q_mean, q_logvar = p_mean + residual, q_logvar.clamp(-8, 4)
        latent = self.sample(q_mean, q_logvar, epsilon)
        return {
            "tokens": self.tokens(proprio, latent),
            "latent": latent,
            "prior_mean": p_mean,
            "prior_logvar": p_logvar,
            "posterior_mean": q_mean,
            "posterior_logvar": q_logvar,
            "kl": diagonal_gaussian_kl(q_mean, q_logvar, p_mean, p_logvar),
        }
