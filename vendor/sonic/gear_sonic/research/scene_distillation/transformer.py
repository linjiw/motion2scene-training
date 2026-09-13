"""Transformer-conditioned BFM prior with explicit masked command tokens."""

import torch
from torch import nn
from torch.nn import functional as F

from gear_sonic.research.hindsight_training.student import mlp
from gear_sonic.research.scene_distillation.commands import MaskedMotionFoundation
from gear_sonic.research.scene_distillation.cvae import masked_controls


class TransformerMotionFoundation(MaskedMotionFoundation):
    """Attend over ten feature blocks and eight command blocks, all public.

    Feature blocks partition native history storage, not chronological frames.
    Only posterior_step accepts privileged teacher inputs. The token adapter has
    no independent history bypass around the conditioned behavior latent.
    """

    def __init__(self, width=256, layers=4, heads=8, latent_dim=64):
        super().__init__(latent_dim)
        if width < 32 or layers < 1 or heads < 1 or width % heads:
            raise ValueError("Invalid transformer dimensions")
        # Replace inherited condition networks; avoid unused trainable parameters.
        del self.proprio, self.command
        self.history_projection = nn.Linear(93, width)
        self.command_projection = nn.Linear(20, width)
        self.position = nn.Parameter(torch.randn(1, 18, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            width * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        # TransformerEncoder clones initial weights; initialize each layer independently.
        for block in self.encoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.readout = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, 192))
        self.prior = mlp([192, width, width, 2 * latent_dim])
        self.posterior = mlp([192 + 1645 + 640, width * 2, width, 2 * latent_dim])
        self.token_adapter = mlp([latent_dim, width * 2, width * 2, 64])

    def _condition(self, proprio, controls=None, control_mask=None):
        if proprio.ndim != 2 or proprio.shape[1] != 930 or not torch.isfinite(proprio).all():
            raise ValueError("Expected finite public B,930 native history")
        if controls is None and control_mask is None:
            controls = proprio.new_zeros(len(proprio), 79)
            control_mask = torch.zeros_like(controls, dtype=torch.bool)
        if controls is None or control_mask is None:
            raise ValueError("Commands and masks must be supplied together")
        if controls.shape != (len(proprio), 79) or control_mask.shape != controls.shape:
            raise ValueError("Expected B,79 commands and availability")
        masked_controls(proprio, controls[:, :8], control_mask[:, :8])
        clean = torch.where(control_mask, controls, 0.0)
        if not torch.isfinite(clean).all():
            raise ValueError("Non-finite available command")
        values = F.pad(clean, (0, 1)).reshape(-1, 8, 10)
        masks = F.pad(control_mask.float(), (0, 1)).reshape(-1, 8, 10)
        command = self.command_projection(torch.cat([values, masks], -1))
        history = self.history_projection(proprio.reshape(-1, 10, 93))
        encoded = self.encoder(torch.cat([history, command], 1) + self.position)
        return self.readout(torch.cat([encoded[:, :10].mean(1), encoded[:, 10:].mean(1)], -1))

    def tokens(self, proprio, latent):
        if latent.shape != (len(proprio), self.latent_dim) or not torch.isfinite(latent).all():
            raise ValueError("Invalid behavior latent")
        continuous = self.token_adapter(latent).tanh() * (31 / 32) - 1 / 32
        return continuous + ((continuous * 16).round() / 16 - continuous).detach()
