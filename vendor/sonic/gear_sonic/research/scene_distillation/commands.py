"""Explicit 79-component root/keypoint/joint interface for the foundation pilot."""

import torch

from gear_sonic.research.hindsight_training.student import mlp
from gear_sonic.research.scene_distillation.cvae import masked_controls
from gear_sonic.research.scene_distillation.foundation import MotionBehaviorFoundation

COMMAND_DIM = 79
PROFILES = {
    "root": (0, 1, 2, 3, 4, 5, 6),
    "navigation": (2, 3, 5, 6),
    "keypoints": tuple(range(8, 50)),
    "joints": tuple(range(50, 79)),
    "full": tuple(range(79)),
}


class MaskedMotionFoundation(MotionBehaviorFoundation):
    """Current target commands; privileged multi-future reference stays in posterior.

    0:8 retain CONTROL_FIELDS; 8:50 are 14 body points in measured anchor frame;
    50:79 are 29 native-order target joint angles. This is our BFM-style interface.
    No command is implicitly available: its mask must be supplied.
    """

    command_dim = COMMAND_DIM

    def __init__(self, latent_dim=32):
        super().__init__(latent_dim)
        self.command = mlp([2 * COMMAND_DIM, 128, 64])

    def _condition(self, proprio, controls, control_mask):
        if controls is None and control_mask is None:
            controls = proprio.new_zeros(len(proprio), COMMAND_DIM)
            control_mask = torch.zeros_like(controls, dtype=torch.bool)
        elif controls is None or control_mask is None:
            raise ValueError("Commands and availability must be supplied together")
        if (
            controls.shape != (len(proprio), COMMAND_DIM)
            or control_mask.shape != controls.shape
            or controls.dtype != proprio.dtype
            or controls.device != proprio.device
            or control_mask.dtype != torch.bool
            or control_mask.device != proprio.device
        ):
            raise ValueError("Expected B,79 commands and boolean availability")
        masked_controls(proprio, controls[:, :8], control_mask[:, :8])
        clean = torch.where(control_mask, controls, 0.0)
        if not torch.isfinite(clean).all():
            raise ValueError("Non-finite available motion command")
        return torch.cat(
            [self._body(proprio), self.command(torch.cat([clean, control_mask.float()], -1))], -1
        )


def sample_command_mask(available, *, generator=None):
    """Equal mixture of declared profiles; intersect with actual available signals."""
    if available.ndim != 2 or available.shape[1] != COMMAND_DIM or available.dtype != torch.bool:
        raise ValueError("Expected B,79 availability")
    profiles = torch.zeros(len(PROFILES), COMMAND_DIM, dtype=torch.bool, device=available.device)
    for i, indices in enumerate(PROFILES.values()):
        profiles[i, list(indices)] = True
    chosen = torch.randint(
        len(PROFILES), (len(available),), device=available.device, generator=generator
    )
    return profiles[chosen] & available
