"""Predict missing reference context while preserving the pretrained SONIC motor path."""

import torch
from torch import nn
from vector_quantize_pytorch import FSQ

from gear_sonic.research.hindsight_training.runtime import load_release_checkpoint, sha
from gear_sonic.research.hindsight_training.student import mlp
from gear_sonic.research.scene_distillation.reference_layout import pack_reference


class AnticipatoryMotor(nn.Module):
    """Full-command-only research student; future references are training targets.

    Initialization repeats the current target frame through the frozen encoder.
    A trainable predictor supplies the remaining nine frames from measured history
    and current public commands. This is deterministic prediction, not flow matching.
    """

    command_dim = 114

    def __init__(self, teacher_checkpoint, teacher_sha256, width=512):
        super().__init__()
        if sha(teacher_checkpoint) != teacher_sha256:
            raise ValueError("Teacher changed")
        self.encoder = mlp([640, 2048, 1024, 512, 512, 64])
        state = load_release_checkpoint(teacher_checkpoint)["policy_state_dict"]
        prefix = "actor_module.encoders.g1.module."
        self.encoder.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}, strict=True
        )
        self.encoder.requires_grad_(False)
        self.quantizer = FSQ([32] * 32)
        self.forecaster = mlp([930 + 114, width, width, 576])
        nn.init.zeros_(self.forecaster[-1].weight)
        nn.init.zeros_(self.forecaster[-1].bias)
        self.register_buffer("input_mean", torch.zeros(1044))
        self.register_buffer("input_std", torch.ones(1044))
        self.register_buffer("frame_std", torch.ones(64))
        self.register_buffer("normalized", torch.tensor(False))

    def initialize_normalization(self, data):
        if self.normalized:
            return
        with torch.no_grad():
            x = torch.cat([data["proprio"], data["controls"]], -1)
            self.input_mean.copy_(x.mean(0))
            self.input_std.copy_(x.std(0).clamp_min(0.05))
            current = torch.cat([data["controls"][:, 50:79], data["controls"][:, 79:]], -1)
            self.frame_std.copy_(current.std(0).clamp_min(0.05))
            self.normalized.fill_(True)

    def encode_reference(self, reference):
        latent = self.encoder(reference.reshape(-1, 640)).reshape(-1, 2, 32)
        tokens, _ = self.quantizer(latent)
        return tokens.reshape(-1, 64)

    def prior_step(self, proprio, controls, control_mask):
        if (
            proprio.shape != (len(controls), 930)
            or controls.shape[1] != 114
            or control_mask.shape != controls.shape
        ):
            raise ValueError("Anticipatory motor requires current full commands")
        required = control_mask.clone()
        required[:, 7] = True
        if not required.all():
            raise ValueError("This motor experiment is full-command only")
        clean = torch.where(control_mask, controls, 0)
        if not torch.isfinite(clean).all() or not torch.isfinite(proprio).all():
            raise ValueError("Nonfinite motor input")
        current = torch.cat([clean[:, 50:79], clean[:, 79:]], -1)
        x = (torch.cat([proprio, clean], -1) - self.input_mean) / self.input_std
        residual = self.forecaster(x.clamp(-20, 20)).reshape(-1, 9, 64) * self.frame_std
        forecast = torch.cat([current[:, None], current[:, None] + residual], 1)
        return dict(tokens=self.encode_reference(pack_reference(forecast)), forecast=forecast)
