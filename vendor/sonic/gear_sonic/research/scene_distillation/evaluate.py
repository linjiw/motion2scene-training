"""Physical public-prior evaluation through the unchanged native tracking scorer."""

import torch

from gear_sonic.research.hindsight_training.qualify import TrackingQualificationCallback
from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.hindsight_training.student import FrozenSonicDecoder
from gear_sonic.research.scene_distillation.collect import native_commands
from gear_sonic.research.scene_distillation.commands import PROFILES, MaskedMotionFoundation


class FoundationEvaluationCallback(TrackingQualificationCallback):
    def __init__(self, *args, student_checkpoint, teacher_sha256, command_profile, **kwargs):
        super().__init__(*args, **kwargs)
        if command_profile not in PROFILES:
            raise ValueError("Unknown foundation command profile")
        self.student_checkpoint = student_checkpoint
        self.teacher_sha256 = teacher_sha256
        self.command_profile = command_profile

    def _get_inference_policy(self, device=None):
        device = self.env.device if device is None else device
        saved = torch.load(self.student_checkpoint, map_location=device, weights_only=False)
        if saved["config"].get("precision") == "fp32":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if saved["stage"] != "foundation" or saved["teacher_sha256"] != self.teacher_sha256:
            raise ValueError("Student/teacher boundary mismatch")
        if sha(saved["config"]["teacher_checkpoint"]) != self.teacher_sha256:
            raise ValueError("Selected teacher file changed")
        foundation = MaskedMotionFoundation().to(device).eval()
        foundation.load_state_dict(saved["model"], strict=True)
        decoder = FrozenSonicDecoder(self.model.policy.state_dict()).to(device).eval()
        self.foundation = foundation
        self.decoder = decoder

        def policy(obs_dict, **kwargs):
            controls, mask = native_commands(
                self.env.motion_command, self.env.env.scene.env_origins
            )
            chosen = torch.zeros_like(mask)
            chosen[:, list(PROFILES[self.command_profile])] = True
            proprio = obs_dict["actor_obs"]
            if self.model.policy.running_mean_std is not None:
                proprio = self.model.policy.running_mean_std(proprio)
            result = foundation.prior_step(proprio, controls, mask & chosen)
            return decoder(result["tokens"], proprio)

        return policy
