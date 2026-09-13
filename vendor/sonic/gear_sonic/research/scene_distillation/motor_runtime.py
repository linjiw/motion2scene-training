"""Native motor evaluation and explicit teacher-lookahead diagnostic controls."""

import math

import torch

from gear_sonic.research.hindsight_training.qualify import TrackingQualificationCallback
from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.collect import native_commands
from gear_sonic.research.scene_distillation.commands import PROFILES
from gear_sonic.research.scene_distillation.motor_training import build_motor
from gear_sonic.research.scene_distillation.reference_layout import pack_reference, unpack_reference
from gear_sonic.research.scene_distillation.train import make_decoder


def current_orientation_observation(actor_module, observation):
    """Read just the current orientation term, including native observation noise."""
    offset = 0
    for name in actor_module.tokenizer_obs_names:
        dims = tuple(actor_module.tokenizer_obs_dims[name])
        if name == "motion_anchor_ori_b_mf_nonflat":
            if dims != (10, 6):
                raise ValueError("Unexpected current-orientation observation schema")
            return observation["tokenizer"][..., offset : offset + 6].reshape(-1, 6)
        offset += math.prod(dims)
    raise ValueError("Missing current public orientation observation")


def motor_commands(command, origins, extended, orientation_observation=None):
    controls, mask = native_commands(command, origins)
    if extended:
        orientation = command.root_rot_dif_l_multi_future.reshape(
            command.num_envs, command.num_future_frames, 6
        )[:, 0]
        if orientation_observation is not None:
            orientation = orientation_observation
        extra = torch.cat([command.joint_vel, orientation], -1)
        controls = torch.cat([controls, extra], -1)
        mask = torch.cat([mask, torch.ones_like(extra, dtype=torch.bool)], -1)
    return controls, mask


def load_motor(path, teacher_sha, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved["stage"] != "motor_foundation" or saved["teacher_sha256"] != teacher_sha:
        raise ValueError("Motor checkpoint boundary mismatch")
    c = saved["config"]
    if sha(c["teacher_checkpoint"]) != teacher_sha:
        raise ValueError("Teacher changed")
    model = build_motor(c).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    return model, make_decoder(c["teacher_checkpoint"], device), c


class MotorEvaluationCallback(TrackingQualificationCallback):
    def __init__(self, *args, student_checkpoint, teacher_sha256, command_profile="full", **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint, self.teacher_sha, self.profile = (
            student_checkpoint,
            teacher_sha256,
            command_profile,
        )
        if command_profile not in PROFILES:
            raise ValueError("Invalid command mode")

    def _get_inference_policy(self, device=None):
        device = device or self.env.device
        model, decoder, config = load_motor(self.checkpoint, self.teacher_sha, device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        def policy(obs_dict, **kwargs):
            controls, mask = motor_commands(
                self.env.motion_command,
                self.env.env.scene.env_origins,
                config.get("current_frame_extension", False),
                (
                    current_orientation_observation(self.model.policy.actor_module, obs_dict)
                    if config.get("current_frame_extension", False)
                    else None
                ),
            )
            if self.profile != "full":
                chosen = torch.zeros_like(mask)
                chosen[:, list(PROFILES[self.profile])] = True
                mask &= chosen
            proprio = obs_dict["actor_obs"]
            if self.model.policy.running_mean_std is not None:
                proprio = self.model.policy.running_mean_std(proprio)
            result = model.prior_step(proprio, controls, mask)
            return decoder(result["tokens"], proprio)

        return policy


def hold_reference_tail(inputs, frames):
    if inputs.shape[-1] != 640 or not 1 <= frames <= 10:
        raise ValueError("Expected ten native 64D reference frames")
    original = unpack_reference(inputs)
    result = original.clone()
    result[..., frames:, :] = original[..., frames - 1 : frames, :]
    return pack_reference(result)


class TeacherHorizonCallback(TrackingQualificationCallback):
    """Teacher input ablation, never reported as a trained public student."""

    def __init__(self, *args, horizon_frames=10, **kwargs):
        super().__init__(*args, **kwargs)
        if not 1 <= horizon_frames <= 10:
            raise ValueError("Invalid teacher horizon")
        self.horizon_frames = horizon_frames

    def _get_inference_policy(self, device=None):
        def modify(module, inputs):
            physical = unpack_reference(inputs[0]).reshape(-1, 10, 64)[:, 0]
            controls, _ = motor_commands(
                self.env.motion_command, self.env.env.scene.env_origins, True
            )
            current = torch.cat([controls[:, 50:79], controls[:, 79:]], -1)
            torch.testing.assert_close(physical[:, :58], current[:, :58], atol=2e-5, rtol=2e-5)
            # The native observation manager adds uniform +/-0.05 rotation-feature noise.
            torch.testing.assert_close(physical[:, 58:], current[:, 58:], atol=0.05001, rtol=0)
            return (hold_reference_tail(inputs[0], self.horizon_frames),)

        self.model.policy.actor_module.encoders["g1"].module[0].register_forward_pre_hook(modify)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return super()._get_inference_policy(device)
