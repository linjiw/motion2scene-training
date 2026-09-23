"""Declared privileged continuation probes from actual navigation arrival states."""

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.motor_runtime import (
    current_orientation_observation,
    load_motor,
    motor_commands,
)
from gear_sonic.research.scene_distillation.navigation_motor_runtime import NavigationMotorCallback


class NavigationTakeoverCallback(NavigationMotorCallback):
    """Switch in the same episode, preserving physics, observation and controller history.

    The diagnostic uses the nominal reference clock without a phase jump. Commands
    are reconstructed at each measured state, never copied from a demonstration.
    These probes do not certify arbitrary recovery or supply deployment inputs.
    """

    supports_observation_ablation = False

    def _begin_task(self, env, teacher, task):
        super()._begin_task(env, teacher, task)
        if sha(self.config["motor_checkpoint"]) != self.config["motor_sha256"]:
            raise ValueError("Takeover motor changed")
        # Constructing an extra model must not perturb simulator observation RNG.
        with torch.random.fork_rng():
            self.recovery_motor = load_motor(
                self.config["motor_checkpoint"], self.config["teacher_sha256"], env.device
            )
        self.probe_tick = 0
        self.switch = None

    def _student_action(self, student, env, teacher, observation, task, noise):
        tick = self.probe_tick
        self.probe_tick += 1
        if tick < self.config["takeover_tick"]:
            return super()._student_action(student, env, teacher, observation, task, noise)
        if self.switch is None:
            position = env.motion_command.robot_anchor_pos_w[0] - env.env.scene.env_origins[0]
            self.switch = dict(
                decision_tick=tick,
                time_s=tick * 0.02,
                root_xyz=position.cpu().tolist(),
                goal_distance_m=float(np.linalg.norm(position.cpu().numpy() - task["goal_xyz"])),
                speed_mps=float(
                    torch.linalg.vector_norm(env.motion_command.robot_anchor_lin_vel_w[0])
                ),
                command_source="nominal-clock reference reconstructed at actual measured state",
                reference_phase_switch=False,
            )
        motor, decoder, config = self.recovery_motor
        commands, available = motor_commands(
            env.motion_command,
            env.env.scene.env_origins,
            config.get("current_frame_extension", False),
            current_orientation_observation(teacher.actor_module, observation),
        )
        proprio = observation["actor_obs"]
        if teacher.running_mean_std is not None:
            proprio = teacher.running_mean_std(proprio)
        return decoder(motor.prior_step(proprio, commands, available)["tokens"], proprio)

    def _complete_task(self, task, score, output):
        trace = np.load(output / "trace.npz")
        # Separate post-switch stabilization from a hold accumulated by the student.
        from gear_sonic.research.scene_distillation.direct_context import score_navigation_task

        start = self.config["takeover_tick"]
        post = None
        if start < len(trace["speed"]):
            post = score_navigation_task(
                task,
                trace["root_xyz"][start:],
                trace["speed"][start:],
                trace["undesired_force"][start:],
                fell=score["fell"],
            )
        write_new(
            output / "takeover.json",
            dict(
                diagnostic_only=True,
                intervention=self.switch,
                post_switch_score=post,
                unassisted_navigation_success=False,
            ),
        )


class OriginalTeacherContinuationCallback(NavigationMotorCallback):
    """Diagnostic original-teacher takeover with matching navigation construction.

    Emits no motor-recovery receipt or training shard. These actions cannot be
    silently used as executed frozen-motor targets by the recovery loader.
    """

    supports_observation_ablation = False

    def _begin_task(self, env, teacher, task):
        super()._begin_task(env, teacher, task)
        if not 0 < self.config["takeover_tick"] < task["deadline_ticks"]:
            raise ValueError("Invalid original-teacher continuation entry")
        self.teacher_tick = 0

    def _student_action(self, student, env, teacher, observation, task, noise):
        tick = self.teacher_tick
        self.teacher_tick += 1
        if tick < self.config["takeover_tick"]:
            return super()._student_action(student, env, teacher, observation, task, noise)
        return teacher.act_inference(obs_dict=observation, skip_episode_attnmask=True)

    def _goal_stop(self, task, roots, speeds, forces, fell):
        from gear_sonic.research.scene_distillation.navigation_recovery import suffix_support

        if len(roots) <= self.config["takeover_tick"]:
            return super()._goal_stop(task, roots, speeds, forces, fell)
        mask, _ = suffix_support(
            task,
            dict(
                root_xyz=np.asarray(roots),
                speed=np.asarray(speeds),
                undesired_force=np.asarray(forces),
            ),
            self.config["takeover_tick"],
            fell,
        )
        return bool(mask.any())

    def _complete_task(self, task, score, output):
        from gear_sonic.research.scene_distillation.navigation_continuation import (
            continuation_outcomes,
        )

        with np.load(output / "trace.npz") as trace:
            result = continuation_outcomes(
                task, trace, min(self.config["takeover_tick"], score["control_steps"])
            )
        write_new(
            output / "teacher-continuation.json",
            dict(
                diagnostic_only=True,
                takeover_tick=self.config["takeover_tick"],
                unassisted_navigation_result=False,
                training_admitted=False,
                provider="original_reference_teacher",
                reference_phase_switch=False,
                outcomes=result,
            ),
        )
