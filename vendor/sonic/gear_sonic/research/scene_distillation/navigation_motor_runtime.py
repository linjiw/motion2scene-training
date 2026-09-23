"""Native goal/map-only evaluation through the common independent task scorer."""

import torch

from gear_sonic.research.scene_distillation.direct_context import task_context
from gear_sonic.research.scene_distillation.direct_scene_runtime import DirectSceneTaskCallback
from gear_sonic.research.scene_distillation.motor_runtime import (
    current_orientation_observation,
    load_motor,
    motor_commands,
)
from gear_sonic.research.scene_distillation.navigation_ablation import (
    ObservationAblation,
    ablation_record,
    observation_task,
    reference_endpoint,
)
from gear_sonic.research.scene_distillation.navigation_localization import CausalLocalization
from gear_sonic.research.scene_distillation.navigation_motor import NavigationInput, load_navigation


class NavigationMotorCallback(DirectSceneTaskCallback):
    # Opt-in stage-config flags (goal_rotation_deg, zero_obstacles) change only the actor's
    # view of the task; scoring, physics and the reference keep the original task.
    ablation = ObservationAblation()
    # Subclasses that record training rows or privileged diagnostics opt out.
    supports_observation_ablation = True

    def _begin_task(self, env, teacher, task):
        ablation = ObservationAblation.from_config(self.config)
        if ablation.active and not self.supports_observation_ablation:
            raise ValueError("Observation ablations are evaluation-only (nav mode)")
        self.ablation = ablation
        self.localization = CausalLocalization()
        self.decision_tick = 0

    def _task_result_fields(self, task, roots):
        if not self.ablation.active:
            return {}
        return dict(
            navigation_ablation=ablation_record(
                task, self.ablation, roots[-1], reference_endpoint(task)
            )
        )

    def _load_student(self, device):
        if self.config.get("actor_profile") not in (
            "nav_goal_map_v1",
            "nav_goal_map_localization_v2",
        ):
            raise ValueError("Navigation actor profile must be explicit")
        student = load_navigation(
            self.config["student_checkpoint"], self.config["teacher_sha256"], device
        )
        if self.config["actor_profile"] != student.actor_profile:
            raise ValueError("Runtime and checkpoint observation profiles differ")
        return student

    def _navigation_actor(self, student, env, teacher, observation, task):
        task = observation_task(task, self.ablation)
        command = env.motion_command
        context = task_context(
            task,
            (command.robot_anchor_pos_w[0] - env.env.scene.env_origins[0]).cpu().numpy(),
            command.robot_anchor_quat_w[0].cpu().numpy(),
        )
        proprio = observation["actor_obs"]
        if teacher.running_mean_std is not None:
            proprio = teacher.running_mean_std(proprio)
        if student.use_localization:
            features = self.localization.update(
                (command.robot_anchor_pos_w[0] - env.env.scene.env_origins[0]).cpu().numpy(),
                command.robot_anchor_quat_w[0].cpu().numpy(),
                self.decision_tick * 0.02,
                task["goal_xyz"],
            )
            context["localization"] = features
            self.decision_tick += 1
        actor = NavigationInput(
            proprio=proprio,
            **{k: torch.as_tensor(v, device=env.device)[None] for k, v in context.items()},
        )
        return actor

    def _student_action(self, student, env, teacher, observation, task, noise):
        actor = self._navigation_actor(student, env, teacher, observation, task)
        return student.navigation_step(actor)["actions"]


class FullMotorTaskCallback(DirectSceneTaskCallback):
    """Matched task control supplied with current full commands, explicitly disclosed."""

    def _load_student(self, device):
        if self.config.get("actor_profile") != "motion_full_current_v2":
            raise ValueError("Full-command task comparator must disclose its inputs")
        return load_motor(self.config["student_checkpoint"], self.config["teacher_sha256"], device)

    def _student_action(self, student, env, teacher, observation, task, noise):
        model, decoder, config = student
        commands, available = motor_commands(
            env.motion_command,
            env.env.scene.env_origins,
            config.get("current_frame_extension", False),
            current_orientation_observation(teacher.actor_module, observation),
        )
        proprio = observation["actor_obs"]
        if teacher.running_mean_std is not None:
            proprio = teacher.running_mean_std(proprio)
        return decoder(model.prior_step(proprio, commands, available)["tokens"], proprio)


class NavigationFullCommandControlCallback(DirectSceneTaskCallback):
    """Use the same checkpoint construction/RNG as navigation, bypassing its adapter.

    This diagnostic receives full commands. It is not goal-only task performance.
    """

    def _load_student(self, device):
        if self.config.get("actor_profile") != "motion_full_current_v2":
            raise ValueError("Matched motor control must disclose full-command inputs")
        return load_navigation(
            self.config["student_checkpoint"], self.config["teacher_sha256"], device
        )

    def _student_action(self, student, env, teacher, observation, task, noise):
        commands, available = motor_commands(
            env.motion_command,
            env.env.scene.env_origins,
            True,
            current_orientation_observation(teacher.actor_module, observation),
        )
        proprio = observation["actor_obs"]
        if teacher.running_mean_std is not None:
            proprio = teacher.running_mean_std(proprio)
        return student.full_step(proprio, commands, available)["actions"]
