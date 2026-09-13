"""Execute and retain supported continuations through the exact frozen motor backend."""

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.direct_context import (
    score_navigation_task,
    task_context,
)
from gear_sonic.research.scene_distillation.motor_runtime import (
    current_orientation_observation,
    load_motor,
    motor_commands,
)
from gear_sonic.research.scene_distillation.navigation_localization import CausalLocalization
from gear_sonic.research.scene_distillation.navigation_motor_runtime import NavigationMotorCallback


def suffix_support(task, trace, switch_tick, fell):
    """Qualify the executed suffix independently of any prefix hold or contact."""
    n = len(trace["speed"])
    mask = np.zeros(n, dtype=bool)
    score = None
    if 0 <= switch_tick < n:
        score = score_navigation_task(
            task,
            trace["root_xyz"][switch_tick:],
            trace["speed"][switch_tick:],
            trace["undesired_force"][switch_tick:],
            fell=fell,
        )
        if score["navigation_success"] and (trace["undesired_force"] <= 1).all():
            mask[switch_tick:] = True
    return mask, score


class MotorRecoveryCollectionCallback(NavigationMotorCallback):
    """Motor demonstration at switch=0; learner-prefix recovery otherwise.

    Only actually motor-executed suffixes that complete the original task are
    admitted. The first supported row is on the navigation learner's distribution;
    subsequent rows are recovery continuation data, not additional DAgger queries.
    """

    def _begin_task(self, env, teacher, task):
        super()._begin_task(env, teacher, task)
        if task["split"] != "train" or self.config.get("teacher_mode"):
            raise ValueError("Motor recovery requires a train-task motor rollout")
        if not 0 <= self.config["takeover_tick"] < task["deadline_ticks"]:
            raise ValueError("Invalid motor recovery switch")
        self.rows = []
        self.record_localization = CausalLocalization()
        self.teacher_input = None

        def capture(module, inputs):
            self.teacher_input = inputs[0].detach().reshape(1, -1).clone()

        layer = teacher.actor_module.decoders["g1_dyn"].module[0]
        self.teacher_handle = layer.register_forward_pre_hook(capture)

    def _load_student(self, device):
        c = self.config
        if sha(c["motor_checkpoint"]) != c["motor_sha256"]:
            raise ValueError("Recovery motor changed")
        if c["takeover_tick"] == 0:
            if c["student_sha256"] != c["motor_sha256"]:
                raise ValueError("Motor demonstration must use its declared motor checkpoint")
            return load_motor(c["motor_checkpoint"], c["teacher_sha256"], device)
        saved = torch.load(c["student_checkpoint"], map_location="cpu", weights_only=False)
        if saved["config"]["motor_sha256"] != c["motor_sha256"]:
            raise ValueError("Navigation and recovery use different motor backends")
        return super()._load_student(device)

    def _student_action(self, student, env, teacher, observation, task, noise):
        tick = len(self.rows)
        if isinstance(student, tuple):
            motor, decoder, _ = student
        else:
            motor, decoder = student.motor, student.decoder
        proprio = observation["actor_obs"]
        if teacher.running_mean_std is not None:
            proprio = teacher.running_mean_std(proprio)
        commands, available = motor_commands(
            env.motion_command,
            env.env.scene.env_origins,
            True,
            current_orientation_observation(teacher.actor_module, observation),
        )
        target = motor.prior_step(proprio, commands, available)
        motor_action = decoder(target["tokens"], proprio)
        if tick < self.config["takeover_tick"]:
            action = super()._student_action(student, env, teacher, observation, task, noise)
        else:
            action = motor_action
        # Original-teacher actions are diagnostics only. Preserve simulator RNG;
        # the chosen physical action and the navigation observation cannot change.
        with torch.random.fork_rng():
            teacher_action = teacher.act_inference(obs_dict=observation, skip_episode_attnmask=True)
        if self.teacher_input is None:
            raise ValueError("Missing original teacher diagnostic")
        position = (
            (env.motion_command.robot_anchor_pos_w[0] - env.env.scene.env_origins[0]).cpu().numpy()
        )
        quaternion = env.motion_command.robot_anchor_quat_w[0].cpu().numpy()
        row = {
            k: v[0].detach().cpu().numpy().copy()
            for k, v in dict(
                proprio=proprio,
                controls=commands,
                control_mask=available,
                motor_actions=motor_action,
                actions=action,
                teacher_actions=teacher_action,
                teacher_tokens=self.teacher_input[:, :64],
            ).items()
        }
        row.update(task_context(task, position, quaternion))
        row.update(
            measured_root_xyz=position.copy(),
            measured_root_wxyz=quaternion.copy(),
            observation_time_s=np.asarray(tick * 0.02, dtype=np.float64),
            localization=self.record_localization.update(
                position, quaternion, tick * 0.02, task["goal_xyz"]
            ),
        )
        if not all(np.isfinite(v).all() for v in row.values()):
            raise ValueError("Nonfinite motor recovery row")
        self.rows.append(row)
        return action

    def _goal_stop(self, task, roots, speeds, forces, fell):
        trace = dict(
            root_xyz=np.asarray(roots), speed=np.asarray(speeds), undesired_force=np.asarray(forces)
        )
        mask, _ = suffix_support(task, trace, self.config["takeover_tick"], fell)
        return bool(mask.any())

    def _complete_task(self, task, score, output):
        self.teacher_handle.remove()
        if len(self.rows) != score["control_steps"]:
            raise ValueError("Recovery rows do not align with physics")
        with np.load(output / "trace.npz") as trace:
            mask, suffix = suffix_support(task, trace, self.config["takeover_tick"], score["fell"])
        arrays = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        arrays["query_mask"] = mask
        arrays["learner_query_mask"] = np.zeros(len(mask), dtype=bool)
        switch = self.config["takeover_tick"]
        if switch > 0 and mask.any():
            arrays["learner_query_mask"][switch] = True
        path = output / "motor-recovery.npz"
        np.savez_compressed(path, **arrays)
        write_new(
            output / "recovery.json",
            dict(
                schema="executed_motor_navigation_recovery_v1",
                supported=bool(mask.any()),
                source="motor_demonstration" if switch == 0 else "navigation_prefix_motor_recovery",
                takeover_tick=switch,
                rows=len(mask),
                supported_rows=int(mask.sum()),
                learner_query_rows=int(arrays["learner_query_mask"].sum()),
                motor_sha256=self.config["motor_sha256"],
                teacher_sha256=self.config["teacher_sha256"],
                behavior_checkpoint=dict(
                    path=self.config["student_checkpoint"], sha256=self.config["student_sha256"]
                ),
                task=dict(path=self.config["task_path"], sha256=sha(self.config["task_path"])),
                score=dict(
                    path=str(output / "task-result.json"), sha256=sha(output / "task-result.json")
                ),
                shard=dict(path=str(path), sha256=sha(path)),
                suffix_score=suffix,
                intervention=switch > 0,
                reference_phase_switch=False,
                not_an_unassisted_navigation_result=True,
            ),
        )
