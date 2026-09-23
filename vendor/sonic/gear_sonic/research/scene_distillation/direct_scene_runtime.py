"""Single-scene task evaluation with no reference-pose termination or hidden reset."""

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.action_native import load_action_student
from gear_sonic.research.scene_distillation.collect import native_commands
from gear_sonic.research.scene_distillation.direct_context import (
    score_navigation_task,
    task_context,
)
from gear_sonic.research.scene_distillation.tasks import validate_task, verify_native_motion
from gear_sonic.research.scene_distillation.train_action_flow import profile_mask


class DirectSceneTaskCallback:
    """A research evaluator, never a scene-teacher qualification certificate.

    Uses pair-resolved 200Hz contact measurements. Stops at goal hold, contact,
    a declared absolute pelvis-height fall guard, or the task deadline. The
    reference cursor is clamped at its last pose to prevent automatic teleport.
    """

    def __init__(self, stage_config, **kwargs):
        self.config = json.loads(Path(stage_config).read_text())

    def _begin_task(self, env, teacher, task):
        pass

    def _record_teacher_query(self, env, teacher, task, observation, action):
        pass

    def _complete_task(self, task, score, output):
        pass

    def _task_result_fields(self, task, roots):
        """Extra task-result.json fields. None by default, so default receipts are unchanged."""
        return {}

    def _goal_stop(self, task, roots, speeds, forces, fell):
        return score_navigation_task(task, roots, speeds, forces, fell=fell)["navigation_success"]

    def _load_student(self, device):
        c = self.config
        return load_action_student(c["student_checkpoint"], c["teacher_sha256"], device)

    def _student_action(self, student, env, teacher, observation, task, noise):
        c = self.config
        command = env.motion_command
        controls, available = native_commands(command, env.env.scene.env_origins)
        context = None
        if student.condition.context_enabled:
            row = task_context(
                task,
                (command.robot_anchor_pos_w[0] - env.env.scene.env_origins[0]).cpu().numpy(),
                command.robot_anchor_quat_w[0].cpu().numpy(),
            )
            context = {k: torch.as_tensor(v, device=env.device)[None] for k, v in row.items()}
        proprio = observation["actor_obs"]
        if teacher.running_mean_std is not None:
            proprio = teacher.running_mean_std(proprio)
        return student.actions(
            proprio,
            controls,
            profile_mask(available, c["command_profile"]),
            context=context,
            noise=noise,
            residual_enabled=c.get("residual_enabled", True),
        )["actions"]

    def on_step_end(self, args, state, control, **kwargs):
        c = self.config
        if c.get("collect_to_deadline", False) and not c.get("teacher_mode", False):
            raise ValueError("Extended collection is restricted to teacher episodes")
        output = Path(c["output"])
        output.mkdir(parents=True, exist_ok=False)
        env, teacher = kwargs["env"], kwargs["model"].policy
        if sha(c["teacher_checkpoint"]) != c["teacher_sha256"]:
            raise ValueError("Teacher changed")
        task = validate_task(json.loads(Path(c["task_path"]).read_text()))
        if env.num_envs != 1 or sha(env.config.scene_usd_path) != task["scene_usd_sha256"]:
            raise ValueError("Scene evaluator requires one exactly bound collision scene")
        teacher.eval()
        teacher.eval_mode()
        env.set_is_evaluating(True)
        observation = env.reset_all()
        teacher.init_rollout()
        verify_native_motion(task, env._motion_lib)
        self._begin_task(env, teacher, task)
        student = None
        if not c.get("teacher_mode", False):
            if sha(c["student_checkpoint"]) != c["student_sha256"]:
                raise ValueError("Student changed")
            student = self._load_student(env.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        command = env.motion_command
        manager = env.env.termination_manager
        original_compute = manager.compute
        original_step = env.env.sim.step
        names = list(env.env.scene["robot"].body_names)
        sensors = [env.env.scene.sensors[f"navigation_contact_{n}"] for n in names]
        for sensor in sensors:
            if sensor.contact_physx_view.filter_count != len(task["obstacles"]) + 1:
                raise ValueError("Missing pair-resolved task contacts")
        nonfeet = [
            i
            for i, n in enumerate(names)
            if n not in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        contact_frames, roots, speeds, forces = [], [], [], []
        measurements = {}

        def sim_step(*a, **kw):
            result = original_step(*a, **kw)
            f = np.stack(
                [
                    s.contact_physx_view.get_contact_force_matrix(dt=0.005)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1, 3)
                    for s in sensors
                ]
            )
            contact_frames.append(f)
            return result

        def compute(*a, **kw):
            # Capture physical data at the native pre-reset boundary. Clear both
            # buffers so the native manager cannot reset on a reference mismatch.
            original_compute(*a, **kw)
            measurements["root"] = (
                (command.robot_anchor_pos_w[0] - env.env.scene.env_origins[0])
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            measurements["speed"] = float(
                torch.linalg.vector_norm(command.robot_anchor_lin_vel_w[0])
            )
            manager._terminated_buf.zero_()
            manager._truncated_buf.zero_()
            return manager._terminated_buf | manager._truncated_buf

        manager.compute = compute
        env.env.sim.step = sim_step
        fell, reason = False, "deadline"
        noise = torch.zeros(1, 29, device=env.device)
        max_steps = min(task["deadline_ticks"], c.get("max_steps", 3000))
        if not 1 <= max_steps <= 3000:
            raise ValueError("Invalid scene task horizon")
        try:
            with torch.no_grad():
                for tick in range(max_steps):
                    # _update_command increments then auto-resamples on exhaustion.
                    # Holding below that boundary keeps reference-only bookkeeping
                    # from resetting the robot in a goal-only evaluation.
                    last = env._motion_lib.get_time_step_total(command.motion_ids) - 2
                    command.time_steps.copy_(
                        torch.minimum(command.time_steps, last - command.motion_start_time_steps)
                    )
                    if student is None:
                        action = teacher.act_inference(
                            obs_dict=observation, skip_episode_attnmask=True
                        )
                        self._record_teacher_query(env, teacher, task, observation, action)
                    else:
                        action = self._student_action(
                            student, env, teacher, observation, task, noise
                        )
                    before = len(contact_frames)
                    measurements.clear()
                    observation, _, done, _ = env.step({"actions": action})
                    if done.any() or not measurements or len(contact_frames) - before != 4:
                        raise RuntimeError("Unexpected reset or missing physical measurements")
                    f = np.linalg.norm(np.stack(contact_frames[before:]), axis=-1)
                    force = max(
                        float(f[:, :, 1:].max(initial=0)), float(f[:, nonfeet, 0].max(initial=0))
                    )
                    roots.append(measurements["root"])
                    speeds.append(measurements["speed"])
                    forces.append(force)
                    fell = bool(roots[-1][2] < 0.25)
                    score = score_navigation_task(task, roots, speeds, forces, fell=fell)
                    if (
                        fell
                        or force > 1
                        or (
                            self._goal_stop(task, roots, speeds, forces, fell)
                            and not c.get("collect_to_deadline", False)
                        )
                    ):
                        reason = "fall" if fell else ("contact" if force > 1 else "goal_hold")
                        break
            score = score_navigation_task(task, roots, speeds, forces, fell=fell)
            score.update(
                state="complete",
                stop_reason=reason,
                teacher_mode=student is None,
                actor_profile=c.get("actor_profile", "legacy_declared_command_profile"),
                teacher_sha256=c["teacher_sha256"],
                student_sha256=c.get("student_sha256"),
                task_sha256=sha(c["task_path"]),
                reference_cursor_clamped=True,
                collection_to_deadline=c.get("collect_to_deadline", False),
                termination_profile="goal_hold_contact_absolute_pelvis_height_0.25_deadline",
                control_dt=0.02,
                contact_dt=0.005,
                scene_teacher_qualified=False,
            )
            score.update(self._task_result_fields(task, roots))
            np.savez_compressed(
                output / "trace.npz",
                root_xyz=roots,
                speed=speeds,
                undesired_force=forces,
                contact_force_w=contact_frames,
                body_names=np.array(names),
            )
            write_new(output / "task-result.json", score)
            self._complete_task(task, score, output)
        finally:
            manager.compute = original_compute
            env.env.sim.step = original_step
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
