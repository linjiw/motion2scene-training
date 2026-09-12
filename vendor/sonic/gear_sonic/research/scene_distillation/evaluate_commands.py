"""Native public-prior command telemetry alongside the existing tracking evaluation."""

from pathlib import Path

import numpy as np
import torch

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.collect import native_commands
from gear_sonic.research.scene_distillation.command_quality import command_metrics
from gear_sonic.research.scene_distillation.evaluate import FoundationEvaluationCallback


class CommandEvaluationCallback(FoundationEvaluationCallback):
    def _pre_evaluate_policy(self, reset_env=True):
        super()._pre_evaluate_policy(reset_env)
        if self.num_total_env_eval_loops != 1:
            raise ValueError("Command telemetry requires one resident motion batch")
        self.command_rows = []
        self.measured_rows = []

    def _pre_eval_env_step(self, actor_state):
        from isaaclab.utils.math import quat_apply_inverse

        actor_state = super()._pre_eval_env_step(actor_state)
        motion = self.env.motion_command
        q = motion.robot_anchor_quat_w
        controls, _ = native_commands(motion, self.env.env.scene.env_origins)
        velocity = quat_apply_inverse(q, motion.robot_anchor_lin_vel_w)
        angular = quat_apply_inverse(q, motion.robot_anchor_ang_vel_w)
        height = motion.robot_anchor_pos_w[:, 2] - self.env.env.scene.env_origins[:, 2]
        measured = torch.stack((velocity[:, 0], velocity[:, 1], angular[:, 2], height), dim=-1)
        self.command_rows.append(controls[:, [2, 3, 6, 5]].detach().cpu().numpy().copy())
        self.measured_rows.append(measured.detach().cpu().numpy().copy())
        return actor_state

    def _post_evaluate_policy(self, eval_res):
        result = super()._post_evaluate_policy(eval_res)
        command, measured = np.stack(self.command_rows), np.stack(self.measured_rows)
        metrics = eval_res["all_metrics_dict"]
        expected = self.env._motion_lib.get_motion_num_steps(self.env.motion_ids).cpu().tolist()
        output = Path(self.output_dir)
        rows = []
        for i, key in enumerate(metrics["motion_keys"]):
            count = min(len(command), int(round(metrics["progress"][i] * expected[i])))
            valid = np.arange(len(command)) < count
            path = output / f"commands-{key}.npz"
            np.savez_compressed(
                path, command=command[:, i], measured=measured[:, i], valid_mask=valid
            )
            rows.append(
                dict(
                    motion_key=key,
                    path=str(path),
                    sha256=sha(path),
                    tracking_completed=not bool(metrics["terminated"][i]),
                    quality=(
                        command_metrics(command[:, i], measured[:, i], valid) if count else None
                    ),
                )
            )
        write_new(
            output / "command-quality.json",
            {
                "termination_profile": (
                    "native_reference_tracking; pose mismatch can censor " "an otherwise valid gait"
                ),
                "command_fields": ["vx_body_mps", "vy_body_mps", "yaw_rate_radps", "height_m"],
                "scene_goal_navigation_evaluation": False,
                "episodes": rows,
            },
        )
        return result
