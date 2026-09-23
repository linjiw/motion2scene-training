"""Separate candidate navigation-query evidence; never a task-positive dataset."""

import numpy as np

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.motor_runtime import (
    current_orientation_observation,
    motor_commands,
)
from gear_sonic.research.scene_distillation.navigation_motor_runtime import NavigationMotorCallback


class NavigationQueryCallback(NavigationMotorCallback):
    """The navigation actor controls every step; privileged targets are logging only.

    Nominal-clock targets are candidates, not automatically task-compatible expert
    advice. Supported admission requires separate same-state continuation evidence.
    """

    supports_observation_ablation = False

    def _begin_task(self, env, teacher, task):
        super()._begin_task(env, teacher, task)
        if task["split"] != "train":
            raise ValueError("Navigation queries are restricted to training tasks")
        self.rows = []

    def _student_action(self, student, env, teacher, observation, task, noise):
        actor = self._navigation_actor(student, env, teacher, observation, task)
        prediction = student.navigation_step(actor)
        # These privileged values never enter navigation_step. They are derived
        # after the action has been chosen, before stepping the physical state.
        commands, mask = motor_commands(
            env.motion_command,
            env.env.scene.env_origins,
            True,
            current_orientation_observation(teacher.actor_module, observation),
        )
        target = student.full_step(actor.proprio, commands, mask)
        values = {
            k: getattr(actor, k)
            for k in actor.__dataclass_fields__
            if getattr(actor, k) is not None
        }
        values.update(
            controls=commands,
            control_mask=mask,
            candidate_motor_actions=target["actions"],
            predicted_commands=prediction["commands"],
            actions=prediction["actions"],
        )
        row = {k: v[0].detach().cpu().numpy().copy() for k, v in values.items()}
        if not all(np.isfinite(v).all() for v in row.values()):
            raise ValueError("Nonfinite navigation query")
        self.rows.append(row)
        return prediction["actions"]

    def _complete_task(self, task, score, output):
        if len(self.rows) != score["control_steps"]:
            raise ValueError("Navigation query / execution length mismatch")
        arrays = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        arrays["supported_query_mask"] = np.zeros(len(self.rows), dtype=bool)
        path = output / "navigation-queries.npz"
        np.savez_compressed(path, **arrays)
        write_new(
            output / "query-collection.json",
            dict(
                schema="navigation_candidate_queries_v1",
                task_positive_eligible=False,
                supported_rows=0,
                rows=len(self.rows),
                actor_profile=self.config["actor_profile"],
                student_sha256=self.config["student_sha256"],
                teacher_sha256=self.config["teacher_sha256"],
                task_path=self.config["task_path"],
                task_sha256=sha(self.config["task_path"]),
                shard=dict(path=str(path), sha256=sha(path)),
                score=dict(
                    path=str(output / "task-result.json"), sha256=sha(output / "task-result.json")
                ),
                target_source="same-state frozen motor under nominal reference clock",
                admission="Candidate-only; successful compatible takeover evidence is required separately",
            ),
        )
