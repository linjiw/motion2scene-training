"""Retain same-state teacher supervision for independently scored stopping probes."""

import numpy as np

from gear_sonic.research.hindsight_training.runtime import sha, write_new
from gear_sonic.research.scene_distillation.collect import native_commands
from gear_sonic.research.scene_distillation.direct_context import task_context
from gear_sonic.research.scene_distillation.direct_scene_runtime import DirectSceneTaskCallback
from gear_sonic.research.scene_distillation.navigation_localization import CausalLocalization


class StoppingTeacherCallback(DirectSceneTaskCallback):
    """Task success labels remain distinct from legacy strict tracking qualification."""

    def _begin_task(self, env, teacher, task):
        if not self.config.get("teacher_mode") or task["split"] != "train":
            raise ValueError("Stopping supervision requires a train-task teacher rollout")
        self.rows, self.handles, self.captured = [], [], {}
        self.localization = CausalLocalization()
        for name, layer in [
            ("future_reference", teacher.actor_module.encoders["g1"].module[0]),
            ("decoder_input", teacher.actor_module.decoders["g1_dyn"].module[0]),
        ]:

            def capture(module, inputs, name=name):
                self.captured[name] = inputs[0].detach().reshape(1, -1).clone()

            self.handles.append(layer.register_forward_pre_hook(capture))

    def _record_teacher_query(self, env, teacher, task, observation, action):
        motion = env.motion_command
        controls, mask = native_commands(motion, env.env.scene.env_origins)
        inputs = self.captured["decoder_input"]
        row = {
            k: v[0].detach().cpu().numpy().copy()
            for k, v in dict(
                proprio=inputs[:, 64:],
                teacher_tokens=inputs[:, :64],
                teacher_actions=action,
                controls=controls,
                control_mask=mask,
                privileged_state=observation["critic_obs"],
                future_reference=self.captured["future_reference"],
            ).items()
        }
        row.update(
            task_context(
                task,
                (motion.robot_anchor_pos_w[0] - env.env.scene.env_origins[0]).cpu().numpy(),
                motion.robot_anchor_quat_w[0].cpu().numpy(),
            )
        )
        position = (motion.robot_anchor_pos_w[0] - env.env.scene.env_origins[0]).cpu().numpy()
        quaternion = motion.robot_anchor_quat_w[0].cpu().numpy()
        timestamp = len(self.rows) * 0.02
        row.update(
            localization=self.localization.update(
                position, quaternion, timestamp, task["goal_xyz"]
            ),
            measured_root_xyz=position.copy(),
            measured_root_wxyz=quaternion.copy(),
            observation_time_s=np.asarray(timestamp, dtype=np.float64),
        )
        if not all(np.isfinite(v).all() for v in row.values()):
            raise ValueError("Nonfinite stopping teacher label")
        self.rows.append(row)

    def _complete_task(self, task, score, output):
        for handle in self.handles:
            handle.remove()
        if len(self.rows) != score["control_steps"]:
            raise ValueError("Teacher labels do not align with physical task steps")
        arrays = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        success = bool(score["navigation_success"])
        arrays["query_mask"] = np.full(len(self.rows), success, dtype=bool)
        path = output / "teacher-episode.npz"
        np.savez_compressed(path, **arrays)
        evidence = output / "task-result.json"
        write_new(
            output / "collection.json",
            dict(
                schema="bfm_executed_foundation_v1",
                teacher_sha256=self.config["teacher_sha256"],
                source="executed_native_teacher_prefix",
                context_schema="complete_known_map_goal_v1",
                scene_task_success=success,
                scene_teacher_qualified=False,
                qualification="Independent goal/hold/contact success; not legacy reference qualification",
                parents=[
                    dict(path=str(evidence), sha256=sha(evidence)),
                    dict(path=self.config["task_path"], sha256=sha(self.config["task_path"])),
                ],
                episodes=[
                    dict(
                        motion_id=task["motion_id"],
                        split="train",
                        path=str(path),
                        sha256=sha(path),
                        rows=len(self.rows),
                        eligible=success,
                        source="executed_native_teacher_prefix",
                        supported_query_rows=int(arrays["query_mask"].sum()),
                        task_path=self.config["task_path"],
                        task_sha256=sha(self.config["task_path"]),
                        terminated=score["fell"],
                        whole_motion_eligible=False,
                        task_success_evidence=dict(path=str(evidence), sha256=sha(evidence)),
                    )
                ],
            ),
        )
