"""Capture measured robot poses for matched mesh replay of native student tasks."""

import numpy as np

from gear_sonic.research.hindsight_training.runtime import write_new
from gear_sonic.research.scene_distillation.navigation_motor_runtime import (
    FullMotorTaskCallback,
    NavigationMotorCallback,
)


class PoseRecordingMixin:
    """Capture post-step physics states without enabling the Isaac RTX renderer."""

    def _begin_task(self, env, teacher, task):
        super()._begin_task(env, teacher, task)
        self.recording_env = env
        self.original_recording_step = env.step
        self.pose_rows = []
        robot = env.env.scene["robot"]
        self.replay_joint_names = list(robot.joint_names)
        self.replay_body_names = list(robot.body_names)

        def step(*args, **kwargs):
            result = self.original_recording_step(*args, **kwargs)
            data = robot.data
            origin = env.env.scene.env_origins[0]
            self.pose_rows.append(
                {
                    k: v.detach().cpu().numpy().copy()
                    for k, v in dict(
                        root_xyz=data.root_link_pos_w[0] - origin,
                        root_wxyz=data.root_link_quat_w[0],
                        joint_pos=data.joint_pos[0],
                        body_xyz=data.body_pos_w[0] - origin,
                        body_wxyz=data.body_quat_w[0],
                    ).items()
                }
            )
            return result

        env.step = step

    def _complete_task(self, task, score, output):
        self.recording_env.step = self.original_recording_step
        if len(self.pose_rows) != score["control_steps"]:
            raise ValueError("Pose recording does not align with task scoring")
        np.savez_compressed(
            output / "robot-poses.npz",
            **{k: np.stack([r[k] for r in self.pose_rows]) for k in self.pose_rows[0]},
            joint_names=np.array(self.replay_joint_names),
            body_names=np.array(self.replay_body_names),
            time_s=(np.arange(len(self.pose_rows)) + 1) * 0.02,
        )
        write_new(
            output / "pose-recording.json",
            dict(
                provenance="measured Isaac physics states; suitable for mesh replay, not camera footage",
                control_dt=0.02,
                steps=len(self.pose_rows),
                post_reset_states_included=False,
            ),
        )
        super()._complete_task(task, score, output)


class RecordFullMotorCallback(PoseRecordingMixin, FullMotorTaskCallback):
    pass


class RecordNavigationMotorCallback(PoseRecordingMixin, NavigationMotorCallback):
    pass
