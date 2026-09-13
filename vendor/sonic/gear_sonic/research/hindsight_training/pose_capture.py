"""Native tracking qualification plus full G1 pose capture for offline mesh replay.

``PoseCaptureQualificationCallback`` behaves exactly like
``TrackingQualificationCallback``: the same native ``<key>.npz`` files,
``trajectory-contract.json`` and ``metrics_eval.json`` are produced by the
unchanged parent classes. It only *reads* simulator/command state after each
evaluation step and additionally writes, per motion key, ``<key>.pose.npz``:

  per recorded step k (same slice as the native arrays, T = motion_num_steps - 1):
    robot_root_pos   (T,3)   root link (pelvis) position, env origin subtracted
    robot_root_quat  (T,4)   root link orientation, wxyz
    robot_joint_pos  (T,29)  articulation (PhysX) joint order -> joint_names_isaac
    ref_root_pos     (T,3)   command reference pelvis position, env origin subtracted
    ref_root_quat    (T,4)   command reference pelvis orientation, wxyz
    ref_joint_pos    (T,29)  motion-library joint order (mujoco_to_isaaclab_dof applied)
    ref_frame        (T,)    motion frame index the command pointed at (start + time_steps)
    motion_id        (T,)    loaded motion index of this env
    term_<name>      (T,)    termination-term flags computed in that step
  full reference track of the motion (independent of failures/resets), F frames:
    ref_track_root_pos (F,3), ref_track_root_quat (F,4), ref_track_joint_pos (F,29),
    ref_track_body_pos (F,14,3)
  scalars/metadata: env_index, env_origin, motion_key, motion_num_steps,
    motion_num_frames, native_progress, native_terminated, dt,
    joint_names_isaac, mujoco_to_isaaclab_dof, body_names, time_out_terms

Frame k is captured after env.step k returns, i.e. at the same instant the
native callback reads ``ref_body_pos_extend``/``rigid_body_pos_extend``. A step
that terminates an env therefore already holds that env's post-reset state;
consumers must censor at ``round(native_progress * motion_num_steps)``.

Requires a single evaluation batch (num_envs >= number of loaded motions).
"""

import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.research.hindsight_training.qualify import TrackingQualificationCallback


def _to_numpy(value):
    return value.detach().to("cpu").numpy().copy()


class PoseCaptureQualificationCallback(TrackingQualificationCallback):
    """TrackingQualificationCallback plus per-step root pose, joints and termination flags."""

    def _pre_evaluate_policy(self, reset_env=True):
        super()._pre_evaluate_policy(reset_env)
        if self.num_total_env_eval_loops != 1:
            raise RuntimeError(
                "PoseCaptureQualificationCallback needs one evaluation batch: "
                f"{self.env._motion_lib._num_unique_motions} motions but "
                f"{self.env.num_envs} envs"
            )
        self._pose_steps = []
        self._pose_per_env = None
        self._pose_num_steps = None

    def _capture_pose_step(self):
        command = self.env.motion_command
        isaac_env = self.env.env
        origins = isaac_env.scene.env_origins
        robot_data = command.robot.data
        terms = isaac_env.termination_manager
        row = {
            "robot_root_pos": robot_data.root_pos_w - origins,
            "robot_root_quat": robot_data.root_quat_w,
            "robot_joint_pos": robot_data.joint_pos,
            "ref_root_pos": command.body_pos_w[:, 0] - origins,
            "ref_root_quat": command.body_quat_w[:, 0],
            "ref_joint_pos": command.joint_pos,
            "ref_frame": command.motion_start_time_steps + command.time_steps,
            "motion_id": command.motion_ids,
        }
        for name in terms.active_terms:
            row[f"term_{name}"] = terms.get_term(name)
        self._pose_steps.append({key: _to_numpy(value) for key, value in row.items()})

    def _post_eval_env_step(self, actor_state):
        self._capture_pose_step()
        num_steps = _to_numpy(self.env._motion_lib.get_motion_num_steps(self.env.motion_ids))
        actor_state = super()._post_eval_env_step(actor_state)
        if actor_state.get("end_eval", False) and self._pose_per_env is None:
            stacked = {
                key: np.stack([step[key] for step in self._pose_steps])
                for key in self._pose_steps[0]
            }
            # Identical per-env slice to ImEvalCallback's native trajectories.
            self._pose_per_env = [
                {key: value[: int(steps) - 1, env] for key, value in stacked.items()}
                for env, steps in enumerate(num_steps)
            ]
            self._pose_num_steps = num_steps
            self._pose_steps = []
        return actor_state

    def _reference_track(self, env_index):
        command = self.env.motion_command
        library = command.motion_lib
        motion_id = command.motion_ids[env_index : env_index + 1]
        frames = int(library.get_time_step_total(motion_id)[0].item())
        steps = torch.arange(frames, dtype=torch.long, device=motion_id.device)
        ids = motion_id.expand(frames)
        body_pos = library.get_body_pos_w(ids, steps)
        return {
            "ref_track_root_pos": _to_numpy(body_pos[:, 0]),
            "ref_track_root_quat": _to_numpy(library.get_body_quat_w(ids, steps)[:, 0]),
            "ref_track_joint_pos": _to_numpy(library.get_dof_pos(ids, steps)),
            "ref_track_body_pos": _to_numpy(body_pos),
            "motion_num_frames": np.int64(frames),
        }

    def _post_evaluate_policy(self, eval_res):
        result = super()._post_evaluate_policy(eval_res)
        output = Path(self.output_dir)
        command = self.env.motion_command
        isaac_env = self.env.env
        terms = isaac_env.termination_manager
        metrics = eval_res["all_metrics_dict"]
        keys = [str(key) for key in metrics["motion_keys"]]
        if self._pose_per_env is None or len(self._pose_per_env) < len(keys):
            raise RuntimeError("pose capture is missing environments for the evaluated motions")
        joint_names = [str(name) for name in command.robot.joint_names]
        time_out_terms = [
            name for name in terms.active_terms if terms.get_term_cfg(name).time_out
        ]
        origins = _to_numpy(isaac_env.scene.env_origins)
        mujoco_to_isaaclab = np.asarray(command.mujoco_to_isaaclab_dof, dtype=np.int64)
        body_names = np.asarray([str(name) for name in command.cmd_body_names])
        dt = float(isaac_env.step_dt)
        for env_index, key in enumerate(keys):
            pose = self._pose_per_env[env_index]
            if int(pose["motion_id"][0]) != env_index:
                raise RuntimeError(f"env {env_index} tracked motion {pose['motion_id'][0]}")
            np.savez_compressed(
                output / f"{key}.pose.npz",
                **pose,
                **self._reference_track(env_index),
                env_index=np.int64(env_index),
                env_origin=origins[env_index],
                motion_key=np.str_(key),
                motion_num_steps=np.int64(self._pose_num_steps[env_index]),
                native_progress=np.float64(metrics["progress"][env_index]),
                native_terminated=np.bool_(metrics["terminated"][env_index]),
                dt=np.float64(dt),
                joint_names_isaac=np.asarray(joint_names),
                mujoco_to_isaaclab_dof=mujoco_to_isaaclab,
                body_names=body_names,
                time_out_terms=np.asarray(time_out_terms),
            )
        (output / "pose-contract.json").write_text(
            json.dumps(
                {
                    "source": "PoseCaptureQualificationCallback (read-only state capture)",
                    "dt": dt,
                    "frames": "T = motion_num_steps - 1; frame k captured after env.step k",
                    "censoring": "valid frames = round(native_progress * motion_num_steps); "
                    "the terminating frame already holds the post-reset state",
                    "reference_alignment": "recorded frame k corresponds to reference frame k+1",
                    "quaternions": "wxyz",
                    "positions": "env-local (scene.env_origins subtracted)",
                    "joint_names_isaac": joint_names,
                    "mujoco_to_isaaclab_dof": mujoco_to_isaaclab.tolist(),
                    "active_terms": list(terms.active_terms),
                    "time_out_terms": time_out_terms,
                    "num_envs": int(self.env.num_envs),
                    "motions": keys,
                },
                indent=2,
            )
        )
        return result
