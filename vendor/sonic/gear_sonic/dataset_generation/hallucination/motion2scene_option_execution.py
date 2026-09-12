"""Opt-in work measurement for unchanged SONIC option execution."""

from pathlib import Path

from isaaclab.utils import configclass
import numpy as np

from .motion2scene_comparison_execution import ComparisonRecorder, ComparisonRecorderCfg


class OptionRecorder(ComparisonRecorder):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._option_torques = []
        self._option_velocities = []

    def record_post_physics_decimation_step(self):
        result = super().record_post_physics_decimation_step()
        robot = self.env.scene["robot"]
        # This is measured input for explicit actuators, estimated PD effort for
        # implicit actuators. The output records that distinction. Bypass cached
        # velocities because the scene timestamp has not advanced at this hook.
        self._option_torques.append(robot.data.applied_torque[0].detach().cpu().numpy().copy())
        self._option_velocities.append(
            robot.root_physx_view.get_dof_velocities()[0].detach().cpu().numpy().copy()
        )
        return result

    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_option_work_saved", False):
            return
        robot = self.env.scene["robot"]
        measured = not any(
            getattr(actuator, "is_implicit_model", True) for actuator in robot.actuators.values()
        )
        np.savez_compressed(
            Path(self.save_dir, "option_mechanical_work.npz"),
            estimated_torque_nm=np.asarray(self._option_torques),
            measured_actuator_effort_available=np.array(measured),
            velocity_rad_s=np.asarray(self._option_velocities),
            physics_steps=np.asarray(self._physics_steps),
            physics_dt_s=self.env.physics_dt,
            joint_names=np.array(robot.joint_names),
        )
        self._option_work_saved = True


@configclass
class OptionRecorderCfg(ComparisonRecorderCfg):
    class_type = OptionRecorder
