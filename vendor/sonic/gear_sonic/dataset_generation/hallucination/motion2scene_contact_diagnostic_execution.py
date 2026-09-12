"""Pair-resolved contact diagnosis for one retained neutral development rollout."""

import json
from pathlib import Path

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import numpy as np

from .motion2scene_long_schedule_execution import (
    LongScheduleEnvCfg,
    LongScheduleRecorder,
    LongScheduleRecorderCfg,
)

DIAGNOSTIC_BODIES = (
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


class ContactDiagnosticEnvCfg(LongScheduleEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        if str(config["forced_option_id"]) != "neutral":
            raise ValueError("this registered contact diagnostic only executes neutral")
        robot_paths = [
            f"/World/envs/env_0/Robot/{name}" for name in config["beam_contact_body_names"]
        ]
        structure = "/World/ground/terrain/Structure"
        filters = robot_paths + [
            f"{structure}/{name}"
            for name in ("Floor", "WallWest", "WallEast", "WallSouth", "WallNorth")
        ]
        for body in DIAGNOSTIC_BODIES:
            setattr(
                self.scene,
                f"diagnostic_{body}",
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_0/Robot/{body}",
                    filter_prim_paths_expr=filters,
                    update_period=0.0,
                    history_length=0,
                    track_air_time=False,
                    force_threshold=0.0,
                ),
            )


class ContactDiagnosticRecorder(LongScheduleRecorder):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._diagnostic_forces = {body: [] for body in DIAGNOSTIC_BODIES}

    def record_post_physics_decimation_step(self):
        result = super().record_post_physics_decimation_step()
        for body, recorded in self._diagnostic_forces.items():
            sensor = self.env.scene.sensors[f"diagnostic_{body}"]
            force = sensor.contact_physx_view.get_contact_force_matrix(dt=self.env.physics_dt)
            recorded.append(force.detach().cpu().numpy().reshape(-1, 3).copy())
        return result

    def _save_diagnostic(self, prefix=""):
        mapping = {}
        for body in DIAGNOSTIC_BODIES:
            sensor = self.env.scene.sensors[f"diagnostic_{body}"]
            mapping[body] = {
                "sensor_body_names": list(sensor.body_names),
                "sensor_native_body_paths": list(sensor.body_physx_view.prim_paths),
                "filter_paths_in_matrix_column_order": list(sensor.cfg.filter_prim_paths_expr),
                "native_filter_count": int(sensor.contact_physx_view.filter_count),
            }
        all_sensor = self.env.scene.sensors["contact_forces"]
        mapping["all_body_sensor"] = {
            "body_names": list(all_sensor.body_names),
            "native_body_paths": list(all_sensor.body_physx_view.prim_paths),
            "robot_articulation_body_names": list(self.env.scene["robot"].body_names),
        }
        np.savez_compressed(
            Path(self.save_dir, f"{prefix}pair_resolved_contacts.npz"),
            **{body: np.asarray(values) for body, values in self._diagnostic_forces.items()},
            physics_steps=np.asarray(self._physics_steps),
            physics_dt_s=self.env.physics_dt,
        )
        Path(self.save_dir, f"{prefix}contact_counterpart_mapping.json").write_text(
            json.dumps(mapping, indent=2) + "\n"
        )

    def preserve_abort(self, reason):
        super().preserve_abort(reason)
        self._save_diagnostic("aborted_")

    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_diagnostic_saved", False):
            return
        self._save_diagnostic()
        self._diagnostic_saved = True


@configclass
class ContactDiagnosticRecorderCfg(LongScheduleRecorderCfg):
    class_type = ContactDiagnosticRecorder
