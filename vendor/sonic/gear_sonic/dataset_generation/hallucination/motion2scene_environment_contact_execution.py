"""Explicit all-body self/floor/wall/beam normal contact measurements at 200 Hz."""

import json
from pathlib import Path

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import numpy as np

from .motion2scene_environment_contacts import environment_counterpart_paths
from .motion2scene_long_schedule_execution import (
    LongScheduleEnvCfg,
    LongScheduleRecorder,
    LongScheduleRecorderCfg,
)


def configure_environment_contact_sensors(scene, body_names, beam_paths=None):
    """One subject body per PhysX view makes every filter column unambiguous."""
    paths = [f"/World/envs/env_0/Robot/{body}" for body in body_names]
    filters = paths + environment_counterpart_paths(beam_paths)
    for body, path in zip(body_names, paths, strict=True):
        setattr(
            scene,
            f"environment_contact_{body}",
            ContactSensorCfg(
                prim_path=path,
                filter_prim_paths_expr=filters,
                update_period=0.0,
                history_length=0,
                track_air_time=False,
                force_threshold=0.0,
            ),
        )


class EnvironmentContactEnvCfg(LongScheduleEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        configure_environment_contact_sensors(
            self.scene,
            list(config["beam_contact_body_names"]),
            config.get("environment_beam_paths"),
        )


class EnvironmentContactRecorder(LongScheduleRecorder):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._environment_body_names = list(env.scene["robot"].body_names)
        self._environment_pairs = []

    def record_post_physics_decimation_step(self):
        result = super().record_post_physics_decimation_step()
        rows = []
        for body in self._environment_body_names:
            sensor = self.env.scene.sensors[f"environment_contact_{body}"]
            forces = sensor.contact_physx_view.get_contact_force_matrix(dt=self.env.physics_dt)
            rows.append(forces.detach().cpu().numpy().reshape(-1, 3).copy())
        self._environment_pairs.append(np.stack(rows))
        return result

    def _save_environment_contacts(self, prefix=""):
        sensors = {}
        for body in self._environment_body_names:
            sensor = self.env.scene.sensors[f"environment_contact_{body}"]
            sensors[body] = {
                "sensor_body_names": list(sensor.body_names),
                "native_body_paths": list(sensor.body_physx_view.prim_paths),
                "filter_paths": list(sensor.cfg.filter_prim_paths_expr),
                "native_filter_count": int(sensor.contact_physx_view.filter_count),
            }
        net_sensor = self.env.scene.sensors["contact_forces"]
        mapping = {
            "schema": "motion2scene_environment_contact_mapping_v1",
            "pair_subject_body_names": self._environment_body_names,
            "sensors": sensors,
            "net_body_names": list(net_sensor.body_names),
            "net_native_body_paths": list(net_sensor.body_physx_view.prim_paths),
            "robot_articulation_body_names": list(self.env.scene["robot"].body_names),
            "normal_forces_only": True,
        }
        np.savez_compressed(
            Path(self.save_dir, f"{prefix}environment_pair_contacts.npz"),
            force_w=np.asarray(self._environment_pairs),
            physics_steps=np.asarray(self._physics_steps),
            physics_dt_s=self.env.physics_dt,
            body_names=np.asarray(self._environment_body_names),
        )
        Path(self.save_dir, f"{prefix}environment_contact_mapping.json").write_text(
            json.dumps(mapping, indent=2) + "\n"
        )

    def preserve_abort(self, reason):
        super().preserve_abort(reason)
        self._save_environment_contacts("aborted_")

    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_environment_saved", False):
            return
        self._save_environment_contacts()
        self._environment_saved = True


@configclass
class EnvironmentContactRecorderCfg(LongScheduleRecorderCfg):
    class_type = EnvironmentContactRecorder
