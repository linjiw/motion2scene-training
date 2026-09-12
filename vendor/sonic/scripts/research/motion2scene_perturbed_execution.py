"""Separate opt-in sensor sensitivity runtime; leaves primary acquisition intact."""

from dataclasses import asdict
import json
from pathlib import Path

from isaaclab.utils import configclass

from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import (
    capture_ray_fan,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_execution import (
    TimedScheduleCommand,
    TimedScheduleEnvCfg,
    TimedScheduleRecorder,
    TimedScheduleRecorderCfg,
)
from scripts.research.motion2scene_sensor_perturbations import (
    PerturbedObservationStream,
    SensorPerturbation,
)

SENSOR_SCHEMA = "motion2scene_sensor_sensitivity_v1"


class PerturbedScheduleCommand(TimedScheduleCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sensor_stream = PerturbedObservationStream(
            SensorPerturbation(**cfg.sensor_perturbation)
        )
        self.observation_history = self.sensor_stream.history

    def _observe_schedule(self, tick, scene_query):
        root = self.robot.data.root_pos_w[0].detach().cpu().numpy()
        quat = self.robot.data.root_quat_w[0].detach().cpu().numpy()
        rays = capture_ray_fan(root, quat, scene_query)
        self._history_cache, delivered = self.sensor_stream.push(rays, tick / 50)
        if self._history_cache["capture_elapsed_s"] != len(self.interface_rows) / 50:
            raise RuntimeError("sensor capture cadence differs from control recording")
        self._history_observation = self.observation_history.snapshot(
            root, quat, self._history_cache["capture_elapsed_s"]
        )
        return (
            bool(self._history_observation["occupied"].any()),
            list(rays[0].origin_w),
            self._history_cache["measurements"],
            delivered,
        )


class PerturbedScheduleEnvCfg(TimedScheduleEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        settings = SensorPerturbation(
            dropout_probability=float(config.get("sensor_dropout_probability", 0)),
            range_noise_std_m=float(config.get("sensor_range_noise_std_m", 0)),
            latency_s=float(config.get("sensor_latency_s", 0)),
            seed=config["sensor_corruption_seed"],
        )
        self.commands.motion.class_type = PerturbedScheduleCommand
        self.commands.motion.sensor_perturbation = asdict(settings)


class PerturbedScheduleRecorder(TimedScheduleRecorder):
    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_sensor_sensitivity_saved", False):
            return
        command = self.env.command_manager.get_term("motion")
        path = Path(self.save_dir) / "reactive_interface.json"
        interface = json.loads(path.read_text())
        interface.update(
            sensor_schema=SENSOR_SCHEMA,
            sensor_settings=asdict(command.sensor_stream.settings),
            sensor_model=(
                "Independent channel dropout and Gaussian hit-range noise, then fixed packet latency; "
                "retained normals and robot pose remain ideal"
            ),
            raw_measurements_role="privileged sensor verification only; not policy inputs",
        )
        path.write_text(json.dumps(interface, indent=2, allow_nan=False) + "\n")
        self._sensor_sensitivity_saved = True


@configclass
class PerturbedScheduleRecorderCfg(TimedScheduleRecorderCfg):
    class_type = PerturbedScheduleRecorder
