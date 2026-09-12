"""Opt-in65-ray duration selector using complete qualified timed schedules."""

from dataclasses import asdict
import json
from pathlib import Path

from isaaclab.utils import configclass
import numpy as np

from .motion2scene_environment_contact_execution import (
    EnvironmentContactEnvCfg,
    EnvironmentContactRecorder,
    EnvironmentContactRecorderCfg,
)
from .motion2scene_long_schedule_execution import LongScheduleCommand
from .motion2scene_observation_delay import ObservationDelay
from .motion2scene_observation_history import (
    FloorCeilingHistory,
    HistoryGrid,
    SensorRay,
    capture_ray_fan,
)
from .motion2scene_timed_history_policy import (
    MODES,
    SCHEMA,
    choose_timed_option,
    load_timed_policy,
    timed_history_features,
    validate_history_bank,
)
from .motion2scene_timed_options import (
    definition_digest,
    legal_timed_actions,
    load_verified_registry,
)


class TimedHistoryCommand(LongScheduleCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.timed_registry_path:
            verified = load_verified_registry(cfg.timed_registry_path, cfg.timed_registry_sha256)
            if definition_digest(verified.request) != definition_digest(self.timed_bank.request):
                raise ValueError("loaded experimental bank differs from the verified registry")
            self.timed_bank = verified
        validate_history_bank(self.timed_bank, cfg.timed_history_mode)
        self.observation_delay = ObservationDelay(cfg.observation_delay_s)
        self.observation_history = FloorCeilingHistory(
            HistoryGrid(max_age_s=cfg.history_max_age_s, max_frames=cfg.history_max_frames)
        )
        self.history_policy = (
            load_timed_policy(cfg.timed_policy_path, cfg.timed_policy_sha256, self.timed_bank)
            if cfg.timed_history_mode == "learned"
            else None
        )
        self.history_feature_names = None
        self._history_cache = {}

    def _observe_schedule(self, tick, scene_query):
        data = self.robot.data
        root = data.root_pos_w[0].detach().cpu().numpy()
        quat = data.root_quat_w[0].detach().cpu().numpy()
        elapsed = len(self.interface_rows) / 50
        measurements = [asdict(ray) for ray in capture_ray_fan(root, quat, scene_query)]
        delivered = self.observation_delay.push(
            {"measurements": measurements, "phase_s": tick / 50}
        )
        if delivered is not None:
            self.observation_history.push(
                [SensorRay(**ray) for ray in delivered["measurements"]],
                delivered["capture_elapsed_s"],
                delivered_time_s=elapsed,
            )
        self._history_observation = self.observation_history.snapshot(root, quat, elapsed)
        self._history_cache = {
            "capture_elapsed_s": elapsed,
            "delivered_capture_elapsed_s": (
                None if delivered is None else delivered["capture_elapsed_s"]
            ),
            "observation_age_s": (
                max(0, elapsed - delivered["capture_elapsed_s"])
                if delivered is not None
                else float(self.cfg.observation_delay_s) + elapsed
            ),
            "measurements": measurements,
            "normal_known_mask": [ray["hit_normal_w"] is not None for ray in measurements],
        }
        # Return compatible recorder containers, without performing legacy37 queries.
        occupied = bool(self._history_observation["occupied"].any())
        return occupied, measurements[0]["origin_w"], measurements, delivered

    def _requested_schedule(self, tick, joint_jumps, root_jumps, packet):
        mode = self.cfg.timed_history_mode
        mask, mandatory = legal_timed_actions(
            self.timed_bank,
            self.timed_state,
            tick,
            joint_jumps,
            root_jumps,
            qualification_only=mode == "forced",
        )
        state = super()._capture_state(tick, packet)["state"]
        active = self.timed_bank.option_ids.index(self.timed_state.active)
        names, features = timed_history_features(
            self._history_observation,
            self.observation_history.grid,
            state,
            tick,
            active,
            mask,
            self._history_cache["observation_age_s"],
        )
        if self.history_feature_names is not None and self.history_feature_names != names:
            raise RuntimeError("timed history feature names changed within an episode")
        self.history_feature_names = names
        if mode == "forced":
            requested = super()._requested_schedule(tick, joint_jumps, root_jumps, packet)
            values = None
        else:
            requested, values = choose_timed_option(
                mode,
                names,
                features,
                mask,
                active,
                tick,
                mandatory,
                self.timed_bank,
                self.history_policy,
            )
        self._history_cache.update(
            features=features.tolist(),
            legal_mask=mask.tolist(),
            policy_values=values,
            selected_option_id=requested,
            floor_observed_cells=int(self._history_observation["floor_observed"].sum()),
            ceiling_observed_cells=int(self._history_observation["ceiling_observed"].sum()),
            occupied_voxels=int(self._history_observation["occupied"].sum()),
        )
        return requested

    def _capture_state(self, tick, packet):
        return {**super()._capture_state(tick, packet), **self._history_cache}


class TimedHistoryEnvCfg(EnvironmentContactEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        command = self.commands.motion
        command.class_type = TimedHistoryCommand
        command.timed_history_mode = str(config.get("timed_history_mode", "scripted_sustained"))
        if command.timed_history_mode not in MODES:
            raise ValueError("unknown timed history mode")
        for key in (
            "timed_registry_path",
            "timed_registry_sha256",
            "timed_policy_path",
            "timed_policy_sha256",
        ):
            setattr(command, key, str(config.get(key) or ""))
        command.observation_delay_s = float(config.get("observation_delay_s", 0))
        command.history_max_age_s = float(config.get("history_max_age_s", 0.5))
        command.history_max_frames = int(config.get("history_max_frames", 26))


class TimedHistoryRecorder(EnvironmentContactRecorder):
    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_timed_history_saved", False):
            return
        command = self.env.command_manager.get_term("motion")
        path = Path(self.save_dir, "reactive_interface.json")
        interface = json.loads(path.read_text())
        interface.update(
            feature_schema=SCHEMA,
            feature_names=list(command.history_feature_names),
            option_ids=list(command.timed_bank.option_ids),
            timed_history_mode=command.cfg.timed_history_mode,
            qualification_only=not command.timed_bank.online_verified,
            timed_registry_path=command.cfg.timed_registry_path,
            timed_registry_sha256=command.cfg.timed_registry_sha256,
            timed_policy_sha256=command.cfg.timed_policy_sha256,
            rule_inputs="causally delivered65 sensor rays/history, measured robot state, current legality",
            limitations="one entry at tick15, exact declared return, finite horizon, no stop or reentry",
            sensor_model="ideal PhysX range returns and hit normals; robot excluded; no scene identities",
        )
        path.write_text(json.dumps(interface, indent=2) + "\n")
        rows = command.interface_rows
        np.savez_compressed(
            Path(self.save_dir, "timed_history_features.npz"),
            schema_version=SCHEMA,
            feature_names=np.asarray(command.history_feature_names),
            option_ids=np.asarray(command.timed_bank.option_ids),
            features=np.asarray([row["features"] for row in rows], dtype=np.float32),
            command_ticks=np.asarray([row["tick"] for row in rows]),
            capture_elapsed_s=np.asarray([row["capture_elapsed_s"] for row in rows]),
            legal_mask=np.asarray([row["legal_mask"] for row in rows], dtype=bool),
            active=np.asarray([row["active"] for row in rows]),
        )
        self._timed_history_saved = True


@configclass
class TimedHistoryRecorderCfg(EnvironmentContactRecorderCfg):
    class_type = TimedHistoryRecorder
