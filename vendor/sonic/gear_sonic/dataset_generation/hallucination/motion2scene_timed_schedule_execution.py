"""Opt-in repeated sensor commitments to complete qualified finite schedules."""

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
from .motion2scene_observation_history import FloorCeilingHistory, HistoryGrid
from .motion2scene_timed_history_execution import TimedHistoryCommand
from .motion2scene_timed_options import (
    definition_digest,
    legal_timed_actions,
    load_verified_registry,
)
from .motion2scene_timed_schedule_learning import SCHEMA, schedule_layout
from .motion2scene_timed_schedule_policy import (
    HISTORY_FRAMES,
    HISTORY_SECONDS,
    MODES,
    choose_schedule_option,
    load_schedule_policy,
    load_script_parameters,
    schedule_history_features,
)


class TimedScheduleCommand(LongScheduleCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        verified = load_verified_registry(cfg.timed_registry_path, cfg.timed_registry_sha256)
        if definition_digest(verified.request) != definition_digest(self.timed_bank.request):
            raise ValueError("actual loaded bank differs from the verified complete schedules")
        self.timed_bank = verified
        self.schedule_phases, _, _ = schedule_layout(verified)
        self.observation_delay = ObservationDelay(cfg.observation_delay_s)
        self.observation_history = FloorCeilingHistory(
            HistoryGrid(max_age_s=HISTORY_SECONDS, max_frames=HISTORY_FRAMES)
        )
        self.history_policy = (
            load_schedule_policy(cfg.timed_policy_path, cfg.timed_policy_sha256, verified)
            if cfg.timed_schedule_mode == "learned"
            else None
        )
        self.script_parameters = (
            load_script_parameters(
                cfg.script_parameters_path, cfg.script_parameters_sha256, verified
            )
            if cfg.timed_schedule_mode == "scripted_multi"
            else None
        )
        self.history_feature_names = None
        self._history_cache = {}

    # The inherited65-ray hook is independent of action count. This does not call
    # the old three-option initializer, readout or fixed entry validation.
    _observe_schedule = TimedHistoryCommand._observe_schedule

    def _requested_schedule(self, tick, joint_jumps, root_jumps, packet):
        mask, mandatory = legal_timed_actions(
            self.timed_bank, self.timed_state, tick, joint_jumps, root_jumps
        )
        state = super()._capture_state(tick, packet)["state"]
        active = self.timed_bank.option_ids.index(self.timed_state.active)
        names, features = schedule_history_features(
            self._history_observation,
            self.observation_history.grid,
            state,
            tick,
            active,
            mask,
            self._history_cache["observation_age_s"],
        )
        if self.history_feature_names is not None and self.history_feature_names != names:
            raise RuntimeError("schedule feature names changed within an episode")
        self.history_feature_names = names
        if self.cfg.timed_schedule_mode == "forced":
            requested = super()._requested_schedule(tick, joint_jumps, root_jumps, packet)
            values = None
        else:
            requested, values = choose_schedule_option(
                self.cfg.timed_schedule_mode,
                names,
                features,
                mask,
                active,
                tick,
                mandatory,
                self.timed_bank,
                policy=self.history_policy,
                script_parameters=self.script_parameters,
                preferred_option_id=self.cfg.preferred_option_id,
                preferred_reference_id=self.cfg.preferred_reference_id,
            )
        self._history_cache.update(
            features=features.tolist(),
            legal_mask=mask.tolist(),
            policy_values=values,
            selected_option_id=requested,
            is_registered_decision_tick=bool(tick in self.schedule_phases),
            floor_observed_cells=int(self._history_observation["floor_observed"].sum()),
            ceiling_observed_cells=int(self._history_observation["ceiling_observed"].sum()),
            occupied_voxels=int(self._history_observation["occupied"].sum()),
        )
        return requested

    def _capture_state(self, tick, packet):
        return {**super()._capture_state(tick, packet), **self._history_cache}


class TimedScheduleEnvCfg(EnvironmentContactEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        command = self.commands.motion
        command.class_type = TimedScheduleCommand
        command.timed_schedule_mode = str(config.get("timed_schedule_mode", "scripted_reference"))
        if command.timed_schedule_mode not in MODES:
            raise ValueError("unknown timed schedule mode")
        for key in (
            "timed_registry_path",
            "timed_registry_sha256",
            "timed_policy_path",
            "timed_policy_sha256",
            "script_parameters_path",
            "script_parameters_sha256",
        ):
            setattr(command, key, str(config.get(key) or ""))
        command.preferred_option_id = str(config.get("preferred_option_id", "neutral"))
        command.preferred_reference_id = str(config.get("preferred_reference_id", "sustained"))
        command.observation_delay_s = float(config.get("observation_delay_s", 0))
        if command.observation_delay_s != 0:
            raise ValueError("initial repeated schedule sensor protocol has zero configured delay")
        command.history_max_age_s = HISTORY_SECONDS
        command.history_max_frames = HISTORY_FRAMES


class TimedScheduleRecorder(EnvironmentContactRecorder):
    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_timed_schedule_saved", False):
            return
        command = self.env.command_manager.get_term("motion")
        path = Path(self.save_dir, "reactive_interface.json")
        interface = json.loads(path.read_text())
        interface.update(
            feature_schema=SCHEMA,
            feature_names=list(command.history_feature_names),
            option_ids=list(command.timed_bank.option_ids),
            phase_ticks=command.schedule_phases.tolist(),
            timed_schedule_mode=command.cfg.timed_schedule_mode,
            qualification_only=False,
            timed_registry_path=command.cfg.timed_registry_path,
            timed_registry_sha256=command.cfg.timed_registry_sha256,
            timed_policy_sha256=command.cfg.timed_policy_sha256,
            script_parameters_sha256=command.cfg.script_parameters_sha256,
            history_max_age_s=HISTORY_SECONDS,
            history_max_frames=HISTORY_FRAMES,
            rule_inputs=(
                "causally delivered65 rays/history, measured robot state "
                "and qualified complete-schedule legality"
            ),
            limitations=(
                "repeated neutral commitment opportunities, at most one adaptation, "
                "exact return, finite reference"
            ),
            sensor_model="ideal PhysX range returns and normals; no environment identity supplied to student",
        )
        path.write_text(json.dumps(interface, indent=2) + "\n")
        rows = command.interface_rows
        np.savez_compressed(
            Path(self.save_dir, "timed_schedule_features.npz"),
            schema_version=SCHEMA,
            feature_names=np.asarray(command.history_feature_names),
            option_ids=np.asarray(command.timed_bank.option_ids),
            features=np.asarray([row["features"] for row in rows], dtype=np.float32),
            command_ticks=np.asarray([row["tick"] for row in rows]),
            capture_elapsed_s=np.asarray([row["capture_elapsed_s"] for row in rows]),
            legal_mask=np.asarray([row["legal_mask"] for row in rows], dtype=bool),
            active_before=np.asarray([row["active_before"] for row in rows]),
            active=np.asarray([row["active"] for row in rows]),
        )
        self._timed_schedule_saved = True


@configclass
class TimedScheduleRecorderCfg(EnvironmentContactRecorderCfg):
    class_type = TimedScheduleRecorder
