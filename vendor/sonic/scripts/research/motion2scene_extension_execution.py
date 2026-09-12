"""Opt-in arm-restricted ridge execution with the complete qualified physical bank."""

from dataclasses import asdict
import json
from pathlib import Path

from isaaclab.utils import configclass

from gear_sonic.dataset_generation.hallucination.motion2scene_long_schedule_execution import (
    LongScheduleCommand,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (
    definition_digest,
    legal_timed_actions,
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_execution import (
    TimedScheduleCommand,
    TimedScheduleEnvCfg,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (
    schedule_layout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (
    schedule_history_features,
)
from scripts.research.motion2scene_extension_policy import load_extension_policy
from scripts.research.motion2scene_extension_teaching import arm_view, choose_arm_schedule
from scripts.research.motion2scene_perturbed_execution import (
    PerturbedScheduleCommand,
    PerturbedScheduleRecorder,
    PerturbedScheduleRecorderCfg,
)
from scripts.research.motion2scene_sensor_perturbations import (
    PerturbedObservationStream,
    SensorPerturbation,
)


class ExtensionScheduleCommand(TimedScheduleCommand):
    def __init__(self, cfg, env):
        # Load every original reference. Only the high-level policy menu changes.
        LongScheduleCommand.__init__(self, cfg, env)
        verified = load_verified_registry(cfg.timed_registry_path, cfg.timed_registry_sha256)
        if definition_digest(verified.request) != definition_digest(self.timed_bank.request):
            raise ValueError("actual loaded physical bank differs from the qualified full bank")
        self.timed_bank = verified
        self.schedule_phases, _, _ = schedule_layout(verified)
        self.history_policy, result = load_extension_policy(
            cfg.extension_result_path, cfg.extension_result_sha256, verified
        )
        if result["policy"] != dict(path=cfg.timed_policy_path, sha256=cfg.timed_policy_sha256):
            raise ValueError("loaded policy archive differs from the assigned fitting result")
        self.extension_arm = result["arm"]
        self.extension_view, _ = arm_view(verified, self.extension_arm)
        self.sensor_stream = PerturbedObservationStream(SensorPerturbation(seed=95001))
        self.observation_history = self.sensor_stream.history
        self.history_feature_names, self._history_cache = None, {}

    _observe_schedule = PerturbedScheduleCommand._observe_schedule

    def _requested_schedule(self, tick, joint_jumps, root_jumps, packet):
        mask, mandatory = legal_timed_actions(
            self.timed_bank, self.timed_state, tick, joint_jumps, root_jumps
        )
        state = LongScheduleCommand._capture_state(self, tick, packet)["state"]
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
            raise RuntimeError("full-bank feature schema changed during arm-policy execution")
        self.history_feature_names = names
        requested, projected = choose_arm_schedule(
            self.timed_bank,
            self.extension_arm,
            self.history_policy,
            names,
            features,
            mask,
            active,
            tick,
            mandatory,
        )
        self._history_cache.update(
            features=features.tolist(),
            legal_mask=mask.tolist(),
            policy_values=None,
            arm_policy=projected,
            selected_option_id=requested,
            is_registered_decision_tick=bool(tick in self.schedule_phases),
            floor_observed_cells=int(self._history_observation["floor_observed"].sum()),
            ceiling_observed_cells=int(self._history_observation["ceiling_observed"].sum()),
            occupied_voxels=int(self._history_observation["occupied"].sum()),
        )
        return requested


class ExtensionScheduleEnvCfg(TimedScheduleEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        if (
            config.get("timed_schedule_mode") != "learned"
            or config.get("learner_family") != "extension_arm_ridge"
        ):
            raise ValueError("explicit learned arm-ridge configuration required")
        if any(
            float(config.get(key, 0)) != 0
            for key in (
                "sensor_dropout_probability",
                "sensor_range_noise_std_m",
                "sensor_latency_s",
            )
        ):
            raise ValueError("equivalent-teaching comparison retains nominal sensing")
        command = self.commands.motion
        command.class_type = ExtensionScheduleCommand
        command.extension_result_path = str(config["extension_result_path"])
        command.extension_result_sha256 = str(config["extension_result_sha256"])


class ExtensionScheduleRecorder(PerturbedScheduleRecorder):
    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_extension_saved", False):
            return
        command = self.env.command_manager.get_term("motion")
        path = Path(self.save_dir) / "reactive_interface.json"
        interface = json.loads(path.read_text())
        interface.update(
            learner_family="extension_arm_ridge",
            extension_arm=command.extension_arm,
            extension_result_sha256=command.cfg.extension_result_sha256,
            logical_option_ids=list(command.extension_view.option_ids),
            logical_request_digest=definition_digest(command.extension_view.request),
            logical_feature_names=command.history_policy["feature_names"].tolist(),
            sensor_settings=asdict(command.sensor_stream.settings),
            policy_format="common ridge on arm-projected features; original full bank remains loaded",
            policy_value_location="observations[*].arm_policy.policy_values",
        )
        path.write_text(json.dumps(interface, indent=2, allow_nan=False) + "\n")
        self._extension_saved = True


@configclass
class ExtensionScheduleRecorderCfg(PerturbedScheduleRecorderCfg):
    class_type = ExtensionScheduleRecorder
