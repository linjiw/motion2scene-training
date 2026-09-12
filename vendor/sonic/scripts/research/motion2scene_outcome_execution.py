"""Opt-in feasibility/time policy runtime with the unchanged traversal interface."""

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
from scripts.research.motion2scene_outcome_policy import choose_schedule, load_policy
from scripts.research.motion2scene_perturbed_execution import (
    PerturbedScheduleCommand,
    PerturbedScheduleRecorder,
    PerturbedScheduleRecorderCfg,
)
from scripts.research.motion2scene_sensor_perturbations import (
    PerturbedObservationStream,
    SensorPerturbation,
)


class OutcomeScheduleCommand(TimedScheduleCommand):
    def __init__(self, cfg, env):
        # Initialize the same physical schedule interface, without loading a ridge
        # archive. The explicit JSON model is the only learned readout used here.
        LongScheduleCommand.__init__(self, cfg, env)
        verified = load_verified_registry(cfg.timed_registry_path, cfg.timed_registry_sha256)
        if definition_digest(verified.request) != definition_digest(self.timed_bank.request):
            raise ValueError("actual loaded bank differs from the verified schedules")
        self.timed_bank = verified
        self.schedule_phases, _, _ = schedule_layout(verified)
        self.outcome_policy = load_policy(cfg.timed_policy_path, cfg.timed_policy_sha256, verified)
        self.sensor_stream = PerturbedObservationStream(
            SensorPerturbation(**cfg.sensor_perturbation)
        )
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
            raise RuntimeError("outcome student feature names changed within an episode")
        self.history_feature_names = names
        requested, estimates = choose_schedule(
            self.outcome_policy, names, features, mask, active, tick, mandatory, self.timed_bank
        )
        self._history_cache.update(
            features=features.tolist(),
            legal_mask=mask.tolist(),
            policy_values=None,
            outcome_predictions=estimates,
            selected_option_id=requested,
            is_registered_decision_tick=bool(tick in self.schedule_phases),
            floor_observed_cells=int(self._history_observation["floor_observed"].sum()),
            ceiling_observed_cells=int(self._history_observation["ceiling_observed"].sum()),
            occupied_voxels=int(self._history_observation["occupied"].sum()),
        )
        return requested


class OutcomeScheduleEnvCfg(TimedScheduleEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        if (
            config.get("timed_schedule_mode") != "learned"
            or config.get("learner_family") != "outcome_tree"
        ):
            raise ValueError("explicit learned outcome-tree configuration required")
        settings = SensorPerturbation(
            dropout_probability=float(config.get("sensor_dropout_probability", 0)),
            range_noise_std_m=float(config.get("sensor_range_noise_std_m", 0)),
            latency_s=float(config.get("sensor_latency_s", 0)),
            seed=config["sensor_corruption_seed"],
        )
        self.commands.motion.class_type = OutcomeScheduleCommand
        self.commands.motion.sensor_perturbation = asdict(settings)


class OutcomeScheduleRecorder(PerturbedScheduleRecorder):
    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_outcome_saved", False):
            return
        command = self.env.command_manager.get_term("motion")
        path = Path(self.save_dir) / "reactive_interface.json"
        interface = json.loads(path.read_text())
        interface.update(
            learner_family="outcome_tree",
            outcome_policy_schema=command.outcome_policy["schema"],
            prediction_interpretation="estimated feasibility and successful time; not a safety certificate",
            policy_format="JSON depth-two feasibility/time trees; no ridge readout",
        )
        path.write_text(json.dumps(interface, indent=2, allow_nan=False) + "\n")
        self._outcome_saved = True


@configclass
class OutcomeScheduleRecorderCfg(PerturbedScheduleRecorderCfg):
    class_type = OutcomeScheduleRecorder
