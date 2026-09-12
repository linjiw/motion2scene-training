"""CPU simulator doubles check the isolated full-bank/arm-policy runtime boundary."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

from motion2scene_extension_teaching import arm_view  # noqa: E402
from motion2scene_fit_extension_teaching import fit_fold  # noqa: E402
from motion2scene_sensor_perturbations import SensorPerturbation  # noqa: E402
from test_motion2scene_extension_teaching import bank, observation  # noqa: E402
from test_motion2scene_fit_extension_teaching import training_fold  # noqa: E402
from test_motion2scene_outcome_execution import runtime as install_simulator_doubles  # noqa: E402


def runtime(monkeypatch):
    install_simulator_doubles(monkeypatch)
    spec = importlib.util.spec_from_file_location(
        "extension_runtime_test", ROOT / "scripts/research/motion2scene_extension_execution.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configuration_requires_the_explicit_family_and_nominal_sensor(monkeypatch):
    module = runtime(monkeypatch)
    settings = dict(
        timed_schedule_mode="learned",
        learner_family="extension_arm_ridge",
        extension_result_path="result.json",
        extension_result_sha256="test",
    )
    cfg = module.ExtensionScheduleEnvCfg(settings)
    assert cfg.commands.motion.class_type is module.ExtensionScheduleCommand
    assert cfg.commands.motion.extension_result_sha256 == "test"
    with pytest.raises(ValueError, match="explicit learned"):
        module.ExtensionScheduleEnvCfg({**settings, "learner_family": "ridge"})
    with pytest.raises(ValueError, match="nominal sensing"):
        module.ExtensionScheduleEnvCfg({**settings, "sensor_latency_s": 0.1})


def test_initialization_keeps_full_bank_but_loads_only_its_declared_arm_model(monkeypatch):
    module = runtime(monkeypatch)
    b = bank()
    fold, training = training_fold(b)
    model, _ = fit_fold(b, "generated", fold, training)
    cfg = SimpleNamespace(
        timed_registry_path="full.json",
        timed_registry_sha256="full",
        extension_result_path="result.json",
        extension_result_sha256="result",
        timed_policy_path="policy.npz",
        timed_policy_sha256="policy",
    )
    monkeypatch.setattr(
        module.LongScheduleCommand, "__init__", lambda self, *_: setattr(self, "timed_bank", b)
    )
    monkeypatch.setattr(module, "load_verified_registry", lambda *args: b)
    monkeypatch.setattr(
        module,
        "load_extension_policy",
        lambda *args: (
            model,
            dict(arm="generated", policy=dict(path="policy.npz", sha256="policy")),
        ),
    )
    command = module.ExtensionScheduleCommand(cfg, object())
    assert command.timed_bank is b and len(command.timed_bank.option_ids) == 5
    assert len(command.extension_view.option_ids) == 3
    assert asdict(command.sensor_stream.settings) == asdict(SensorPerturbation(seed=95001))
    cfg.timed_policy_sha256 = "different"
    with pytest.raises(ValueError, match="archive differs"):
        module.ExtensionScheduleCommand(cfg, object())


def test_runtime_records_full_inputs_and_actual_projected_choice(monkeypatch):
    module = runtime(monkeypatch)
    b = bank()
    fold, training = training_fold(b)
    model, _ = fit_fold(b, "generated", fold, training)
    names, x, legal = observation(b)
    command = module.ExtensionScheduleCommand.__new__(module.ExtensionScheduleCommand)
    command.timed_bank, command.extension_arm, command.history_policy = b, "generated", model
    command.timed_state = SimpleNamespace(active="neutral")
    command.schedule_phases = np.array([15, 50])
    command._history_cache = dict(observation_age_s=0)
    command.history_feature_names = None
    command.observation_history = SimpleNamespace(grid=object())
    command._history_observation = {
        k: np.array([True]) for k in ("floor_observed", "ceiling_observed", "occupied")
    }
    monkeypatch.setattr(module, "legal_timed_actions", lambda *args: (legal, None))
    monkeypatch.setattr(
        module.LongScheduleCommand, "_capture_state", lambda *args: dict(state={}), raising=False
    )
    monkeypatch.setattr(module, "schedule_history_features", lambda *args: (names, x))
    selected = command._requested_schedule(15, {}, {}, {})
    assert selected == "generated_00_e15"
    assert command._history_cache["features"] == x.tolist()
    assert command._history_cache["legal_mask"] == legal.tolist()
    assert command._history_cache["policy_values"] is None
    assert len(command._history_cache["arm_policy"]["features"]) == 106
    assert command._history_cache["arm_policy"]["original_option_indices"] == [0, 1, 2]


def test_recorder_keeps_logical_menu_separate_from_loaded_bank(monkeypatch, tmp_path):
    module = runtime(monkeypatch)
    b = bank()
    fold, training = training_fold(b)
    model, _ = fit_fold(b, "generated", fold, training)
    path = tmp_path / "reactive_interface.json"
    path.write_text(json.dumps(dict(option_ids=list(b.option_ids))))
    command = SimpleNamespace(
        extension_arm="generated",
        extension_view=arm_view(b, "generated")[0],
        history_policy=model,
        cfg=SimpleNamespace(extension_result_sha256="result"),
        sensor_stream=SimpleNamespace(settings=SensorPerturbation(seed=95001)),
    )
    recorder = module.ExtensionScheduleRecorder()
    recorder._initialized, recorder.save_dir = True, tmp_path
    recorder.env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    recorder.close_writers()
    saved = json.loads(path.read_text())
    assert saved["option_ids"] == list(b.option_ids)
    assert saved["logical_option_ids"] == list(command.extension_view.option_ids)
    assert saved["sensor_settings"] == asdict(SensorPerturbation(seed=95001))
    assert saved["learner_family"] == "extension_arm_ridge"
    recorder.close_writers()
    assert json.loads(path.read_text()) == saved
