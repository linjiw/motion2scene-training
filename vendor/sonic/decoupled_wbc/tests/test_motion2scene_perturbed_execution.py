"""CPU hook tests with explicit simulator doubles, not native rollout evidence."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

from motion2scene_sensor_perturbations import SensorPerturbation  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import (  # noqa: E402
    SensorRay,
)


class Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.value)


def load_double_runtime(monkeypatch):
    parent = ModuleType("parent")

    class Command:
        def __init__(self, cfg, env):
            self.cfg, self.robot, self.interface_rows = cfg, env.robot, []

    class EnvCfg:
        def __init__(self, config, **kwargs):
            self.commands = SimpleNamespace(motion=SimpleNamespace())

    class Recorder:
        def close_writers(self):
            pass

    parent.TimedScheduleCommand = Command
    parent.TimedScheduleEnvCfg = EnvCfg
    parent.TimedScheduleRecorder = Recorder
    parent.TimedScheduleRecorderCfg = object
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    monkeypatch.setitem(
        sys.modules,
        "gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_execution",
        parent,
    )
    path = ROOT / "scripts/research/motion2scene_perturbed_execution.py"
    spec = importlib.util.spec_from_file_location("sensor_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_hook_uses_corrupted_history_and_handles_every_ray_missing(monkeypatch):
    runtime = load_double_runtime(monkeypatch)
    ray = SensorRay((0, 0, 1), (1, 0, 0), 4, 1, (-1, 0, 0))
    monkeypatch.setattr(runtime, "capture_ray_fan", lambda *args: [ray] * 65)
    robot = SimpleNamespace(
        data=SimpleNamespace(root_pos_w=[Tensor([0, 0, 1])], root_quat_w=[Tensor([1, 0, 0, 0])])
    )
    cfg = SimpleNamespace(sensor_perturbation=asdict(SensorPerturbation(dropout_probability=1)))
    command = runtime.PerturbedScheduleCommand(cfg, SimpleNamespace(robot=robot))
    occupied, origin, measurements, packet = command._observe_schedule(1, None)
    assert not occupied and measurements == packet["measurements"] == []
    assert origin == [0, 0, 1]
    assert command._history_observation["unknown"].all()
    assert len(command._history_cache["raw_measurements"]) == 65
    assert command.observation_history is command.sensor_stream.history


def test_runtime_configuration_retains_declared_corruption_and_rejects_fractional_seed(monkeypatch):
    runtime = load_double_runtime(monkeypatch)
    cfg = runtime.PerturbedScheduleEnvCfg(dict(sensor_corruption_seed=91, sensor_latency_s=0.1))
    assert cfg.commands.motion.sensor_perturbation["latency_s"] == 0.1
    assert cfg.commands.motion.class_type is runtime.PerturbedScheduleCommand
    with pytest.raises(ValueError):
        runtime.PerturbedScheduleEnvCfg(dict(sensor_corruption_seed=1.5))


def test_recorder_does_not_describe_corrupted_measurements_as_ideal(monkeypatch, tmp_path):
    runtime = load_double_runtime(monkeypatch)
    interface = tmp_path / "reactive_interface.json"
    interface.write_text(json.dumps(dict(sensor_model="nominal")))
    recorder = runtime.PerturbedScheduleRecorder()
    recorder._initialized, recorder.save_dir = True, tmp_path
    command = SimpleNamespace(
        sensor_stream=SimpleNamespace(settings=SensorPerturbation(range_noise_std_m=0.03))
    )
    recorder.env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    recorder.close_writers()
    saved = json.loads(interface.read_text())
    assert saved["sensor_schema"] == runtime.SENSOR_SCHEMA
    assert saved["sensor_settings"]["range_noise_std_m"] == 0.03
    assert "Gaussian" in saved["sensor_model"]
    recorder.close_writers()
    assert json.loads(interface.read_text()) == saved
