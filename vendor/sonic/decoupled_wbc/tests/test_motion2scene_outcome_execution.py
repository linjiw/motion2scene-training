"""CPU adapter tests with simulator doubles; physical integration is separate."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

from motion2scene_sensor_perturbations import SensorPerturbation  # noqa: E402


def runtime(monkeypatch):
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    long = ModuleType("long")
    long.LongScheduleCommand = type("LongScheduleCommand", (), {})
    monkeypatch.setitem(
        sys.modules,
        "gear_sonic.dataset_generation.hallucination.motion2scene_long_schedule_execution",
        long,
    )
    timed = ModuleType("timed")
    timed.TimedScheduleCommand = type("TimedScheduleCommand", (long.LongScheduleCommand,), {})

    class Env:
        def __init__(self, config, **kwargs):
            self.commands = SimpleNamespace(
                motion=SimpleNamespace(timed_policy_path=config.get("timed_policy_path"))
            )

    timed.TimedScheduleEnvCfg = Env
    monkeypatch.setitem(
        sys.modules,
        "gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_execution",
        timed,
    )
    perturbed = ModuleType("perturbed")

    class Command:
        def _observe_schedule(self, tick, query):
            return tick, query

    class Recorder:
        def close_writers(self):
            pass

    perturbed.PerturbedScheduleCommand = Command
    perturbed.PerturbedScheduleRecorder = Recorder
    perturbed.PerturbedScheduleRecorderCfg = object
    monkeypatch.setitem(sys.modules, "scripts.research.motion2scene_perturbed_execution", perturbed)
    spec = importlib.util.spec_from_file_location(
        "outcome_runtime_test", ROOT / "scripts/research/motion2scene_outcome_execution.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_uses_explicit_outcome_family_without_loading_ridge(monkeypatch):
    module = runtime(monkeypatch)
    cfg = module.OutcomeScheduleEnvCfg(
        dict(
            timed_schedule_mode="learned",
            learner_family="outcome_tree",
            timed_policy_path="policy.json",
            sensor_corruption_seed=95001,
            sensor_latency_s=0.1,
        )
    )
    assert cfg.commands.motion.class_type is module.OutcomeScheduleCommand
    assert cfg.commands.motion.timed_policy_path == "policy.json"
    assert cfg.commands.motion.sensor_perturbation == asdict(
        SensorPerturbation(latency_s=0.1, seed=95001)
    )
    with pytest.raises(ValueError, match="explicit learned"):
        module.OutcomeScheduleEnvCfg(dict(timed_schedule_mode="learned", learner_family="ridge"))


def test_recording_identifies_tree_predictions_instead_of_ridge_values(monkeypatch, tmp_path):
    module = runtime(monkeypatch)
    path = tmp_path / "reactive_interface.json"
    path.write_text(json.dumps(dict(timed_schedule_mode="learned")))
    command = SimpleNamespace(outcome_policy=dict(schema="test_tree"))
    recorder = module.OutcomeScheduleRecorder()
    recorder._initialized, recorder.save_dir = True, tmp_path
    recorder.env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    recorder.close_writers()
    saved = json.loads(path.read_text())
    assert (
        saved["learner_family"] == "outcome_tree" and saved["outcome_policy_schema"] == "test_tree"
    )
    assert "no ridge readout" in saved["policy_format"]
    recorder.close_writers()
    assert json.loads(path.read_text()) == saved
