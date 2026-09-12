"""CPU checks of the separately declared sensor episode and causal verifier."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

import motion2scene_sensor_episode as episode
from motion2scene_sensor_perturbations import PerturbedObservationStream, SensorPerturbation

from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import SensorRay
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (
    schedule_history_features,
)


def fixture_context(monkeypatch):
    base = dict(
        cell_id="learned_neutral",
        output="/nominal",
        runtime_seed=91,
        hydra_overrides=[
            "++manager_env._target_=" + episode.nominal.RUNTIME + ".TimedScheduleEnvCfg",
            "++manager_env.recorders.trajectory._target_="
            + episode.nominal.RUNTIME
            + ".TimedScheduleRecorderCfg",
            "++manager_env.config.observation_delay_s=0.0",
            "++seed=91",
        ],
        command=["runner", "--out", "/nominal", "--extra", ""],
    )
    base["command"][-1] = " ".join(base["hydra_overrides"])
    cell = copy.deepcopy(base)
    cell["output"] = cell["command"][2] = "/sensitivity"
    settings = SensorPerturbation(0.3, 0.01, 0.04, 92)
    updates = {key: episode.RUNTIME + "." + names[0] for key, names in episode.TARGETS.items()}
    updates.update(
        {"manager_env.config." + episode.SENSOR_KEYS[k]: v for k, v in asdict(settings).items()}
    )
    cell = episode.replace_overrides(cell, updates)
    calls = []
    monkeypatch.setattr(
        episode.nominal, "validate_collection_context", lambda *a, **kw: calls.append((a, kw))
    )
    manifest = dict(schema=episode.SCHEMA, cell=cell, sensor_settings=asdict(settings))
    return manifest, dict(cells=[base]), SimpleNamespace(), dict(split="development"), calls


def test_sensor_intervention_preserves_nominal_policy_scene_seed_and_criteria(monkeypatch):
    manifest, common, bank, scene, calls = fixture_context(monkeypatch)
    assert episode.validate_context(manifest, common, bank, scene).latency_s == 0.04
    assert len(calls) == 1 and calls[0][1] == dict(actual=False)
    assert manifest["cell"]["command"][-1].count("sensor_latency_s=0.04") == 1


@pytest.mark.parametrize(
    "change", ["physics_seed", "sensor_setting", "duplicate", "command", "reserved"]
)
def test_unintended_changes_are_rejected(monkeypatch, change):
    manifest, common, bank, scene, _ = fixture_context(monkeypatch)
    if change == "physics_seed":
        manifest["cell"] = episode.replace_overrides(manifest["cell"], dict(seed=93))
    elif change == "sensor_setting":
        manifest["sensor_settings"]["latency_s"] = 0.1
    elif change == "duplicate":
        manifest["cell"]["hydra_overrides"].append("++seed=91")
        manifest["cell"]["command"][-1] = " ".join(manifest["cell"]["hydra_overrides"])
    elif change == "command":
        manifest["cell"]["command"][0] = "different_runner"
    else:
        scene["split"] = "reserved_evaluation_v3"
    with pytest.raises(ValueError):
        episode.validate_context(manifest, common, bank, scene)


def captured_interface(settings):
    bank = SimpleNamespace(option_ids=("neutral", "adapt"))
    stream = PerturbedObservationStream(settings)
    observations = []
    for tick in range(1, 8):
        rays = [SensorRay([0, 0, 1], [0, 0, 1], 4, 0.3, [0, 0, -1])] * 65
        cache, _ = stream.push(rays, tick / 50)
        row = dict(
            root_pos_w=[0, 0, 1],
            root_quat_w=[1, 0, 0, 0],
            tick=tick,
            state=dict(
                dof_pos=[0] * 29,
                dof_vel=[0] * 29,
                projected_gravity_b=[0, 0, -1],
                root_lin_vel_w=[0, 0, 0],
                root_ang_vel_w=[0, 0, 0],
            ),
            active_before="neutral",
            legal_mask=[True, True],
            **cache
        )
        observation = stream.history.snapshot(
            row["root_pos_w"], row["root_quat_w"], cache["capture_elapsed_s"]
        )
        names, features = schedule_history_features(
            observation,
            stream.history.grid,
            row["state"],
            tick,
            0,
            np.asarray(row["legal_mask"]),
            cache["observation_age_s"],
        )
        observations.append(dict(row, features=features.tolist()))
    return bank, json.loads(
        json.dumps(
            dict(
                sensor_schema=episode.SENSOR_SCHEMA,
                sensor_settings=asdict(settings),
                feature_names=names,
                observations=observations,
            )
        )
    )


def test_complete_corrupted_capture_reconstructs_and_rejects_feature_leakage():
    settings = SensorPerturbation(0.3, 0.01, 0.04, 92)
    bank, interface = captured_interface(settings)
    assert episode.verify_sensor_capture(interface, bank, settings) == dict(
        reconstructed_control_rows=7, exact_features=True
    )
    interface["observations"][0]["features"][0] += 1
    with pytest.raises(ValueError, match="student features"):
        episode.verify_sensor_capture(interface, bank, settings)


def test_future_packet_or_changed_dropout_is_rejected():
    settings = SensorPerturbation(dropout_probability=0.3, latency_s=0.04)
    bank, interface = captured_interface(settings)
    interface["observations"][0]["delivered_capture_elapsed_s"] = 0
    with pytest.raises(ValueError, match="delivered_capture_elapsed_s"):
        episode.verify_sensor_capture(interface, bank, settings)


def test_run_retains_charged_timeout_and_never_retries(tmp_path, monkeypatch):
    manifest = dict(
        cell=dict(output=str(tmp_path / "rollout"), command=["native"]),
        limits=dict(minimum_free_gpu_mib=7500, timeout_s=375),
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        episode,
        "verify",
        lambda *a, **kw: (manifest, dict(expected_physics_steps=1192), None, None),
    )
    monkeypatch.setattr(episode.nominal, "free_gpu_mib", lambda: 10000)

    def timeout(command, **kwargs):
        raise episode.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(episode.nominal, "run_with_process_group", timeout)
    monkeypatch.setattr(episode, "analyze", lambda out: dict(partial_retained=True))
    assert episode.run(tmp_path) == dict(partial_retained=True)
    attempt = json.loads((tmp_path / "rollout/attempt.json").read_text())
    assert attempt["exit_status"] == 124 and attempt["command"] == ["native"]
    assert (
        json.loads((tmp_path / "launch.json").read_text())["charged_maximum_physics_steps"] == 1192
    )
    with pytest.raises(ValueError, match="automatic retry forbidden"):
        episode.run(tmp_path)


def test_resource_wait_does_not_charge_or_launch(tmp_path, monkeypatch):
    manifest = dict(
        cell=dict(output=str(tmp_path / "rollout")), limits=dict(minimum_free_gpu_mib=7500)
    )
    monkeypatch.setattr(episode, "verify", lambda *a, **kw: (manifest, {}, None, None))
    monkeypatch.setattr(episode.nominal, "free_gpu_mib", lambda: 7000)
    with pytest.raises(RuntimeError, match="remains unlaunched"):
        episode.run(tmp_path)
    assert not (tmp_path / "launch.json").exists()
    assert not (tmp_path / "rollout").exists()
