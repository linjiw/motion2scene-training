"""The learner intervention must preserve the paired physical experiment."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

import motion2scene_outcome_episode as episode
from motion2scene_sensor_perturbations import SensorPerturbation


def fixture_context(monkeypatch, tmp_path):
    base = dict(
        cell_id="learned_neutral",
        timed_schedule_mode="learned",
        output="/nominal",
        runtime_seed=91,
        hydra_overrides=[
            "++manager_env._target_=" + episode.nominal.RUNTIME + ".TimedScheduleEnvCfg",
            "++manager_env.recorders.trajectory._target_="
            + episode.nominal.RUNTIME
            + ".TimedScheduleRecorderCfg",
            "++manager_env.config.timed_policy_path=/original.npz",
            "++manager_env.config.timed_policy_sha256=original",
            "++seed=91",
        ],
        command=["runner", "--out", "/nominal", "--extra", ""],
    )
    base["command"][-1] = " ".join(base["hydra_overrides"])
    policy = dict(path="/outcome.json", sha256="new_model")
    settings = SensorPerturbation(seed=95001)
    manifest = dict(
        schema=episode.SCHEMA,
        policy=policy,
        sensor_settings=asdict(settings),
        cell=episode.expected_cell(base, policy, settings, tmp_path),
    )
    calls = []
    monkeypatch.setattr(episode, "load_policy", lambda *a: dict(schema="portable"))
    monkeypatch.setattr(
        episode.nominal, "validate_collection_context", lambda *a, **kw: calls.append((a, kw))
    )
    return manifest, dict(cells=[base], policy=policy), object(), dict(split="development"), calls


def test_only_learner_changes_and_shared_physical_validation_is_used(monkeypatch, tmp_path):
    manifest, common, bank, scene, calls = fixture_context(monkeypatch, tmp_path)
    original = copy.deepcopy(common["cells"])
    episode.validate_context(manifest, common, bank, scene)
    assert common["cells"] == original
    assert len(calls) == 1 and calls[0][1] == dict(actual=False)
    normalized = calls[0][0][0]["hydra_overrides"]
    assert "++seed=91" in normalized
    assert "++manager_env.config.timed_policy_path=/outcome.json" in normalized
    assert not any("sensor_" in value or "learner_family" in value for value in normalized)


@pytest.mark.parametrize("change", ["seed", "duplicate", "runtime", "reserved", "command"])
def test_unrelated_interventions_rejected(monkeypatch, tmp_path, change):
    manifest, common, bank, scene, _ = fixture_context(monkeypatch, tmp_path)
    if change == "seed":
        manifest["cell"] = episode.replace_overrides(manifest["cell"], dict(seed=92))
    elif change == "runtime":
        manifest["cell"] = episode.replace_overrides(
            manifest["cell"], {"manager_env._target_": "unrelated.Runtime"}
        )
    elif change == "duplicate":
        manifest["cell"]["hydra_overrides"].append("++seed=91")
        manifest["cell"]["command"][-1] = " ".join(manifest["cell"]["hydra_overrides"])
    elif change == "command":
        manifest["cell"]["command"][0] = "unrelated"
    else:
        scene["split"] = "reserved"
    with pytest.raises(ValueError):
        episode.validate_context(manifest, common, bank, scene)


def test_timeout_is_charged_and_retained_without_retry(tmp_path, monkeypatch):
    manifest = dict(
        cell=dict(output=str(tmp_path / "rollout"), command=["native"]),
        limits=dict(minimum_free_gpu_mib=7500, timeout_s=375),
    )
    monkeypatch.setattr(
        episode,
        "verify",
        lambda *a, **kw: (manifest, dict(expected_physics_steps=1192), None, None),
    )
    monkeypatch.setattr(episode.nominal, "free_gpu_mib", lambda: 10000)

    def timeout(command, **kwargs):
        raise episode.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(episode.nominal, "run_with_process_group", timeout)
    monkeypatch.setattr(episode, "analyze", lambda out: dict(retained=True))
    assert episode.run(tmp_path) == dict(retained=True)
    assert json.loads((tmp_path / "rollout/attempt.json").read_text())["exit_status"] == 124
    with pytest.raises(ValueError, match="automatic retry"):
        episode.run(tmp_path)
