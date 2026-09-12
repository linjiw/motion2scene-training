"""Assigned equivalent-teaching episodes preserve the matched physical comparison."""

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_extension_episode as episode
from motion2scene_extension_teaching import arm_view, choose_arm_schedule
from motion2scene_fit_extension_teaching import fit_fold
from test_motion2scene_extension_teaching import bank, observation
from test_motion2scene_fit_extension_teaching import training_fold

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import definition_digest


def fixture_context(monkeypatch, tmp_path):
    base = dict(
        cell_id="forced_neutral",
        role="complete_schedule_teacher",
        policy_id="forced",
        timed_schedule_mode="forced",
        forced_option_id="neutral",
        output="/nominal",
        runtime_seed=97001,
        hydra_overrides=["++seed=97001", "++manager_env.config.timed_schedule_mode=forced"],
        command=["native", "--out", "/nominal", "--extra", ""],
    )
    base["command"][-1] = " ".join(base["hydra_overrides"])
    registry = dict(path="/registry.json", sha256="registry")
    template = dict(path="/held/manifest.json", sha256="template")
    ref = dict(path="/fit/result.json", sha256="result")
    result = dict(
        arm="generated",
        registry=registry,
        policy=dict(path="/fit/policy.npz", sha256="policy"),
        study=dict(path="/study.json", sha256="study"),
        analysis=dict(fold=dict(training_task_ids=["train"], evaluation_task_ids=["held"])),
    )
    study = dict(
        policy_evaluation_seed=97001, prepared=dict(path="/prepared.json", sha256="prepared")
    )
    prepared = dict(tasks=[dict(task_id="held", collection=template)])
    monkeypatch.setattr(
        episode, "read", lambda r: study if r["path"] == "/study.json" else prepared
    )
    monkeypatch.setattr(episode, "load_extension_policy", lambda *args: ({}, result))
    calls = []
    monkeypatch.setattr(
        episode.nominal,
        "validate_collection_context",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    manifest = dict(
        schema=episode.SCHEMA,
        sensor_settings=episode.asdict(episode.SensorPerturbation(seed=95001)),
        model_result=ref,
        policy=result["policy"],
        nominal_template=template,
        cell=episode.expected_cell(base, ref, result, tmp_path),
    )
    common = dict(cells=[base], registry=registry, policy=result["policy"])
    scene = dict(split="development", scene_id="held")
    return manifest, common, object(), scene, calls


def test_wrapper_changes_only_declared_policy_components(monkeypatch, tmp_path):
    manifest, common, b, scene, calls = fixture_context(monkeypatch, tmp_path)
    original = copy.deepcopy(common)
    episode.validate_context(manifest, common, b, scene)
    assert common == original
    assert calls[0][1] == dict(actual=False)
    normalized = calls[0][0][0]
    assert normalized["timed_schedule_mode"] == "learned"
    assert "++seed=97001" in normalized["hydra_overrides"]
    assert not any(
        "extension_result" in x or "learner_family" in x for x in normalized["hydra_overrides"]
    )


@pytest.mark.parametrize(
    "change", ["seed", "task", "reserved", "policy", "command", "duplicate", "runtime"]
)
def test_unassigned_or_changed_experiment_is_rejected(monkeypatch, tmp_path, change):
    manifest, common, b, scene, _ = fixture_context(monkeypatch, tmp_path)
    if change == "seed":
        manifest["cell"]["runtime_seed"] = 97002
    elif change == "task":
        scene["scene_id"] = "train"
    elif change == "reserved":
        scene["split"] = "reserved"
    elif change == "policy":
        manifest["policy"] = dict(path="/other.npz", sha256="other")
    elif change == "command":
        manifest["cell"]["command"][0] = "different"
    elif change == "runtime":
        manifest["cell"] = episode.replace_overrides(
            manifest["cell"], {"manager_env._target_": "Different.Runtime"}
        )
    else:
        manifest["cell"]["hydra_overrides"].append("++seed=97001")
        manifest["cell"]["command"][-1] = " ".join(manifest["cell"]["hydra_overrides"])
    with pytest.raises(ValueError):
        episode.validate_context(manifest, common, b, scene)


def policy_trace():
    b = bank()
    fold, training = training_fold(b)
    model, _ = fit_fold(b, "generated", fold, training)
    view, _ = arm_view(b, "generated")
    result = dict(arm="generated", logical_request_digest=definition_digest(view.request))
    ref = dict(path="/fit/result.json", sha256="result")
    names, x, legal = observation(b)
    selected, projected = choose_arm_schedule(b, "generated", model, names, x, legal, 0, 15, None)
    interface = dict(
        learner_family="extension_arm_ridge",
        extension_arm="generated",
        extension_result_sha256="result",
        logical_option_ids=list(view.option_ids),
        logical_request_digest=result["logical_request_digest"],
        logical_feature_names=model["feature_names"].tolist(),
        feature_names=list(names),
        observations=[
            dict(
                active_before="neutral",
                tick=15,
                features=x.tolist(),
                legal_mask=legal.tolist(),
                selected_option_id=selected,
                policy_values=None,
                arm_policy=projected,
            )
        ],
    )
    return interface, model, result, ref, b


def test_actual_policy_trace_checks_decisions_inputs_and_numerical_values():
    interface, model, result, ref, b = policy_trace()
    assert episode.verify_policy_trace(interface, model, result, ref, b) == 1
    original = copy.deepcopy(interface)
    interface["observations"][0]["arm_policy"]["policy_values"][0] += 1e-14
    assert episode.verify_policy_trace(interface, model, result, ref, b) == 1
    for change in ("choice", "input", "value", "arm"):
        altered = copy.deepcopy(original)
        row = altered["observations"][0]
        if change == "choice":
            row["selected_option_id"] = "authored_00_e15"
        elif change == "input":
            row["arm_policy"]["features"][0] += 1
        elif change == "value":
            row["arm_policy"]["policy_values"][0] += 1e-6
        else:
            altered["extension_arm"] = "authored"
        with pytest.raises(ValueError):
            episode.verify_policy_trace(altered, model, result, ref, b)


def test_forced_adaptation_is_not_a_neutral_policy_template():
    with pytest.raises(ValueError, match="forced-neutral"):
        episode.expected_cell(
            dict(timed_schedule_mode="forced", forced_option_id="generated_00_e15"),
            {},
            {},
            Path("/tmp/test"),
        )


def test_timeout_is_charged_and_not_automatically_retried(tmp_path, monkeypatch):
    manifest = dict(
        cell=dict(output=str(tmp_path / "rollout"), command=["native"]),
        limits=dict(minimum_free_gpu_mib=7500, timeout_s=375),
    )
    monkeypatch.setattr(
        episode,
        "verify",
        lambda *args, **kwargs: (manifest, dict(expected_physics_steps=1192), None, None),
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


def test_low_capacity_keeps_episode_unlaunched(tmp_path, monkeypatch):
    manifest = dict(
        cell=dict(output=str(tmp_path / "rollout"), command=["native"]),
        limits=dict(minimum_free_gpu_mib=7500),
    )
    monkeypatch.setattr(episode, "verify", lambda *args, **kwargs: (manifest, {}, None, None))
    monkeypatch.setattr(episode.nominal, "free_gpu_mib", lambda: 7000)
    with pytest.raises(RuntimeError, match="unlaunched"):
        episode.run(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("complete", [True, False])
def test_scorer_uses_shared_physical_criteria_and_does_not_claim_partial_trace(
    complete, tmp_path, monkeypatch
):
    folder = tmp_path / "rollout"
    folder.mkdir()
    (folder / "attempt.json").write_text(json.dumps(dict(command=["native"], exit_status=0)))
    (tmp_path / "manifest.json").write_text("{}")
    manifest = dict(cell=dict(output=str(folder), command=["native"]))
    monkeypatch.setattr(episode, "verify", lambda *args: (manifest, {}, None, None))

    def common_scorer(*args, execution_validator):
        assert execution_validator is episode.validate_context
        if not complete:
            raise ValueError("incomplete capture")
        return dict(outcome=dict(task_outcome="pass"))

    monkeypatch.setattr(episode, "analyze_complete", common_scorer)
    monkeypatch.setattr(
        episode.nominal,
        "analyze_incomplete",
        lambda *args: dict(outcome=dict(task_outcome="failure")),
    )
    ref = episode.analyze(tmp_path)
    saved = json.loads(Path(ref["path"]).read_text())
    assert saved["complete_policy_trace_verified"] is complete
    assert saved["row"]["outcome"]["task_outcome"] == ("pass" if complete else "failure")
