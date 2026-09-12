"""Portable learner input admission preserves WAIT, split and chronology semantics."""

import copy
import json

import numpy as np
import pytest

from scripts.research.motion2scene_dataset_baseline import verify_teacher_row
from scripts.research.motion2scene_export_option_dataset import MULTI_SCHEMA


def fixture(tmp_path):
    folder = tmp_path / "episodes/input"
    folder.mkdir(parents=True)
    np.savez_compressed(
        folder / "student_history.npz",
        schema_version=MULTI_SCHEMA,
        features=np.array([[0.25, 0.5]]),
        feature_names=np.array(["sensor_a", "sensor_b"]),
        option_ids=np.array(["neutral", "duck"]),
        phase_s=np.array([0.2]),
        active_before=np.array([0]),
        legal_mask=np.array([[True, True]]),
    )
    (folder / "sensor_alignment.json").write_text(json.dumps({"packet_eligible": [True]}))
    episode = {
        "split": "development_only",
        "directory": "episodes/input",
        "student_input": {"file": "student_history.npz"},
        "source": 1,
        "physics_seed": 3,
        "condition": "present",
        "beam": {"height": 1.2},
        "registered_option_ids": ["neutral", "duck"],
        "measurement_admitted": True,
        "pass": True,
        "reset_count": 0,
        "failure_flags": {"fall_observed": False},
        "executed_option": {
            "first_episode_final_option_index": 0,
            "stayed_neutral": False,
            "return_time_s": 3.3,
            "option_index": 1,
        },
        "costs": {"passage_time_s": 2.9},
    }
    episodes = {"input": copy.deepcopy(episode), "future_duck": copy.deepcopy(episode)}
    target = {
        "student_input": {
            "episode_id": "input",
            "file": "episodes/input/student_history.npz",
            "row_index": 0,
            "phase_s": 0.2,
        },
        "phase_s": 0.2,
        "pass_labels": [True, True],
        "passage_time_s": [2.9, 2.9],
        "admitted": [True, True],
        "legal_mask": [True, True],
        "continuation_episode_ids": ["future_duck", "future_duck"],
        "admitted_continuation_counts": [1, 1],
        "expected_continuation_counts": [1, 1],
        "matching_audit": [
            {"episode_id": "future_duck", "matched": True, "prefix": {"exact_match": True}}
        ],
    }
    return target, episodes


def test_wait_keeps_future_adaptation_label_and_sensor_only_features(tmp_path):
    target, episodes = fixture(tmp_path)
    x, names, ids = verify_teacher_row(tmp_path, target, episodes)
    assert np.array_equal(x, [0.25, 0.5])
    assert names == ["sensor_a", "sensor_b"]
    assert ids == ["neutral", "duck"]
    assert episodes[target["continuation_episode_ids"][0]]["executed_option"]["option_index"] == 1


@pytest.mark.parametrize(
    "change", ["test_split", "different_scene", "missing_return", "unmatched", "missing_schedule"]
)
def test_teacher_target_rejects_nontraining_or_invalid_continuation(tmp_path, change):
    target, episodes = fixture(tmp_path)
    if change == "test_split":
        episodes["future_duck"]["split"] = "evaluation"
    elif change == "different_scene":
        episodes["future_duck"]["beam"]["height"] = 1.5
    elif change == "missing_return":
        episodes["future_duck"]["executed_option"]["return_time_s"] = None
    elif change == "unmatched":
        target["matching_audit"][0]["prefix"]["exact_match"] = False
    else:
        target["expected_continuation_counts"][0] = 2
    with pytest.raises(ValueError):
        verify_teacher_row(tmp_path, target, episodes)


def test_post_reset_packet_cannot_become_training_input(tmp_path):
    target, episodes = fixture(tmp_path)
    (tmp_path / "episodes/input/sensor_alignment.json").write_text(
        json.dumps({"packet_eligible": [False]})
    )
    with pytest.raises(ValueError, match="ineligible sensor packet"):
        verify_teacher_row(tmp_path, target, episodes)


def test_offline_fit_writes_only_training_report_and_portable_sensor_model(tmp_path, monkeypatch):
    from scripts.research import motion2scene_dataset_baseline as baseline

    data = {
        "features": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "feature_names": ["sensor_a", "sensor_b"],
        "option_ids": ["neutral", "duck"],
        "passed": np.array([[True, True], [False, True]]),
        "times": [[1.0, 1.5], [np.nan, 1.5]],
        "admitted": np.ones((2, 2), dtype=bool),
        "legal": np.ones((2, 2), dtype=bool),
        "datasets": [],
        "inputs": [],
        "semantics": "WAIT with finite continuation",
    }
    monkeypatch.setattr(baseline, "load_training_datasets", lambda _: data)
    out = tmp_path / "fit"
    summary = baseline.fit([], out, 1e-6)
    assert summary["new_physics_steps"] == 0
    report = json.loads((out / "training_report.json").read_text())
    assert report["evaluation_episodes"] == 0
    with np.load(out / "policy.npz", allow_pickle=False) as model:
        assert model["feature_names"].tolist() == ["sensor_a", "sensor_b"]
        assert model["weights"].shape == (2, 2)
        assert "beam" not in model and "pass_labels" not in model
    with pytest.raises(FileExistsError):
        baseline.fit([], out, 1e-6)
