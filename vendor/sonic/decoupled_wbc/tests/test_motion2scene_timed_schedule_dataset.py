"""Portable input separation and future-continuation identity checks."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from decoupled_wbc.tests.test_motion2scene_timed_schedule_training_audit import (
    TRAINER,
    collection as build_collection,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "timed_schedule_export", ROOT / "scripts/research/motion2scene_export_timed_schedule_dataset.py"
)
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


@pytest.fixture
def collection(tmp_path, monkeypatch):
    return build_collection.__wrapped__(tmp_path, monkeypatch)


def portable_teacher(fixture):
    teacher = TRAINER.audit_collection(fixture["path"], fixture["bank"], fixture["registry"])
    identifiers = ["episode_" + str(i) for i in range(len(fixture["bank"].option_ids))]
    teacher.update(
        episode_ids=identifiers,
        recorded_history_group=[
            fixture["registry"]["sha256"],
            fixture["manifest"]["scene_definition"]["sha256"],
            teacher["physics_seed"],
        ],
    )
    episodes = {}
    for i, option_id in enumerate(fixture["bank"].option_ids):
        row, payload, interface = fixture["actual"][option_id]
        episodes[identifiers[i]] = (
            dict(
                configured_forced_option_id=option_id,
                physics_seed=8731,
                collection_sha256=teacher["collection"]["sha256"],
                assessment=row,
                original_cell_id=option_id,
                physics_steps=row["physics_steps"],
            ),
            payload,
            interface,
        )
    for target in teacher["targets"]:
        target["continuation_episode_ids"] = [
            None if i is None else identifiers[i] for i in target["continuation_option_indices"]
        ]
    return teacher, episodes


def test_wait_continuation_is_recomputed_from_portable_measured_schedules(collection):
    teacher, episodes = portable_teacher(collection)
    assert EXPORTER.verify_teacher_group(teacher, episodes, collection["bank"]) == 3
    assert teacher["targets"][0]["continuation_episode_ids"][0] == "episode_6"
    damaged = copy.deepcopy(teacher)
    damaged["targets"][0]["continuation_episode_ids"][0] = "episode_0"
    with pytest.raises(ValueError, match="continuation episode"):
        EXPORTER.verify_teacher_group(damaged, episodes, collection["bank"])
    damaged = copy.deepcopy(teacher)
    damaged["targets"][0]["passage_time_s"][0] = 1.0
    with pytest.raises(ValueError, match="actual continuation"):
        EXPORTER.verify_teacher_group(damaged, episodes, collection["bank"])


def test_portable_teacher_cannot_drop_a_future_branch_or_reuse_a_different_seed(collection):
    teacher, episodes = portable_teacher(collection)
    teacher["episode_ids"] = teacher["episode_ids"][:-1]
    with pytest.raises(ValueError, match="distinct episode"):
        EXPORTER.verify_teacher_group(teacher, episodes, collection["bank"])
    teacher, episodes = portable_teacher(collection)
    episodes["episode_6"][0]["physics_seed"] = 8732
    with pytest.raises(ValueError, match="branch identities"):
        EXPORTER.verify_teacher_group(teacher, episodes, collection["bank"])


def test_student_input_whitelist_preserves_preaction_but_excludes_postaction_and_labels(tmp_path):
    names = EXPORTER.expected_feature_names(7)
    x = np.zeros((1, 114), dtype=np.float32)
    mask = np.array([[True, False, False, False, False, True, True]])
    source, destination = tmp_path / "source.npz", tmp_path / "student_inputs.npz"
    np.savez_compressed(
        source,
        schema_version=EXPORTER.INPUT_SCHEMA,
        feature_names=names,
        option_ids=["neutral", "a", "b", "c", "d", "e", "f"],
        features=x,
        command_ticks=[70],
        capture_elapsed_s=[1.38],
        legal_mask=mask,
        active_before=["neutral"],
        active=["e"],
        scene_geometry=[1.2],
        passed=[True],
        policy_values=[2],
    )
    interface = dict(
        option_ids=["neutral", "a", "b", "c", "d", "e", "f"],
        observations=[
            dict(
                features=x[0].tolist(),
                tick=70,
                active_before="neutral",
                legal_mask=mask[0].tolist(),
            )
        ],
    )
    result = EXPORTER.export_inputs(source, destination, interface)
    assert set(result) == EXPORTER.INPUT_KEYS
    assert result["active_before"].tolist() == ["neutral"]
    assert "active" not in result and "scene_geometry" not in result and "passed" not in result
    interface["observations"][0]["active_before"] = "e"
    with pytest.raises(ValueError, match="preaction input"):
        EXPORTER.export_inputs(source, destination, interface)


def test_pickle_free_metadata_reconstructs_nested_physical_arrays(tmp_path):
    source = {
        "fps": 50,
        "root_pos_w": np.arange(6).reshape(2, 3),
        "nested": {"value": np.array([1.5], dtype=np.float32)},
        "note": "recorded",
    }
    metadata, excluded = EXPORTER.save_pickle_free(tmp_path / "trajectory.npz", source)
    assert not excluded
    EXPORTER.write_json(tmp_path / "trajectory_metadata.json", metadata)
    reconstructed = EXPORTER.portable_payload(tmp_path)
    assert reconstructed["fps"] == 50 and reconstructed["note"] == "recorded"
    np.testing.assert_array_equal(reconstructed["root_pos_w"], source["root_pos_w"])
    np.testing.assert_array_equal(reconstructed["nested"]["value"], source["nested"]["value"])
