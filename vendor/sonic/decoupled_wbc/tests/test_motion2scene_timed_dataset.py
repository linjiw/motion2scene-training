"""Portable timed-data boundaries: actual identity, commands and causal inputs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "timed_export", ROOT / "scripts/research/motion2scene_export_timed_dataset.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_result(root, suffix, trajectory=None, manifest_path=None):
    if trajectory is None:
        trajectory = root / suffix / "rollouts/cell/trajectories/000000.trajectory.pkl"
        trajectory.parent.mkdir(parents=True)
        trajectory.write_bytes(b"identical actual capture bytes")
        save(trajectory.parent / "reactive_interface.json", {"observations": []})
    if manifest_path is None:
        manifest_path = root / suffix / "manifest.json"
        save(
            manifest_path, {"cells": [{"cell_id": "cell", "output": str(trajectory.parent.parent)}]}
        )
    result = root / suffix / "result.json"
    save(
        result,
        {
            "schema": "motion2scene_timed_history_collection_v1",
            "manifest": MODULE.ref(manifest_path),
            "rows": [{"cell_id": "cell", "trajectory": MODULE.ref(trajectory), "physics_steps": 4}],
        },
    )
    return result, trajectory, manifest_path


def test_distinct_actual_captures_with_identical_bytes_are_retained(tmp_path):
    a, _, _ = make_result(tmp_path, "first")
    b, _, _ = make_result(tmp_path, "second")
    records, _ = MODULE.collect_sources([a, b])
    assert len(records) == 2
    assert records[0]["trajectory_ref"]["sha256"] == records[1]["trajectory_ref"]["sha256"]


def test_same_capture_multiple_assessments_is_counted_once(tmp_path):
    a, trajectory, manifest = make_result(tmp_path, "first")
    b, _, _ = make_result(tmp_path, "second", trajectory, manifest)
    records, studies = MODULE.collect_sources([a, b])
    assert len(records) == 1 and len(studies) == 2
    assert len(records[0]["assessments"]) == 2


def test_configured_schedule_does_not_replace_actual_switches():
    interface = {
        "forced_option_id": "sustained",
        "switches": [
            {"tick": 15, "from": "neutral", "to": "short"},
            {"tick": 265, "from": "short", "to": "neutral"},
        ],
        "observations": [
            {"tick": 14, "active_before": "neutral", "active": "neutral"},
            {"tick": 15, "active_before": "neutral", "active": "short"},
            {"tick": 265, "active_before": "short", "active": "neutral"},
        ],
    }
    actual = MODULE.executed_schedule(interface)
    assert actual["actual_entered_option_ids"] == ["short"]
    assert actual["configured_forced_option_id"] == "sustained"
    interface["switches"] = []
    interface["observations"] = [{"tick": 15, "active_before": "neutral", "active": "neutral"}]
    assert MODULE.executed_schedule(interface)["no_entry_neutral"]
    interface["observations"][0]["active"] = "sustained"
    with pytest.raises(ValueError, match="switch history"):
        MODULE.executed_schedule(interface)


def test_sensor_identity_removed_and_normal_known_preserved():
    record = {
        "time_s": 0.3,
        "measurements": [
            {
                "origin_w": [0, 0, 1],
                "direction_w": [1, 0, 0],
                "hit_distance_m": 0.5,
                "hit_normal_w": [0, 0, -1],
                "object_id": "beam",
                "scene": "secret",
            }
        ],
        "normal_known_mask": [True],
        "features": [99],
        "selected_option_id": "short",
    }
    packet = MODULE.causal_packets({"observations": [record]})[0]
    assert packet["normal_known_mask"] == [True]
    assert "features" not in packet and "selected_option_id" not in packet
    assert "object_id" not in packet["measurements"][0] and "scene" not in packet["measurements"][0]
    record["normal_known_mask"] = [False]
    with pytest.raises(ValueError, match="Normal-known"):
        MODULE.causal_packets({"observations": [record]})


def test_exact106d_input_whitelist_excludes_scene_labels(tmp_path):
    source, output = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    output.mkdir()
    names = MODULE.expected_feature_names()
    feature = np.zeros((1, 106), dtype=np.float32)
    mask = np.ones((1, 3), dtype=bool)
    np.savez_compressed(
        source / "timed_history_features.npz",
        schema_version=MODULE.TIMED_SCHEMA,
        feature_names=names,
        option_ids=["neutral", "short", "sustained"],
        features=feature,
        command_ticks=[15],
        capture_elapsed_s=[0.28],
        legal_mask=mask,
        active=["neutral"],
        scene_ground_truth=[1.2],
    )
    interface = {
        "feature_schema": MODULE.TIMED_SCHEMA,
        "feature_names": names,
        "option_ids": ["neutral", "short", "sustained"],
        "observations": [
            {"tick": 15, "features": feature[0].tolist(), "legal_mask": mask[0].tolist()}
        ],
    }
    result = MODULE.export_features(output, source, interface, {"packet_eligible": [True]})
    assert result["dimension"] == 106
    with np.load(output / "student_inputs.npz", allow_pickle=False) as values:
        assert set(values.files) == MODULE.FEATURE_KEYS
        assert "scene_ground_truth" not in values.files and "active" not in values.files


def test_portable_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        MODULE.safe_path(tmp_path, "../secret")


def make_comparison_suite(tmp_path):
    child, _, manifest = make_result(tmp_path, "child")
    body = json.loads(child.read_text())
    body["physics_steps"] = 4
    save(child, body)
    _, studies = MODULE.collect_sources([child])
    registered = {
        "manifest": MODULE.ref(manifest),
        "mode": "learned",
        "scene_index": 0,
        "scene_definition": {"scene": "development"},
    }
    plan = tmp_path / "plan.json"
    save(
        plan,
        {
            "schema": "motion2scene_timed_policy_comparison_v1",
            "expected_episodes": 1,
            "expected_physics_steps": 4,
            "implementation": [],
        },
    )
    registration = tmp_path / "registration.json"
    save(registration, {"plan": MODULE.ref(plan), "children": [registered]})
    suite = tmp_path / "result.json"
    save(
        suite,
        {
            "registration": MODULE.ref(registration),
            "episodes": 1,
            "physics_steps": 4,
            "groups": [
                {
                    **registered,
                    "result": MODULE.ref(child),
                    "rows": body["rows"],
                    "physics_steps": 4,
                }
            ],
        },
    )
    return suite, studies


def test_comparison_parent_is_preserved_without_double_counting(tmp_path):
    suite, studies = make_comparison_suite(tmp_path)
    result = MODULE.export_comparison_suite(suite, studies, tmp_path / "portable/suite", tmp_path)
    assert result["counted_as_additional_episodes"] is False
    assert json.loads((tmp_path / "portable/suite/result.json").read_text())["episodes"] == 1


@pytest.mark.parametrize("fault", ["missing_child", "duplicate_child", "changed_label"])
def test_comparison_suite_cannot_omit_or_replace_actual_children(tmp_path, fault):
    suite, studies = make_comparison_suite(tmp_path)
    body = json.loads(suite.read_text())
    if fault == "missing_child":
        studies.clear()
    elif fault == "duplicate_child":
        body["groups"].append(body["groups"][0])
    else:
        body["groups"][0]["rows"][0]["pass"] = True
    save(suite, body)
    with pytest.raises(ValueError):
        MODULE.export_comparison_suite(suite, studies, tmp_path / "portable/suite", tmp_path)
