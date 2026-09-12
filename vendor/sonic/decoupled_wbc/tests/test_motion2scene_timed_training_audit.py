"""Adversarial provenance checks around the timed trainer's independent re-audit."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))
SPEC = importlib.util.spec_from_file_location(
    "timed_trainer_audit", ROOT / "scripts/research/motion2scene_train_timed_policy.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return MODULE.artifact(path)


@pytest.fixture
def collection(tmp_path, monkeypatch):
    ids = ("neutral", "short", "sustained")
    bank = SimpleNamespace(option_ids=ids, request={"fixture": "shared_reference_registry"})
    registry = save(tmp_path / "registry.json", {"fixture": True})
    scene_asset = save(tmp_path / "scene.usda", {"fixture": "same room"})
    scene = save(
        tmp_path / "scene.json",
        {"split": "development", "scene_id": "test_scene", "scene": scene_asset},
    )
    names = MODULE.expected_feature_names()
    features = np.zeros(106)
    features[names.index("phase_s")] = 0.3
    features[names.index("active_option_0")] = 1
    features[-3:] = 1
    payload = {
        key: np.zeros((15, 3))
        for key in (
            "dof_pos",
            "dof_vel",
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "applied_joint_action",
            "action_motion_token",
            "reference_g1_qpos",
        )
    }
    payload["motion_time_s"] = np.arange(15) / 50
    interface = {
        "observations": [
            {
                "tick": tick,
                "features": features.tolist(),
                "legal_mask": [True] * 3,
                "measurements": [],
            }
            for tick in range(1, 16)
        ]
    }
    rows, cells, actual = [], [], {}
    for option in ids:
        folder = tmp_path / "rollouts" / option
        artifact_refs = {}
        for key in (
            "trajectory",
            "sensor",
            "features",
            "all_body_contacts",
            "environment_pairs",
            "environment_mapping",
            "physics_beam_contacts",
            "attempt",
        ):
            artifact_refs[key] = save(folder / "trajectories" / (key + ".json"), {"fixture": key})
        command = ["bash", "fixture.sh", "--extra", "++seed=8731"]
        artifact_refs["attempt"] = save(
            folder / "attempt.json", {"exit_status": 0, "command": command}
        )
        save(
            folder / "success_manifest.json",
            {
                "capture_context": {
                    "scene_id": "test_scene",
                    "scene": {
                        "path": scene_asset["path"],
                        "resolved": scene_asset["path"],
                        "hash": scene_asset["sha256"],
                    },
                },
                "runtime_config_hash": "synthetic_fixture",
            },
        )
        row = {
            "cell_id": option,
            "forced_option_id": option,
            "measurement_admitted": True,
            "pass": True,
            "costs": {"passage_time_s": 3.0},
            "schedule_audit": {"valid": True},
            "observation_timing": {"fixture": True},
            "physics_steps": 60,
            **artifact_refs,
        }
        rows.append(row)
        cells.append(
            {
                "cell_id": option,
                "forced_option_id": option,
                "timed_history_mode": "forced",
                "runtime_seed": 8731,
                "hydra_overrides": ["++seed=8731"],
                "command": command,
                "output": str(folder),
                "scene": scene_asset,
            }
        )
        actual[option] = (copy.deepcopy(row), copy.deepcopy(payload), copy.deepcopy(interface))
    teacher = {
        "entry_tick": 15,
        "option_ids": list(ids),
        "feature_names": list(names),
        "features": features.tolist(),
        "legality": [True] * 3,
        "passed": [True] * 3,
        "passage_time_s": [3.0] * 3,
        "admitted": [True] * 3,
        "exact_shared_entry_observation": True,
    }
    manifest = {
        "schema": MODULE.COLLECTION_SCHEMA,
        "split": "development",
        "registry": registry,
        "request_digest": MODULE.definition_digest(bank.request),
        "scene_definition": scene,
        "cells": cells,
    }
    result = {"schema": MODULE.COLLECTION_SCHEMA, "teacher": teacher, "rows": rows}

    def commit():
        result["manifest"] = save(tmp_path / "manifest.json", manifest)
        save(tmp_path / "result.json", result)
        return tmp_path / "result.json"

    monkeypatch.setattr(MODULE, "analyze_cell", lambda cell, *_: actual[cell["forced_option_id"]])
    return SimpleNamespace(
        bank=bank,
        registry=registry,
        manifest=manifest,
        result=result,
        actual=actual,
        commit=commit,
        root=tmp_path,
    )


def test_audit_fixture_has_matched_prefix_and_complete_teacher(collection):
    result = MODULE.audit_collection(collection.commit(), collection.bank, collection.registry)
    assert result["admitted"] == [True] * 3
    assert result["physics_steps"] == 180


def test_mixed_branch_seeds_are_rejected(collection):
    collection.manifest["cells"][1]["runtime_seed"] = 9999
    with pytest.raises(ValueError):
        MODULE.audit_collection(collection.commit(), collection.bank, collection.registry)


def test_duplicate_cells_are_rejected(collection):
    collection.manifest["cells"].append(copy.deepcopy(collection.manifest["cells"][0]))
    with pytest.raises(ValueError):
        MODULE.audit_collection(collection.commit(), collection.bank, collection.registry)


def test_duplicate_result_rows_are_rejected(collection):
    collection.result["rows"].append(copy.deepcopy(collection.result["rows"][0]))
    with pytest.raises(ValueError):
        MODULE.audit_collection(collection.commit(), collection.bank, collection.registry)


def test_stored_row_cannot_name_an_unrelated_valid_capture(collection):
    unrelated = save(collection.root / "unrelated.json", {"another": "capture"})
    collection.result["rows"][1]["trajectory"] = unrelated
    with pytest.raises(ValueError):
        MODULE.audit_collection(collection.commit(), collection.bank, collection.registry)


def test_actual_scene_hash_must_match_registration(collection):
    folder = Path(collection.manifest["cells"][1]["output"])
    save(
        folder / "success_manifest.json",
        {
            "capture_context": {
                "scene_id": "test_scene",
                "scene": {
                    "path": "different_room.usda",
                    "resolved": "different_room.usda",
                    "hash": "sha256:" + "f" * 64,
                },
            },
            "runtime_config_hash": "synthetic_fixture",
        },
    )
    with pytest.raises(ValueError):
        MODULE.audit_collection(collection.commit(), collection.bank, collection.registry)


def test_actual_seed_command_must_match_declared_override(collection):
    for cell in collection.manifest["cells"]:
        cell["runtime_seed"] = 9999
        cell["hydra_overrides"] = ["++seed=9999"]
    with pytest.raises(ValueError):
        MODULE.audit_collection(collection.commit(), collection.bank, collection.registry)


def test_cached_cost_cannot_replace_actual_measured_time(collection):
    collection.result["rows"][1]["costs"]["passage_time_s"] = 0.1
    with pytest.raises(ValueError, match="costs"):
        MODULE.audit_collection(collection.commit(), collection.bank, collection.registry)


@pytest.mark.parametrize("change", ["missing", "unadmitted", "wrong_seed", "wrong_prefix"])
def test_failed_future_continuation_cannot_be_omitted_from_wait(change):
    from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (
        TimedScheduledOutcome,
        timed_schedule_teacher,
    )

    bank = SimpleNamespace(
        online_verified=True,
        option_ids=("neutral", "now", "later", "last"),
        request={
            "max_entries_per_episode": 1,
            "options": [
                {"option_id": name, "entry_tick": tick}
                for name, tick in [("now", 15), ("later", 50), ("last", 70)]
            ],
        },
    )
    branches = [
        TimedScheduledOutcome(
            str(i), i, 8731, i == 2, 3.0 if i == 2 else None, True, {15: "same"}, 1192
        )
        for i in range(4)
    ]
    if change == "missing":
        branches.pop()
    else:
        branches[-1] = TimedScheduledOutcome(
            "3",
            3,
            9999 if change == "wrong_seed" else 8731,
            False,
            None,
            change != "unadmitted",
            {15: "different" if change == "wrong_prefix" else "same"},
            1192,
        )
    result = timed_schedule_teacher(
        bank, branches, 15, "same", 8731, np.array([True, True, False, False])
    )
    assert result["expected_continuation_counts"][0] == 3
    assert result["admitted"][0] is False
    assert result["teacher_action"] is None
