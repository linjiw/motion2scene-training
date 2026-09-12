"""Prepare-only synthetic artifact tests; no simulator or physical qualification."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from decoupled_wbc.tests.test_motion2scene_timed_history_policy import bank_fixture
from decoupled_wbc.tests.test_motion2scene_timed_options import artifact

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/research"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "timed_collector", SCRIPTS / "motion2scene_collect_timed_history.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def inputs(tmp_path):
    bank = bank_fixture(tmp_path)
    # Explicit synthetic contact reports, never physical qualification.
    refs = []
    for option_id, ref in bank.definition["evidence"].items():
        evidence_path = Path(ref["path"])
        evidence = json.loads(evidence_path.read_text())
        audit_path = tmp_path / (option_id + "_environment_fixture.json")
        audit_path.write_text(
            json.dumps(
                {
                    "schema": "motion2scene_environment_contact_audit_v1",
                    "complete_synchronized_streams": True,
                    "no_undesired_measured_contact": True,
                    "normal_force_threshold_n": 1.0,
                    "self_contacts": [],
                }
            )
        )
        evidence["artifacts"]["environment_contact_audit"] = artifact(audit_path)
        evidence_path.write_text(json.dumps(evidence))
        refs.append(artifact(evidence_path))
    from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (
        make_verified_registry,
    )

    (tmp_path / "registry.json").write_text(json.dumps(make_verified_registry(bank.request, refs)))
    request = tmp_path / "request.json"
    request.write_text(json.dumps(bank.request))
    scene = tmp_path / "synthetic_scene.usda"
    scene.write_text("Synthetic preparation fixture, never executed")
    definition = tmp_path / "scene.json"
    definition.write_text(
        json.dumps(
            {
                "schema": collector.SCENE_SCHEMA,
                "split": "development",
                "scene_id": "synthetic_scene",
                "scene": {**artifact(scene), "scene_id": "synthetic_scene"},
                "beam_collision_enabled": True,
                "beam": {
                    "center_xy_m": [2.0, 0.0],
                    "yaw_rad": 0.0,
                    "length_m": 0.5,
                    "width_m": 1.0,
                    "thickness_m": 0.1,
                    "underside_m": 1.25,
                },
            }
        )
    )
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "implementation": {
                    "checkpoint": bank.request["controller"],
                    "python": "/not/executed/python",
                },
                "cells": [
                    {
                        "cell_id": "neutral",
                        "motion": bank.request["references"][0]["motion"],
                        "hydra_overrides": [
                            "++seed=1",
                            "++manager_env.config.expected_reference_frames=199",
                        ],
                    }
                ],
            }
        )
    )
    return SimpleNamespace(
        registry=tmp_path / "registry.json",
        request=request,
        scene_definition=definition,
        template=template,
        cell="neutral",
        out=tmp_path / "prepared",
        policy_mode="forced",
        policy=None,
        seed=8731,
    )


def test_preparation_binds_three_full_schedules_and_no_wrap_budget(tmp_path):
    args = inputs(tmp_path)
    collector.prepare(args)
    manifest, bank, scene = collector.verify_manifest(args.out)
    assert manifest["expected_recorded_control_steps"] == 298
    assert manifest["expected_physics_steps"] == 1192
    assert [cell["forced_option_id"] for cell in manifest["cells"]] == list(bank.option_ids)
    assert scene["split"] == "development"
    assert len(manifest["feature_names"]) == 106
    for cell in manifest["cells"]:
        command = cell["command"]
        assert command[command.index("--max-steps") + 1] == "299"
        assert "++seed=1" not in cell["hydra_overrides"]
        assert all(
            "expected_reference_frames=199" not in value for value in cell["hydra_overrides"]
        )
        assert not Path(cell["output"]).exists()
    args.request.write_text("changed request")
    with pytest.raises(ValueError, match="hash mismatch"):
        collector.verify_manifest(args.out)


def test_prepare_rejects_unverified_or_wrong_source_before_creating_output(tmp_path):
    args = inputs(tmp_path)
    original_registry = args.registry
    args.registry = args.request
    with pytest.raises(ValueError, match="registr"):
        collector.prepare(args)
    assert not args.out.exists()
    args.registry = original_registry
    template = json.loads(args.template.read_text())
    template["cells"][0]["motion"] = artifact(args.scene_definition)
    args.template.write_text(json.dumps(template))
    with pytest.raises(ValueError, match="source/controller"):
        collector.prepare(args)
    assert not args.out.exists()


def test_collision_api_presence_does_not_enable_disabled_placeholder():
    disabled = {"collision": True, "attributes": {"physics:collisionEnabled": "False"}}
    assert collector.imported_collision_enabled(disabled) is False
    disabled["attributes"]["physics:collisionEnabled"] = "True"
    assert collector.imported_collision_enabled(disabled) is True
    disabled["attributes"].clear()
    with pytest.raises(ValueError, match="recorded explicitly"):
        collector.imported_collision_enabled(disabled)
