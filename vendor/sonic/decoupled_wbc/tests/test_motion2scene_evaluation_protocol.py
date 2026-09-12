"""Adversarial artifact/protocol changes must not authorize reserved execution."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_evaluation_protocol import (
    ADOPTED,
    ADOPTION_GATES,
    SCHEMA,
    _canonical_beam,
    protocol_spec_digest,
    validate_reserved_execution,
)


@pytest.fixture
def protocol_case(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value, sort_keys=True))
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    runtime = write("runtime.py", "runtime_v1")
    scoring = write("scoring.py", "scoring_v1")
    registry = write("registry.json", {"registry": 7})
    request = write("request.json", {"complete_schedules": 7})
    controller = write("controller.pt", "fixed_controller")
    model = write("model.npz", "frozen_model")
    usd = write("scene.usda", "audited_native_geometry")
    lock_path = (
        Path(__file__).resolve().parents[2] / "docs/motion2scene/TRAVERSAL_EVALUATION_LOCK_V3.json"
    )
    lock = {"path": str(lock_path), "sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest()}
    layouts = json.loads(lock_path.read_text())["evaluation"]["layouts"]
    scenes = []
    for layout in layouts:
        for variant in layout["fixed_world_variants"]:
            beams = variant["beams"]
            scenes.append(
                {
                    "layout_id": layout["layout_id"],
                    "variant_id": variant["offset_id"],
                    "scene_id": layout["layout_id"] + "__" + variant["offset_id"],
                    "beams": [_canonical_beam(b) for b in beams],
                    "scene": usd,
                    "beam_collision_enabled": [True] * len(beams),
                }
            )
    policy = {
        "policy_id": "common_learner",
        "mode": "learned",
        "preferred_option_id": "neutral",
        "preferred_reference_id": "sustained",
        "model": model,
    }
    protocol = {
        "schema": SCHEMA,
        "status": ADOPTED,
        "adoption_requirements": {key: True for key in ADOPTION_GATES},
        "adoption_receipt": None,
        "geometry_lock": lock,
        "scenes": scenes,
        "enabled_variant_ids": [v["offset_id"] for v in layouts[0]["fixed_world_variants"]],
        "physics_seeds": [94301, 94302],
        "nominal_episodes_per_policy": 36,
        "stress_episodes_per_policy": 288,
        "implementation": {
            "registry": registry,
            "request": request,
            "controller": controller,
            "runtime_artifacts": [runtime],
            "scoring_artifacts": [scoring],
        },
        "sensor": {"feature_names": [f"feature_{i}" for i in range(114)]},
        "policies": [policy],
    }

    def register(value):
        value["adoption_receipt"] = write(
            "adoption.json",
            {
                "status": "ADOPTED",
                "evaluation_outcomes_inspected": False,
                "protocol_spec_sha256": protocol_spec_digest(value),
                "gate_evidence": {
                    key: [write(f"evidence_{key}.json", {"gate": key, "passed": True})]
                    for key in ADOPTION_GATES
                },
            },
        )
        return write("protocol.json", value)

    definition = {
        **copy.deepcopy(scenes[-1]),
        "schema": "motion2scene_timed_schedule_scene_v1",
        "split": "reserved_evaluation_v3",
        "evaluation_protocol": register(protocol),
    }
    context = {
        **copy.deepcopy(protocol["implementation"]),
        **copy.deepcopy(policy),
        "physics_seed": 94301,
        "reference_frames": 299,
        "recorded_control_steps": 298,
        "feature_names": protocol["sensor"]["feature_names"].copy(),
    }
    return protocol, definition, context, register, write


def test_exact_adopted_scene_and_actual_context_pass(protocol_case):
    _, scene, context, _, _ = protocol_case
    result = validate_reserved_execution(scene, context)
    assert result["scene"]["layout_id"] == scene["layout_id"]
    assert len(result["scene"]["beams"]) == 2


def test_proposed_protocol_is_rejected_even_with_valid_hash(protocol_case):
    protocol, scene, context, register, _ = protocol_case
    protocol["status"] = "PROPOSED_NOT_ADOPTED_NO_EXECUTION_AUTHORITY"
    scene["evaluation_protocol"] = register(protocol)
    with pytest.raises(ValueError, match="proposed"):
        validate_reserved_execution(scene, context)


@pytest.mark.parametrize("key", ["registry", "request", "controller", "model"])
def test_another_validly_hashed_execution_artifact_is_rejected(protocol_case, key):
    _, scene, context, _, write = protocol_case
    context[key] = write("other_" + key, "different_but_hash_valid")
    with pytest.raises(ValueError, match="identities differ"):
        validate_reserved_execution(scene, context)


@pytest.mark.parametrize("key", ["runtime_artifacts", "scoring_artifacts"])
def test_actual_runtime_or_scorer_substitution_is_rejected(protocol_case, key):
    _, scene, context, _, write = protocol_case
    context[key] = [write("other.py", "different_implementation")]
    with pytest.raises(ValueError, match=key):
        validate_reserved_execution(scene, context)


@pytest.mark.parametrize("change", ["missing_beam", "disabled_beam", "other_variant", "other_usd"])
def test_scene_substitution_preserves_no_loophole(protocol_case, change):
    _, scene, context, _, write = protocol_case
    if change == "missing_beam":
        scene["beams"].pop()
    elif change == "disabled_beam":
        scene["beam_collision_enabled"][1] = False
    elif change == "other_variant":
        scene["variant_id"] = "variant_07"
    else:
        scene["scene"] = write("other.usda", "other_geometry")
    with pytest.raises(ValueError):
        validate_reserved_execution(scene, context)


def test_geometry_cannot_be_narrowed_even_by_new_adoption_receipt(protocol_case):
    protocol, scene, context, register, _ = protocol_case
    protocol["scenes"].pop(0)
    scene["evaluation_protocol"] = register(protocol)
    with pytest.raises(ValueError, match="all18"):
        validate_reserved_execution(scene, context)


def test_edited_spec_requires_new_exact_adoption_receipt(protocol_case):
    protocol, scene, context, _, write = protocol_case
    protocol["scoring"] = {"changed_terminal_policy": True}
    scene["evaluation_protocol"] = write("protocol.json", protocol)
    with pytest.raises(ValueError, match="adoption receipt"):
        validate_reserved_execution(scene, context)


def test_missing_gate_and_unfrozen_other_model_block_baseline_launch(protocol_case):
    protocol, scene, context, register, _ = protocol_case
    protocol["adoption_requirements"].pop("exact114_runtime_smoke_complete")
    scene["evaluation_protocol"] = register(protocol)
    with pytest.raises(ValueError, match="every implementation gate"):
        validate_reserved_execution(scene, context)
    protocol["adoption_requirements"] = {key: True for key in ADOPTION_GATES}
    protocol["policies"].append({"policy_id": "future_unfrozen", "mode": "learned", "model": None})
    scene["evaluation_protocol"] = register(protocol)
    with pytest.raises(ValueError, match="explicit artifact"):
        validate_reserved_execution(scene, context)


@pytest.mark.parametrize(
    "key,value",
    [
        ("physics_seed", 8731),
        ("physics_seed", True),
        ("recorded_control_steps", 199),
        ("reference_frames", 300),
        ("preferred_option_id", "prior_splice_e050_r265"),
    ],
)
def test_command_context_cannot_change_seed_horizon_or_policy(protocol_case, key, value):
    _, scene, context, _, _ = protocol_case
    context[key] = value
    with pytest.raises(ValueError):
        validate_reserved_execution(scene, context)


def test_multi_option_script_parameters_are_bound_to_actual_command(protocol_case):
    protocol, scene, context, register, write = protocol_case
    parameters = write("script_parameters.json", {"threshold": 1.25})
    protocol["policies"][0].update(mode="scripted_multi", model=None, script_parameters=parameters)
    context.update(mode="scripted_multi", model=None, script_parameters=parameters)
    scene["evaluation_protocol"] = register(protocol)
    validate_reserved_execution(scene, context)
    context["script_parameters"] = write("other_parameters.json", {"threshold": 1.20})
    with pytest.raises(ValueError, match="identities differ"):
        validate_reserved_execution(scene, context)


def test_retained_stress_geometry_does_not_enable_optional_followup(protocol_case):
    protocol, scene, context, register, _ = protocol_case
    protocol["enabled_variant_ids"] = ["nominal"]
    scene["evaluation_protocol"] = register(protocol)
    with pytest.raises(ValueError, match="not enabled"):
        validate_reserved_execution(scene, context)


def test_pending_development_constant_selection_blocks_adoption(protocol_case):
    protocol, scene, context, register, _ = protocol_case
    protocol["policies"].append(
        {
            "policy_id": "constant_development_selected",
            "mode": "constant_option",
            "preferred_option_id": None,
            "selection_artifact": None,
        }
    )
    scene["evaluation_protocol"] = register(protocol)
    with pytest.raises(ValueError, match="explicit artifact"):
        validate_reserved_execution(scene, context)


def test_internally_consistent_replacement_lock_is_rejected(protocol_case):
    protocol, scene, context, register, write = protocol_case
    lock = json.loads(Path(protocol["geometry_lock"]["path"]).read_text())
    lock["evaluation"]["layouts"][-1]["fixed_world_variants"][-1]["beams"][0]["center_xyz_m"][
        0
    ] += 0.1
    protocol["geometry_lock"] = write("replacement_lock.json", lock)
    beams = lock["evaluation"]["layouts"][-1]["fixed_world_variants"][-1]["beams"]
    protocol["scenes"][-1]["beams"] = [_canonical_beam(b) for b in beams]
    scene["beams"] = protocol["scenes"][-1]["beams"]
    scene["evaluation_protocol"] = register(protocol)
    with pytest.raises(ValueError, match="canonical original V3"):
        validate_reserved_execution(scene, context)


@pytest.mark.parametrize("fault", ["missing", "empty", "changed_content"])
def test_adoption_booleans_require_hash_bound_evidence(protocol_case, fault):
    protocol, scene, context, _, write = protocol_case
    receipt = json.loads(Path(protocol["adoption_receipt"]["path"]).read_text())
    key = ADOPTION_GATES[0]
    if fault == "missing":
        receipt["gate_evidence"].pop(key)
    elif fault == "empty":
        receipt["gate_evidence"][key] = []
    else:
        Path(receipt["gate_evidence"][key][0]["path"]).write_text("changed after receipt")
    protocol["adoption_receipt"] = write("adoption.json", receipt)
    scene["evaluation_protocol"] = write("protocol.json", protocol)
    with pytest.raises(ValueError):
        validate_reserved_execution(scene, context)
