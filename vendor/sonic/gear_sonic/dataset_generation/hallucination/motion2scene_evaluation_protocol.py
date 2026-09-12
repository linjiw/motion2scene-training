"""Explicit adoption and artifact binding for the reserved V3 evaluation panel.

This is a read-only preflight. It does not adopt a proposal, authorize physics,
or infer an episode outcome from missing measurements.
"""

import hashlib
import json
from pathlib import Path

SCHEMA = "motion2scene_reserved_evaluation_protocol_v3"
ADOPTED = "ADOPTED_READY_FOR_REGISTERED_EXECUTION"
TRUSTED_V3_LOCK_SHA256 = "509817600075888ef5cf681c1033c9dcd6796d7a08561e30ee4157049aff5181"
ADOPTION_GATES = (
    "seven_schedules_physically_qualified",
    "exact114_runtime_smoke_complete",
    "two_beam_full_horizon_runtime_qualified",
    "terminal_and_failure_scorer_tested",
    "protocol_validator_integrated",
    "strong_multi_option_script_development_validated",
    "all_comparison_models_and_thresholds_frozen",
    "complete_source_closure_and_environment_frozen",
    "no_reserved_outcomes_inspected",
)


def protocol_spec_digest(protocol):
    """Bind an adoption receipt without a receipt/protocol hash cycle."""
    body = {key: value for key, value in protocol.items() if key != "adoption_receipt"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def checked_identity(ref):
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise ValueError("explicit artifact path and SHA256 required")
    if not isinstance(ref.get("sha256"), str):
        raise ValueError("explicit artifact SHA256 string required")
    digest = ref["sha256"].removeprefix("sha256:")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("canonical SHA256 digest required")
    path = Path(ref["path"]).resolve(strict=True)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("evaluation artifact content differs from its pinned SHA256")
    return {"path": str(path), "sha256": digest}


def _json(ref):
    return json.loads(Path(checked_identity(ref)["path"]).read_text())


def _same_artifact(actual, expected):
    if checked_identity(actual) != checked_identity(expected):
        raise ValueError("actual and registered evaluation artifact identities differ")


def _identity_set(refs):
    if not isinstance(refs, list) or not refs:
        raise ValueError("complete nonempty implementation artifact list required")
    identities = [checked_identity(ref) for ref in refs]
    paths = [ref["path"] for ref in identities]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate implementation artifact identity")
    return {(ref["path"], ref["sha256"]) for ref in identities}


def _canonical_beam(beam):
    return {
        "center_xy_m": beam["center_xyz_m"][:2],
        "length_m": beam["full_dimensions_xyz_m"][0],
        "width_m": beam["full_dimensions_xyz_m"][1],
        "thickness_m": beam["full_dimensions_xyz_m"][2],
        "underside_m": beam["center_xyz_m"][2] - beam["full_dimensions_xyz_m"][2] / 2,
        "yaw_rad": beam["yaw_rad"],
    }


def _validate_panel(protocol):
    if checked_identity(protocol["geometry_lock"])["sha256"] != TRUSTED_V3_LOCK_SHA256:
        raise ValueError("canonical original V3 geometry-lock identity required")
    lock = _json(protocol["geometry_lock"])
    if lock.get("schema") != "motion2scene_six_second_world_geometry_lock_v3":
        raise ValueError("the exact V3 world-geometry lock is required")
    layouts = lock["evaluation"]["layouts"]
    expected = {}
    for layout in layouts:
        for variant in layout["fixed_world_variants"]:
            key = (layout["layout_id"], variant["offset_id"])
            if key in expected:
                raise ValueError("duplicate locked geometry identity")
            expected[key] = [_canonical_beam(beam) for beam in variant["beams"]]
    scenes = protocol["scenes"]
    keys = [(row["layout_id"], row["variant_id"]) for row in scenes]
    if (
        len(layouts) != 18
        or len(expected) != 162
        or len(keys) != len(set(keys))
        or set(keys) != set(expected)
        or protocol["physics_seeds"] != lock["evaluation"]["physics_seeds"]
        or protocol["physics_seeds"] != [94301, 94302]
    ):
        raise ValueError("all18 layouts, nine variants and both original physics seeds required")
    for row in scenes:
        key = (row["layout_id"], row["variant_id"])
        if (
            row["beams"] != expected[key]
            or row["scene_id"] != "__".join(key)
            or row["beam_collision_enabled"] != [True] * len(expected[key])
        ):
            raise ValueError("reserved geometry or collision settings differ from the V3 lock")
    if (
        protocol["nominal_episodes_per_policy"] != 36
        or protocol["stress_episodes_per_policy"] != 288
    ):
        raise ValueError("reserved denominators cannot be narrowed")


def validate_reserved_execution(scene_definition, context):
    """Bind one proposed execution to an adopted complete evaluation protocol.

    ``context`` comes from the actual prepared command/manifest, not the scene
    file. It supplies seed, policy ID/mode/preference/model, registry, request,
    controller, runtime and scoring artifact lists, and exact feature names.
    Call before registration/execution and independently before analysis.
    """
    protocol = _json(scene_definition.get("evaluation_protocol"))
    if protocol.get("schema") != SCHEMA or protocol.get("status") != ADOPTED:
        raise ValueError("proposed, pending or unrecognized protocols cannot enable evaluation")
    gates = protocol.get("adoption_requirements")
    if (
        not isinstance(gates, dict)
        or set(gates) != set(ADOPTION_GATES)
        or any(value is not True for value in gates.values())
        or not isinstance(protocol.get("adoption_receipt"), dict)
    ):
        raise ValueError("every implementation gate and an explicit adoption receipt are required")
    receipt = _json(protocol["adoption_receipt"])
    if (
        receipt.get("status") != "ADOPTED"
        or receipt.get("evaluation_outcomes_inspected") is not False
        or receipt.get("protocol_spec_sha256") != protocol_spec_digest(protocol)
    ):
        raise ValueError("pre-evaluation adoption receipt required")
    evidence = receipt.get("gate_evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(ADOPTION_GATES):
        raise ValueError("hash-bound evidence for every adoption gate required")
    for refs in evidence.values():
        _identity_set(refs)
    _validate_panel(protocol)
    if (
        scene_definition.get("schema") != "motion2scene_timed_schedule_scene_v1"
        or scene_definition.get("split") != "reserved_evaluation_v3"
        or type(context.get("physics_seed")) is not int
        or context["physics_seed"] not in protocol["physics_seeds"]
    ):
        raise ValueError("exact reserved scene schema, split and physics seed required")
    selected = [
        row for row in protocol["scenes"] if row["scene_id"] == scene_definition.get("scene_id")
    ]
    if len(selected) != 1:
        raise ValueError("scene identity is absent from the complete reserved panel")
    selected = selected[0]
    enabled = protocol.get("enabled_variant_ids")
    if not isinstance(enabled, list) or selected["variant_id"] not in enabled:
        raise ValueError("this retained geometry variant is not enabled in the adopted experiment")
    for key in ("layout_id", "variant_id", "beams", "beam_collision_enabled"):
        if scene_definition.get(key) != selected[key]:
            raise ValueError(f"actual reserved scene differs in {key}")
    _same_artifact(scene_definition["scene"], selected["scene"])
    implementation = protocol["implementation"]
    for key in ("registry", "request", "controller"):
        _same_artifact(context.get(key), implementation[key])
    for key in ("runtime_artifacts", "scoring_artifacts"):
        if _identity_set(context.get(key)) != _identity_set(implementation[key]):
            raise ValueError(f"actual complete {key} differ from the evaluation lock")
    if context.get("feature_names") != protocol["sensor"]["feature_names"]:
        raise ValueError("actual feature order differs from the exact114D schema")
    if len(context["feature_names"]) != 114 or len(set(context["feature_names"])) != 114:
        raise ValueError("the common114D schema is required")
    if context.get("reference_frames") != 299 or context.get("recorded_control_steps") != 298:
        raise ValueError("qualified finite reference and recorded horizon required")
    policies = protocol["policies"]
    if len({row["policy_id"] for row in policies}) != len(policies):
        raise ValueError("duplicate registered policy identity")
    for entry in policies:
        if entry["mode"] == "learned":
            checked_identity(entry.get("model"))
        elif entry["mode"] == "scripted_multi":
            checked_identity(entry.get("script_parameters"))
        elif entry["mode"] == "constant_option":
            checked_identity(entry.get("selection_artifact"))
            if entry.get("preferred_option_id") not in protocol["schedules"]["option_ids"][1:]:
                raise ValueError("constant selection must name one qualified adapting schedule")
    selected_policy = [row for row in policies if row["policy_id"] == context.get("policy_id")]
    if len(selected_policy) != 1:
        raise ValueError("actual policy is not in the adopted comparison")
    policy = selected_policy[0]
    for key in ("mode", "preferred_option_id", "preferred_reference_id"):
        if context.get(key) != policy[key]:
            raise ValueError(f"actual policy configuration differs in {key}")
    if policy["mode"] == "learned":
        _same_artifact(context.get("model"), policy.get("model"))
    elif context.get("model") is not None or policy.get("model") is not None:
        raise ValueError("a fixed baseline cannot silently consume a learned model")
    if policy["mode"] == "scripted_multi":
        _same_artifact(context.get("script_parameters"), policy.get("script_parameters"))
    return {"protocol": protocol, "scene": selected, "policy": policy}
