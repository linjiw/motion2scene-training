#!/usr/bin/env python3
"""Register one or two declared beams with repeated qualified sensor commitments."""

import argparse
import copy
import json
from pathlib import Path
import pickle
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from hallucination.run_approved_manifest import free_gpu_mib, run_with_process_group  # noqa: E402
from motion2scene_scene_resolution import validate_scene_handoff  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_course import (  # noqa: E402
    audit_imported_beams,
    beam_prim_name,
    score_course,
    validate_beams,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_environment_contacts import (  # noqa: E402
    audit_environment_contacts,
    validated_beam_paths,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_evaluation_protocol import (  # noqa: E402
    validate_reserved_execution,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_observation_timing import (  # noqa: E402
    audit_decision_visibility,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import (  # noqa: E402
    score_passage,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_sensor_alignment import (  # noqa: E402
    audit_sensor_alignment,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_course_measurements import (  # noqa: E402
    synchronize_pair_beam_forces,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_measurements import (  # noqa: E402
    audit_whole_body_contacts,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    checked_artifact,
    definition_digest,
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_outcome import (  # noqa: E402
    classify_timed_attempt,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    SCHEMA,
    schedule_layout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    HISTORY_FRAMES,
    HISTORY_SECONDS,
    MODES,
    audit_executed_schedule,
    expected_feature_names,
    load_schedule_policy,
    load_script_parameters,
)

RUNTIME = "gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_execution"
SCENE_SCHEMA = "motion2scene_timed_schedule_scene_v1"
COLLECTION_SCHEMA = "motion2scene_timed_schedule_collection_v1"


def validate_scene(definition):
    """Validate explicit geometry without consulting any outcome or hidden layout."""
    if (
        definition.get("schema") != SCENE_SCHEMA
        or definition.get("split") not in ("development", "reserved_evaluation_v3")
        or not isinstance(definition.get("scene_id"), str)
        or re.fullmatch(r"[A-Za-z0-9_-]+", definition["scene_id"]) is None
    ):
        raise ValueError("explicit scene schema, split and scene identifier required")
    checked_artifact({key: definition["scene"][key] for key in ("path", "sha256")})
    beams, enabled = definition.get("beams"), definition.get("beam_collision_enabled")
    validate_beams(beams)
    if (
        len(beams) not in (1, 2)
        or not isinstance(enabled, list)
        or len(enabled) != len(beams)
        or any(type(v) is not bool for v in enabled)
    ):
        raise ValueError("one or two beams with explicit per-beam collision booleans required")
    if len(beams) == 2 and not all(enabled):
        raise ValueError("a two-beam course requires both declared constraints enabled")
    if definition["split"] != "development":
        checked_artifact(definition.get("evaluation_protocol"))
    return definition


def scene_beam_paths(scene):
    return validated_beam_paths(
        [f"/World/ground/terrain/{beam_prim_name(i)}" for i in range(len(scene["beams"]))]
    )


def audit_scene_geometry(shapes, scene):
    paths = scene_beam_paths(scene)
    imported = {value["path"]: value for value in shapes}
    # Every native beam must be declared and separately instrumented.
    actual = {path for path in imported if path.startswith(paths[0])}
    if actual != set(paths):
        raise ValueError("native beam set differs from the complete declared scene")
    for path, enabled in zip(paths, scene["beam_collision_enabled"], strict=True):
        if imported_collision_enabled(imported[path]) != enabled:
            raise ValueError("actual beam collision setting differs from registration")
    return audit_imported_beams(shapes, scene["beams"])


def score_scene(payload, forces, scene, bank, *, commands_valid):
    if len(scene["beams"]) == 1:
        return score_passage(payload, forces[:, 0], scene["beams"][0])
    result = score_course(
        payload,
        forces,
        scene["beams"],
        commands_valid=commands_valid,
        timeout_s=(bank.frame_count - 2) / 50,
        reference_frames=bank.frame_count,
        final_reference_phase_s=float(np.asarray(payload["motion_time_s"])[-1]),
    )
    # The recorder stops before the loader's wrap callback. This safe captured
    # horizon is one tick earlier than the full native bank's last phase.
    if result["passage_finish_frame_exclusive"] is None:
        result["failure_reasons"].append("safe_captured_reference_horizon_before_course_completion")
    return result


def validate_collection_context(cell, manifest, bank, scene, *, actual=False):
    """Derive effective execution identity before any reserved protocol check."""
    command = cell["command"]
    if command.count("--extra") != 1 or command[command.index("--extra") + 1] != " ".join(
        cell["hydra_overrides"]
    ):
        raise ValueError("actual command and registered overrides differ")
    effective = {}
    for item in cell["hydra_overrides"]:
        key, value = item.lstrip("+").split("=", 1)
        if key in effective:
            raise ValueError("ambiguous duplicate effective override")
        effective[key] = value
    cfg = "manager_env.config."
    expected = {
        "seed": str(cell["runtime_seed"]),
        "manager_env._target_": f"{RUNTIME}.TimedScheduleEnvCfg",
        "manager_env.recorders.trajectory._target_": f"{RUNTIME}.TimedScheduleRecorderCfg",
        cfg + "timed_schedule_mode": cell["timed_schedule_mode"],
        cfg + "forced_option_id": cell["forced_option_id"],
        cfg + "preferred_option_id": cell["preferred_option_id"],
        cfg + "preferred_reference_id": cell["preferred_reference_id"],
        cfg + "timed_registry_path": manifest["registry"]["path"],
        cfg + "timed_registry_sha256": manifest["registry"]["sha256"],
        cfg + "qualification_request_path": manifest["request"]["path"],
        cfg + "qualification_request_sha256": manifest["request"]["sha256"],
        cfg + "expected_reference_frames": str(bank.frame_count),
        cfg + "observation_delay_s": "0.0",
        cfg + "history_max_age_s": str(HISTORY_SECONDS),
        cfg + "history_max_frames": str(HISTORY_FRAMES),
    }
    for prefix, ref in (
        ("timed_policy", manifest.get("policy")),
        ("script_parameters", manifest.get("script_parameters")),
    ):
        if prefix == "script_parameters" and "script_parameters" not in manifest:
            continue  # Original smoke had the same absent-script default.
        for field in ("path", "sha256"):
            expected[cfg + prefix + "_" + field] = ref[field] if ref else ""
    if any(effective.get(key) != value for key, value in expected.items()):
        raise ValueError("effective policy/seed/interface differs from registration")
    if json.loads(effective[cfg + "environment_beam_paths"]) != scene_beam_paths(scene):
        raise ValueError("actual beam instrumentation differs from scene")
    for flag, value in (
        ("--max-steps", str(bank.frame_count)),
        ("--scene", scene["scene_id"]),
        ("--scene-package", str(Path(scene["scene"]["path"]).parent)),
        ("--motion", bank.request["references"][0]["motion"]["path"]),
        ("--checkpoint", bank.request["controller"]["path"]),
        ("--out", cell["output"]),
    ):
        if command.count(flag) != 1 or command[command.index(flag) + 1] != value:
            raise ValueError(f"registered command has different {flag}")
    checked_artifact({key: scene["scene"][key] for key in ("path", "sha256")})
    validate_scene_handoff(command, scene)
    if actual:
        folder = Path(cell["output"])
        attempt = json.loads((folder / "attempt.json").read_text())
        if attempt["command"] != command:
            raise ValueError("actual attempted invocation differs from registration")
        success = folder / "success_manifest.json"
        if success.exists():
            runtime = json.loads(success.read_text())
            captured = runtime["capture_context"]["scene"]
            if (
                runtime["capture_context"]["scene_id"] != scene["scene_id"]
                or captured["hash"] != scene["scene"]["sha256"]
                or Path(captured["resolved"]).resolve() != Path(scene["scene"]["path"]).resolve()
            ):
                raise ValueError("actual native scene differs from registered geometry")
    if scene["split"] == "development":
        return {"scope": "explicit development scene; no reserved evaluation execution"}
    context = dict(
        physics_seed=int(effective["seed"]),
        policy_id=cell.get("policy_id"),
        mode=effective[cfg + "timed_schedule_mode"],
        preferred_option_id=effective[cfg + "preferred_option_id"],
        preferred_reference_id=effective[cfg + "preferred_reference_id"],
        model=manifest.get("policy"),
        script_parameters=manifest.get("script_parameters"),
        registry=manifest["registry"],
        request=manifest["request"],
        controller=bank.request["controller"],
        runtime_artifacts=manifest.get("runtime_artifacts"),
        scoring_artifacts=manifest.get("scoring_artifacts"),
        feature_names=manifest["feature_names"],
        reference_frames=int(effective[cfg + "expected_reference_frames"]),
        recorded_control_steps=manifest["expected_recorded_control_steps"],
    )
    return validate_reserved_execution(scene, context)


def phase_visibility(observations, alignment, scene, phases):
    """Retain explicit unavailable phases instead of inventing later observations."""
    eligible = alignment["packet_eligible"]
    prefix = []
    for row, admitted in zip(observations, eligible, strict=True):
        if not admitted:
            break
        prefix.append(row)
    result = []
    for tick in phases:
        try:
            beams = [
                audit_decision_visibility(prefix, beam, int(tick), collision_enabled=enabled)
                for beam, enabled in zip(
                    scene["beams"], scene["beam_collision_enabled"], strict=True
                )
            ]
            result.append(dict(entry_tick=int(tick), beams=beams))
        except (ValueError, KeyError, TypeError, IndexError) as error:
            result.append(
                dict(entry_tick=int(tick), available=False, beams=None, reason=str(error))
            )
    return result


def analyze_incomplete(cell, manifest, bank, scene, attempt, error):
    """Keep an assigned attempt and independently verifiable terminal evidence."""
    folder = Path(cell["output"]) / "trajectories"
    raw_artifacts, diagnostics = {}, [error]
    payload, interface, pairs, contacts, mapping = None, {}, None, None, None
    paths = list(folder.glob("*.trajectory.pkl"))
    if len(paths) == 1:
        raw_artifacts["trajectory"] = artifact(paths[0])
        try:
            # Trusted local recorder file, immediately SHA-bound in this receipt.
            with paths[0].open("rb") as handle:
                payload = pickle.load(handle)  # noqa: S301
        except (OSError, ValueError, pickle.UnpicklingError) as failure:
            diagnostics.append(str(failure))
    else:
        raw = folder / "aborted_raw_recording.pkl"
        if raw.exists():
            raw_artifacts["aborted_raw_recording"] = artifact(raw)
            try:
                with raw.open("rb") as handle:
                    frames = pickle.load(handle)  # noqa: S301 - trusted bound recorder data.
                if not isinstance(frames, dict) or set(frames) != {0}:
                    raise ValueError("abort must identify the single recorded environment")
                payload = dict(frames[0], fps=50)
            except (OSError, ValueError, TypeError, pickle.UnpicklingError) as failure:
                diagnostics.append(str(failure))
    for name in ("reactive_interface.json", "aborted_interface.json"):
        path = folder / name
        if path.exists():
            raw_artifacts["sensor"] = artifact(path)
            try:
                interface = json.loads(path.read_text())
            except (ValueError, OSError) as failure:
                diagnostics.append(str(failure))
            break
    for key, filename in (
        ("environment_pairs", "environment_pair_contacts.npz"),
        ("all_body_contacts", "all_body_contacts.npz"),
    ):
        for prefix in ("", "aborted_"):
            path = folder / (prefix + filename)
            if path.exists():
                raw_artifacts[key] = artifact(path)
                try:
                    with np.load(path, allow_pickle=False) as data:
                        values = {key: data[key].copy() for key in data.files}
                    if key == "environment_pairs":
                        pairs = values
                        map_path = folder / (prefix + "environment_contact_mapping.json")
                        raw_artifacts["environment_mapping"] = artifact(map_path)
                        mapping = json.loads(map_path.read_text())
                    else:
                        contacts = values
                except (ValueError, OSError) as failure:
                    diagnostics.append(str(failure))
                break
    contact_audit = None
    if pairs is not None and contacts is not None and mapping is not None:
        contact_audit = audit_environment_contacts(
            pairs,
            contacts,
            mapping,
            len(pairs.get("physics_steps", [])),
            neutral_self_pairs=manifest.get("declared_neutral_self_pairs", []),
            beam_paths=scene_beam_paths(scene),
        )
    steps = pairs.get("physics_steps") if pairs is not None else None
    if steps is None and contacts is not None:
        steps = contacts.get("physics_steps")
    if steps is None:
        path = folder / "aborted_physics.npz"
        if path.exists():
            raw_artifacts["aborted_physics"] = artifact(path)
            try:
                with np.load(path, allow_pickle=False) as data:
                    steps = data["physics_steps"].copy()
            except (ValueError, OSError, KeyError) as failure:
                diagnostics.append(str(failure))
    observations = interface.get("observations", [])
    outcome = classify_timed_attempt(
        payload,
        observations,
        reference_frames=bank.frame_count,
        phase_ticks=manifest["phase_ticks"],
        exit_status=attempt["exit_status"],
        contact_audit=contact_audit,
        physics_steps=steps,
    )
    return dict(
        cell_id=cell["cell_id"],
        mode=cell["timed_schedule_mode"],
        forced_option_id=cell["forced_option_id"],
        status=(
            "failed_attempt" if attempt["exit_status"] != 0 else "invalid_or_partial_measurement"
        ),
        measurement_admitted=False,
        task_outcome_admitted=outcome["task_outcome"] != "unknown",
        **{"pass": False},
        physics_steps=outcome["physics_steps_recorded"],
        sensor_queries=None,
        sensor_queries_in_recorded_packets_only=sum(
            len(row.get("measurements", [])) for row in observations
        ),
        outcome=outcome,
        contact_audit=contact_audit,
        analysis_errors=diagnostics,
        raw_artifacts=raw_artifacts,
        partial=raw_artifacts.get("sensor"),
        costs=dict(
            passage_time_s=None,
            whole_episode_time_s=None,
            switch_count=len(interface.get("switches", [])),
            positive_mechanical_work_j=None,
        ),
    )


def prepare(args):
    bank = load_verified_registry(args.registry, artifact(args.registry)["sha256"])
    phases, _, _ = schedule_layout(bank)
    if args.policy_mode not in MODES:
        raise ValueError("unsupported registered policy mode")
    preferred_option_id = getattr(args, "preferred_option_id", "neutral")
    preferred_reference_id = getattr(args, "preferred_reference_id", "sustained")
    if args.policy_mode == "constant_option" and preferred_option_id not in bank.option_ids:
        raise ValueError("constant mode requires a complete qualified option ID")
    if args.policy_mode == "scripted_reference" and preferred_reference_id not in [
        ref["reference_id"] for ref in bank.request["references"][1:]
    ]:
        raise ValueError("scripted mode requires an adapting qualified reference ID")
    contact_reports = {}
    for option_id, evidence_ref in bank.definition["evidence"].items():
        evidence = json.loads(checked_artifact(evidence_ref).read_text())
        audit_ref = evidence["artifacts"].get("environment_contact_audit")
        if audit_ref is None:
            raise ValueError(
                "timed sensor collection requires separately audited environmental counterparts"
            )
        audit = json.loads(checked_artifact(audit_ref).read_text())
        if (
            audit.get("schema") != "motion2scene_environment_contact_audit_v1"
            or audit.get("complete_synchronized_streams") is not True
            or audit.get("no_undesired_measured_contact") is not True
            or audit.get("normal_force_threshold_n") != 1.0
        ):
            raise ValueError(
                "qualified environmental contact audit differs from the registered criterion"
            )
        contact_reports[option_id] = {"artifact": audit_ref, "audit": audit}
    neutral_pairs = [
        row["body_pair"] for row in contact_reports["neutral"]["audit"]["self_contacts"]
    ]
    request_ref = artifact(args.request)
    if json.loads(checked_artifact(request_ref).read_text()) != bank.request:
        raise ValueError("request artifact differs from the physically verified registry")
    scene = validate_scene(json.loads(args.scene_definition.read_text()))
    template = json.loads(args.template.read_text())
    base = next(cell for cell in template["cells"] if cell["cell_id"] == args.cell)
    if (
        template["implementation"]["checkpoint"] != bank.request["controller"]
        or base["motion"] != bank.request["references"][0]["motion"]
    ):
        raise ValueError("template source/controller differs from verified timed qualification")
    script_ref = None
    script_path = getattr(args, "script_parameters", None)
    if args.policy_mode == "scripted_multi":
        if script_path is None:
            raise ValueError("scripted_multi requires an immutable script-parameters JSON")
        script_ref = artifact(script_path)
        load_script_parameters(script_path, script_ref["sha256"], bank)
    elif script_path is not None:
        raise ValueError("script parameters are only consumed by scripted_multi")
    runtime_assets_ref = None
    runtime_assets = []
    if getattr(args, "runtime_assets", None) is not None:
        runtime_assets_ref = artifact(args.runtime_assets)
        runtime_assets = json.loads(checked_artifact(runtime_assets_ref).read_text())
        if not isinstance(runtime_assets, list) or not runtime_assets:
            raise ValueError("runtime asset declaration must be a nonempty artifact list")
        for ref in runtime_assets:
            checked_artifact(ref)
    policy_ref = None
    if args.policy_mode == "learned":
        if args.policy is None:
            raise ValueError("learned mode requires a SHA-bound schedule policy")
        policy_ref = artifact(args.policy)
        load_schedule_policy(args.policy, policy_ref["sha256"], bank)
    elif args.policy is not None:
        raise ValueError("policy artifact is only used by learned mode")
    if args.seed < 0:
        raise ValueError("nonnegative explicit runtime seed required")
    subset = getattr(args, "forced_option_ids", None)
    if subset is not None and (
        args.policy_mode != "forced"
        or not subset
        or len(set(subset)) != len(subset)
        or any(option_id not in bank.option_ids for option_id in subset)
    ):
        raise ValueError("forced subset must name distinct qualified complete schedules")
    ids = (
        tuple(subset)
        if subset is not None
        else (bank.option_ids if args.policy_mode == "forced" else ("neutral",))
    )
    cells = []
    for option_id in ids:
        identifier = f"{args.policy_mode}_{option_id}"
        cell = copy.deepcopy(base)
        cell.pop("option", None)
        cell.update(
            cell_id=identifier,
            condition=scene["scene_id"],
            role=(
                "complete_schedule_teacher"
                if args.policy_mode == "forced"
                else "repeated_schedule_policy"
            ),
            encounter_action=int(option_id != "neutral"),
            declared_schedule=next(
                (item for item in bank.request["options"] if item["option_id"] == option_id), None
            ),
            timed_schedule_mode=args.policy_mode,
            policy_id=getattr(args, "policy_id", None) or args.policy_mode,
            forced_option_id=option_id,
            beam=scene["beams"][0],
            beams=scene["beams"],
            preferred_option_id=preferred_option_id,
            preferred_reference_id=preferred_reference_id,
            scene={**scene["scene"], "scene_id": scene["scene_id"]},
            runtime_seed=args.seed,
            motion=bank.request["references"][0]["motion"],
            alternate_motion=bank.request["references"][1]["motion"],
            output=str(args.out.resolve() / "rollouts" / identifier),
        )
        updates = {
            "manager_env._target_": f"{RUNTIME}.TimedScheduleEnvCfg",
            "manager_env.recorders.trajectory._target_": f"{RUNTIME}.TimedScheduleRecorderCfg",
            "seed": args.seed,
        }
        for key, value in {
            "reactive_alternate_path": cell["alternate_motion"]["path"],
            "qualification_request_path": request_ref["path"],
            "qualification_request_sha256": request_ref["sha256"],
            "timed_registry_path": str(args.registry.resolve()),
            "timed_registry_sha256": artifact(args.registry)["sha256"],
            "timed_schedule_mode": args.policy_mode,
            "forced_option_id": option_id,
            "preferred_option_id": preferred_option_id,
            "preferred_reference_id": preferred_reference_id,
            "environment_beam_paths": json.dumps(scene_beam_paths(scene), separators=(",", ":")),
            "expected_reference_frames": bank.frame_count,
            "encounter_action": int(option_id != "neutral"),
            "decision_time_s": int(phases[0]) / 50,
            "observation_delay_s": 0.0,
            "history_max_age_s": HISTORY_SECONDS,
            "history_max_frames": HISTORY_FRAMES,
            "timed_policy_path": policy_ref["path"] if policy_ref else "",
            "timed_policy_sha256": policy_ref["sha256"] if policy_ref else "",
            "script_parameters_path": script_ref["path"] if script_ref else "",
            "script_parameters_sha256": script_ref["sha256"] if script_ref else "",
        }.items():
            updates[f"manager_env.config.{key}"] = value
        overrides = [
            value
            for value in base["hydra_overrides"]
            if value.lstrip("+").split("=", 1)[0] not in updates
        ]
        cell["hydra_overrides"] = overrides + [f"++{key}={value}" for key, value in updates.items()]
        cell["command"] = [
            "bash",
            str(ROOT / "scripts/research/run_kimodo_sonic_rollout.sh"),
            "--scene",
            scene["scene_id"],
            "--scene-package",
            str(Path(scene["scene"]["path"]).parent),
            "--scene-usd",
            str(Path(scene["scene"]["path"]).resolve()),
            "--motion",
            cell["motion"]["path"],
            "--out",
            cell["output"],
            "--checkpoint",
            bank.request["controller"]["path"],
            "--python",
            template["implementation"]["python"],
            "--trajectory-only",
            "--max-steps",
            str(bank.frame_count),
            "--extra",
            " ".join(cell["hydra_overrides"]),
        ]
        cells.append(cell)
    runtime_sources = closure([ROOT / (RUNTIME.replace(".", "/") + ".py")]) | {
        ROOT / "scripts/research/run_kimodo_sonic_rollout.sh",
        ROOT / "gear_sonic/eval_agent_trl.py",
    }
    scoring_sources = closure([Path(__file__)])
    sources = runtime_sources | scoring_sources
    dependencies = []
    # Validate reserved execution before creating its registration directory.
    # The protocol receives independently enumerated actual code and explicit
    # asset declarations. Native asset completeness remains an adoption gate.
    for path in sorted(sources):
        dependencies.append(artifact(path))
    manifest = {
        "schema": COLLECTION_SCHEMA,
        "split": scene["split"],
        "request": request_ref,
        "request_digest": definition_digest(bank.request),
        "registry": artifact(args.registry),
        "scene_definition": artifact(args.scene_definition),
        "template": artifact(args.template),
        "policy": policy_ref,
        "script_parameters": script_ref,
        "runtime_assets_declaration": runtime_assets_ref,
        "runtime_artifacts": [artifact(path) for path in sorted(runtime_sources)] + runtime_assets,
        "scoring_artifacts": [artifact(path) for path in sorted(scoring_sources)],
        "qualified_environment_audits": {
            key: value["artifact"] for key, value in contact_reports.items()
        },
        "declared_neutral_self_pairs": neutral_pairs,
        "feature_schema": SCHEMA,
        "feature_names": list(expected_feature_names(len(bank.option_ids))),
        "option_ids": list(bank.option_ids),
        "phase_ticks": phases.tolist(),
        "expected_recorded_control_steps": bank.frame_count - 1,
        "expected_physics_steps": 4 * (bank.frame_count - 1),
        "implementation": template["implementation"],
        "dependencies": dependencies,
        "cells": cells,
        "limits": {"minimum_free_gpu_mib": 7500, "timeout_s": 375, "serial": True},
        "sensor": {
            "rays_per_tick": 65,
            "history_seconds": HISTORY_SECONDS,
            "history_frames": HISTORY_FRAMES,
            "delay_s": 0,
        },
        "environment_beam_paths": scene_beam_paths(scene),
        "scoring": (
            "passage plus whole captured horizon: stable, "
            "no undesired measured contact, exact declared return"
        ),
        "failure_policy": "retain all attempted cells and partial raw captures; never automatically retry",
        "scope": (
            "declared one/two-beam finite course; repeated neutral commitment at qualified "
            "ticks, at most one adaptation and mandatory exact return"
        ),
    }
    for cell in cells:
        validate_collection_context(cell, manifest, bank, scene)
    args.out.mkdir(parents=True, exist_ok=False)
    for dependency in dependencies:
        path = Path(dependency["path"])
        snapshot = args.out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
        dependency["snapshot"] = artifact(snapshot)
    write_new(args.out / "manifest.json", manifest)
    print(json.dumps({"prepared": len(cells), "manifest": artifact(args.out / "manifest.json")}))


def verify_manifest(out, *, execution=True):
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != COLLECTION_SCHEMA:
        raise ValueError("unsupported timed collection schema")
    bank = load_verified_registry(
        **{
            "path": manifest["registry"]["path"],
            "expected_sha256": manifest["registry"]["sha256"],
        }
    )
    for ref in (manifest["request"], manifest["template"], manifest["scene_definition"]):
        checked(Path(ref["path"]), ref["sha256"])
    for dependency in manifest["dependencies"]:
        ref = dependency if execution else dependency["snapshot"]
        checked(Path(ref["path"]), ref["sha256"])
    for key in ("policy", "script_parameters", "runtime_assets_declaration"):
        if manifest.get(key):
            checked_artifact(manifest[key])
    if "runtime_artifacts" in manifest:
        assets = (
            json.loads(checked_artifact(manifest["runtime_assets_declaration"]).read_text())
            if manifest.get("runtime_assets_declaration")
            else []
        )
        for ref in assets:
            checked_artifact(ref)
        if execution:
            runtime_sources = closure([ROOT / (RUNTIME.replace(".", "/") + ".py")]) | {
                ROOT / "scripts/research/run_kimodo_sonic_rollout.sh",
                ROOT / "gear_sonic/eval_agent_trl.py",
            }
            if manifest["runtime_artifacts"] != [
                artifact(path) for path in sorted(runtime_sources)
            ] + assets or manifest["scoring_artifacts"] != [
                artifact(path) for path in sorted(closure([Path(__file__)]))
            ]:
                raise ValueError(
                    "declared runtime/scoring closure differs from actual executable sources"
                )
        # Archived analysis checks its recorded code roles against the complete
        # SHA-bound source snapshot, without pretending current code is old code.
        code = {(ref["path"], ref["sha256"]) for ref in manifest["dependencies"]}
        declared = {
            (ref["path"], ref["sha256"])
            for ref in manifest["runtime_artifacts"] + manifest["scoring_artifacts"]
        }
        extra = {(ref["path"], ref["sha256"]) for ref in assets}
        if declared != code | extra:
            raise ValueError(
                "archived implementation roles do not cover the complete source snapshot"
            )
    scene = validate_scene(json.loads(checked_artifact(manifest["scene_definition"]).read_text()))
    phases, _, _ = schedule_layout(bank)
    if (
        manifest["request_digest"] != definition_digest(bank.request)
        or manifest["option_ids"] != list(bank.option_ids)
        or manifest["phase_ticks"] != phases.tolist()
        or manifest["feature_names"] != list(expected_feature_names(len(bank.option_ids)))
        or manifest["environment_beam_paths"] != scene_beam_paths(scene)
        or manifest["expected_recorded_control_steps"] != bank.frame_count - 1
        or manifest["expected_physics_steps"] != 4 * (bank.frame_count - 1)
    ):
        raise ValueError("collection schema/constants differ from verified registry and scene")
    for cell in manifest["cells"]:
        validate_collection_context(cell, manifest, bank, scene)
    return manifest, bank, scene


def imported_collision_enabled(shape):
    """Inventory collision records API presence; the schema attribute enables it."""
    value = shape["attributes"].get("physics:collisionEnabled")
    if type(value) is bool:
        return value
    if value in ("True", "False"):
        return value == "True"
    raise ValueError("actual imported physics:collisionEnabled must be recorded explicitly")


def analyze_cell(cell, manifest, bank, scene):
    validate_collection_context(cell, manifest, bank, scene, actual=True)
    phases, _, _ = schedule_layout(bank)
    folder = Path(cell["output"]) / "trajectories"
    paths = list(folder.glob("*.trajectory.pkl"))
    if len(paths) != 1:
        raise ValueError("exactly one complete physical capture required")
    payload = load_reset_capture(paths[0])
    interface = json.loads((folder / "reactive_interface.json").read_text())
    if (
        interface["timed_schedule_mode"] != cell["timed_schedule_mode"]
        or interface["timed_policy_sha256"]
        != (manifest["policy"]["sha256"] if manifest["policy"] else "")
        or interface.get("script_parameters_sha256", "")
        != (manifest["script_parameters"]["sha256"] if manifest.get("script_parameters") else "")
        or interface["feature_schema"] != SCHEMA
        or interface["option_ids"] != list(bank.option_ids)
        or tuple(interface["feature_names"]) != expected_feature_names(len(bank.option_ids))
        or interface["timed_request_digest"] != definition_digest(bank.request)
        or interface["timed_registry_sha256"] != manifest["registry"]["sha256"]
        or interface["phase_ticks"] != phases.tolist()
        or interface["history_max_age_s"] != HISTORY_SECONDS
        or interface["history_max_frames"] != HISTORY_FRAMES
        or interface["qualification_only"] is not False
    ):
        raise ValueError("captured policy/feature/reference schema differs from registration")
    observations = interface["observations"]
    if any(
        len(row["features"]) != 100 + 2 * len(bank.option_ids)
        or not np.isfinite(row["features"]).all()
        or len(row["measurements"]) != 65
        for row in observations
    ):
        raise ValueError("incomplete100+2K features or65-ray measurements")
    with np.load(folder / "timed_schedule_features.npz", allow_pickle=False) as data:
        if (
            str(data["schema_version"]) != SCHEMA
            or tuple(data["feature_names"]) != expected_feature_names(len(bank.option_ids))
            or not np.array_equal(data["features"], [row["features"] for row in observations])
            or not np.array_equal(
                data["active_before"], [row["active_before"] for row in observations]
            )
            or not np.array_equal(data["active"], [row["active"] for row in observations])
            or not np.array_equal(data["command_ticks"], [row["tick"] for row in observations])
            or not np.array_equal(data["legal_mask"], [row["legal_mask"] for row in observations])
        ):
            raise ValueError("feature archive and raw sensor interface differ")
    alignment = audit_sensor_alignment(payload, observations, reference_frames=bank.frame_count)
    with np.load(folder / "all_body_contacts.npz", allow_pickle=False) as data:
        contacts = {key: data[key].copy() for key in data.files}
    legacy_contact_diagnostic = audit_whole_body_contacts(
        contacts, manifest["expected_physics_steps"]
    )
    with np.load(folder / "environment_pair_contacts.npz", allow_pickle=False) as data:
        pairs = {key: data[key].copy() for key in data.files}
    mapping = json.loads((folder / "environment_contact_mapping.json").read_text())
    contact_audit = audit_environment_contacts(
        pairs,
        contacts,
        mapping,
        manifest["expected_physics_steps"],
        neutral_self_pairs=manifest.get("declared_neutral_self_pairs", []),
        beam_paths=scene_beam_paths(scene),
    )
    forces, sync = synchronize_pair_beam_forces(
        pairs, mapping, scene_beam_paths(scene), contacts["control_steps"]
    )
    inventory = json.loads((folder / "native_collision_inventory.json").read_text())
    geometry_audit = audit_scene_geometry(inventory["shapes"], scene)
    observation_timing = phase_visibility(observations, alignment, scene, phases)
    runtime = json.loads((folder.parent / "success_manifest.json").read_text())
    source_audit = json.loads((folder / "loaded_reference_bank.json").read_text())
    source_bound = (
        runtime["checkpoint_hash"] == bank.request["controller"]["sha256"]
        and runtime["capture_context"]["motion"]["hash"]
        == bank.request["references"][0]["motion"]["sha256"]
        and runtime["capture_context"]["use_encoder"] == "g1"
        and max(source_audit["source_route_max_error_m"]) <= 1e-4
        and interface["loaded_reference_ids"]
        == [value["reference_id"] for value in bank.request["references"]]
        and interface["loaded_reference_frames"]
        == [bank.frame_count] * len(bank.request["references"])
    )
    admitted = bool(
        source_bound
        and all(alignment["packet_eligible"])
        and len(observations) == bank.frame_count - 1
        and contact_audit["complete_synchronized_streams"]
        and np.array_equal(contacts["control_steps"], [row["physics_step"] for row in observations])
    )
    stable = bool(
        (
            (np.asarray(payload["root_pos_w"])[:, 2] >= 0.5)
            & (-np.asarray(payload["projected_gravity_b"])[:, 2] >= 0.5)
        ).all()
    )
    schedule_audit = audit_executed_schedule(
        bank,
        observations,
        interface["switches"],
        cell["timed_schedule_mode"],
        cell["forced_option_id"],
    )
    passage = score_scene(payload, forces, scene, bank, commands_valid=schedule_audit["valid"])
    no_refusals = schedule_audit["no_refusals"]
    returned = observations[-1]["active"] == "neutral"
    passed = bool(
        admitted
        and passage["pass"]
        and stable
        and returned
        and no_refusals
        and contact_audit["no_undesired_measured_contact"]
        and schedule_audit["valid"]
    )
    finish = passage["passage_finish_frame_exclusive"]
    result = {
        "cell_id": cell["cell_id"],
        "mode": cell["timed_schedule_mode"],
        "forced_option_id": cell["forced_option_id"],
        "measurement_admitted": admitted,
        "pass": passed,
        "passage": passage,
        "whole_horizon_stable": stable,
        "returned_to_neutral": returned,
        "no_refusals": no_refusals,
        "physics_steps": contact_audit["recorded_physics_steps"],
        "sensor_queries": 65 * len(observations),
        "phase_observation_receipt": [
            {
                "command_tick": row["tick"],
                "active_before": row["active_before"],
                "selected_option_id": row["selected_option_id"],
                "legal_mask": row["legal_mask"],
                "physical_reference_phase_s": (row["tick"] - 1) / 50,
                "capture_elapsed_s": row["capture_elapsed_s"],
                "delivered_capture_elapsed_s": row["delivered_capture_elapsed_s"],
                "ceiling_observed_cells": row["ceiling_observed_cells"],
                "upper_occupied_bands": [
                    name
                    for name, value in zip(
                        expected_feature_names(len(bank.option_ids)), row["features"], strict=True
                    )
                    if name.endswith("upper_hit") and value > 0
                ],
                "measurement_count": len(row["measurements"]),
                "normal_known_count": sum(row["normal_known_mask"]),
            }
            for row in observations
            if row["tick"] in {int(t) + offset for t in phases for offset in (-1, 0)}
        ],
        "first_upper_occupied_evidence_tick": next(
            (
                row["tick"]
                for row in observations
                if any(
                    value > 0
                    for name, value in zip(
                        expected_feature_names(len(bank.option_ids)), row["features"], strict=True
                    )
                    if name.endswith("upper_hit")
                )
            ),
            None,
        ),
        "sensor_alignment": alignment,
        "contact_audit": contact_audit,
        "legacy_net_nonfoot_diagnostic": legacy_contact_diagnostic,
        "environment_pairs": artifact(folder / "environment_pair_contacts.npz"),
        "environment_mapping": artifact(folder / "environment_contact_mapping.json"),
        "schedule_audit": schedule_audit,
        "beam_contact_synchronization": sync,
        "imported_geometry_audit": geometry_audit,
        "observation_timing": observation_timing,
        "trajectory": artifact(paths[0]),
        "sensor": artifact(folder / "reactive_interface.json"),
        "features": artifact(folder / "timed_schedule_features.npz"),
        "all_body_contacts": artifact(folder / "all_body_contacts.npz"),
        "physics_beam_contacts": artifact(folder / "physics_beam_contacts.npz"),
        "costs": {
            "passage_time_s": (finish - 1) / 50 if passed else None,
            "whole_episode_time_s": len(observations) / 50,
            "switch_count": len(interface["switches"]),
            "positive_mechanical_work_j": None,
            "work_measurement": "implicit actuator estimated PD effort only",
        },
    }
    outcome = classify_timed_attempt(
        payload,
        observations,
        reference_frames=bank.frame_count,
        phase_ticks=phases.tolist(),
        exit_status=0,
        measurement_admitted=admitted,
        passage=passage,
        contact_audit=contact_audit,
        schedule_audit=schedule_audit,
        physics_steps=pairs["physics_steps"],
    )
    result.update(outcome=outcome, task_outcome_admitted=outcome["task_outcome"] != "unknown")
    if passed != outcome["complete_pass"]:
        raise ValueError("complete scorer and independent physical outcome classification disagree")
    return result, payload, interface


def analyze(out):
    manifest, bank, scene = verify_manifest(out, execution=False)
    sources = []
    for source in sorted(closure([Path(__file__)])):
        destination = out / "analysis_source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError("immutable analysis source snapshot already differs")
        if not destination.exists():
            shutil.copyfile(source, destination)
        sources.append({**artifact(source), "snapshot": artifact(destination)})
    rows = []
    for cell in manifest["cells"]:
        folder = Path(cell["output"])
        attempt = json.loads((folder / "attempt.json").read_text())
        validate_collection_context(cell, manifest, bank, scene, actual=True)
        if attempt["exit_status"] != 0:
            row = analyze_incomplete(cell, manifest, bank, scene, attempt, "nonzero_process_exit")
        else:
            try:
                row, payload, interface = analyze_cell(cell, manifest, bank, scene)
            except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
                row = analyze_incomplete(cell, manifest, bank, scene, attempt, str(error))
        row["attempt"] = artifact(folder / "attempt.json")
        rows.append(row)
    write_new(
        out / "result.json",
        {
            "schema": COLLECTION_SCHEMA,
            "manifest": artifact(out / "manifest.json"),
            "rows": rows,
            "teacher": None,
            "teacher_scope": "future schedule teacher requires separately audited matched prefixes",
            "analysis_implementation": sources,
            "physics_steps": sum(row["physics_steps"] or 0 for row in rows),
            "unmeasured_failed_attempts": sum(row["physics_steps"] is None for row in rows),
            "scope": (
                "complete declared episodes; physical failures retained "
                "independently from measurement admission"
            ),
        },
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "passed": sum(row["pass"] for row in rows),
            }
        )
    )


def run(out):
    manifest, _, scene = verify_manifest(out)
    for cell in manifest["cells"]:
        folder = Path(cell["output"])
        if (folder / "attempt.json").exists():
            continue
        if folder.exists():
            raise RuntimeError(f"unfinished attempted cell retained: {folder}")
        checked_artifact({key: scene["scene"][key] for key in ("path", "sha256")})
        if free_gpu_mib() < manifest["limits"]["minimum_free_gpu_mib"]:
            raise RuntimeError("resource preflight paused; remaining cells unlaunched")
        started = time.monotonic()
        status = run_with_process_group(cell["command"], timeout=manifest["limits"]["timeout_s"])
        folder.mkdir(parents=True, exist_ok=True)
        write_new(
            folder / "attempt.json",
            {
                "exit_status": status,
                "wall_seconds": time.monotonic() - started,
                "command": cell["command"],
            },
        )
        print(json.dumps({"cell_id": cell["cell_id"], "exit_status": status}), flush=True)
    analyze(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--scene-definition", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--cell", default="neutral")
    parser.add_argument("--policy-mode", choices=MODES, default="forced")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--policy-id", help="Explicit adopted policy identity, or development mode")
    parser.add_argument("--script-parameters", type=Path)
    parser.add_argument(
        "--runtime-assets", type=Path, help="Explicit JSON list of SHA-bound native/runtime assets"
    )
    parser.add_argument(
        "--forced-option-ids",
        nargs="+",
        help="Explicit forced subset; default is the complete bank",
    )
    parser.add_argument("--preferred-option-id", default="neutral")
    parser.add_argument("--preferred-reference-id", default="sustained")
    parser.add_argument("--seed", type=int, default=8731)
    args = parser.parse_args()
    if args.action == "prepare":
        if any(
            getattr(args, key) is None
            for key in ("registry", "request", "scene_definition", "template")
        ):
            parser.error("prepare requires --registry --request --scene-definition --template")
        prepare(args)
    elif args.action == "run":
        run(args.out)
    else:
        analyze(args.out)
