#!/usr/bin/env python3
"""Register one declared development scene with verified timed sensor policies."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from hallucination.run_approved_manifest import free_gpu_mib, run_with_process_group  # noqa: E402
from motion2scene_reactive_interface import physics_windows  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import (  # noqa: E402
    paired_prefix,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_course import (  # noqa: E402
    audit_imported_beams,
    validate_beams,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_environment_contacts import (  # noqa: E402
    audit_environment_contacts,
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
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_history_policy import (  # noqa: E402
    ENTRY_TICK,
    MODES,
    SCHEMA,
    audit_executed_schedule,
    expected_feature_names,
    load_timed_policy,
    validate_history_bank,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_measurements import (  # noqa: E402
    audit_whole_body_contacts,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    checked_artifact,
    definition_digest,
    load_verified_registry,
)

RUNTIME = "gear_sonic.dataset_generation.hallucination.motion2scene_timed_history_execution"
SCENE_SCHEMA = "motion2scene_timed_history_scene_v1"
COLLECTION_SCHEMA = "motion2scene_timed_history_collection_v1"


def validate_scene(definition):
    if (
        definition.get("schema") != SCENE_SCHEMA
        or definition.get("split") != "development"
        or type(definition.get("beam_collision_enabled")) is not bool
        or not isinstance(definition.get("scene_id"), str)
        or not definition["scene_id"]
    ):
        raise ValueError("explicit development scene, geometry and collision setting required")
    checked_artifact({key: definition["scene"][key] for key in ("path", "sha256")})
    validate_beams([definition["beam"]])
    return definition


def prepare(args):
    bank = load_verified_registry(args.registry, artifact(args.registry)["sha256"])
    validate_history_bank(bank, args.policy_mode)
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
    policy_ref = None
    if args.policy_mode == "learned":
        if args.policy is None:
            raise ValueError("learned mode requires a SHA-bound106D policy")
        policy_ref = artifact(args.policy)
        load_timed_policy(args.policy, policy_ref["sha256"], bank)
    elif args.policy is not None:
        raise ValueError("policy artifact is only used by learned mode")
    if args.seed < 0:
        raise ValueError("nonnegative explicit runtime seed required")
    ids = bank.option_ids if args.policy_mode == "forced" else ("neutral",)
    cells = []
    for option_id in ids:
        identifier = f"{args.policy_mode}_{option_id}"
        cell = copy.deepcopy(base)
        cell.pop("option", None)
        cell.update(
            cell_id=identifier,
            condition=scene["scene_id"],
            role=(
                "development_duration_teacher"
                if args.policy_mode == "forced"
                else "development_duration_policy"
            ),
            encounter_action=int(option_id != "neutral"),
            declared_schedule=next(
                (item for item in bank.request["options"] if item["option_id"] == option_id), None
            ),
            timed_history_mode=args.policy_mode,
            forced_option_id=option_id,
            beam=scene["beam"],
            scene={**scene["scene"], "scene_id": scene["scene_id"]},
            runtime_seed=args.seed,
            motion=bank.request["references"][0]["motion"],
            alternate_motion=bank.request["references"][1]["motion"],
            output=str(args.out.resolve() / "rollouts" / identifier),
        )
        updates = {
            "manager_env._target_": f"{RUNTIME}.TimedHistoryEnvCfg",
            "manager_env.recorders.trajectory._target_": f"{RUNTIME}.TimedHistoryRecorderCfg",
            "seed": args.seed,
        }
        for key, value in {
            "reactive_alternate_path": cell["alternate_motion"]["path"],
            "qualification_request_path": request_ref["path"],
            "qualification_request_sha256": request_ref["sha256"],
            "timed_registry_path": str(args.registry.resolve()),
            "timed_registry_sha256": artifact(args.registry)["sha256"],
            "timed_history_mode": args.policy_mode,
            "forced_option_id": option_id,
            "expected_reference_frames": bank.frame_count,
            "encounter_action": int(option_id != "neutral"),
            "decision_time_s": 0.3,
            "observation_delay_s": 0.0,
            "history_max_age_s": 0.5,
            "history_max_frames": 26,
            "timed_policy_path": policy_ref["path"] if policy_ref else "",
            "timed_policy_sha256": policy_ref["sha256"] if policy_ref else "",
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
    args.out.mkdir(parents=True, exist_ok=False)
    sources = closure([Path(__file__), ROOT / (RUNTIME.replace(".", "/") + ".py")]) | {
        ROOT / "scripts/research/run_kimodo_sonic_rollout.sh",
        ROOT / "gear_sonic/eval_agent_trl.py",
    }
    dependencies = []
    for path in sorted(sources):
        snapshot = args.out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
        dependencies.append({**artifact(path), "snapshot": artifact(snapshot)})
    manifest = {
        "schema": COLLECTION_SCHEMA,
        "split": "development",
        "request": request_ref,
        "request_digest": definition_digest(bank.request),
        "registry": artifact(args.registry),
        "scene_definition": artifact(args.scene_definition),
        "template": artifact(args.template),
        "policy": policy_ref,
        "qualified_environment_audits": {
            key: value["artifact"] for key, value in contact_reports.items()
        },
        "declared_neutral_self_pairs": neutral_pairs,
        "feature_schema": SCHEMA,
        "feature_names": list(expected_feature_names()),
        "option_ids": list(bank.option_ids),
        "entry_tick": ENTRY_TICK,
        "expected_recorded_control_steps": bank.frame_count - 1,
        "expected_physics_steps": 4 * (bank.frame_count - 1),
        "implementation": template["implementation"],
        "dependencies": dependencies,
        "cells": cells,
        "limits": {"minimum_free_gpu_mib": 7500, "timeout_s": 375, "serial": True},
        "sensor": {"rays_per_tick": 65, "history_seconds": 0.5, "history_frames": 26, "delay_s": 0},
        "scoring": (
            "passage plus whole captured horizon: stable, "
            "no undesired measured contact, exact declared return"
        ),
        "failure_policy": "retain all attempted cells and partial raw captures; never automatically retry",
        "scope": "single development scene; one duration decision at.3s; no stopping, reentry or learned timing",
    }
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
    if manifest["policy"]:
        checked_artifact(manifest["policy"])
    scene = validate_scene(json.loads(checked_artifact(manifest["scene_definition"]).read_text()))
    if manifest["request_digest"] != definition_digest(bank.request):
        raise ValueError("collection request digest differs from verified registry")
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
    folder = Path(cell["output"]) / "trajectories"
    paths = list(folder.glob("*.trajectory.pkl"))
    if len(paths) != 1:
        raise ValueError("exactly one complete physical capture required")
    payload = load_reset_capture(paths[0])
    interface = json.loads((folder / "reactive_interface.json").read_text())
    if (
        interface["timed_history_mode"] != cell["timed_history_mode"]
        or interface["timed_policy_sha256"]
        != (manifest["policy"]["sha256"] if manifest["policy"] else "")
        or interface["feature_schema"] != SCHEMA
        or interface["option_ids"] != list(bank.option_ids)
        or tuple(interface["feature_names"]) != expected_feature_names()
        or interface["timed_request_digest"] != definition_digest(bank.request)
        or interface["timed_registry_sha256"] != manifest["registry"]["sha256"]
    ):
        raise ValueError("captured policy/feature/reference schema differs from registration")
    observations = interface["observations"]
    if any(
        len(row["features"]) != 106
        or not np.isfinite(row["features"]).all()
        or len(row["measurements"]) != 65
        for row in observations
    ):
        raise ValueError("incomplete106D features or65-ray measurements")
    with np.load(folder / "timed_history_features.npz", allow_pickle=False) as data:
        if (
            str(data["schema_version"]) != SCHEMA
            or tuple(data["feature_names"]) != expected_feature_names()
            or not np.array_equal(data["features"], [row["features"] for row in observations])
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
    )
    with np.load(folder / "beam_contacts.npz", allow_pickle=False) as data:
        sampled = data["force_w"].copy()
    with np.load(folder / "physics_beam_contacts.npz", allow_pickle=False) as data:
        _, forces, sync = physics_windows(data, sampled)
    inventory = json.loads((folder / "native_collision_inventory.json").read_text())
    shapes = copy.deepcopy(inventory["shapes"])
    beam_shape = next(
        value for value in shapes if value["path"] == "/World/ground/terrain/CounterfactualBeam"
    )
    if imported_collision_enabled(beam_shape) != scene["beam_collision_enabled"]:
        raise ValueError("actual beam collision setting differs from registration")
    geometry_audit = audit_imported_beams(shapes, [scene["beam"]])
    observation_timing = audit_decision_visibility(
        observations, scene["beam"], ENTRY_TICK, collision_enabled=scene["beam_collision_enabled"]
    )
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
        and interface["loaded_reference_frames"] == [bank.frame_count] * 3
    )
    flags = ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
    unchanged = all(all(row["transition"][key] for key in flags) for row in observations)
    admitted = bool(
        source_bound
        and unchanged
        and all(alignment["packet_eligible"])
        and len(observations) == bank.frame_count - 1
        and contact_audit["complete_synchronized_streams"]
        and np.array_equal(contacts["control_steps"], [row["physics_step"] for row in observations])
    )
    passage = score_passage(payload, forces, scene["beam"])
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
        cell["timed_history_mode"],
        cell["forced_option_id"],
    )
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
        "mode": cell["timed_history_mode"],
        "forced_option_id": cell["forced_option_id"],
        "measurement_admitted": admitted,
        "pass": passed,
        "passage": passage,
        "whole_horizon_stable": stable,
        "returned_to_neutral": returned,
        "no_refusals": no_refusals,
        "physics_steps": contact_audit["recorded_physics_steps"],
        "sensor_queries": 65 * len(observations),
        "entry_observation_receipt": [
            {
                "command_tick": row["tick"],
                "physical_reference_phase_s": (row["tick"] - 1) / 50,
                "capture_elapsed_s": row["capture_elapsed_s"],
                "delivered_capture_elapsed_s": row["delivered_capture_elapsed_s"],
                "ceiling_observed_cells": row["ceiling_observed_cells"],
                "upper_occupied_bands": [
                    name
                    for name, value in zip(expected_feature_names(), row["features"], strict=True)
                    if name.endswith("upper_hit") and value > 0
                ],
                "measurement_count": len(row["measurements"]),
                "normal_known_count": sum(row["normal_known_mask"]),
            }
            for row in observations
            if row["tick"] in (14, 15)
        ],
        "first_upper_occupied_evidence_tick": next(
            (
                row["tick"]
                for row in observations
                if any(
                    value > 0
                    for name, value in zip(expected_feature_names(), row["features"], strict=True)
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
        "features": artifact(folder / "timed_history_features.npz"),
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
    rows, captures = [], {}
    for cell in manifest["cells"]:
        folder = Path(cell["output"])
        attempt = json.loads((folder / "attempt.json").read_text())
        if attempt["exit_status"] != 0:
            partial = folder / "trajectories/aborted_interface.json"
            capture = json.loads(partial.read_text()) if partial.exists() else {}
            rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "mode": cell["timed_history_mode"],
                    "status": "failed_attempt",
                    "measurement_admitted": False,
                    "pass": False,
                    "physics_steps": capture.get("physical_steps_recorded"),
                    "sensor_queries": None,
                    "sensor_queries_in_recorded_packets_only": (
                        65 * len(capture["observations"]) if "observations" in capture else None
                    ),
                    "attempt": artifact(folder / "attempt.json"),
                    "partial": artifact(partial) if partial.exists() else None,
                }
            )
            continue
        row, payload, interface = analyze_cell(cell, manifest, bank, scene)
        row["attempt"] = artifact(folder / "attempt.json")
        captures[cell["forced_option_id"]] = (row, payload, interface)
        rows.append(row)
    teacher = None
    if all(cell["timed_history_mode"] == "forced" for cell in manifest["cells"]) and set(
        captures
    ) == set(bank.option_ids):
        neutral = captures["neutral"]
        decisions = []
        prefixes = []
        for option_id in bank.option_ids:
            _, payload, interface = captures[option_id]
            target = [row for row in interface["observations"] if row["tick"] == ENTRY_TICK]
            if len(target) != 1:
                raise ValueError("exactly one first-episode entry packet required")
            decisions.append(target[0])
            prefixes.append(paired_prefix(neutral[1], payload, ENTRY_TICK / 50))
        exact = all(p["exact_match"] and p["frames"] >= ENTRY_TICK for p in prefixes) and all(
            np.array_equal(row["features"], decisions[0]["features"])
            and row["measurements"] == decisions[0]["measurements"]
            for row in decisions
        )
        teacher = {
            "entry_tick": ENTRY_TICK,
            "option_ids": list(bank.option_ids),
            "feature_names": list(expected_feature_names()),
            "features": decisions[0]["features"],
            "legality": decisions[0]["legal_mask"],
            "passed": [captures[key][0]["pass"] for key in bank.option_ids],
            "passage_time_s": [
                captures[key][0]["costs"]["passage_time_s"] for key in bank.option_ids
            ],
            "admitted": [
                bool(exact and captures[key][0]["measurement_admitted"]) for key in bank.option_ids
            ],
            "matched_prefixes": prefixes,
            "exact_shared_entry_observation": exact,
            "scope": (
                "one complete forced schedule per action; matched single entry, "
                "no timing continuation approximation"
            ),
        }
    write_new(
        out / "result.json",
        {
            "schema": COLLECTION_SCHEMA,
            "manifest": artifact(out / "manifest.json"),
            "rows": rows,
            "teacher": teacher,
            "analysis_implementation": sources,
            "physics_steps": sum(row["physics_steps"] or 0 for row in rows),
            "unmeasured_failed_attempts": sum(row["physics_steps"] is None for row in rows),
            "scope": (
                "complete development episodes; physical failures retained "
                "independently from measurement admission"
            ),
        },
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "passed": sum(row["pass"] for row in rows),
                "teacher": teacher is not None,
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
