#!/usr/bin/env python3
"""Register and execute complete forced finite schedules in a matched empty room."""

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
from hallucination.run_approved_manifest import (  # noqa: E402
    free_gpu_mib,
    run_with_process_group,
)
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import (  # noqa: E402
    paired_prefix,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_measurements import (  # noqa: E402
    audit_whole_body_contacts,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    EVIDENCE_SCHEMA,
    checked_artifact,
    definition_digest,
    validate_evidence,
    validate_request,
)
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    CollisionCapsule,
    body_capsules_world,
)

DATA = ROOT.parent / "research-data/groot-wbc"


def prepare(args):
    request = json.loads(args.request.read_text())
    bank = validate_request(request)
    if request.get("minimum_recovery_ticks") != 15 or any(
        option["recovery_end_tick"] - option["return_tick"] < 15 for option in request["options"]
    ):
        raise ValueError("every requested schedule needs at least15 physical recovery ticks")
    for reference in request["references"]:
        checked_artifact(reference["motion"])
    checked_artifact(request["controller"])
    parent = json.loads(args.template.read_text())
    if request["controller"] != parent["implementation"]["checkpoint"]:
        raise ValueError("request and template frozen controllers differ")
    geometry = json.loads(args.geometry.read_text())
    for layer in geometry["layers"]:
        checked(Path(layer["path"]), layer["sha256"])
    args.out.mkdir(parents=True, exist_ok=False)
    cells = []
    for option in bank.option_ids:
        cell = copy.deepcopy(parent["cells"][0])
        cell.pop("option", None)
        cell.update(
            cell_id=option,
            forced_option_id=option,
            role="forced_complete_schedule_development_qualification",
            encounter_action=int(option != "neutral"),
            declared_schedule=next(
                (value for value in request["options"] if value["option_id"] == option), None
            ),
            motion=request["references"][0]["motion"],
            alternate_motion=request["references"][1]["motion"],
            output=str(args.out.resolve() / "rollouts" / option),
        )
        prefixes = (
            "++manager_env._target_=",
            "++manager_env.recorders.trajectory._target_=",
            "++manager_env.config.reactive_alternate_path=",
            "++manager_env.config.encounter_action=",
            "++manager_env.config.expected_reference_frames=",
        )
        cell["hydra_overrides"] = [
            value for value in cell["hydra_overrides"] if not value.startswith(prefixes)
        ] + [
            "++manager_env._target_=gear_sonic.dataset_generation.hallucination.motion2scene_long_schedule_execution.LongScheduleEnvCfg",
            "++manager_env.recorders.trajectory._target_=gear_sonic.dataset_generation.hallucination.motion2scene_long_schedule_execution.LongScheduleRecorderCfg",
            f"++manager_env.config.reactive_alternate_path={request['references'][1]['motion']['path']}",
            f"++manager_env.config.encounter_action={int(option != 'neutral')}",
            f"++manager_env.config.expected_reference_frames={bank.frame_count}",
            f"++manager_env.config.qualification_request_path={args.request.resolve()}",
            f"++manager_env.config.qualification_request_sha256={artifact(args.request)['sha256']}",
            f"++manager_env.config.forced_option_id={option}",
        ]
        cell["command"] = [
            "bash",
            str(ROOT / "scripts/research/run_kimodo_sonic_rollout.sh"),
            "--scene",
            cell["scene"]["scene_id"],
            "--scene-package",
            str(Path(cell["scene"]["path"]).parent),
            "--motion",
            cell["motion"]["path"],
            "--out",
            cell["output"],
            "--checkpoint",
            request["controller"]["path"],
            "--python",
            parent["implementation"]["python"],
            "--trajectory-only",
            "--max-steps",
            str(bank.frame_count),
            "--extra",
            " ".join(cell["hydra_overrides"]),
        ]
        cells.append(cell)
    sources = closure(
        [
            Path(__file__),
            ROOT
            / "gear_sonic/dataset_generation/hallucination/motion2scene_long_schedule_execution.py",
        ]
    ) | {
        ROOT / "scripts/research/run_kimodo_sonic_rollout.sh",
        ROOT / "gear_sonic/eval_agent_trl.py",
    }
    dependencies = []
    for path in sorted(sources):
        destination = args.out / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        dependencies.append({**artifact(path), "snapshot": artifact(destination)})
    write_new(
        args.out / "manifest.json",
        {
            "schema": "motion2scene_long_schedule_qualification_v1",
            "request": artifact(args.request),
            "request_digest": definition_digest(request),
            "template": artifact(args.template),
            "geometry": artifact(args.geometry),
            "implementation": parent["implementation"],
            "dependencies": dependencies,
            "cells": cells,
            "expected_recorded_control_steps": bank.frame_count - 1,
            "expected_physics_steps": 4 * (bank.frame_count - 1),
            "limits": {"minimum_free_gpu_mib": 7500, "timeout_s": 375, "serial": True},
            "measurement_contract": {
                "floor_bounds_xy_m": [[-10, 10], [-4, 4]],
                "required_outer_envelope_wall_clearance_m": 0.5,
                "nonfoot_normal_force_threshold_n": 1.0,
                "unexplained_foot_force_threshold_n": 1.0,
                "whole_episode_root_height_min_m": 0.5,
                "whole_episode_upright_gravity_min": 0.5,
                "recovery_tail_maximum_root_height_error_vs_neutral_m": 0.03,
                "recovery_tail_mean_outer_height_error_vs_neutral_m": 0.03,
                "recovery_tail_minimum_physical_span_s": 0.30,
                "recovery_tail_window": "final16samples ending at declared recovery_end_tick",
                "mechanical_work": "unavailable for implicit PhysX actuator; estimated PD effort retained",
            },
            "scope": (
                "Matched forced experimental development schedules; "
                "no online qualification before complete evidence"
            ),
            "failure_policy": (
                "Retain every scientific or technical failure and raw partial arrays; "
                "never automatically retry an attempted cell"
            ),
        },
    )
    print(json.dumps({"prepared": len(cells), "manifest": artifact(args.out / "manifest.json")}))


def run(out):
    manifest = json.loads((out / "manifest.json").read_text())
    for ref in [manifest["request"], manifest["geometry"], *manifest["dependencies"]]:
        checked(Path(ref["path"]), ref["sha256"])
    request = json.loads(Path(manifest["request"]["path"]).read_text())
    for ref in [request["controller"], *(r["motion"] for r in request["references"])]:
        checked_artifact(ref)
    for cell in manifest["cells"]:
        folder = Path(cell["output"])
        if (folder / "attempt.json").exists():
            continue
        if folder.exists():
            raise RuntimeError(f"unfinished attempt retained: {folder}")
        checked(Path(cell["scene"]["path"]), cell["scene"]["sha256"])
        if free_gpu_mib() < manifest["limits"]["minimum_free_gpu_mib"]:
            raise RuntimeError("resource preflight paused; remaining cells unlaunched")
        started = time.monotonic()
        status = run_with_process_group(cell["command"], timeout=manifest["limits"]["timeout_s"])
        write_new(
            folder / "attempt.json",
            {
                "exit_status": status,
                "wall_seconds": time.monotonic() - started,
                "command": cell["command"],
            },
        )
        print(json.dumps({"cell": cell["cell_id"], "exit_status": status}), flush=True)
    analyze(out)


def analyze(out):
    manifest = json.loads((out / "manifest.json").read_text())
    request = json.loads(checked_artifact(manifest["request"]).read_text())
    bank = validate_request(request)
    raw_geometry = json.loads(checked_artifact(manifest["geometry"]).read_text())
    shapes = {}
    for shape in raw_geometry["shapes"]:
        shapes.setdefault(shape["owner"], []).append(
            CollisionCapsule(tuple(shape["start"]), tuple(shape["end"]), shape["radius"])
        )
    rows, captures = [], {}
    for cell in manifest["cells"]:
        folder = Path(cell["output"])
        attempt = json.loads((folder / "attempt.json").read_text())
        if attempt["exit_status"] != 0:
            partial = folder / "trajectories/aborted_interface.json"
            steps = (
                json.loads(partial.read_text())["physical_steps_recorded"]
                if partial.exists()
                else None
            )
            rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "qualified": False,
                    "status": "incomplete_failed_attempt",
                    "attempt": artifact(folder / "attempt.json"),
                    "physics_steps": steps,
                }
            )
            continue
        path = folder / "trajectories"
        trajectory = next(path.glob("*.trajectory.pkl"))
        payload = load_reset_capture(trajectory)
        interface = json.loads((path / "reactive_interface.json").read_text())
        observations = interface["observations"]
        times = np.asarray(payload["motion_time_s"]).reshape(-1)
        ticks = np.asarray([value["tick"] for value in observations])
        with np.load(path / "all_body_contacts.npz") as handle:
            contacts = {name: handle[name].copy() for name in handle.files}
        contact_audit = audit_whole_body_contacts(contacts, manifest["expected_physics_steps"])
        starts, ends, radii, _ = body_capsules_world(
            payload["body_pos_w"], payload["body_quat_w"], payload["body_names"], capsules=shapes
        )
        height = np.max(np.maximum(starts[:, :, 2], ends[:, :, 2]) + radii, axis=1)
        minimum = np.minimum(starts[:, :, :2], ends[:, :, :2]) - radii[..., None]
        maximum = np.maximum(starts[:, :, :2], ends[:, :, :2]) + radii[..., None]
        wall_clearance = float(min((minimum + [10, 4]).min(), ([10, 4] - maximum).min()))
        roots = np.asarray(payload["root_pos_w"])
        sensor_roots = np.asarray([value["root_pos_w"] for value in observations])
        sensor_quats = np.asarray([value["root_quat_w"] for value in observations])
        aligned = (
            np.array_equal(roots, sensor_roots)
            and np.array_equal(payload["root_quat_w"], sensor_quats)
            and np.array_equal(
                contacts["control_steps"], [value["physics_step"] for value in observations]
            )
            and np.allclose(ticks / 50 - times, 0.02, atol=1e-7, rtol=0)
        )
        clock = {
            "root_quaternion_physics_counter_match": bool(aligned),
            "command_ticks": ticks.tolist(),
            "physical_reference_phase_s": times.tolist(),
            "no_physical_reset": bool(np.all(np.diff(times) > 0)),
            "no_sensor_wrap": bool(np.array_equal(ticks, np.arange(1, bank.frame_count))),
        }
        geometry_audit = {
            "geometry": manifest["geometry"],
            "support_plane_z_m": request["support_plane_z_m"],
            "height_measurement": request["height_measurement"],
            "executed_body_height_m": (height - request["support_plane_z_m"]).tolist(),
            "outer_wall_clearance_m": wall_clearance,
            "scope": (
                "Native-shape enclosing capsules transformed by actual50Hzbody poses; "
                "finite sampled geometry, not continuous collision guarantee"
            ),
        }
        write_new(path / "clock_audit.json", clock)
        write_new(path / "geometry_audit.json", geometry_audit)
        write_new(path / "all_contact_audit.json", contact_audit)
        captures[cell["cell_id"]] = (
            cell,
            payload,
            interface,
            height,
            clock,
            geometry_audit,
            contact_audit,
            trajectory,
        )
    neutral = captures.get("neutral")
    for option_id, (
        cell,
        payload,
        interface,
        height,
        clock,
        geometry,
        contacts,
        trajectory,
    ) in captures.items():
        path = Path(cell["output"]) / "trajectories"
        option = next(
            (value for value in request["options"] if value["option_id"] == option_id), None
        )
        entry_tick = (
            option["entry_tick"]
            if option
            else min(value["entry_tick"] for value in request["options"])
        )
        prefix = (
            paired_prefix(neutral[1], payload, entry_tick / 50)
            if neutral
            else {"frames": 0, "exact_match": False}
        )
        write_new(path / "prefix_audit.json", prefix)
        ticks = np.asarray(clock["command_ticks"])
        recovery = option is None
        recovery_measurements = None
        if option and neutral:
            tail = (ticks >= option["recovery_end_tick"] - 15) & (
                ticks <= option["recovery_end_tick"]
            )
            root_error = abs(
                np.asarray(payload["root_pos_w"])[tail, 2]
                - np.asarray(neutral[1]["root_pos_w"])[tail, 2]
            )
            height_error = abs(height[tail] - neutral[3][tail])
            recovery = bool(
                tail.sum() >= 16
                and root_error.max(initial=0) <= 0.03
                and height_error.mean() <= 0.03
            )
            recovery_measurements = {
                "tail_samples": int(tail.sum()),
                "maximum_root_height_error_m": float(root_error.max(initial=0)),
                "mean_outer_height_error_m": float(height_error.mean()),
            }
        stable = bool(
            (
                (np.asarray(payload["root_pos_w"])[:, 2] >= 0.5)
                & (-np.asarray(payload["projected_gravity_b"])[:, 2] >= 0.5)
            ).all()
        )
        source_audit = json.loads((path / "loaded_reference_bank.json").read_text())
        runtime = json.loads((path.parent / "success_manifest.json").read_text())
        checks = {
            "matched_approach_prefix": prefix["exact_match"],
            "all_contacts_audited": contacts["complete_synchronized_streams"]
            and contacts["no_undesired_measured_contact"]
            and geometry["outer_wall_clearance_m"] >= 0.5,
            "whole_episode_stable": stable,
            "full_horizon": len(ticks) == bank.frame_count - 1,
            "no_reset": clock["no_physical_reset"],
            "no_wrap": clock["no_sensor_wrap"],
            "source_bank_bound": (
                max(source_audit["source_route_max_error_m"]) <= 1e-4
                and runtime["capture_context"]["motion"]["hash"]
                == request["references"][0]["motion"]["sha256"]
                and interface["timed_request_digest"] == definition_digest(request)
            ),
            "controller_bound": (
                runtime["checkpoint_hash"] == request["controller"]["sha256"]
                and runtime["capture_context"]["use_encoder"] == "g1"
            ),
            "sensor_clock_aligned": clock["root_quaternion_physics_counter_match"],
            "reference_guard_verified": all(
                s["joint_jump_rad"] <= 0.05 and s["root_jump_m"] <= 0.01
                for s in interface["switches"]
            ),
            "recovery_verified": recovery,
            "no_refusals": all(r["transition"]["allowed"] for r in interface["observations"]),
        }
        evidence = {
            "schema": EVIDENCE_SCHEMA,
            "request_digest": definition_digest(request),
            "controller": request["controller"],
            "reference_motions": [reference["motion"] for reference in request["references"]],
            "option_id": option_id,
            "height_measurement": request["height_measurement"],
            "support_plane_z_m": request["support_plane_z_m"],
            "checks": checks,
            "loaded_reference_ids": interface["loaded_reference_ids"],
            "loaded_reference_frames": interface["loaded_reference_frames"],
            "fps": 50,
            "command_ticks": ticks.tolist(),
            "active_option_ids": [row["active"] for row in interface["observations"]],
            "switches": interface["switches"],
            "matched_prefix_frames": prefix["frames"],
            "executed_body_height_m": geometry["executed_body_height_m"],
            "recorded_physics_steps": contacts["recorded_physics_steps"],
            "recovery_measurements": recovery_measurements,
            "artifacts": {
                "trajectory": artifact(trajectory),
                "interface": artifact(path / "reactive_interface.json"),
                "physics_contacts": artifact(path / "all_body_contacts.npz"),
                "clock_audit": artifact(path / "clock_audit.json"),
                "prefix_audit": artifact(path / "prefix_audit.json"),
                "geometry_audit": artifact(path / "geometry_audit.json"),
                "contact_audit": artifact(path / "all_contact_audit.json"),
            },
        }
        failure = None
        try:
            validate_evidence(request, evidence)
        except ValueError as error:
            failure = str(error)
        write_new(path / "qualification_evidence.json", evidence)
        rows.append(
            {
                "cell_id": option_id,
                "qualified": failure is None,
                "qualification_error": failure,
                "checks": checks,
                "physics_steps": contacts["recorded_physics_steps"],
                "recovery_measurements": recovery_measurements,
                "evidence": artifact(path / "qualification_evidence.json"),
            }
        )
    write_new(
        out / "result.json",
        {
            "schema": "motion2scene_long_schedule_qualification_v1",
            "manifest": artifact(out / "manifest.json"),
            "rows": rows,
            "physics_steps": sum(row["physics_steps"] or 0 for row in rows),
            "unmeasured_failed_attempts": sum(row["physics_steps"] is None for row in rows),
            "scope": "Finite forced development schedule evidence; rejected/partial outcomes retained",
        },
    )
    print(
        json.dumps(
            {"rows": [{"id": row["cell_id"], "qualified": row["qualified"]} for row in rows]}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument(
        "--template", type=Path, default=DATA / "m2s-longer-neutral-qualification-v2/manifest.json"
    )
    parser.add_argument(
        "--geometry", type=Path, default=DATA / "m2s-native-beam-audit-v1/geometry.json"
    )
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "run":
        run(args.out)
    else:
        analyze(args.out)
