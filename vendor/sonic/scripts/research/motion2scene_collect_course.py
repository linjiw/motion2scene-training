#!/usr/bin/env python3
"""Register and run fresh development overhead courses within one reference pass.

No locked layouts are consumed or modified. Scene truth is restricted to
authoring, contact instrumentation and scoring; the existing student is reused.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
import joblib  # noqa: E402
from motion2scene_collect_multi_options import analyze_cell as analyze_multi_cell  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_course import (  # noqa: E402
    COURSE_SCHEMA,
    author_course,
    beam_prim_name,
    reference_frame_budget,
    score_course,
    synchronize_course_forces,
    validate_course_definition,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_policy import (  # noqa: E402
    load_multi_policy,
    load_option_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_sensor_alignment import (  # noqa: E402
    audit_sensor_alignment,
)

RUNTIME = "gear_sonic.dataset_generation.hallucination.motion2scene_course_execution"


def prepare(args):
    parent = json.loads(args.template.read_text())
    selected = [cell for cell in parent["cells"] if cell["cell_id"] == args.template_cell]
    if len(selected) != 1:
        raise ValueError("requires one explicitly selected development template cell")
    base = selected[0]
    registry_ref = artifact(args.registry)
    registry = load_option_registry(args.registry, registry_ref["sha256"])
    if (
        registry["controller"] != parent["implementation"]["checkpoint"]
        or registry["source"] != base["generation_seed"]
    ):
        raise ValueError("course source/controller differ from qualified option registry")
    neutral = registry["references"][0]["motion"]
    if any(base["motion"][key] != neutral[key] for key in ("path", "sha256")):
        raise ValueError("template primary motion differs from qualified neutral")
    references = []
    for item in registry["references"]:
        motion = item["motion"]
        library = joblib.load(checked(Path(motion["path"]), motion["sha256"]))
        if len(library) != 1:
            raise ValueError("course requires one clip per qualified reference")
        references.append(next(iter(library.values())))
    source_frames = len(references[0]["root_trans_offset"])
    source_fps = float(references[0]["fps"])
    if (
        any(
            float(entry["fps"]) != source_fps
            or len(entry["root_trans_offset"]) != source_frames
            or len(entry["dof"]) != source_frames
            for entry in references
        )
        or not np.isfinite(source_fps)
        or source_fps <= 0
    ):
        raise ValueError(
            "course references need equal existing source durations; no padding permitted"
        )
    budget = reference_frame_budget(source_frames, source_fps)
    frames = budget["frames"]
    if not 0 <= args.option_index < len(references):
        raise ValueError("requested option outside registry")
    if (
        args.entry_time_s
        not in registry["references"][args.option_index]["qualified_entry_times_s"]
    ):
        raise ValueError("requested entry time lacks physical qualification")
    route = np.asarray(references[0]["root_trans_offset"], dtype=float)[:, :2]
    if any(
        not np.allclose(np.asarray(entry["root_trans_offset"])[:, :2], route, atol=1e-7, rtol=0)
        for entry in references[1:]
    ):
        raise ValueError("course options must retain the qualified common approach route")
    progress = np.r_[0, np.linalg.norm(np.diff(route, axis=0), axis=1).cumsum()]
    if not np.isfinite(route).all() or progress[-1] <= 0:
        raise ValueError("requires a finite nonzero qualified approach route")
    progress /= progress[-1]
    stations = np.asarray(args.stations)
    if (
        stations.ndim != 1
        or len(stations) < 2
        or not np.isfinite(stations).all()
        or np.any(stations <= 0)
        or np.any(stations >= 1)
        or np.any(np.diff(stations) <= 0)
    ):
        raise ValueError("course stations must increase strictly inside the known route")
    if len({len(args.stations), len(args.lengths), len(args.undersides)}) != 1:
        raise ValueError("supply one length and underside per station")
    direction = route[-1] - route[0]
    yaw = float(np.arctan2(direction[1], direction[0]))
    beams = [
        {
            "route_progress_fraction": float(station),
            "center_xy_m": [
                float(np.interp(station, progress, route[:, axis])) for axis in range(2)
            ],
            "yaw_rad": yaw,
            "length_m": float(length),
            "width_m": args.width,
            "thickness_m": args.thickness,
            "underside_m": float(underside),
        }
        for station, length, underside in zip(stations, args.lengths, args.undersides, strict=True)
    ]
    definition = {
        "schema": COURSE_SCHEMA,
        "split": "development",
        "course_id": args.course_id,
        "motion_mode": "one_authored_reference_pass",
        "beams": beams,
        "reference_budget": budget,
        "timeout_s": budget["duration_s"],
        "source": registry["source"],
        "registry": registry_ref,
        "neutral_motion": neutral,
        "template_scene": base["scene"],
        "construction": "fresh development parameters; no geometric or physical acceptance implied",
        "unqualified": [
            "reference extension",
            "repeated adaptation entries",
            "protective stopping",
            "general navigation",
        ],
    }
    validate_course_definition(definition)
    scene_source = checked(Path(base["scene"]["path"]), base["scene"]["sha256"])
    scene_text = author_course(scene_source.read_text(), beams, course_id=args.course_id)
    policy = artifact(args.policy) if args.policy else None
    option_ids = [item["name"] for item in registry["references"]]
    if "learned" in args.modes:
        if policy is None:
            raise ValueError("learned course mode requires a fixed multi-option policy")
        load_multi_policy(policy["path"], policy["sha256"], option_ids)
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(args.out / "course_definition.json", definition)
    definition_ref = artifact(args.out / "course_definition.json")
    scene = args.out / "scene" / f"{args.course_id}.usda"
    scene.parent.mkdir()
    scene.write_text(scene_text)
    scene_ref = {**artifact(scene), "scene_id": args.course_id}
    cells = []
    for mode in args.modes:
        index = 0 if mode == "always_walk" else args.option_index
        cell = copy.deepcopy(base)
        for key in ("option", "history_mode", "encounter_action"):
            cell.pop(key, None)
        identifier = f"{args.course_id}_{mode}_k{index}"
        cell.update(
            cell_id=identifier,
            base_cell_id=args.course_id,
            condition="development_course",
            scene=scene_ref,
            beam=beams[0],
            course_definition=definition_ref,
            option_index=index,
            multi_option_mode=mode,
            runtime_seed=args.seed,
            decision_time_s=args.entry_time_s,
            reject_sensor_clock_wrap=True,
            alternate_motion=registry["references"][1]["motion"],
            output=str(args.out.resolve() / "rollouts" / identifier),
        )
        keys = (
            "manager_env._target_=",
            "manager_env.recorders.trajectory._target_=",
            "manager_env.config.reactive_alternate_path=",
            "manager_env.config.option_registry_path=",
            "manager_env.config.option_registry_sha256=",
            "manager_env.config.multi_option_mode=",
            "manager_env.config.preferred_option_index=",
            "manager_env.config.forced_entry_time_s=",
            "manager_env.config.multi_policy_path=",
            "manager_env.config.multi_policy_sha256=",
            "manager_env.config.course_definition_path=",
            "manager_env.config.course_definition_sha256=",
            "manager_env.config.observation_delay_s=",
            "seed=",
        )
        cell["hydra_overrides"] = [
            value for value in base["hydra_overrides"] if not value.lstrip("+").startswith(keys)
        ] + [
            f"++manager_env._target_={RUNTIME}.CourseEnvCfg",
            f"++manager_env.recorders.trajectory._target_={RUNTIME}.CourseRecorderCfg",
            f"++manager_env.config.reactive_alternate_path={registry['references'][1]['motion']['path']}",
            f"++manager_env.config.option_registry_path={registry_ref['path']}",
            f"++manager_env.config.option_registry_sha256={registry_ref['sha256']}",
            f"++manager_env.config.course_definition_path={definition_ref['path']}",
            f"++manager_env.config.course_definition_sha256={definition_ref['sha256']}",
            f"++manager_env.config.multi_option_mode={mode}",
            f"++manager_env.config.preferred_option_index={index}",
            f"++manager_env.config.forced_entry_time_s={args.entry_time_s}",
            "++manager_env.config.observation_delay_s=0.0",
            f"++seed={args.seed}",
        ]
        if mode == "learned":
            cell["hydra_overrides"] += [
                f"++manager_env.config.multi_policy_path={policy['path']}",
                f"++manager_env.config.multi_policy_sha256={policy['sha256']}",
            ]
        cells.append(cell)
    dependencies = closure([Path(__file__), ROOT / (RUNTIME.replace(".", "/") + ".py")])
    dependencies.add(ROOT / "scripts/research/run_kimodo_sonic_rollout.sh")
    snapshots = []
    for path in sorted(dependencies):
        snapshot = args.out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        snapshots.append({**artifact(path), "snapshot": artifact(snapshot)})
    write_new(
        args.out / "manifest.json",
        {
            "schema": "motion2scene_development_course_acquisition_v1",
            "purpose": "two or more constraints within one existing reference pass; fresh development only",
            "template": artifact(args.template),
            "registry": registry_ref,
            "course_definition": definition_ref,
            "option_ids": option_ids,
            "implementation": parent["implementation"],
            "dependencies": snapshots,
            "policy": policy,
            "cells": cells,
            "limits": {
                "max_steps": frames,
                "timeout_s": 375,
                "minimum_free_gpu_mib": 7500,
                "serial": True,
            },
        },
    )
    print(
        json.dumps(
            {
                "prepared": len(cells),
                "beams": len(beams),
                "reference_seconds": budget["duration_s"],
                "out": str(args.out),
            }
        ),
        flush=True,
    )


def analyze_cell(cell, option_ids):
    row = analyze_multi_cell(cell, option_ids)
    folder = Path(cell["output"]) / "trajectories"
    definition_ref = cell["course_definition"]
    definition = json.loads(
        checked(Path(definition_ref["path"]), definition_ref["sha256"]).read_text()
    )
    validate_course_definition(definition)
    capture = json.loads((folder / "course_capture.json").read_text())
    if (
        capture["definition_sha256"] != definition_ref["sha256"]
        or capture["definition"] != definition
        or capture["reference_exhaustion_events"]
    ):
        raise ValueError("course definition or reference-budget audit failed")
    with np.load(folder / "course_contacts.npz", allow_pickle=False) as data:
        if list(data["beam_names"]) != [beam_prim_name(i) for i in range(len(definition["beams"]))]:
            raise ValueError("missing or reordered course force channels")
        forces, sync = synchronize_course_forces(data)
        physics_steps = len(data["physics_steps"])
    payload = load_reset_capture(
        checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
    )
    interface = json.loads(Path(row["sensor"]["path"]).read_text())
    alignment = audit_sensor_alignment(
        payload,
        interface["observations"],
        reference_frames=definition["reference_budget"]["frames"],
    )
    row["measurement_admitted"] = row["measurement_admitted"] and all(alignment["packet_eligible"])
    first_end = row["first_episode_frames"]
    final_phase = interface["observations"][first_end - 1]["time_s"]
    scoring = score_course(
        payload,
        forces,
        definition["beams"],
        commands_valid=row["measurement_admitted"],
        timeout_s=definition["timeout_s"],
        reference_frames=definition["reference_budget"]["frames"],
        final_reference_phase_s=final_phase,
    )
    row.update(scoring)
    row.pop("beam")
    row.update(
        course_id=definition["course_id"],
        beams=definition["beams"],
        course_definition=definition_ref,
        course_contacts=artifact(folder / "course_contacts.npz"),
        course_capture=artifact(folder / "course_capture.json"),
        all_beam_contact_synchronization_error=sync,
        sensor_alignment=alignment,
        acquisition={"physics_steps": physics_steps, "evaluated_branches": 1},
        costs={
            "course_time_s": (
                scoring["elapsed_through_scoring_horizon_s"] if scoring["pass"] else None
            ),
            "positive_mechanical_work_j": None,
            "work_measurement": "implicit actuator effort unavailable",
            "switch_count": len(interface["switches"]),
        },
    )
    return row


def analyze(out):
    manifest = json.loads((out / "manifest.json").read_text())
    rows = [
        json.loads((Path(cell["output"]) / "course_row.json").read_text())
        for cell in manifest["cells"]
    ]
    write_new(
        out / "result.json",
        {
            "schema": "motion2scene_development_course_result_v1",
            "manifest": artifact(out / "manifest.json"),
            "rows": rows,
            "physics_steps": sum(row["acquisition"]["physics_steps"] for row in rows),
            "wall_seconds": sum(row["wall_seconds"] for row in rows),
            "scope": (
                "actual development course execution within one finite reference; "
                "no repeated entry or navigation claim"
            ),
        },
    )


def execute(out):
    from hallucination.run_approved_manifest import free_gpu_mib, run_with_process_group

    manifest = json.loads((out / "manifest.json").read_text())
    for reference in manifest["dependencies"]:
        checked(Path(reference["path"]), reference["sha256"])
    registry = manifest["registry"]
    load_option_registry(registry["path"], registry["sha256"])
    checkpoint = manifest["implementation"]["checkpoint"]
    checked(Path(checkpoint["path"]), checkpoint["sha256"])
    for cell in manifest["cells"]:
        folder = Path(cell["output"])
        if (folder / "course_row.json").exists():
            continue
        if folder.exists():
            raise RuntimeError("unfinished course attempt retained; prepare a new manifest")
        for key in ("motion", "alternate_motion", "scene", "course_definition"):
            checked(Path(cell[key]["path"]), cell[key]["sha256"])
        if free_gpu_mib() < manifest["limits"]["minimum_free_gpu_mib"]:
            raise RuntimeError("insufficient free GPU memory; remaining courses stay pending")
        scene = Path(cell["scene"]["path"])
        command = [
            "bash",
            str(ROOT / "scripts/research/run_kimodo_sonic_rollout.sh"),
            "--scene",
            cell["scene"]["scene_id"],
            "--scene-package",
            str(scene.parent),
            "--motion",
            cell["motion"]["path"],
            "--out",
            str(folder),
            "--checkpoint",
            checkpoint["path"],
            "--python",
            manifest["implementation"]["python"],
            "--max-steps",
            str(manifest["limits"]["max_steps"]),
            "--trajectory-only",
            "--extra",
            " ".join(cell["hydra_overrides"]),
        ]
        start = time.monotonic()
        status = run_with_process_group(command, timeout=manifest["limits"]["timeout_s"])
        elapsed = time.monotonic() - start
        write_new(
            folder / "attempt.json",
            {"exit_status": status, "wall_seconds": elapsed, "command": command},
        )
        if status:
            raise RuntimeError(
                f"course infrastructure/reference-guard failure retained at {folder}"
            )
        row = analyze_cell(cell, manifest["option_ids"])
        row["wall_seconds"] = elapsed
        write_new(folder / "course_row.json", row)
        print(
            json.dumps(
                {
                    "cell": cell["cell_id"],
                    "pass": row["pass"],
                    "completed_beams": row["completed_beams"],
                    "failure_reasons": row["failure_reasons"],
                }
            ),
            flush=True,
        )
    analyze(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--template-cell")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--course-id", default="development_two_beams")
    parser.add_argument("--stations", type=float, nargs="+", default=[0.43, 0.57])
    parser.add_argument("--lengths", type=float, nargs="+", default=[0.12, 0.18])
    parser.add_argument("--undersides", type=float, nargs="+", default=[1.30, 1.27])
    parser.add_argument("--width", type=float, default=1.2)
    parser.add_argument("--thickness", type=float, default=0.1)
    parser.add_argument("--option-index", type=int, default=4)
    parser.add_argument("--entry-time-s", type=float, default=0.3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("forced", "scripted", "always_walk", "always_adapt", "learned"),
        default=["forced"],
    )
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--seed", type=int, default=8731)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.template is None or not args.template_cell or args.registry is None:
            parser.error("prepare requires template, template-cell and registry")
        prepare(args)
    elif args.stage == "run":
        execute(args.out)
    else:
        analyze(args.out)
