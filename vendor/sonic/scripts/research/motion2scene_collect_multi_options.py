#!/usr/bin/env python3
"""Acquire or evaluate real multi-option sensor policies with matched branch records."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_reactive_interface import physics_windows  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import (  # noqa: E402
    paired_prefix,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_policy import (  # noqa: E402
    SCHEMA,
    load_option_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import (  # noqa: E402
    score_passage,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)

RUNTIME = "gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_execution"


def validate_template_bindings(parent, cells, registry):
    checkpoint = parent["implementation"]["checkpoint"]
    if registry["controller"] != checkpoint:
        raise ValueError("execution checkpoint differs from option qualification")
    checked(Path(checkpoint["path"]), checkpoint["sha256"])
    neutral = registry["references"][0]["motion"]
    for cell in cells:
        if cell["generation_seed"] != registry["source"]:
            raise ValueError("template source differs from qualified option registry")
        motion = cell["motion"]
        if (
            Path(motion["path"]).resolve() != Path(neutral["path"]).resolve()
            or motion["sha256"] != neutral["sha256"]
        ):
            raise ValueError("template neutral motion differs from qualified registry")
        checked(Path(motion["path"]), motion["sha256"])


def qualified_reference_frames(registry):
    """Use the physically loaded bank, not source frame count divided by fps."""
    counts, evidence = set(), []
    for entry in registry["references"]:
        ref = entry["qualification"]
        qualification = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        row = next(
            row
            for row in qualification["rows"]
            if row["qualified"]
            and row["source"] == registry["source"]
            and row["option"]["reference"] == entry["name"]
        )
        bank = row["bank"]
        with np.load(checked(Path(bank["path"]), bank["sha256"]), allow_pickle=False) as data:
            roots, joints = data["root_xyz"], data["joint_pos"]
            if roots.ndim != 3 or joints.shape[:2] != roots.shape[:2] or float(data["fps"]) != 50:
                raise ValueError("qualification bank requires complete 50 Hz loaded references")
            counts.add(roots.shape[1])
        evidence.append({"option": entry["name"], "loaded_bank": bank})
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise ValueError("qualified references must share one finite loaded frame budget")
    return next(iter(counts)), evidence


def sensor_clock_audit(states, payload):
    """Separate post-command reference phase from same-row physical recording."""
    phases = np.asarray([state["time_s"] for state in states], dtype=float)
    wraps = (np.flatnonzero(np.diff(phases) <= 0) + 1).tolist()
    end = wraps[0] if wraps else len(states)
    roots = np.asarray([state["root_pos_w"] for state in states[:end]])
    physical = np.asarray(payload["root_pos_w"])[:end]
    offsets = phases[:end] - np.asarray(payload["motion_time_s"])[:end]
    return {
        "phase_wrap_indices": wraps,
        "pre_wrap_rows": end,
        "pre_wrap_root_state_exact_match": bool(np.array_equal(roots, physical)),
        "pre_wrap_command_phase_advance_s": (
            [float(offsets.min()), float(offsets.max())] if end else None
        ),
        "relation": (
            "sensor row i shares physical state with trajectory row i before wrap; "
            "its command reference phase follows the one-tick command update"
        ),
    }


def prepare(args):
    parent = json.loads(args.template.read_text())
    registry_ref = artifact(args.registry)
    registry = load_option_registry(args.registry, registry_ref["sha256"])
    selected = [c for c in parent["cells"] if c["cell_id"] in args.cells]
    if len(selected) != len(set(args.cells)):
        raise ValueError("missing or duplicate template cell")
    validate_template_bindings(parent, selected, registry)
    reference_frames, frame_evidence = qualified_reference_frames(registry)
    if args.mode == "learned" and args.policy is None:
        raise ValueError("learned mode requires trained policy")
    options = (
        args.force_options
        if args.force_options is not None
        else list(range(len(registry["references"])))
    )
    indices = options if args.mode == "forced" else [args.preferred_option]
    cells = []
    for base in selected:
        for index in indices:
            if not 0 <= index < len(registry["references"]):
                raise ValueError("option outside registry")
            cell = copy.deepcopy(base)
            identifier = f"{base['cell_id']}_{args.mode}_k{index}"
            cell.update(
                cell_id=identifier,
                base_cell_id=base["cell_id"],
                option_index=index,
                multi_option_mode=args.mode,
                runtime_seed=args.seed,
                decision_time_s=args.entry_time_s,
                reject_sensor_clock_wrap=True,
                reference_frame_budget=reference_frames,
                output=str(args.out.resolve() / "rollouts" / identifier),
            )
            cell["alternate_motion"] = registry["references"][1]["motion"]
            prefixes = (
                "manager_env._target_=",
                "manager_env.recorders.trajectory._target_=",
                "manager_env.config.closed_loop_mode=",
                "manager_env.config.encounter_action=",
                "manager_env.config.reactive_alternate_path=",
                "manager_env.config.forced_entry_time_s=",
                "manager_env.config.observation_delay_s=",
                "manager_env.config.learned_policy_path=",
                "manager_env.config.learned_policy_sha256=",
                "seed=",
            )
            cell["hydra_overrides"] = [
                v for v in base["hydra_overrides"] if not v.lstrip("+").startswith(prefixes)
            ] + [
                f"++manager_env._target_={RUNTIME}.MultiOptionEnvCfg",
                f"++manager_env.recorders.trajectory._target_={RUNTIME}.MultiOptionRecorderCfg",
                f"++manager_env.config.reactive_alternate_path={registry['references'][1]['motion']['path']}",
                f"++manager_env.config.option_registry_path={registry_ref['path']}",
                f"++manager_env.config.option_registry_sha256={registry_ref['sha256']}",
                f"++manager_env.config.multi_option_mode={args.mode}",
                f"++manager_env.config.preferred_option_index={index}",
                f"++manager_env.config.forced_entry_time_s={args.entry_time_s}",
                "++manager_env.config.observation_delay_s=0.0",
                f"++seed={args.seed}",
            ]
            if args.policy is not None:
                policy = artifact(args.policy)
                cell["hydra_overrides"] += [
                    f"++manager_env.config.multi_policy_path={policy['path']}",
                    f"++manager_env.config.multi_policy_sha256={policy['sha256']}",
                ]
            cells.append(cell)
    args.out.mkdir(parents=True, exist_ok=False)
    code = closure([Path(__file__), ROOT / (RUNTIME.replace(".", "/") + ".py")])
    code.add(ROOT / "scripts/research/run_kimodo_sonic_rollout.sh")
    dependencies = []
    for path in sorted(code):
        snapshot = args.out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        dependencies.append({**artifact(path), "snapshot": artifact(snapshot)})
    write_new(
        args.out / "manifest.json",
        {
            "schema": "motion2scene_multi_option_acquisition_v1",
            "purpose": "development multi-option execution and teacher acquisition",
            "template": artifact(args.template),
            "registry": registry_ref,
            "option_ids": [r["name"] for r in registry["references"]],
            "reference_frame_budget": reference_frames,
            "reference_frame_evidence": frame_evidence,
            "capture_endpoint": (
                "driver max_steps equals loaded reference frames; captures one fewer rows "
                "and terminates before command resampling; no motion extension"
            ),
            "implementation": parent["implementation"],
            "dependencies": dependencies,
            "cells": cells,
            "limits": {"timeout_s": 375, "minimum_free_gpu_mib": 7500, "serial": True},
        },
    )
    print(json.dumps({"prepared": len(cells), "out": str(args.out)}), flush=True)


def analyze_cell(cell, option_ids):
    folder = Path(cell["output"]) / "trajectories"
    paths = list(folder.glob("*.trajectory.pkl"))
    if len(paths) != 1:
        raise ValueError("exactly one trajectory required")
    payload = load_reset_capture(paths[0])
    with np.load(folder / "beam_contacts.npz") as data:
        sampled = data["force_w"].copy()
    with np.load(folder / "physics_beam_contacts.npz") as data:
        _, forces, sync = physics_windows(data, sampled)
        steps = len(data["physics_steps"])
    passage = score_passage(payload, forces, cell["beam"])
    interface = json.loads((folder / "reactive_interface.json").read_text())
    if interface["feature_schema"] != SCHEMA or interface["option_names"] != option_ids:
        raise ValueError("wrong online option feature/identity schema")
    states = interface["observations"]
    size = len(interface["feature_names"])
    if len(states) != len(payload["motion_time_s"]) or any(
        len(s["features"]) != size for s in states
    ):
        raise ValueError("incomplete sensor/state recording")
    clock = sensor_clock_audit(states, payload)
    flags = ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
    unchanged = all(all(s["transition"][key] for key in flags) for s in states)
    no_refusal = all(not s["transition"]["attempted"] or s["transition"]["allowed"] for s in states)
    entered = [s for s in interface["switches"] if s["to"] != 0]
    expected = cell["option_index"]
    executed = (
        (
            not entered
            if expected == 0
            else len(entered) == 1
            and entered[0]["to"] == expected
            and abs(entered[0]["time_s"] - cell["decision_time_s"]) < 1e-8
        )
        if cell["multi_option_mode"] == "forced"
        else True
    )
    measurement_admitted = (
        sync <= 1e-6
        and unchanged
        and no_refusal
        and executed
        and clock["pre_wrap_root_state_exact_match"]
        and (not cell.get("reject_sensor_clock_wrap", False) or not clock["phase_wrap_indices"])
    )
    return {
        "cell_id": cell["cell_id"],
        "base_cell_id": cell["base_cell_id"],
        "option_index": expected,
        "configured_preferred_option_index": expected,
        "executed_option_index": (
            entered[0]["to"] if len(entered) == 1 else (0 if not entered else None)
        ),
        "executed_option_sequence": [0] + [event["to"] for event in interface["switches"]],
        "mode": cell["multi_option_mode"],
        "source": cell["generation_seed"],
        "physics_seed": cell["runtime_seed"],
        "condition": cell["condition"],
        "beam": cell["beam"],
        "decision_time_s": cell["decision_time_s"],
        **passage,
        "measurement_admitted": bool(measurement_admitted),
        "sensor_clock": clock,
        "switches": interface["switches"],
        "costs": {
            "passage_time_s": (
                (passage["passage_finish_frame_exclusive"] - 1) / 50 if passage["pass"] else None
            ),
            "positive_mechanical_work_j": None,
            "work_measurement": "implicit actuator effort unavailable",
            "switch_count": len(interface["switches"]),
        },
        "trajectory": artifact(paths[0]),
        "sensor": artifact(folder / "reactive_interface.json"),
        "features": artifact(folder / "closed_loop_features.npz"),
        "bank": artifact(folder / "loaded_reference_bank.npz"),
        "acquisition": {"physics_steps": steps, "evaluated_branches": 1},
    }


def analyze(out):
    manifest = json.loads((out / "manifest.json").read_text())
    rows = [
        json.loads((Path(c["output"]) / "multi_option_row.json").read_text())
        for c in manifest["cells"]
    ]
    examples = []
    for base in dict.fromkeys(row["base_cell_id"] for row in rows):
        group = [r for r in rows if r["base_cell_id"] == base and r["mode"] == "forced"]
        if not group:
            continue
        packets = []
        prefix_matches = []
        first = load_reset_capture(
            checked(Path(group[0]["trajectory"]["path"]), group[0]["trajectory"]["sha256"])
        )
        for row in group:
            current = load_reset_capture(
                checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
            )
            prefix_matches.append(
                paired_prefix(first, current, row["decision_time_s"])["exact_match"]
            )
            document = json.loads(
                checked(Path(row["sensor"]["path"]), row["sensor"]["sha256"]).read_text()
            )
            packet = [
                p
                for p in document["observations"][: row["first_episode_frames"]]
                if abs(p["time_s"] - row["decision_time_s"]) < 1e-8
            ]
            if len(packet) != 1:
                raise ValueError("missing or duplicate teacher decision frame")
            packets.append(packet[0])
        matched = all(prefix_matches) and all(
            p["features"] == packets[0]["features"] and p["state"] == packets[0]["state"]
            for p in packets
        )
        count = len(manifest["option_ids"])
        passing, times, admission = [False] * count, [None] * count, [False] * count
        for row in group:
            i = row["option_index"]
            passing[i], times[i] = row["pass"], row["costs"]["passage_time_s"]
            admission[i] = matched and row["measurement_admitted"]
        examples.append(
            {
                "group_id": base,
                "matched_state_and_features": matched,
                "features": packets[0]["features"],
                "feature_names": document["feature_names"],
                "option_ids": manifest["option_ids"],
                "passages": passing,
                "passage_time_s": times,
                "admission": admission,
                "legality": packets[0]["legal_mask"],
                "branch_cell_ids": [r["cell_id"] for r in group],
            }
        )
    write_new(
        out / "result.json",
        {
            "manifest": artifact(out / "manifest.json"),
            "rows": rows,
            "teacher_examples": examples,
            "physics_steps": sum(r["acquisition"]["physics_steps"] for r in rows),
            "scope": "actual development multi-option sensor-policy executions; no held-out-course claim",
        },
    )


def execute(out):
    from hallucination.run_approved_manifest import free_gpu_mib, run_with_process_group

    manifest = json.loads((out / "manifest.json").read_text())
    for ref in manifest["dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    checkpoint = manifest["implementation"]["checkpoint"]
    checked(Path(checkpoint["path"]), checkpoint["sha256"])
    for cell in manifest["cells"]:
        folder = Path(cell["output"])
        if (folder / "multi_option_row.json").exists():
            continue
        if folder.exists():
            raise RuntimeError("unfinished attempt retained; explicit new manifest required")
        if free_gpu_mib() < manifest["limits"]["minimum_free_gpu_mib"]:
            raise RuntimeError("insufficient free GPU memory")
        for key in ("motion", "alternate_motion", "scene"):
            checked(Path(cell[key]["path"]), cell[key]["sha256"])
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
            "--trajectory-only",
            "--extra",
            " ".join(cell["hydra_overrides"]),
        ]
        if "reference_frame_budget" in manifest:
            command += ["--max-steps", str(manifest["reference_frame_budget"])]
        start = time.monotonic()
        status = run_with_process_group(command, timeout=manifest["limits"]["timeout_s"])
        write_new(
            folder / "attempt.json",
            {"exit_status": status, "wall_seconds": time.monotonic() - start, "command": command},
        )
        if status:
            raise RuntimeError(f"infrastructure failure retained at {folder}")
        row = analyze_cell(cell, manifest["option_ids"])
        write_new(folder / "multi_option_row.json", row)
        print(
            json.dumps(
                {
                    "cell": cell["cell_id"],
                    "pass": row["pass"],
                    "admitted": row["measurement_admitted"],
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
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--cells", nargs="+")
    parser.add_argument(
        "--mode",
        choices=("forced", "scripted", "always_walk", "always_adapt", "learned"),
        default="forced",
    )
    parser.add_argument("--force-options", type=int, nargs="+")
    parser.add_argument("--preferred-option", type=int, default=4)
    parser.add_argument("--entry-time-s", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=8731)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "run":
        execute(args.out)
    else:
        analyze(args.out)
