#!/usr/bin/env python3
"""Collect matched physical sensor-history branches for a one-encounter teacher.

Manifest preparation precedes physical execution; failed attempts are retained.
Only newly recorded matched state/feature prefixes support imitation targets.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from motion2scene_reactive_interface import physics_windows  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import (  # noqa: E402
    paired_prefix,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import (  # noqa: E402
    score_passage,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)

RUNTIME = "gear_sonic.dataset_generation.hallucination.motion2scene_closed_loop_execution"
MODES = ("forced_walk", "forced_adapt", "always_walk", "always_adapt", "scripted", "learned")


def prepare(args):
    parent = json.loads(args.template.read_text())
    selected = [cell for cell in parent["cells"] if cell["cell_id"] in args.cells]
    if {cell["cell_id"] for cell in selected} != set(args.cells):
        raise ValueError("requested template cells missing")
    if len(selected) != len(args.cells):
        raise ValueError("duplicate template/requested cell ids")
    if "learned" in args.modes and args.policy is None:
        raise ValueError("learned mode requires a policy artifact")
    policy = artifact(args.policy) if args.policy else None
    cells = []
    for base in selected:
        for mode in args.modes:
            cell = copy.deepcopy(base)
            identifier = f"{base['cell_id']}_{mode}_t{round(args.entry_time_s * 100):03d}"
            cell.update(
                cell_id=identifier,
                base_cell_id=base["cell_id"],
                history_mode=mode,
                encounter_action=int(mode == "forced_adapt"),
                runtime_seed=args.seed,
                decision_time_s=args.entry_time_s,
                output=str(args.out.resolve() / "rollouts" / identifier),
            )
            prefixes = (
                "manager_env._target_=",
                "manager_env.recorders.trajectory._target_=",
                "manager_env.config.closed_loop_mode=",
                "manager_env.config.encounter_action=",
                "manager_env.config.forced_entry_time_s=",
                "manager_env.config.history_max_age_s=",
                "manager_env.config.history_max_frames=",
                "manager_env.config.observation_delay_s=",
                "manager_env.config.learned_policy_path=",
                "manager_env.config.learned_policy_sha256=",
                "seed=",
            )
            cell["hydra_overrides"] = [
                value
                for value in base["hydra_overrides"]
                if not value.lstrip("+").startswith(prefixes)
            ] + [
                f"++manager_env._target_={RUNTIME}.ClosedLoopEnvCfg",
                f"++manager_env.recorders.trajectory._target_={RUNTIME}.ClosedLoopRecorderCfg",
                f"++manager_env.config.closed_loop_mode={'forced' if mode.startswith('forced_') else mode}",
                f"++manager_env.config.encounter_action={int(mode == 'forced_adapt')}",
                f"++manager_env.config.forced_entry_time_s={args.entry_time_s}",
                f"++manager_env.config.history_max_age_s={args.history_age_s}",
                "++manager_env.config.history_max_frames=26",
                f"++manager_env.config.observation_delay_s={args.sensor_delay_s}",
                f"++seed={args.seed}",
            ]
            if mode == "learned":
                cell["hydra_overrides"] += [
                    f"++manager_env.config.learned_policy_path={policy['path']}",
                    f"++manager_env.config.learned_policy_sha256={policy['sha256']}",
                ]
            cells.append(cell)
    dependencies = [Path(__file__), ROOT / "scripts/research/run_kimodo_sonic_rollout.sh"]
    module = ROOT / "gear_sonic/dataset_generation/hallucination"
    dependencies.extend(
        module / name
        for name in (
            "motion2scene_closed_loop_execution.py",
            "motion2scene_overhang_eval_execution.py",
            "motion2scene_overhang_execution.py",
            "motion2scene_reactive_execution.py",
            "motion2scene_reactive_query_execution.py",
            "motion2scene_beam_execution.py",
            "motion2scene_observation_history.py",
            "motion2scene_closed_loop_policy.py",
            "motion2scene_action_contract.py",
            "motion2scene_observation_delay.py",
            "motion2scene_ray_observer.py",
            "motion2scene_overhang_observer.py",
            "motion2scene_reference_bank.py",
            "motion2scene_reset_capture.py",
            "motion2scene_passage.py",
        )
    )
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "manifest.json",
        {
            "schema": "motion2scene_sensor_history_acquisition_v1",
            "purpose": "development-only sensor-history and physical teacher acquisition",
            "template": artifact(args.template),
            "implementation": parent["implementation"],
            "dependencies": [artifact(path) for path in sorted(set(dependencies))],
            "policy": policy,
            "cells": cells,
            "limits": {"timeout_s": 375, "minimum_free_gpu_mib": 7500, "serial": True},
        },
    )
    print(json.dumps({"prepared": len(cells), "out": str(args.out)}), flush=True)


def execute(out):
    from hallucination.run_approved_manifest import free_gpu_mib, run_with_process_group

    manifest = json.loads((out / "manifest.json").read_text())
    for dependency in manifest["dependencies"]:
        checked(Path(dependency["path"]), dependency["sha256"])
    checkpoint = manifest["implementation"]["checkpoint"]
    checked(Path(checkpoint["path"]), checkpoint["sha256"])
    for cell in manifest["cells"]:
        folder = Path(cell["output"])
        if (folder / "history_row.json").exists():
            continue
        if folder.exists():
            raise RuntimeError(f"unfinished attempt retained at {folder}; prepare a new manifest")
        for key in ("motion", "alternate_motion", "scene"):
            checked(Path(cell[key]["path"]), cell[key]["sha256"])
        if free_gpu_mib() < manifest["limits"]["minimum_free_gpu_mib"]:
            raise RuntimeError("insufficient free GPU memory; remaining cells stay pending")
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
        started = time.monotonic()
        status = run_with_process_group(command, timeout=manifest["limits"]["timeout_s"])
        elapsed = time.monotonic() - started
        write_new(
            folder / "attempt.json",
            {"exit_status": status, "wall_seconds": elapsed, "command": command},
        )
        if status:
            raise RuntimeError(f"physical runtime failed; attempt retained at {folder}")
        row = analyze_cell(cell)
        row["wall_seconds"] = elapsed
        write_new(folder / "history_row.json", row)
        print(
            json.dumps(
                {"cell": cell["cell_id"], "pass": row["pass"], "sensor": row["sensor_summary"]}
            ),
            flush=True,
        )
    analyze(out)


def analyze_cell(cell):
    folder = Path(cell["output"]) / "trajectories"
    trajectories = list(folder.glob("*.trajectory.pkl"))
    if len(trajectories) != 1:
        raise ValueError("requires exactly one trajectory")
    payload = load_reset_capture(trajectories[0])
    with np.load(folder / "beam_contacts.npz") as archive:
        sampled = archive["force_w"]
    with np.load(folder / "physics_beam_contacts.npz") as archive:
        _, force, sync = physics_windows(archive, sampled)
        physics_steps = archive["physics_steps"].copy()
    passage = score_passage(payload, force, cell["beam"])
    interface = json.loads((folder / "reactive_interface.json").read_text())
    expected_mode = "forced" if cell["history_mode"].startswith("forced_") else cell["history_mode"]
    if (
        interface["mode"] != expected_mode
        or interface.get("feature_schema") != "motion2scene_history_linear_v1"
    ):
        raise ValueError("wrong runtime mode or sensor feature schema")
    observations = interface["observations"][: passage["first_episode_frames"]]
    if not observations or any(len(row["measurements"]) != 65 for row in observations):
        raise ValueError("missing sensor fan capture")
    if any(
        not np.isfinite(row["features"]).all() or len(row["features"]) != 100
        for row in observations
    ):
        raise ValueError("invalid pre-decision features")
    if sync > 1e-6:
        raise ValueError("contact clocks do not align")
    upper_indices = [
        i for i, name in enumerate(interface["feature_names"]) if name.endswith("upper_hit")
    ]
    upper_visible = [
        row for row in observations if any(row["features"][index] for index in upper_indices)
    ]
    first_episode_s = passage["first_episode_frames"] / 50
    return {
        "cell_id": cell["cell_id"],
        "base_cell_id": cell["base_cell_id"],
        "mode": cell["history_mode"],
        "decision_time_s": cell["decision_time_s"],
        "source": cell["generation_seed"],
        "beam": cell["beam"],
        "condition": cell["condition"],
        **passage,
        "sensor_summary": {
            "recorded_first_episode_frames": len(observations),
            "frames_with_floor": sum(row["floor_observed_cells"] > 0 for row in observations),
            "frames_with_ceiling": sum(row["ceiling_observed_cells"] > 0 for row in observations),
            "rays_with_ideal_query_normal": sum(
                ray["hit_normal_w"] is not None
                for row in observations
                for ray in row["measurements"]
            ),
            "minimum_delivered_age_s": min(row["observation_age_s"] for row in observations),
            "maximum_delivered_age_s": max(row["observation_age_s"] for row in observations),
            "first_delivered_upper_hit_phase_s": (
                upper_visible[0]["time_s"] if upper_visible else None
            ),
            "upper_visible_by_last_entry": any(row["time_s"] <= 0.4 for row in upper_visible),
            "causal_capture": all(
                row["delivered_capture_elapsed_s"] is None
                or row["delivered_capture_elapsed_s"] <= row["capture_elapsed_s"]
                for row in observations
            ),
        },
        "switches": interface["switches"],
        "costs": {
            "passage_time_s": (
                (passage["passage_finish_frame_exclusive"] - 1) / 50 if passage["pass"] else None
            ),
            "positive_mechanical_work_j": None,
            "work_measurement": "runtime uses implicit actuator PD estimates; actual work unavailable",
            "switch_count": len(interface["switches"]),
            "scored_horizon_s": first_episode_s,
        },
        "acquisition": {"physics_steps": len(physics_steps), "evaluated_branches": 1},
        "trajectory": artifact(trajectories[0]),
        "sensor": artifact(folder / "reactive_interface.json"),
        "features": artifact(folder / "closed_loop_features.npz"),
        "bank": artifact(folder / "loaded_reference_bank.npz"),
        "scope": "one-encounter development execution; no unseen-course or navigation claim",
    }


def analyze(out):
    manifest = json.loads((out / "manifest.json").read_text())
    rows = [
        json.loads((Path(cell["output"]) / "history_row.json").read_text())
        for cell in manifest["cells"]
    ]
    teachers = []
    for base_id in dict.fromkeys(row["base_cell_id"] for row in rows):
        grouped = {row["mode"]: row for row in rows if row["base_cell_id"] == base_id}
        if not {"forced_walk", "forced_adapt"}.issubset(grouped):
            continue
        pair = [grouped[mode] for mode in ("forced_walk", "forced_adapt")]
        phase = pair[0]["decision_time_s"]
        payloads = [
            load_reset_capture(
                checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
            )
            for row in pair
        ]
        prefix = paired_prefix(*payloads, phase)
        packets = []
        for row in pair:
            document = json.loads(
                checked(Path(row["sensor"]["path"]), row["sensor"]["sha256"]).read_text()
            )
            choices = [
                packet
                for packet in document["observations"][: row["first_episode_frames"]]
                if abs(packet["time_s"] - phase) < 1e-8
            ]
            if len(choices) != 1:
                raise ValueError("missing or duplicate first-episode teacher decision tick")
            packets.append(choices[0])
        same_state = packets[0]["state"] == packets[1]["state"] and all(
            packets[0][key] == packets[1][key]
            for key in ("root_pos_w", "root_quat_w", "active_before")
        )
        same_features = packets[0]["features"] == packets[1]["features"]
        eligible = prefix["exact_match"] and same_state and same_features
        admitted = [
            bool(
                eligible
                and packet["transition"]["allowed"]
                and packet["transition"]["requested"] == action
                and packet["active"] == action
                and all(
                    packet["transition"][key]
                    for key in ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
                )
            )
            for action, packet in enumerate(packets)
        ]
        feasible = [i for i, row in enumerate(pair) if row["pass"]]
        selected = (
            min(feasible, key=lambda i: (pair[i]["costs"]["passage_time_s"], i))
            if all(admitted) and feasible
            else None
        )
        cost_tie = (
            len(feasible) == 2
            and pair[0]["costs"]["passage_time_s"] == pair[1]["costs"]["passage_time_s"]
        )
        if cost_tie:
            selected = None
        teachers.append(
            {
                "base_cell_id": base_id,
                "phase_s": phase,
                "paired_prefix": prefix,
                "decision_state_exact_match": same_state,
                "decision_features_exact_match": same_features,
                "admitted": admitted,
                "eligible_for_imitation": all(admitted) and bool(feasible) and not cost_tie,
                "teacher_action": selected,
                "pass_labels": [row["pass"] for row in pair],
                "passage_times_s": [row["costs"]["passage_time_s"] for row in pair],
                "features": packets[0]["features"],
                "branch_cell_ids": [row["cell_id"] for row in pair],
                "ranking": (
                    "passage first, then measured passage time; "
                    "both-fail and exact cost ties have no target"
                ),
            }
        )
    result = {
        "schema": "motion2scene_sensor_history_acquisition_v1",
        "manifest": artifact(out / "manifest.json"),
        "rows": rows,
        "teachers": teachers,
        "physics_steps": sum(row["acquisition"]["physics_steps"] for row in rows),
        "wall_seconds": sum(row["wall_seconds"] for row in rows),
        "claim_limit": "new development sensor corpus; two tested motion branches per matched teacher state",
    }
    write_new(out / "result.json", result)
    print(
        json.dumps(
            {
                "result": str(out / "result.json"),
                "teacher_examples": sum(t["eligible_for_imitation"] for t in teachers),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--cells", nargs="+")
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=["forced_walk", "forced_adapt"]
    )
    parser.add_argument("--entry-time-s", type=float, default=0.3)
    parser.add_argument("--history-age-s", type=float, default=0.5)
    parser.add_argument("--sensor-delay-s", type=float, default=0)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--seed", type=int, default=8731)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.template is None or not args.cells:
            parser.error("prepare requires --template and --cells")
        prepare(args)
    elif args.mode == "run":
        execute(args.out)
    else:
        analyze(args.out)
