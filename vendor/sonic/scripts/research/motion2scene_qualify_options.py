#!/usr/bin/env python3
"""Prepare, execute and measure a finite SONIC option qualification sweep.

Uses the installed phase-aligned interface without changing tracker weights. Every
branch and failure is retained. Empty-scene qualification does not certify beams.
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
from motion2scene_reactive_interface import physics_windows  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import (  # noqa: E402
    paired_prefix,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_motion_options import (  # noqa: E402
    MotionOption,
    mechanical_work,
    qualification_predicates,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import (  # noqa: E402
    score_passage,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)

DATA = ROOT.parent / "research-data/groot-wbc"
TEMPLATE = DATA / "m2s-development-transition-bank-v2/manifest.json"


def prepare(args):
    parent = json.loads(args.template.read_text())
    options = [MotionOption("walk", "neutral")]
    for specification in args.options:
        reference, entry = specification.split(":")
        options.append(
            MotionOption(f"{reference}_t{round(float(entry) * 100):03d}", reference, float(entry))
        )
    if len({o.option_id for o in options}) != len(options):
        raise ValueError("duplicate options")
    cells = []
    for source in args.sources:
        candidates = [
            c
            for c in parent["cells"]
            if c["generation_seed"] == source
            and (
                c["cell_id"] == args.template_cell
                if args.template_cell
                else c["condition"] == "absent"
            )
        ]
        if not candidates:
            raise ValueError(f"no template cell for source {source}")
        base = candidates[0]
        for option in options:
            cell = copy.deepcopy(base)
            alternate_path = Path(base["alternate_motion"]["path"]).with_name(
                f"{option.reference if option.action else 'd040'}.pkl"
            )
            provenance_path = alternate_path.with_suffix(".pkl.manifest.json")
            provenance = json.loads(provenance_path.read_text())
            checked(alternate_path, provenance["output"]["sha256"])
            checked(Path(provenance["input"]["path"]), provenance["input"]["sha256"])
            neutral = next(
                iter(
                    joblib.load(
                        checked(Path(base["motion"]["path"]), base["motion"]["sha256"])
                    ).values()
                )
            )
            alternate = next(iter(joblib.load(alternate_path).values()))
            if float(neutral["fps"]) != float(alternate["fps"]) or len(neutral["dof"]) != len(
                alternate["dof"]
            ):
                raise ValueError("installed runtime requires equal duration reference libraries")
            cell_id = f"source{source}_{option.option_id}"
            cell.update(
                cell_id=cell_id,
                option=option.as_dict(),
                generation_seed=source,
                runtime_seed=args.seed,
                encounter_action=option.action,
                decision_time_s=option.entry_time_s,
                output=str(args.out.resolve() / "rollouts" / cell_id),
                alternate_motion={
                    **artifact(alternate_path),
                    "conversion_provenance": artifact(provenance_path),
                    "reference": provenance["input"],
                },
            )
            prefixes = (
                "++seed=",
                "++manager_env.config.encounter_action=",
                "++manager_env.config.decision_time_s=",
                "++manager_env.config.reactive_alternate_path=",
                "++manager_env.recorders.trajectory._target_=",
                "++manager_env._target_=",
                "++manager_env.config.learned_policy_path=",
                "++manager_env.config.learned_policy_sha256=",
            )
            cell["hydra_overrides"] = [
                v for v in base["hydra_overrides"] if not v.startswith(prefixes)
            ] + [
                f"++seed={args.seed}",
                f"++manager_env.config.encounter_action={option.action}",
                f"++manager_env.config.decision_time_s={option.entry_time_s}",
                f"++manager_env.config.reactive_alternate_path={alternate_path}",
                "++manager_env._target_=gear_sonic.dataset_generation.hallucination.motion2scene_comparison_execution.ComparisonEnvCfg",
                "++manager_env.recorders.trajectory._target_=gear_sonic.dataset_generation.hallucination.motion2scene_option_execution.OptionRecorderCfg",
            ]
            cells.append(cell)
    args.out.mkdir(parents=True, exist_ok=False)
    code = closure(
        [
            Path(__file__),
            ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_option_execution.py",
            ROOT
            / "gear_sonic/dataset_generation/hallucination/motion2scene_comparison_execution.py",
        ]
    ) | {ROOT / "scripts/research/run_kimodo_sonic_rollout.sh"}
    dependencies = []
    for path in sorted(code):
        snapshot = args.out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        dependencies.append({**artifact(path), "snapshot": artifact(snapshot)})
    write_new(
        args.out / "manifest.json",
        {
            "schema": "motion2scene_option_qualification_v1",
            "purpose": "development option qualification; not held-out traversal evidence",
            "template": artifact(args.template),
            "implementation": parent["implementation"],
            "dependencies": dependencies,
            "options": [o.as_dict() for o in options],
            "cells": cells,
            "limits": {"timeout_s": 375, "minimum_free_gpu_mib": 7500, "serial": True},
            "unsupported": [
                "variable return request",
                "speed scaling",
                "motion looping",
                "route steering",
                "stop",
            ],
        },
    )
    print(json.dumps({"prepared": len(cells), "out": str(args.out)}), flush=True)


def execute(out):
    from hallucination.run_approved_manifest import free_gpu_mib, run_with_process_group

    manifest = json.loads((out / "manifest.json").read_text())
    for ref in manifest["dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    checkpoint = manifest["implementation"]["checkpoint"]
    checked(Path(checkpoint["path"]), checkpoint["sha256"])
    for cell in manifest["cells"]:
        folder = Path(cell["output"])
        if (folder / "qualification_row.json").exists():
            continue
        if folder.exists():
            raise RuntimeError(f"unfinished attempt retained at {folder}; no implicit retry")
        for ref in (cell["motion"], cell["alternate_motion"], cell["scene"]):
            checked(Path(ref["path"]), ref["sha256"])
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
            raise RuntimeError(f"infrastructure failure retained at {folder}")
        row = analyze_cell(cell)
        row["wall_seconds"] = elapsed
        write_new(folder / "qualification_row.json", row)
        print(
            json.dumps(
                {
                    "cell": cell["cell_id"],
                    "qualified": row["qualified"],
                    "pass": row["pass"],
                    "predicates": row["predicates"],
                }
            ),
            flush=True,
        )
    analyze(out)


def analyze_cell(cell):
    folder = Path(cell["output"]) / "trajectories"
    trajectories = list(folder.glob("*.trajectory.pkl"))
    if len(trajectories) != 1:
        raise ValueError("exactly one complete trajectory required")
    payload = load_reset_capture(trajectories[0])
    with np.load(folder / "beam_contacts.npz") as f:
        sampled = f["force_w"]
    with np.load(folder / "physics_beam_contacts.npz") as f:
        _, force, sync = physics_windows(f, sampled)
        control_steps = f["control_steps"].copy()
        physics_steps = f["physics_steps"].copy()
    passage = score_passage(payload, force, cell["beam"])
    interface = json.loads((folder / "reactive_interface.json").read_text())
    option = MotionOption(**{k: v for k, v in cell["option"].items() if k != "action"})
    predicates = qualification_predicates(option, interface, passage)
    predicates["contact_synchronization"] = sync <= 1e-6
    with np.load(folder / "option_mechanical_work.npz") as f:
        if not np.array_equal(f["physics_steps"], physics_steps):
            raise ValueError("work/contact physics clocks differ")
        # Report the same first-episode horizon for failures, passage horizon for successes.
        horizon = passage["passage_finish_frame_exclusive"] or passage["first_episode_frames"]
        keep = (f["physics_steps"] >= control_steps[0] - 3) & (
            f["physics_steps"] <= control_steps[horizon - 1]
        )
        if bool(f["measured_actuator_effort_available"]):
            work = mechanical_work(
                f["estimated_torque_nm"][keep], f["velocity_rad_s"][keep], float(f["physics_dt_s"])
            )
        else:
            work = {
                "positive_mechanical_work_j": None,
                "absolute_mechanical_work_j": None,
                "net_mechanical_work_j": None,
                "measurement": "unavailable: implicit PhysX actuator effort is only estimated by IsaacLab",
            }
        physical_steps = len(physics_steps)
    return {
        "cell_id": cell["cell_id"],
        "source": cell["generation_seed"],
        "option": option.as_dict(),
        "condition": cell["condition"],
        "beam": cell["beam"],
        "physics_seed": cell["runtime_seed"],
        **passage,
        "predicates": predicates,
        "qualified": all(predicates.values()),
        "costs": {
            **work,
            "passage_time_s": None if not passage["pass"] else (horizon - 1) / 50,
            "switch_count": len(interface["switches"]),
            "scored_horizon_s": horizon / 50,
        },
        "acquisition": {
            "physics_steps": physical_steps,
            "recorded_control_steps": len(payload["motion_time_s"]),
            "evaluated_branches": 1,
        },
        "trajectory": artifact(trajectories[0]),
        "sensor": artifact(folder / "reactive_interface.json"),
        "decision": artifact(folder / "decision_capture.json"),
        "work": artifact(folder / "option_mechanical_work.npz"),
        "bank": artifact(folder / "loaded_reference_bank.npz"),
        "inventory": artifact(folder / "native_collision_inventory.json"),
        "scope": "source/seed/scene-specific execution; empty qualification is not obstacle robustness",
    }


def analyze(out):
    manifest = json.loads((out / "manifest.json").read_text())
    rows = [
        json.loads((Path(c["output"]) / "qualification_row.json").read_text())
        for c in manifest["cells"]
    ]
    prefixes = []
    for row in rows:
        baseline = next(
            r for r in rows if r["source"] == row["source"] and r["option"]["action"] == 0
        )
        prefix = paired_prefix(
            load_reset_capture(
                checked(Path(baseline["trajectory"]["path"]), baseline["trajectory"]["sha256"])
            ),
            load_reset_capture(
                checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
            ),
            row["option"]["entry_time_s"],
        )
        prefixes.append({"cell_id": row["cell_id"], **prefix})
        row["predicates"]["matched_approach_prefix"] = prefix["exact_match"]
        row["qualified"] = all(row["predicates"].values())
    write_new(
        out / "result.json",
        {
            "schema": "motion2scene_option_qualification_v1",
            "manifest": artifact(out / "manifest.json"),
            "rows": rows,
            "paired_prefixes": prefixes,
            "assigned": len(rows),
            "qualified": sum(r["qualified"] for r in rows),
            "physics_steps": sum(r["acquisition"]["physics_steps"] for r in rows),
            "wall_seconds": sum(r["wall_seconds"] for r in rows),
            "claim_limit": "finite development qualification; no held-out passage improvement established",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--template-cell")
    parser.add_argument("--sources", type=int, nargs="+", default=[41002])
    parser.add_argument(
        "--options",
        nargs="+",
        default=["d040:0.2", "d040:0.3", "d040:0.4", "d055:0.3", "d070:0.3", "d085:0.3"],
    )
    parser.add_argument("--seed", type=int, default=8731)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "run":
        execute(args.out)
    else:
        analyze(args.out)
