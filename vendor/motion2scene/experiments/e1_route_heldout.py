"""Prepare and analyze the registered eight-carrier route-retention validation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from motion2scene.motion.route_retention import assess_route_retention, compare_routes
from motion2scene.motion.route_semantics import classify_route

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path: Path, expected: str) -> Path:
    if sha(path) != expected:
        raise ValueError(f"artifact hash mismatch: {path}")
    return path


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(design: dict, design_path: Path, manifest_path: Path) -> None:
    from gear_sonic.dataset_generation.kimodo_motion_adapter import (
        load_kimodo_qpos_csv,
        qpos_to_sonic_motion_entry,
        save_sonic_motion_file,
    )
    from gear_sonic.dataset_generation.reference_gate import screen_reference

    source = Path(design["source_repository"]["path"])
    generation = design["generation"]
    data = Path(generation["output_root"])
    census_path = data / "reference_census.json"
    if census_path.exists() or manifest_path.exists() or (data / "sonic").exists():
        raise FileExistsError("preparation outputs already exist; refusing to overwrite")
    files = sorted((data / "motions").glob("*.csv"))
    if len(files) != generation["registered_reference_count"]:
        raise ValueError("generated reference count differs from registration")
    rows = []
    seen = set()
    template = json.loads(
        (ROOT / "experiments/registrations/E1_CONTROLLED_DUCK_Q3_V1_MANIFEST.json").read_text()
    )
    cells = []
    for path in files:
        sidecar_path = path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text())
        seed = sidecar["generation_seed"]
        if seed not in generation["seeds"] or seed in seen:
            raise ValueError("unexpected or duplicate generation seed")
        seen.add(seed)
        if (
            sidecar["prompt_design_version"] != generation["prompt_design_version"]
            or sidecar["num_frames"] != 120
            or sidecar["denoising_steps"] != generation["denoising_steps"]
        ):
            raise ValueError("generation settings differ from registration")
        qpos = load_kimodo_qpos_csv(path)
        gate = screen_reference(qpos, path.stem, "walk")
        route = classify_route(qpos[:, :2], "straight", fps=30.0)
        admitted = (
            gate.embodiment_feasible
            and gate.self_collision_free
            and route.validity_class == "valid_straight"
        )
        cell_id = f"m2s_route_heldout_seed_{seed}"
        row = {
            "generation_seed": seed,
            "cell_id": cell_id,
            "csv": str(path),
            "csv_sha256": sha(path),
            "sidecar_sha256": sha(sidecar_path),
            "q0": "passed" if gate.embodiment_feasible else "failed",
            "q1": "passed" if gate.self_collision_free else "failed",
            "gate_notes": list(gate.notes),
            "reference_route": route.to_dict(),
            "admitted": admitted,
        }
        if admitted:
            motion = data / "sonic" / f"seed_{seed}.pkl"
            save_sonic_motion_file(
                motion, motion_key=cell_id, motion_entry=qpos_to_sonic_motion_entry(qpos)
            )
            provenance = motion.with_suffix(".pkl.manifest.json")
            write_new(provenance, {
                "schema_version": "motion2scene_sonic_conversion_provenance_v1",
                "scene_start_xyz": [0.0, 0.0, 0.0],
                "scene_yaw": 0.0,
                "canonicalize_horizontal_origin": True,
                "source_fps": 30.0,
                "input": {"path": str(path), "sha256": sha(path)},
                "output": {"path": str(motion), "sha256": sha(motion)},
                "converter_source_sha256": sha(
                    source / "gear_sonic/dataset_generation/kimodo_motion_adapter.py"
                ),
                "design_sha256": sha(design_path),
            })
            cell = copy.deepcopy(template["cells"][0])
            cell.update({
                "cell_id": cell_id,
                "base_carrier_id": f"heldout_seed_{seed}",
                "ladder_group_id": f"heldout_seed_{seed}",
                "runtime_seed": design["q3"]["runtime_seed"],
                "hydra_overrides": [f"++seed={design['q3']['runtime_seed']}"],
                "output": str(data / "q3" / "rollouts" / cell_id),
                "motion": {
                    "path": str(motion), "sha256": sha(motion),
                    "conversion_provenance": str(provenance),
                    "conversion_provenance_sha256": sha(provenance),
                    "scene_start_xyz": [0.0, 0.0, 0.0],
                },
            })
            cells.append(cell)
        rows.append(row)
        print(f"seed={seed}: Q0={row['q0']} Q1={row['q1']} route={route.validity_class}")
    write_new(census_path, {
        "schema_version": "motion2scene_route_heldout_reference_census_v1",
        "design_sha256": sha(design_path),
        "driver_sha256": sha(Path(__file__)),
        "registered": len(generation["seeds"]), "generated": len(rows),
        "admitted": len(cells), "rows": rows,
    })
    template.update({
        "experiment": design["experiment"],
        "purpose": design["purpose"],
        "cells": cells,
        "eligibility": {
            "reference_gate": {"path": str(census_path), "sha256": sha(census_path)},
            "rule": "all registered Q0/Q1-valid straight neutral references",
        },
        "registered_predictions": {
            "path": str(ROOT / design["registered_predictions"]["path"]),
            "sha256": design["registered_predictions"]["sha256"],
        },
    })
    template["execution_policy"]["timing_override"] = (
        "Fresh neutral seeds and relative route gate frozen before generation; "
        "one rollout per admitted carrier at seed 7700, serial trajectory-only."
    )
    template["execution_policy"]["cost_ceiling"].update({
        "rollouts": len(cells), "gpu_hours_contended": len(cells) * 375 / 3600,
    })
    template["stop_conditions"] = [
        "refuse artifact hash changes", "yield below 7500 MiB free VRAM",
        "stop on infrastructure failure", "retain all scientific rejections without retry",
        "neutral-only route validation cannot admit adapted motions to Q4",
    ]
    write_new(manifest_path, template)
    print(f"Prepared {len(cells)}/{len(rows)} registered neutral controls")


def analyze(design: dict, design_path: Path, manifest_path: Path) -> None:
    from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

    data = Path(design["generation"]["output_root"])
    manifest = json.loads(manifest_path.read_text())
    census_path = checked(
        data / "reference_census.json", manifest["eligibility"]["reference_gate"]["sha256"]
    )
    census = json.loads(census_path.read_text())
    run_path = data / "q3/run_record.json"
    run = json.loads(run_path.read_text())
    if run["manifest_sha256"] != sha(manifest_path):
        raise ValueError("run record manifest mismatch")
    rows = []
    for ref in census["rows"]:
        row = dict(ref)
        record = run["cells"].get(ref["cell_id"], {})
        row["execution_status"] = record.get("status", "not_measured")
        row["tracker_survived"] = None
        row["route_retained"] = None
        if record.get("status") == "completed":
            scientific = record["scientific"]
            artifacts = scientific["artifacts"]
            path = checked(Path(artifacts["trajectory"]), artifacts["trajectory_sha256"])
            with path.open("rb") as handle:
                payload, _ = best_evaluable_payload(pickle.load(handle))
            reference = np.loadtxt(checked(Path(ref["csv"]), ref["csv_sha256"]), delimiter=",")
            result = compare_routes(
                reference[:, :2], np.asarray(payload["root_pos_w"])[:, :2],
                expected_route="straight", reference_fps=30.0,
                achieved_fps=float(payload["fps"]),
            )
            decision = assess_route_retention(result)
            row.update({
                "tracker_survived": scientific["outcome"] == "accepted",
                "tracker_rejection_reasons": scientific["rejection_reasons"],
                "route_retained": decision.retained,
                "retention": result.to_dict(), "route_decision": decision.to_dict(),
                "trajectory_sha256": sha(path),
            })
        rows.append(row)
    survivors = [row for row in rows if row["tracker_survived"] is True]
    retained = [row for row in survivors if row["route_retained"] is True]
    rate = len(retained) / len(survivors) if survivors else None
    complete = run["status"] == "completed" and all(
        row["execution_status"] == "completed" for row in rows if row["admitted"]
    )
    output = {
        "schema_version": "motion2scene_route_heldout_result_v2",
        "run_status": run["status"], "analysis_complete": complete,
        "design_sha256": sha(design_path), "manifest_sha256": sha(manifest_path),
        "run_record_sha256": sha(run_path), "driver_sha256": sha(Path(__file__)),
        "registered": census["registered"], "generated": census["generated"],
        "reference_admitted": census["admitted"],
        "completed": sum(row["execution_status"] == "completed" for row in rows),
        "tracker_survivors": len(survivors), "retained_survivors": len(retained),
        "retention_rate_among_survivors": rate,
        "prediction_4_met": rate >= 0.8 if complete and rate is not None else None,
        "q4_admitted_ladders": 0,
        "rows": rows,
    }
    write_new(data / "q3/relative_retention_result_v2.json", output)
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "analyze"))
    parser.add_argument("--design", type=Path, default=ROOT / "configs/e1_route_retention_heldout_v1.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/registrations/E1_ROUTE_RETENTION_HELDOUT_V1_MANIFEST.json")
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    for field in ("analysis_implementation", "registered_predictions"):
        checked(ROOT / design[field]["path"], design[field]["sha256"])
    sys.path.insert(0, design["source_repository"]["path"])
    {"prepare": prepare, "analyze": analyze}[args.mode](design, args.design, args.manifest)


if __name__ == "__main__":
    main()
