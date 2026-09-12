#!/usr/bin/env python3
"""Prepare a hash-bound, serial Q3 manifest from controlled duck candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    source_repo = args.source_repo.resolve()
    driver = source_repo / "scripts/research/run_kimodo_sonic_rollout.sh"
    manifest_driver = source_repo / "scripts/research/hallucination/run_approved_manifest.py"
    checkpoint = source_repo / "sonic_release/last.pt"
    scene = source_repo / "gear_sonic/data/assets/scenes/g1_counterfactual/screen_empty.usda"
    cells = []
    for ladder in candidates["ladders"]:
        if any(level["q0_status"] != "passed" or level["q1_status"] != "passed" for level in ladder["levels"][:3]):
            raise ValueError(f"first three levels are not Q0/Q1 qualified: {ladder['ladder_group_id']}")
        neutral_id = f"m2s_e1q3_{ladder['ladder_group_id']}__neutral"
        for level in ladder["levels"][:3]:
            label = level["label"]
            cell_id = f"m2s_e1q3_{ladder['ladder_group_id']}__{label}"
            motion_path = Path(level["sonic_motion"])
            provenance = motion_path.with_suffix(motion_path.suffix + ".manifest.json")
            task = {
                "neutral": "A humanoid walks straight at a steady pace",
                "d040": "A humanoid walks straight and makes a shallow crouch near mid-route",
                "d055": "A humanoid walks straight and makes a medium crouch near mid-route",
            }[label]
            cell = {
                "cell_id": cell_id,
                "base_carrier_id": ladder["base_carrier_id"],
                "ladder_group_id": ladder["ladder_group_id"],
                "ladder_level": level["ladder_level"],
                "body_mode": "walk" if label == "neutral" else "controlled_crouch",
                "turn": "straight",
                "runtime_seed": 7600,
                "hydra_overrides": ["++seed=7600"],
                "task_prompt": task,
                "scene": {
                    "scene_id": "screen_empty",
                    "path": "gear_sonic/data/assets/scenes/g1_counterfactual/screen_empty.usda",
                    "sha256": sha256(scene),
                },
                "motion": {
                    "path": str(motion_path),
                    "sha256": level["sonic_motion_sha256"],
                    "conversion_provenance": str(provenance),
                    "conversion_provenance_sha256": sha256(provenance),
                    "scene_start_xyz": [0.0, 0.0, 0.0],
                },
                "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                "expectation": {
                    "status": "registered_measurement_without_cellwise_directional_prediction"
                },
                "verdict_policy": "reference_trackability",
                "output": str(args.rollout_root / "rollouts" / cell_id),
            }
            if label != "neutral":
                cell["depends_on_acceptance_of"] = neutral_id
            cells.append(cell)

    if len(cells) != 24:
        raise ValueError(f"registered design requires 24 cells, observed {len(cells)}")
    payload = {
        "schema_version": "motion2scene_physics_manifest_v1",
        "experiment": "M2S-E1-controlled-duck-q3-v1",
        "purpose": (
            "Single-seed obstacle-absent Q3 for neutral, 40 mm, and 55 mm levels across "
            "eight same-carrier controlled duck groups."
        ),
        "authorization": {
            "status": "self_approved",
            "authority": "docs/hallucination/GOVERNANCE.md",
            "claim5_isolation_verified": True,
            "note": (
                "Fresh Motion2Scene lineage authorized by the user; no historical protected "
                "claim inputs are modified."
            ),
        },
        "execution_policy": {
            "not_authorized": False,
            "timing_override": (
                "Candidates, three levels, one common physics seed, serial trajectory-only "
                "capture, 7500 MiB launch floor, dependencies and predictions were frozen "
                "before any cell launch."
            ),
            "runtime": {
                "free_gpu_mib_required": 7500,
                "hang_timeout_seconds": 1800,
                "serial_execution": True,
                "render_ego": False,
                "success_requires": [
                    "SONIC_EVAL_SUCCESS marker",
                    "success_manifest.json",
                    "exactly one trajectory pickle",
                    "recorded fps=50",
                ],
            },
            "cost_ceiling": {
                "rollouts": 24,
                "seconds_per_rollout_contended": 375,
                "gpu_hours_contended": 2.5,
                "daily_envelope_gpu_hours": 8.0,
            },
        },
        "implementation": {
            "source_repository": str(source_repo),
            "source_commit": candidates["source_commit"],
            "rollout_driver": {"path": str(driver), "sha256": sha256(driver)},
            "manifest_driver": {
                "path": str(manifest_driver),
                "sha256": sha256(manifest_driver),
            },
            "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
            "python": str(source_repo / ".venv_isaaclab/bin/python"),
        },
        "eligibility": {
            "reference_gate": {
                "path": str(args.candidates.resolve()),
                "sha256": sha256(args.candidates),
            },
            "rule": (
                "all eight Q0/Q1-valid straight neutral carriers; neutral/d040/d055 are S3 "
                "and strictly ordered within every selected prefix"
            ),
        },
        "registered_predictions": {
            "path": str(args.predictions.resolve()),
            "sha256": sha256(args.predictions),
        },
        "stop_conditions": [
            "refuse launch if any pinned artifact hash differs",
            "yield without spending if free GPU memory is below 7500 MiB",
            "execute serially and stop on the first infrastructure failure",
            "retain scientific rejections without retry",
            "do not promote tracker survival to S4 without achieved-state analysis",
            "do not promote any result beyond Q3 single-seed evidence",
        ],
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(cells)} Q3 cells -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
