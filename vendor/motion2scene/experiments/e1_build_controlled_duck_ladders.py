#!/usr/bin/env python3
"""Build same-carrier duck ladders from preregistered Q0/Q1-valid walk references.

The operator changes only adaptation severity while preserving each carrier's horizontal
route, root orientation, timing, and frame count. Outputs are reference-level candidates;
no ladder is admitted to Motion2Scene-DB until its rungs pass Q3/Q4 and retain semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from motion2scene.inverse import solve_overhead_interval
from motion2scene.motion.paired_semantics import (
    PairedSemanticPolicy,
    assess_paired_reduction,
)

STATIONS = np.linspace(0.0, 1.0, 101)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def status_rank(value: str) -> int:
    order = {
        "not_measured": -1,
        "absent": 0,
        "elicited": 1,
        "localized": 2,
        "route_aligned": 3,
        "controller_retained": 4,
    }
    return order[value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    source_repo = args.source_repo.resolve()
    sys.path.insert(0, str(source_repo))
    from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints
    from gear_sonic.dataset_generation.hallucination.motion_envelope import extract_envelope
    from gear_sonic.dataset_generation.kimodo_motion_adapter import (
        qpos_to_sonic_motion_entry,
        save_sonic_motion_file,
    )
    from gear_sonic.dataset_generation.local_adaptation import (
        _silhouette,
        active_frames,
        local_crouch,
    )
    from gear_sonic.dataset_generation.reference_gate import screen_reference
    from gear_sonic.dataset_generation.reference_payload import payload_from_reference

    parameters = registration["operator"]
    target_drops = tuple(float(value) for value in parameters["target_drops_m"])
    station = float(parameters["station_progress"])
    window = float(parameters["window_progress"])
    minimum_effect = float(registration["semantic_policy"]["minimum_effect_m"])
    safety_margin = float(registration["analytic_interval"]["target_safety_margin_m"])
    strike_margin = float(registration["analytic_interval"]["weaker_strike_margin_m"])
    mjcf = Path(registration["mjcf"]["path"])
    if sha256(mjcf) != registration["mjcf"]["sha256"].removeprefix("sha256:"):
        raise ValueError("MJCF hash differs from registration")

    args.out.mkdir(parents=True, exist_ok=True)
    ladder_rows: list[dict] = []
    for position, candidate in enumerate(registration["base_carriers"], start=1):
        source = Path(candidate["csv"])
        if sha256(source) != candidate["csv_sha256"].removeprefix("sha256:"):
            raise ValueError(f"base carrier hash differs from registration: {source}")
        nominal = np.loadtxt(source, delimiter=",")
        nominal_gate = screen_reference(
            nominal,
            candidate["motion_id"],
            "walk",
            mjcf_path=mjcf,
        )
        if not nominal_gate.worth_a_rollout:
            raise ValueError(f"registered base carrier no longer passes Q0/Q1: {source}")

        ladder_id = f"duck_seed_{int(candidate['generation_seed'])}"
        ladder_dir = args.out / ladder_id
        ladder_dir.mkdir(parents=True, exist_ok=True)
        levels: list[dict] = []
        qpos_by_label: dict[str, np.ndarray] = {}
        silhouette_by_label: dict[str, np.ndarray] = {}

        for level, target_drop in enumerate((0.0, *target_drops)):
            label = "neutral" if level == 0 else f"d{round(1000 * target_drop):03d}"
            if level == 0:
                qpos = nominal.copy()
                operator_report = None
            else:
                qpos, report = local_crouch(
                    nominal,
                    station,
                    target_drop_m=target_drop,
                    window=window,
                    mjcf_path=mjcf,
                )
                operator_report = asdict(report)
                if not report.root_path_preserved:
                    raise ValueError(f"operator changed the carrier route for {ladder_id}/{label}")

            csv_path = ladder_dir / f"{label}.csv"
            np.savetxt(csv_path, qpos, delimiter=",", fmt="%.10f")
            motion_path = ladder_dir / f"{label}.pkl"
            motion_key = f"{ladder_id}__{label}"
            save_sonic_motion_file(
                motion_path,
                motion_key=motion_key,
                motion_entry=qpos_to_sonic_motion_entry(qpos, source_fps=30.0),
            )

            gate = screen_reference(qpos, motion_key, "walk", mjcf_path=mjcf)
            payload = payload_from_reference(qpos, fps=30.0, mjcf_path=mjcf)
            tracks = extract_keypoints(payload)
            envelope = extract_envelope(tracks, motion_key, fractions=STATIONS)
            qpos_by_label[label] = qpos
            silhouette_by_label[label] = _silhouette(qpos, mjcf)
            level_row = {
                "ladder_level": level,
                "label": label,
                "target_drop_m": target_drop,
                "motion_key": motion_key,
                "csv": str(csv_path),
                "csv_sha256": f"sha256:{sha256(csv_path)}",
                "sonic_motion": str(motion_path),
                "sonic_motion_sha256": f"sha256:{sha256(motion_path)}",
                "q0_status": "passed" if gate.embodiment_feasible else "failed",
                "q1_status": "passed" if gate.self_collision_free else "failed",
                "operator_report": operator_report,
                "whole_body_top_m": envelope.up_m.tolist(),
                "semantic_assessment": {
                    "semantic_status": "not_measured",
                    "reason": "neutral ladder level",
                },
            }
            if level > 0:
                semantic = assess_paired_reduction(
                    STATIONS,
                    envelope.up_m,
                    np.asarray(levels[0]["whole_body_top_m"]),
                    route_valid=True,
                    policy=PairedSemanticPolicy(minimum_effect=minimum_effect),
                )
                level_row["semantic_assessment"] = semantic.to_dict()
            levels.append(level_row)

        reductions = [
            float(level["semantic_assessment"]["peak_reduction"])
            for level in levels[1:]
        ]
        ordered = all(second > first for first, second in pairwise(reductions))
        intervals = []
        for weaker, target in pairwise(levels):
            mask = active_frames(nominal, qpos_by_label[target["label"]])
            weaker_reach = float(silhouette_by_label[weaker["label"]][mask].max())
            target_reach = float(silhouette_by_label[target["label"]][mask].max())
            interval = solve_overhead_interval(
                target_motion_id=target["motion_key"],
                weaker_motion_id=weaker["motion_key"],
                target_reach_m=target_reach,
                weaker_reach_m=weaker_reach,
                safety_margin_m=safety_margin,
                strike_margin_m=strike_margin,
            )
            intervals.append(
                {
                    "weaker_level": weaker["ladder_level"],
                    "target_level": target["ladder_level"],
                    "active_frame_count": int(mask.sum()),
                    **interval.to_dict(),
                }
            )

        reference_qualified_levels = sum(
            level["q0_status"] == "passed"
            and level["q1_status"] == "passed"
            and (
                level["ladder_level"] == 0
                or status_rank(level["semantic_assessment"]["semantic_status"]) >= 2
            )
            for level in levels
        )
        ladder_rows.append(
            {
                "ladder_group_id": ladder_id,
                "base_carrier_id": candidate["motion_id"],
                "generation_seed": candidate["generation_seed"],
                "source_csv": str(source),
                "source_csv_sha256": candidate["csv_sha256"],
                "route": candidate["route"],
                "levels": levels,
                "monotonicity_status": "ordered" if ordered else "not_ordered",
                "reference_qualified_levels": reference_qualified_levels,
                "reference_ladder_candidate": ordered and reference_qualified_levels >= 3,
                "q3_q4_admitted": False,
                "critical_intervals": intervals,
            }
        )
        print(f"built {position}/{len(registration['base_carriers'])}: {ladder_id}")

    source_files = {
        "local_adaptation": source_repo / "gear_sonic/dataset_generation/local_adaptation.py",
        "motion_envelope": source_repo
        / "gear_sonic/dataset_generation/hallucination/motion_envelope.py",
        "reference_gate": source_repo / "gear_sonic/dataset_generation/reference_gate.py",
    }
    output = {
        "schema_version": "motion2scene_e1_controlled_duck_ladders_v1",
        "evidence_scope": "reference_only_q0_q1_s0_s3_and_analytic_intervals",
        "registration": str(args.registration.resolve()),
        "registration_sha256": f"sha256:{sha256(args.registration)}",
        "analysis_driver_sha256": f"sha256:{sha256(Path(__file__))}",
        "source_repo": str(source_repo),
        "source_commit": git_head(source_repo),
        "source_files": {
            name: {"path": str(path), "sha256": f"sha256:{sha256(path)}"}
            for name, path in source_files.items()
        },
        "attempted_ladder_groups": len(ladder_rows),
        "reference_ladder_candidates": sum(
            row["reference_ladder_candidate"] for row in ladder_rows
        ),
        "q3_q4_admitted_ladders": 0,
        "attempted_intervals": sum(len(row["critical_intervals"]) for row in ladder_rows),
        "nonempty_reference_intervals": sum(
            interval["nonempty"]
            for row in ladder_rows
            for interval in row["critical_intervals"]
        ),
        "dataset_eligible_intervals": 0,
        "ladders": ladder_rows,
    }
    manifest = args.out / "candidates.json"
    manifest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{output['reference_ladder_candidates']}/{output['attempted_ladder_groups']} "
        "reference ladder candidates; "
        f"{output['nonempty_reference_intervals']}/{output['attempted_intervals']} "
        f"nonempty reference intervals; 0 dataset eligible -> {manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
