"""Construct matched duck references using one common time map per carrier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from e1_route_heldout import checked, sha, write_new

from motion2scene.motion.paired_semantics import PairedSemanticPolicy, assess_paired_reduction
from motion2scene.motion.route_semantics import classify_route


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.registration.read_text())
    source = Path(design["source_repository"])
    sys.path.insert(0, str(source))
    from gear_sonic.dataset_generation.deployable_retiming import (
        departure_profile,
        resample_clip,
        time_map,
    )
    from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints
    from gear_sonic.dataset_generation.hallucination.motion_envelope import extract_envelope
    from gear_sonic.dataset_generation.kimodo_motion_adapter import (
        qpos_to_sonic_motion_entry,
        save_sonic_motion_file,
    )
    from gear_sonic.dataset_generation.reference_gate import screen_reference
    from gear_sonic.dataset_generation.reference_payload import payload_from_reference

    candidates = json.loads(checked(
        Path(design["source_candidates"]), design["source_candidates_sha256"]
    ).read_text())
    output = Path(design["output"])
    if output.exists():
        raise FileExistsError("refusing to overwrite shared-clock artifacts")
    stations = np.linspace(0.0, 1.0, 101)
    event = (stations >= 0.45) & (stations <= 0.55)
    ladders = []
    for ladder in candidates["ladders"]:
        originals = {level["label"]: level for level in ladder["levels"]}
        clips = {
            label: np.loadtxt(checked(
                Path(originals[label]["csv"]), originals[label]["csv_sha256"]
            ), delimiter=",") for label in design["levels"]
        }
        activity = departure_profile(clips["neutral"], clips["d085"])
        tau = time_map(1.0 - (1.0 - design["common_time_map"]["minimum_pace_ratio"]) * activity)
        warped = {label: resample_clip(clip, tau) for label, clip in clips.items()}
        nominal = warped["neutral"]
        root_matched = all(
            np.array_equal(clip[:, :2], nominal[:, :2])
            and np.array_equal(clip[:, 3:7], nominal[:, 3:7])
            for clip in warped.values()
        )
        if not root_matched:
            raise ValueError("shared-clock horizontal route or root orientation mismatch")
        route = classify_route(nominal[:, :2], "straight", fps=30.0)
        directory = output / ladder["ladder_group_id"]
        directory.mkdir(parents=True)
        np.save(directory / "source_frame_map.npy", tau)
        levels = []
        for index, (label, qpos) in enumerate(warped.items()):
            key = f"shared_clock_{ladder['ladder_group_id']}__{label}"
            csv = directory / f"{label}.csv"
            motion = directory / f"{label}.pkl"
            np.savetxt(csv, qpos, delimiter=",", fmt="%.10f")
            save_sonic_motion_file(motion, motion_key=key, motion_entry=qpos_to_sonic_motion_entry(qpos))
            gate = screen_reference(qpos, key, "walk")
            envelope = extract_envelope(
                extract_keypoints(payload_from_reference(qpos, fps=30.0)), key, fractions=stations
            )
            row = {
                "label": label, "ladder_level": index, "motion_key": key,
                "csv": str(csv), "csv_sha256": sha(csv),
                "sonic_motion": str(motion), "sonic_motion_sha256": sha(motion),
                "q0_status": "passed" if gate.embodiment_feasible else "failed",
                "q1_status": "passed" if gate.self_collision_free else "failed",
                "whole_body_top_m": envelope.up_m.tolist(),
                "semantic_assessment": {"semantic_status": "not_applicable"},
                "source_csv_sha256": originals[label]["csv_sha256"],
            }
            if index:
                row["semantic_assessment"] = assess_paired_reduction(
                    stations, envelope.up_m, np.asarray(levels[0]["whole_body_top_m"]),
                    route_valid=route.validity_class == "valid_straight",
                    policy=PairedSemanticPolicy(minimum_effect=0.05),
                ).to_dict()
            levels.append(row)
        profiles = np.asarray([level["whole_body_top_m"] for level in levels])
        minimum_gap = float(np.min(profiles[:-1, event] - profiles[1:, event]))
        eligible = minimum_gap > 0 and all(
            level["q0_status"] == "passed" and level["q1_status"] == "passed"
            and (index == 0 or level["semantic_assessment"]["semantic_status"] == "route_aligned")
            for index, level in enumerate(levels)
        )
        ladders.append({
            "ladder_group_id": ladder["ladder_group_id"],
            "base_carrier_id": ladder["base_carrier_id"],
            "generation_seed": ladder["generation_seed"],
            "root_route_and_clock_matched": root_matched,
            "frames": len(nominal), "duration_scale": (len(nominal) - 1) / 119,
            "reference_route": route.to_dict(),
            "minimum_central_adjacent_height_gap_m": minimum_gap,
            "reference_ladder_candidate": eligible,
            "q3_q4_admitted": False, "levels": levels,
            "time_map_sha256": sha(directory / "source_frame_map.npy"),
        })
        print(f"{ladder['ladder_group_id']}: frames={len(nominal)} min gap={minimum_gap:.4f} eligible={eligible}")
    write_new(output / "candidates.json", {
        "schema_version": "motion2scene_shared_clock_duck_candidates_v1",
        "source_repo": str(source), "source_commit": design["source_commit"],
        "registration_sha256": sha(args.registration), "driver_sha256": sha(Path(__file__)),
        "attempted_ladders": len(ladders),
        "reference_ladder_candidates": sum(row["reference_ladder_candidate"] for row in ladders),
        "q3_q4_admitted_ladders": 0, "ladders": ladders,
    })


if __name__ == "__main__":
    main()
