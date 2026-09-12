#!/usr/bin/env python3
"""Freeze new world-space evaluation geometry; never query robot outcomes."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/home/linjiw/research-data/groot-wbc")
NEUTRAL = DATA / "m2s-longer-reference-development-v1/neutral.pkl"
NEUTRAL_SHA = "ef33c2139f7a6278d670c012edc9db70671d08b6842e8757d15dc6a6b61b3990"
V2 = ROOT / "docs/motion2scene/TRAVERSAL_EVALUATION_LOCK_V2.json"
V2_SHA = "7898862a0a02ab7d65a5e21f245c15a77b82a74c86245e88a611ad48ffd58dfc"
SEED = 202609081820


def reference(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def world_beam(route, spec):
    """Fix each box on the named source route, with positive lateral to the left."""
    route = np.asarray(route, dtype=np.float64)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2 or not np.isfinite(route).all():
        raise ValueError("finite XY route required")
    distance = np.r_[0.0, np.linalg.norm(np.diff(route, axis=0), axis=1).cumsum()]
    direction = route[-1] - route[0]
    if distance[-1] <= 0 or np.linalg.norm(direction) <= 1e-9:
        raise ValueError("nonzero route length and net direction required")
    progress = distance / distance[-1]
    keep = np.r_[np.diff(progress) > 0, True]
    heading = float(np.arctan2(direction[1], direction[0]))
    xy = np.array(
        [
            np.interp(spec["route_progress_fraction"], progress[keep], route[keep, i])
            for i in range(2)
        ]
    )
    xy += spec["lateral_offset_m"] * np.array([-np.sin(heading), np.cos(heading)])
    return {
        **spec,
        "center_xy_m": xy.tolist(),
        "center_xyz_m": [*xy.tolist(), spec["underside_m"] + spec["thickness_m"] / 2],
        "yaw_rad": heading + spec["yaw_offset_rad"],
        "full_dimensions_xyz_m": [spec["length_m"], spec["width_m"], spec["thickness_m"]],
        "collision_enabled": True,
        "placement_authority": "these fixed world coordinates; never relocate from another reference",
    }


def build_lock(created_utc):
    if reference(NEUTRAL)["sha256"] != NEUTRAL_SHA or reference(V2)["sha256"] != V2_SHA:
        raise ValueError("pinned reference or original lock changed")
    old = json.loads(V2.read_text())
    library = joblib.load(NEUTRAL)
    if len(library) != 1:
        raise ValueError("one qualified six-second neutral required")
    motion = next(iter(library.values()))
    route = np.asarray(motion["root_trans_offset"], dtype=np.float64)[:, :2]
    if len(route) != 180:
        raise ValueError("expected the named 180-frame source, not a substitute carrier")
    rng = random.Random(SEED)
    common = old["generation"]["common_ranges"]

    def sample(station_range, length_range):
        ranges = [
            station_range,
            common["underside_m"],
            length_range,
            common["width_m"],
            common["thickness_m"],
            common["lateral_offset_m"],
            common["yaw_offset_rad"],
        ]
        keys = old["generation"]["draw_order"]
        return dict(zip(keys, [round(rng.uniform(*bounds), 9) for bounds in ranges], strict=True))

    families = [
        {"name": "short", "count": 4, "station_range": [0.26, 0.77], "length_range_m": [0.1, 0.25]},
        {
            "name": "sustained",
            "count": 4,
            "station_range": [0.32, 0.68],
            "length_range_m": [0.5, 0.9],
        },
        {
            "name": "early_constraint",
            "count": 4,
            "station_range": [0.15, 0.35],
            "length_range_m": [0.1, 0.25],
        },
    ]
    layouts = []
    for family in families:
        for _ in range(family["count"]):
            layouts.append(
                {
                    "layout_id": f"locked_v3_single_{len(layouts):02d}",
                    "source_id": 41002,
                    "family": family["name"],
                    "beams": [sample(family["station_range"], family["length_range_m"])],
                }
            )
    for i in range(6):
        layouts.append(
            {
                "layout_id": f"locked_v3_course_{i:02d}",
                "source_id": 41002,
                "family": "short_then_short" if i < 3 else "short_then_sustained",
                "beams": [
                    sample([0.24, 0.35], [0.1, 0.25]),
                    sample([0.65, 0.77], [0.1, 0.25] if i < 3 else [0.5, 0.9]),
                ],
            }
        )
    offsets = copy.deepcopy(old["evaluation"]["perturbation_set"])
    for layout in layouts:
        raw = layout.pop("beams")
        layout["split"] = "reserved_evaluation_v3"
        layout["nominal_beams"] = [world_beam(route, beam) for beam in raw]
        layout["fixed_world_variants"] = []
        for offset in offsets:
            beams = []
            for beam in raw:
                perturbed = dict(beam)
                for key in (
                    "route_progress_fraction",
                    "underside_m",
                    "lateral_offset_m",
                    "yaw_offset_rad",
                ):
                    delta_key = "delta_" + {
                        "lateral_offset_m": "lateral_m",
                        "yaw_offset_rad": "yaw_rad",
                    }.get(key, key)
                    perturbed[key] += offset[delta_key]
                beams.append(world_beam(route, perturbed))
            layout["fixed_world_variants"].append({"offset_id": offset["id"], "beams": beams})
    scoring = copy.deepcopy(old["scoring"])
    scoring["inherited_timeouts_not_adopted_s"] = {
        "single": scoring.pop("single_timeout_s"),
        "course": scoring.pop("course_timeout_s"),
    }
    scoring["finite_horizon_terminal_censoring_policy"] = {
        "status": "PENDING_PRE_EVALUATION_AGREEMENT_AND_IMPLEMENTATION_LOCK",
        "qualified_reference_end_phase_s": 5.96,
        "qualified_captured_physical_end_s": 5.94,
        "issue": (
            "Define incomplete passage, incomplete upright hold, and incomplete recovery at "
            "the finite horizon before any evaluation; retain all assigned layouts and all "
            "attempted outcomes."
        ),
        "execution_blocked_until_resolved": True,
        "no_reference_wrap_or_post_reset_sensor_as_continuation": True,
    }
    return {
        "schema": "motion2scene_six_second_world_geometry_lock_v3",
        "created_utc": created_utc,
        "status": "GEOMETRY_LOCKED_IMPLEMENTATION_AND_TERMINAL_POLICY_PENDING_ZERO_EVALUATION_RUNS",
        "relationship_to_v2": {
            "original": reference(V2),
            "replaces_original": False,
            "counts_as_original_72_episode_completion": False,
            "reason": (
                "New 180-frame carrier and separately sampled panel; original four-second "
                "route geometry is not reinterpreted."
            ),
        },
        "qualification_scope": (
            "Six-second neutral execution only. Short/sustained adaptation, entry/return "
            "timing, common learned options and full course sequences remain unqualified."
        ),
        "carrier": {
            "motion": reference(NEUTRAL),
            "source_id": 41002,
            "source_frames": 180,
            "source_fps": 30,
            "loaded_reference_frames": 299,
            "neutral_qualification": reference(
                DATA / "m2s-longer-neutral-qualification-v2/result.json"
            ),
            "source_transfer_claim": False,
            "ancestry": "Reused development source41002; new duration-conditioned Kimodo generation.",
        },
        "generation": {
            "seed": SEED,
            "rng": "Python random.Random MT19937; uniform, each draw rounded to9 decimals",
            "draw_order": old["generation"]["draw_order"],
            "single_families_in_order": families,
            "course_order": "three short-then-short, then three short-then-sustained; first beam before second",
            "course_station_ranges": [[0.24, 0.35], [0.65, 0.77]],
            "common_ranges": common,
            "rejection_sampling": False,
            "clearance_queries": 0,
            "physics_steps": 0,
            "outcomes_used": False,
            "route_xy_m": route.tolist(),
            "placement": (
                "Original named source XY arc length; interpolate last duplicate station; "
                "global net heading; positive lateral left. Pin nominal and every stress "
                "variant in world space at creation."
            ),
            "driver": reference(Path(__file__)),
        },
        "evaluation": {
            "physics_seeds": [94301, 94302],
            "execution_order_seed": SEED + 1,
            "layouts": layouts,
            "nominal_episodes_per_policy": 36,
            "single_episodes_per_policy": 24,
            "course_episodes_per_policy": 12,
            "stress_episodes_per_policy": 288,
            "perturbation_set": offsets,
            "stress_reporting": (
                "Separate from nominal denominator; use stored fixed world variants, same "
                "delta on both beams."
            ),
            "retention": (
                "All18 nominal layouts and8 stress variants retained, no feasibility or "
                "outcome filter. Unsupported runtime is not_run."
            ),
            "student_input_exclusions": old["evaluation"]["student_input_exclusions"],
            "training_exclusion": (
                "No teacher, clearance, policy or physical outcome queries on these layouts "
                "or variants during development/tuning."
            ),
            "course_gate": (
                "Separate actual full-sequence transition/horizon qualification and common "
                "interface lock before any course evaluation."
            ),
        },
        "comparison_matrix": old["comparison_matrix"],
        "fixed_baselines": old["fixed_baselines"],
        "isolating_ablations": old["isolating_ablations"],
        "acquisition": old["acquisition"],
        "scoring": scoring,
        "analysis": old["analysis"],
        "implementation_lock": {
            "status": "PENDING",
            "required_fields": (
                old["common_implementation"]["required_implementation_lock_fields"]
            ),
            "execution_authorized_by_geometry_lock": False,
        },
        "amendments": {
            "rule": (
                "Never overwrite this lock. Any geometry or scoring amendment must cite "
                "this SHA and disclose prior outcome inspection; pin "
                "implementation/terminal policy before evaluation."
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lock = build_lock(datetime.now(timezone.utc).isoformat())
    with args.out.open("x") as stream:
        stream.write(json.dumps(lock, indent=2, allow_nan=False) + "\n")
    checksum = args.out.with_suffix(".sha256")
    with checksum.open("x") as stream:
        stream.write(f"{reference(args.out)['sha256']}  {args.out.name}\n")
    print(
        json.dumps(
            {"lock": reference(args.out), "layouts": 18, "world_variants": 162, "physics_steps": 0}
        )
    )


if __name__ == "__main__":
    main()
