#!/usr/bin/env python3
"""One bounded complementary late-beam proposal from qualified executions.

Authored schedules must retain outer clearance over all81 offsets; both prior
schedules must nominally interfere. No physical or observation label is created.
"""

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
import joblib  # noqa: E402
from motion2scene_materialize_evaluation import place_beams  # noqa: E402
from motion2scene_timed_scene_screen import beam_clearance  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_course import (  # noqa: E402
    author_course,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    CollisionCapsule,
    body_capsules_world,
)

IDS = [
    "neutral",
    "short_e015_r265",
    "sustained_e015_r255",
    "prior_splice_e015_r265",
    "prior_splice_e050_r265",
    "short_e070_r265",
    "sustained_e070_r255",
]
PRIOR = (3, 4)
AUTHORED = (1, 2, 5, 6)


def read_bound(ref):
    return json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())


def specification(station, length, height):
    return dict(
        route_progress_fraction=float(station),
        lateral_offset_m=0.0,
        yaw_offset_rad=0.0,
        length_m=float(length),
        width_m=1.2,
        thickness_m=0.1,
        underside_m=float(height),
    )


def beam_key(beam):
    return tuple(
        np.round(
            [
                *beam["center_xy_m"],
                beam["yaw_rad"],
                beam["length_m"],
                beam["width_m"],
                beam["thickness_m"],
                beam["underside_m"],
            ],
            9,
        )
    )


def run(result_path, lock_path, out):
    source = json.loads(result_path.read_text())
    manifest = read_bound(source["manifest"])
    geometry = read_bound(manifest["geometry"])
    references = [artifact(result_path), source["manifest"], manifest["geometry"]]
    for layer in geometry["layers"]:
        checked(Path(layer["path"]), layer["sha256"])
        references.append(layer)
    shape_sets = {"outer": {}, "inner": {}}
    for shape in geometry["shapes"]:
        capsule = CollisionCapsule(tuple(shape["start"]), tuple(shape["end"]), shape["radius"])
        shape_sets["outer"].setdefault(shape["owner"], []).append(capsule)
        if shape["role"] == "native_primitive_subset":
            shape_sets["inner"].setdefault(shape["owner"], []).append(capsule)
    executions = {}
    for row in source["rows"]:
        evidence = read_bound(row["evidence"])
        if not row["qualified"] or not all(evidence["checks"].values()):
            raise ValueError("all seven source schedules must be physically qualified")
        ref = evidence["artifacts"]["trajectory"]
        payload = load_reset_capture(checked(Path(ref["path"]), ref["sha256"]))
        if len(payload["root_pos_w"]) != 298:
            raise ValueError("complete finite six-second capture required")
        references.extend([row["evidence"], ref])
        executions[row["cell_id"]] = {
            kind: body_capsules_world(
                payload["body_pos_w"],
                payload["body_quat_w"],
                payload["body_names"],
                capsules=shapes,
            )
            for kind, shapes in shape_sets.items()
        }
    if set(executions) != set(IDS):
        raise ValueError("exact preregistered seven-option bank required")
    neutral_ref, scene_ref = (manifest["cells"][0][key] for key in ("motion", "scene"))
    neutral = next(
        iter(joblib.load(checked(Path(neutral_ref["path"]), neutral_ref["sha256"])).values())
    )
    route = np.asarray(neutral["root_trans_offset"])[:, :2]
    template = checked(Path(scene_ref["path"]), scene_ref["sha256"]).read_text()
    grid = [
        specification(station / 1000, length, height / 1000)
        for station, length, height in itertools.product(
            range(450, 901, 25), (0.1, 0.2, 0.4, 0.6, 0.8), range(1120, 1401, 10)
        )
    ]
    beams = place_beams(route, grid)
    # Use the frozen lock solely to exclude duplicate geometry. No reserved
    # trajectory, clearance value, physical label or sensor stream is loaded.
    lock = json.loads(lock_path.read_text())
    reserved = {
        beam_key(beam)
        for layout in lock["evaluation"]["layouts"]
        for variant in layout["fixed_world_variants"]
        for beam in variant["beams"]
    }
    if any(beam_key(beam) in reserved for beam in beams):
        raise ValueError("development proposal duplicates a reserved beam")
    offsets = np.asarray(
        list(
            itertools.product(
                (-0.02, 0, 0.02), (-0.02, 0, 0.02), (-0.005, 0, 0.005), (-0.02, 0, 0.02)
            )
        )
    )
    out.mkdir(parents=True, exist_ok=False)
    snapshots = []
    for path in sorted(closure([Path(__file__)])):
        dest = out / "source_snapshot" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        snapshots.append({**artifact(path), "snapshot": artifact(dest)})
    registration = out / "registration.json"
    write_new(
        registration,
        dict(
            schema="motion2scene_complementary_late_option_screen_v1",
            split="development",
            source_ancestry="source41002; prior projection/splice is declared authored derivation",
            references=references + [neutral_ref, scene_ref],
            implementation=snapshots,
            grid=grid,
            grid_world_beams=beams,
            offsets_world_xyz_yaw=offsets.tolist(),
            option_ids=IDS,
            positive_indices=AUTHORED,
            negative_indices=PRIOR,
            margin_m=0.01,
            geometric_clearance_cap_m=0.20,
            selection=(
                "For each authored option shortlist at most16 nominal candidates with outer>=.01 "
                "and both prior native-inner clearances<=-.01. Rank by minimum nominal signed "
                "slack, then grid index. Screen all81 offsets for each shortlisted positive. "
                "Select ONE maximum robust signed slack across every option/length; tie lower "
                "grid then lower option index. If none qualifies, materialize no scene. "
                "No fallback changes to the registered grid."
            ),
            proposed_future_physics_seed=8731,
            proposed_future_branch_order=IDS,
            purpose=(
                "Development complementary capability hypothesis from measured empty execution; "
                "no arbitrary behavior penalty, held-out outcome or primary acquisition label"
            ),
            reserved_exclusion=dict(
                lock=artifact(lock_path),
                compared_world_beams=len(reserved),
                duplicate_beams=0,
                outcomes_read=False,
            ),
            scope=(
                "finite recorded50Hz primitive geometry only; no continuous-time guarantee, "
                "physical passage label, motion-option promotion or observation availability"
            ),
        ),
    )
    print(json.dumps({"registration": artifact(registration), "grid_size": len(grid)}), flush=True)
    start, queries = time.monotonic(), 0
    nominal = np.zeros((len(beams), len(IDS), 2))
    for i, beam in enumerate(beams):
        for a, option in enumerate(IDS):
            for k, kind in enumerate(("outer", "inner")):
                nominal[i, a, k] = beam_clearance(executions[option][kind], beam, [0, 0, 0, 0])
                queries += 1
        if i % 100 == 0:
            print(json.dumps({"completed": i, "seconds": time.monotonic() - start}), flush=True)
    candidates = []
    for positive in AUTHORED:
        eligible = [
            i
            for i in range(len(beams))
            if nominal[i, positive, 0] >= 0.01 and np.all(nominal[i, PRIOR, 1] <= -0.01)
        ]
        nominal_slack = lambda i: min(  # noqa: E731
            nominal[i, positive, 0] - 0.01,
            float((-nominal[i, PRIOR, 1] - 0.01).min()),
        )
        shortlisted = sorted(eligible, key=lambda i: (-nominal_slack(i), i))[:16]
        for i in shortlisted:
            values = [
                beam_clearance(executions[IDS[positive]]["outer"], beams[i], offset)
                for offset in offsets
            ]
            queries += len(offsets)
            worst = min(values)
            candidates.append(
                dict(
                    grid_index=i,
                    positive=IDS[positive],
                    positive_index=positive,
                    negatives=[IDS[a] for a in PRIOR],
                    nominal_candidates_for_positive=len(eligible),
                    positive_offset_clearance_m=values,
                    minimum_positive_clearance_m=worst,
                    negative_nominal_clearance_m={IDS[a]: float(nominal[i, a, 1]) for a in PRIOR},
                    eligible=worst >= 0.01,
                    robust_slack_m=min(worst - 0.01, float((-nominal[i, PRIOR, 1] - 0.01).min())),
                )
            )
    accepted = [r for r in candidates if r["eligible"]]
    chosen = (
        max(accepted, key=lambda r: (r["robust_slack_m"], -r["grid_index"], -r["positive_index"]))
        if accepted
        else None
    )
    np.savez_compressed(
        out / "nominal.npz",
        clearance_m=nominal,
        option_ids=np.asarray(IDS),
        kinds=np.asarray(["outer", "inner"]),
    )

    def materialize(scene_id, scene_beams, provenance):
        scene = out / "scenes" / f"{scene_id}.usda"
        scene.parent.mkdir(exist_ok=True)
        scene.write_text(author_course(template, scene_beams, course_id=scene_id))
        definition = out / f"{scene_id}.json"
        write_new(
            definition,
            dict(
                schema="motion2scene_timed_schedule_scene_v1",
                split="development",
                scene_id=scene_id,
                scene=artifact(scene),
                beams=scene_beams,
                beam_collision_enabled=[True] * len(scene_beams),
                provenance=dict(registration=artifact(registration), **provenance),
                physical_labels=None,
                observation_timing_status="unmeasured",
            ),
        )
        return artifact(definition)

    scene = None
    if chosen is not None:
        i = chosen["grid_index"]
        scene = materialize(
            f"complementary_late_development_{i:04d}",
            [beams[i]],
            {"grid_index": i, "positive_option_id": chosen["positive"]},
        )
    write_new(
        out / "result.json",
        dict(
            schema="motion2scene_complementary_late_option_screen_result_v1",
            registration=artifact(registration),
            nominal=artifact(out / "nominal.npz"),
            candidates=candidates,
            selected=chosen,
            scene=scene,
            nominal_eligible_counts={
                IDS[a]: int(
                    np.sum(
                        (nominal[:, a, 0] >= 0.01) & np.all(nominal[:, PRIOR, 1] <= -0.01, axis=1)
                    )
                )
                for a in AUTHORED
            },
            selected_beam=None if chosen is None else beams[chosen["grid_index"]],
            all_selected_nominal_clearance_m=(
                None
                if chosen is None
                else {
                    option: {
                        "outer": float(nominal[chosen["grid_index"], i, 0]),
                        "inner": float(nominal[chosen["grid_index"], i, 1]),
                    }
                    for i, option in enumerate(IDS)
                }
            ),
            clearance_queries=queries,
            seconds=time.monotonic() - start,
            physics_steps=0,
            physical_labels_created=0,
            observation_timing_screened=False,
            scope="Bounded geometric search only; absent candidate is retained without widening the grid",
        ),
    )
    print(
        json.dumps(
            dict(
                scene=scene,
                selected=(
                    None
                    if chosen is None
                    else {k: v for k, v in chosen.items() if k != "positive_offset_clearance_m"}
                ),
                queries=queries,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.result, args.lock, args.out)
