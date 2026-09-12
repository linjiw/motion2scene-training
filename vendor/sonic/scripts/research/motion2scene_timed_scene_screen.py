#!/usr/bin/env python3
"""Construct development beams from measured complete timed executions.

A finite geometry screen does not admit an option or create physical labels.
Observation availability is deliberately left pending native scene execution.
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
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance  # noqa: E402
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


def beam_clearance(capsules, beam, offset):
    """Exact primitive clearance after world XY/Z and yaw offsets."""
    starts, ends, radii, _ = capsules
    yaw = beam["yaw_rad"] + offset[3]
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    center = np.r_[beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2]
    center += offset[:3]
    a, b = (starts - center) @ rotation, (ends - center) @ rotation
    half = np.array([beam["length_m"], beam["width_m"], beam["thickness_m"]]) / 2
    # Exact lower bound on distance to every capsule's enclosing AABB. Query all
    # capsules capable of reducing the current .20m capped minimum; no omitted
    # capsule can change any +/- .01m eligibility decision.
    delta = np.maximum(np.maximum(np.minimum(a, b) - half, -half - np.maximum(a, b)), 0)
    lower = np.linalg.norm(delta, axis=-1) - radii
    take = lower <= 0.20
    rr = np.broadcast_to(radii, lower.shape)
    values = capsule_box_clearance(a[take], b[take], rr[take], -half, half)
    return float(values.min(initial=0.20))


def run(result_path, out):
    source = json.loads(result_path.read_text())
    manifest = json.loads(
        checked(Path(source["manifest"]["path"]), source["manifest"]["sha256"]).read_text()
    )
    geometry_ref = manifest["geometry"]
    geometry = json.loads(checked(Path(geometry_ref["path"]), geometry_ref["sha256"]).read_text())
    refs = [artifact(result_path), source["manifest"], geometry_ref]
    shape_sets = {"outer": {}, "inner": {}}
    for layer in geometry["layers"]:
        checked(Path(layer["path"]), layer["sha256"])
        refs.append(layer)
    for shape in geometry["shapes"]:
        capsule = CollisionCapsule(tuple(shape["start"]), tuple(shape["end"]), shape["radius"])
        shape_sets["outer"].setdefault(shape["owner"], []).append(capsule)
        if shape["role"] == "native_primitive_subset":
            shape_sets["inner"].setdefault(shape["owner"], []).append(capsule)
    executions, admission = {}, {}
    for row in source["rows"]:
        evidence = json.loads(
            checked(Path(row["evidence"]["path"]), row["evidence"]["sha256"]).read_text()
        )
        ref = evidence["artifacts"]["trajectory"]
        payload = load_reset_capture(checked(Path(ref["path"]), ref["sha256"]))
        if len(payload["root_pos_w"]) != 298 or not evidence["checks"]["sensor_clock_aligned"]:
            raise ValueError("requires complete aligned measured six-second schedules")
        refs.extend([row["evidence"], ref])
        executions[row["cell_id"]] = {
            name: body_capsules_world(
                payload["body_pos_w"],
                payload["body_quat_w"],
                payload["body_names"],
                capsules=shapes,
            )
            for name, shapes in shape_sets.items()
        }
        admission[row["cell_id"]] = row["qualified"]
    ids = ["neutral", "short_e015_r265", "sustained_e015_r255"]
    if set(executions) != set(ids):
        raise ValueError("requires neutral, short and sustained measured alternatives")
    neutral_ref = manifest["cells"][0]["motion"]
    motion = next(
        iter(joblib.load(checked(Path(neutral_ref["path"]), neutral_ref["sha256"])).values())
    )
    route = np.asarray(motion["root_trans_offset"])[:, :2]
    scene_ref = manifest["cells"][0]["scene"]
    scene_text = checked(Path(scene_ref["path"]), scene_ref["sha256"]).read_text()
    grid = [
        dict(
            route_progress_fraction=float(station),
            lateral_offset_m=0.0,
            yaw_offset_rad=0.0,
            length_m=length,
            width_m=1.2,
            thickness_m=0.1,
            underside_m=float(height),
        )
        for station, length, height in itertools.product(
            np.linspace(0.4, 0.75, 15), [0.1, 0.5, 1.0], np.linspace(1.20, 1.34, 29)
        )
    ]
    offsets = np.asarray(
        [[0, 0, 0, 0]]
        + [
            list(v)
            for v in itertools.product(
                [-0.02, 0, 0.02], [-0.02, 0, 0.02], [-0.005, 0, 0.005], [-0.02, 0, 0.02]
            )
            if any(v)
        ]
    )
    out.mkdir(parents=True, exist_ok=False)
    snapshots = []
    for path in sorted(closure([Path(__file__)])):
        dest = out / "source_snapshot" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        snapshots.append({**artifact(path), "snapshot": artifact(dest)})
    write_new(
        out / "registration.json",
        dict(
            schema="motion2scene_timed_scene_screen_v1",
            split="development",
            references=refs + [neutral_ref, scene_ref],
            implementation=snapshots,
            grid=grid,
            offsets_world_xyz_yaw=offsets.tolist(),
            option_ids=ids,
            margin_m=0.01,
            source_qualification=admission,
            geometric_clearance_cap_m=0.2,
            selection=(
                "max-min geometric slack per length and positive/negative pair; "
                "deterministic first-index ties"
            ),
            observability=(
                "pending native sensor measurements before entry tick15; "
                "no geometric proxy creates availability"
            ),
            scope=(
                "recorded 50 Hz native primitive geometry; no option promotion, "
                "new physical labels or held-out evaluation"
            ),
        ),
    )
    beams = place_beams(route, grid)
    nominal = np.zeros((len(beams), len(ids), 2))
    start = time.monotonic()
    queries = 0
    for i, beam in enumerate(beams):
        for a, option in enumerate(ids):
            for k, kind in enumerate(("outer", "inner")):
                nominal[i, a, k] = beam_clearance(executions[option][kind], beam, offsets[0])
                queries += 1
        if i % 100 == 0:
            print(
                json.dumps(
                    {
                        "nominal_completed": i,
                        "total": len(beams),
                        "seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
    candidates = []
    for positive, negative in ((1, 0), (2, 0), (2, 1), (1, 2)):
        selected = np.flatnonzero(
            (nominal[:, positive, 0] >= 0.01) & (nominal[:, negative, 1] <= -0.01)
        )
        for i in selected:
            clearances = [
                beam_clearance(executions[ids[positive]]["outer"], beams[i], offset)
                for offset in offsets
            ]
            queries += len(offsets)
            worst = min(clearances)
            candidates.append(
                dict(
                    grid_index=int(i),
                    positive=ids[positive],
                    negative=ids[negative],
                    positive_min_m=worst,
                    negative_nominal_m=float(nominal[i, negative, 1]),
                    eligible=bool(worst >= 0.01),
                    slack_m=min(worst - 0.01, -nominal[i, negative, 1] - 0.01),
                )
            )
    np.savez_compressed(
        out / "nominal.npz",
        clearance_m=nominal,
        option_ids=np.asarray(ids),
        kinds=np.asarray(["outer", "inner"]),
    )
    chosen = []
    for length, pair in itertools.product(
        [0.1, 0.5, 1.0], [(ids[1], ids[0]), (ids[2], ids[0]), (ids[2], ids[1]), (ids[1], ids[2])]
    ):
        eligible = [
            r
            for r in candidates
            if r["eligible"]
            and (r["positive"], r["negative"]) == pair
            and beams[r["grid_index"]]["length_m"] == length
        ]
        if eligible:
            chosen.append(max(eligible, key=lambda r: r["slack_m"]))
    # Materialize unique deterministic selected layouts, without issuing labels.
    scenes = []
    for i in sorted({r["grid_index"] for r in chosen}):
        scene_id = f"timed_development_{i:04d}"
        scene_path = out / "scenes" / f"{scene_id}.usda"
        scene_path.parent.mkdir(exist_ok=True)
        scene_path.write_text(author_course(scene_text, [beams[i]], course_id=scene_id))
        definition = dict(
            schema="motion2scene_timed_history_scene_v1",
            split="development",
            scene_id=scene_id,
            scene=artifact(scene_path),
            beam=beams[i],
            beam_collision_enabled=True,
            provenance=dict(screen_registration=artifact(out / "registration.json"), grid_index=i),
            physical_labels=None,
        )
        path = out / f"{scene_id}.json"
        write_new(path, definition)
        scenes.append(artifact(path))
    write_new(
        out / "result.json",
        dict(
            registration=artifact(out / "registration.json"),
            nominal=artifact(out / "nominal.npz"),
            candidates=candidates,
            selected=chosen,
            scenes=scenes,
            clearance_queries=queries,
            seconds=time.monotonic() - start,
            physical_rollout_steps=0,
            observation_timing_screened=False,
            physical_labels_acquired=0,
        ),
    )
    print(json.dumps({"selected": chosen, "scenes": scenes, "queries": queries}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.result, args.out)
