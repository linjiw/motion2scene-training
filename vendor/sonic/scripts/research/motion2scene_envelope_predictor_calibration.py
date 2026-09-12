#!/usr/bin/env python3
"""Calibrate the executed-envelope screen against measured schedule outcomes.

The constructor proposes scenes with a geometric predicate over executed whole-body envelopes, then
physical execution decides what actually happens. This study asks how well the first predicts the
second, on encounters that have ALREADY been executed. It reads frozen candidate geometry and frozen
7-branch teacher outcome tables and computes a confusion matrix. It adds no physics, no fit and no
new candidate search.

The property of interest is the one the completed M8 development panel identified: an encounter is
"prior-free solvable" when every member of the corpus's covering set fails and some other schedule
passes. Those are the encounters that force a non-covering response, and corpora containing them are
the only ones whose M8 policies stopped being constant functions.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)

COVERING_SET = ("prior_splice_e015_r265", "prior_splice_e050_r265")
POSITIVE_MARGIN_M = 0.010
NEGATIVE_MARGIN_M = 0.010
REQUIRED_OFFSETS = 81
SWEEP_MM = (0.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0)


def load_geometry(pool):
    """Frozen per-candidate envelope queries, keyed by candidate id."""
    geometry = {}
    for path in sorted(pool.glob("seed_*/geometry_queries.npz")):
        arrays = np.load(path)
        options = [str(o) for o in arrays["option_ids"]]
        for index, candidate in enumerate(arrays["candidate_ids"]):
            geometry[str(candidate)] = dict(
                options=options,
                nominal_inner=arrays["nominal_inner_clearance_m"][index],
                outer_by_offset=arrays["outer_clearance_by_offset_m"][index],
                offset_counts=arrays["evaluated_offset_counts"][index],
            )
    return geometry


def predicate(entry, negative_margin_m=NEGATIVE_MARGIN_M, positive_margin_m=POSITIVE_MARGIN_M):
    """The constructor's targeting screen, evaluated exactly as the queue applies it."""
    options = entry["options"]
    covering = [options.index(name) for name in COVERING_SET]
    negative = all(entry["nominal_inner"][i] <= -negative_margin_m for i in covering)
    positive = any(
        entry["offset_counts"][j] == REQUIRED_OFFSETS
        and float(entry["outer_by_offset"][:, j].min()) >= positive_margin_m
        for j, name in enumerate(options)
        if name not in COVERING_SET
    )
    return dict(fires=negative and positive, negative_ok=negative, positive_ok=positive)


def measured(outcomes):
    """Prior-free solvable: every covering schedule failed and some other schedule passed."""
    covering_fails = all(outcomes.get(name) == "failure" for name in COVERING_SET)
    other_passes = any(v == "pass" for k, v in outcomes.items() if k not in COVERING_SET)
    return covering_fails and other_passes


def calibrate(acquisition, pool, out):
    tables = read_checked(artifact(acquisition))
    geometry = load_geometry(pool)
    encounters, corpora = {}, {}
    for corpus in tables["corpora"]:
        ids = []
        for task in corpus["tasks"]:
            encounters.setdefault(task["scene_id"], task["outcomes"])
            ids.append(task["scene_id"])
        corpora[corpus["run_id"]] = ids

    rows = []
    for scene_id, outcomes in sorted(encounters.items()):
        entry = geometry.get(scene_id)
        if entry is None:
            continue
        screen = predicate(entry)
        covering = [entry["options"].index(name) for name in COVERING_SET]
        rows.append(
            dict(
                scene_id=scene_id,
                measured_prior_free_solvable=measured(outcomes),
                **screen,
                covering_nominal_inner_mm=[
                    round(float(entry["nominal_inner"][i]) * 1000, 3) for i in covering
                ],
                measured_passing_schedules=sorted(k for k, v in outcomes.items() if v == "pass"),
            )
        )

    counts = Counter((row["measured_prior_free_solvable"], row["fires"]) for row in rows)
    tp, fn = counts[True, True], counts[True, False]
    fp, tn = counts[False, True], counts[False, False]

    sweep = []
    for millimetres in SWEEP_MM:
        fires = [
            predicate(geometry[row["scene_id"]], negative_margin_m=millimetres / 1000)["fires"]
            for row in rows
        ]
        hit = sum(f and row["measured_prior_free_solvable"] for f, row in zip(fires, rows))
        sweep.append(
            dict(
                negative_margin_mm=millimetres,
                fires=sum(fires),
                true_positives=hit,
                recall_of_measured_property=f"{hit}/{tp + fn}",
            )
        )

    misses = [row for row in rows if row["measured_prior_free_solvable"] and not row["fires"]]
    unreachable = [row for row in misses if max(row["covering_nominal_inner_mm"]) > 0]
    positive_bar_only = [row for row in misses if row["negative_ok"] and not row["positive_ok"]]

    out.mkdir(parents=True, exist_ok=False)
    return write_new(
        out / "result.json",
        dict(
            schema="motion2scene_envelope_predictor_calibration_v1",
            implementation=artifact(Path(__file__)),
            acquisition=artifact(acquisition),
            candidate_pool=artifact(pool / "registration.json"),
            covering_set=list(COVERING_SET),
            screen=dict(
                negative_margin_m=NEGATIVE_MARGIN_M,
                positive_margin_m=POSITIVE_MARGIN_M,
                required_offsets=REQUIRED_OFFSETS,
            ),
            executed_encounters=len(rows),
            confusion=dict(
                true_positive=tp, false_negative=fn, false_positive=fp, true_negative=tn
            ),
            recall=f"{tp}/{tp + fn}",
            precision=f"{tp}/{tp + fp}" if tp + fp else "0/0",
            negative_margin_sweep=sweep,
            missed=misses,
            missed_unreachable_by_any_negative_margin=[row["scene_id"] for row in unreachable],
            missed_rejected_by_positive_robustness_bar=[
                row["scene_id"] for row in positive_bar_only
            ],
            corpora_encounter_ids=corpora,
            rows=rows,
            interpretation=(
                "Precision and recall answer different questions. High precision means an encounter "
                "the screen selects really does carry the property. Low recall means the screen "
                "cannot be used to CONCENTRATE a corpus on that property, because most encounters "
                "carrying it are never proposed. Encounters whose covering-schedule clearance is "
                "predicted positive yet measured failing are unreachable by any negative-margin "
                "relaxation; they bound what threshold tuning can achieve."
            ),
            new_physics_steps=0,
            new_fits=0,
            new_candidate_searches=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.acquisition, args.pool, args.out)))
