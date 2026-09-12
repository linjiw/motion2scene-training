#!/usr/bin/env python3
"""Finite teaching study with delayed access to recorded scene-summary features.

This is an information ablation, not a simulation of camera latency or missing
ray returns. All 28 corridor features are masked before a declared decision;
recorded proprioception, phase and legal actions are unchanged.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import complete_schedules, feature_keys, read_checked  # noqa: E402
from motion2scene_observation_teacher import observation_consistent_teacher  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    schedule_layout,
)


def mask_scene_features(features, names, phases, reveal_tick):
    values = np.asarray(features, dtype=float).copy()
    indices = [i for i, name in enumerate(names) if name.startswith("corridor_")]
    if len(indices) != 28 or values.shape[1:] != (len(phases), len(names)):
        raise ValueError("exact recorded corridor/state interface required")
    if reveal_tick is not None and reveal_tick not in phases:
        raise ValueError("reveal must be a legal decision or never")
    for k, tick in enumerate(phases):
        if reveal_tick is None or tick < reveal_tick:
            values[:, k, indices] = 0
    return values


def maximum_causal_passages(keys, phases, entries, passed, times, admitted):
    """Exact passage-count optimum over deterministic policies on finite groups.

    Use the teacher's validation of complete inputs and refining information
    groups. No inferred outcome replaces an unknown physical branch.
    """
    rows = observation_consistent_teacher(keys, phases, entries, passed, times, admitted)
    if not all(row["complete"] for row in rows):
        raise ValueError("exact finite capability requires complete measured branches")
    values = {}
    for k in reversed(range(len(phases))):
        for row in (r for r in rows if r["phase_tick"] == phases[k]):
            ids = row["encounter_indices"]
            action_counts = []
            for action in row["legal_actions"]:
                if action or k == len(phases) - 1:
                    count = int(np.asarray(passed)[ids, action].sum())
                else:
                    successors = {str(keys[i, k + 1]) for i in ids}
                    count = sum(values[k + 1, key] for key in successors)
                action_counts.append(count)
            values[k, row["information_key"]] = max(action_counts)
    return sum(values[0, str(key)] for key in set(keys[:, 0]))


def run(source, out):
    original = read_checked(artifact(source / "result.json"))
    registration = read_checked(original["registration"])
    groups = read_checked(original["teachers"])
    for group in groups:
        if read_checked(read_checked(group["collection"])["manifest"])["split"] != "development":
            raise ValueError("development information ablation only")
    ref = registration["registry"]
    bank = load_verified_registry(ref["path"], ref["sha256"])
    phases, _, layout = schedule_layout(bank)
    entries = [layout[i] for i in range(len(layout))]
    for group in groups:
        group["targets"].sort(key=lambda row: row["phase_tick"])
        if [r["phase_tick"] for r in group["targets"]] != phases.tolist():
            raise ValueError("all neutral decision summaries required")
    names = groups[0]["targets"][0]["feature_names"]
    features = np.array([[r["features"] for r in g["targets"]] for g in groups])
    passed, times, admitted = complete_schedules(groups, layout)
    out.mkdir(parents=True, exist_ok=False)
    experiment = write_new(
        out / "experiment.json",
        dict(
            source_result=artifact(source / "result.json"),
            source_teachers=original["teachers"],
            implementation=artifact(Path(__file__)),
            teacher_implementation=artifact(
                ROOT / "scripts/research/motion2scene_observation_teacher.py"
            ),
            intervention=(
                "zero all corridor summary coordinates before the declared reveal decision; "
                "retain every other feature"
            ),
            reveal_ticks=[int(k) for k in phases] + [None],
            interpretation="finite recorded-data information ablation, not a noisy sensor or latency rollout",
            new_physics_steps=0,
        ),
    )
    results = []
    for reveal in [int(k) for k in phases] + [None]:
        values = mask_scene_features(features, names, phases, reveal)
        keys = feature_keys(values)
        teacher = observation_consistent_teacher(keys, phases, entries, passed, times, admitted)
        first = [r for r in teacher if r["phase_tick"] == phases[0]]
        results.append(
            dict(
                reveal_tick=reveal,
                groups_per_phase=[len(set(keys[:, k])) for k in range(len(phases))],
                maximum_causal_passages=maximum_causal_passages(
                    keys, phases, entries, passed, times, admitted
                ),
                all_contexts_jointly_solvable=all(r["teacher_action"] is not None for r in first),
                initial_teacher_actions=[r["teacher_action"] for r in first],
                first_phase_information_conflicts=sum(r["information_conflict"] for r in first),
                teacher=teacher,
            )
        )
    return write_new(
        out / "result.json",
        dict(
            experiment=experiment,
            contexts=len(groups),
            scene_ids=[g["scene_id"] for g in groups],
            individual_schedule_capability=int(passed.any(axis=1).sum()),
            strongest_constant_passages=int(passed.sum(axis=0).max()),
            conditions=results,
            new_physics_steps=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.out)))
