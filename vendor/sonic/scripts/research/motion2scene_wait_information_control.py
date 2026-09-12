#!/usr/bin/env python3
"""Compare WAIT with one-time choice over the identical complete schedule bank.

Finite recorded-branch information limits, not learned or executed policies.
Initial choice may select any full schedule, including delayed entries. This
holds repertoire fixed while varying when observations can inform selection.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import complete_schedules, feature_keys, read_checked  # noqa: E402
from motion2scene_information_ablation import (  # noqa: E402
    mask_scene_features,
    maximum_causal_passages,
)
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


def one_time_schedule_limit(initial_keys, passed, times, admitted):
    """Best shared full-schedule choice in each initial information group.

    All complete schedules are candidates regardless of their future entry time.
    Among equal passage counts use the sum of successful passage times; equal-count
    ties have equal denominators. All unknowns must be resolved for an exact limit.
    """
    keys = np.asarray(initial_keys)
    p, t, known = np.asarray(passed), np.asarray(times, float), np.asarray(admitted)
    if (
        keys.ndim != 1
        or not len(keys)
        or keys.dtype.kind not in "US"
        or p.ndim != 2
        or p.shape[0] != len(keys)
        or p.shape[1] == 0
        or p.dtype.kind != "b"
        or known.dtype.kind != "b"
        or t.shape != p.shape
        or known.shape != p.shape
        or not known.all()
        or not np.isfinite(t[p]).all()
        or np.any(t[p] < 0)
    ):
        raise ValueError("complete measured branch tables and initial information keys required")
    groups = []
    for key in sorted(set(keys)):
        ids = np.flatnonzero(keys == key)
        counts = p[ids].sum(axis=0)
        costs = np.where(p[ids], t[ids], 0).sum(axis=0)
        best_count = counts.max()
        tied = np.flatnonzero(counts == best_count)
        best_cost = costs[tied].min()
        optima = [int(a) for a in tied if costs[a] == best_cost]
        groups.append(
            dict(
                information_key=str(key),
                encounter_indices=ids.tolist(),
                maximum_passages=int(best_count),
                optimal_schedule_indices=optima,
                selected_schedule_index=optima[0],
                successful_time_sum_s=float(best_cost),
            )
        )
    return dict(
        assigned_contexts=len(keys),
        information_groups=len(groups),
        groups=groups,
        maximum_passages=sum(g["maximum_passages"] for g in groups),
    )


def run(source, out):
    source_ref = artifact(source / "result.json")
    original = read_checked(source_ref)
    registration = read_checked(original["registration"])
    groups = read_checked(original["teachers"])
    for group in groups:
        manifest = read_checked(read_checked(group["collection"])["manifest"])
        if manifest["split"] != "development":
            raise ValueError("only development data may set information interventions")
    ref = registration["registry"]
    bank = load_verified_registry(ref["path"], ref["sha256"])
    phases, _, layout = schedule_layout(bank)
    entries = [layout[i] for i in range(len(layout))]
    for group in groups:
        group["targets"].sort(key=lambda row: row["phase_tick"])
        if [r["phase_tick"] for r in group["targets"]] != phases.tolist():
            raise ValueError("all recorded neutral decision observations required")
    names = groups[0]["targets"][0]["feature_names"]
    if any(r["feature_names"] != names for g in groups for r in g["targets"]):
        raise ValueError("common causal feature interface required")
    features = np.asarray([[r["features"] for r in g["targets"]] for g in groups])
    passed, times, admitted = complete_schedules(groups, layout)
    conditions = []
    for reveal in [int(k) for k in phases] + [None]:
        values = mask_scene_features(features, names, phases, reveal)
        keys = feature_keys(values)
        initial = one_time_schedule_limit(keys[:, 0], passed, times, admitted)
        sequential = maximum_causal_passages(keys, phases, entries, passed, times, admitted)
        conditions.append(
            dict(
                reveal_tick=reveal,
                information_group_sizes_by_phase=[
                    sorted(int((keys[:, k] == key).sum()) for key in set(keys[:, k]))
                    for k in range(len(phases))
                ],
                initial_choice=initial,
                sequential_wait_maximum_passages=sequential,
                sequential_minus_initial_passages=sequential - initial["maximum_passages"],
            )
        )
    out.mkdir(parents=True, exist_ok=False)
    return write_new(
        out / "result.json",
        dict(
            schema="motion2scene_wait_information_control_v1",
            source_result=source_ref,
            source_teachers=original["teachers"],
            registry=ref,
            implementations=[
                artifact(Path(__file__)),
                *[
                    artifact(Path(__file__).with_name(name))
                    for name in (
                        "motion2scene_information_ablation.py",
                        "motion2scene_observation_teacher.py",
                        "motion2scene_decision_study.py",
                    )
                ],
            ],
            scene_ids=[g["scene_id"] for g in groups],
            schedule_ids=list(bank.option_ids),
            complete_schedule_choices_in_both_controls=len(bank.option_ids),
            individual_bank_capability=int(passed.any(axis=1).sum()),
            conditions=conditions,
            new_physics_steps=0,
            new_fits=0,
            intervention="existing 28-corridor-coordinate reveal mask, exact available-vector equality",
            evidence="finite recorded-table information limits; no executed or learned policy comparison",
            interpretation=(
                "one-time choice includes all late-entry schedules and executes neutral until entry; "
                "WAIT can exploit later information, never stopping; "
                "nominal singleton groups do not prove robustness"
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.out)))
