#!/usr/bin/env python3
"""Candidate-level capability from the unchanged fixed extension task panel."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_extension_coverage import summarize as summarize_banks, union_bounds  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)


def added_bounds(added, base):
    """Exact missing-outcome bounds for disjoint added/base schedule sets."""
    a, b = union_bounds(added), union_bounds(base)
    return [int(a[0] == 1 and b[1] == 0), int(a[1] == 1 and b[0] == 0)]


def summarize(options, conditions):
    """Group entry schedules by actual reference, never by successful outcome."""
    ids = ["neutral", *[o["option_id"] for o in options]]
    banks = summarize_banks(ids, conditions)  # Validate complete allocation and costs.
    groups = {}
    for i, option in enumerate(options, start=1):
        candidate = option["reference_id"]
        arm = candidate.split("_", 1)[0]
        if arm not in ("generated", "authored") or not ids[i].startswith(candidate + "_"):
            raise ValueError("schedule must retain its declared candidate reference and arm")
        groups.setdefault(candidate, dict(arm=arm, indices=[]))["indices"].append(i)

    candidates = {}
    for candidate, group in groups.items():
        selected = group["indices"]
        others = [
            i
            for name, other in groups.items()
            if name != candidate and other["arm"] == group["arm"]
            for i in other["indices"]
        ]
        rows, paired_times = [], []
        for condition in conditions:
            states, times = condition["states"], condition["times_s"]
            added = [states[i] for i in selected]
            row = dict(
                task_id=condition["task_id"],
                candidate_alone=union_bounds(added),
                candidate_plus_neutral=union_bounds([states[0], *added]),
                incremental_over_neutral=added_bounds(added, states[:1]),
                unique_within_arm=added_bounds(added, [states[0], *[states[i] for i in others]]),
                paired_best_time_difference_s=None,
            )
            # Compare two complete, mutually passing banks, both retaining neutral.
            if states[0] == 1 and all(s != -1 for s in added):
                best = min(times[i] for i in [0, *selected] if states[i] == 1)
                row["paired_best_time_difference_s"] = best - times[0]
                paired_times.append(best - times[0])
            rows.append(row)

        def total(key):
            return [sum(r[key][i] for r in rows) for i in (0, 1)]

        candidates[candidate] = dict(
            arm=group["arm"],
            schedule_ids=[ids[i] for i in selected],
            assigned_tasks=len(conditions),
            assigned_branches=len(conditions) * len(selected),
            unknown_outcomes=sum(c["states"][i] == -1 for c in conditions for i in selected),
            candidate_alone_bounds=total("candidate_alone"),
            candidate_plus_neutral_bounds=total("candidate_plus_neutral"),
            incremental_over_neutral_bounds=total("incremental_over_neutral"),
            unique_within_arm_bounds=total("unique_within_arm"),
            mutually_passing_complete_tasks=len(paired_times),
            mean_paired_best_time_difference_s=(
                sum(paired_times) / len(paired_times) if paired_times else None
            ),
            task_results=rows,
        )
    return dict(
        bank_summary=banks,
        candidates=candidates,
        arms={
            arm: dict(
                assigned_candidates=sum(g["arm"] == arm for g in groups.values()),
                candidates_with_incremental_coverage_bounds=[
                    sum(
                        c["incremental_over_neutral_bounds"][i] > 0
                        for c in candidates.values()
                        if c["arm"] == arm
                    )
                    for i in (0, 1)
                ],
            )
            for arm in ("generated", "authored")
        },
        interpretation=(
            "Each motion reference is one candidate; entry schedules are not independent samples. "
            "Unique coverage removes both entry schedules together from its own arm, retaining "
            "neutral and every other candidate. Missing-outcome bounds are not confidence "
            "intervals. Best-schedule costs are capability readouts, not learned performance."
        ),
    )


def run(coverage_path, output):
    ref = artifact(coverage_path)
    coverage = read_checked(ref)
    prepared = read_checked(coverage["prepared"])
    study = read_checked(prepared["study"])
    registry = read_checked(study["registry"])
    conditions = coverage["conditions"]
    if [c["task_id"] for c in conditions] != [t["task_id"] for t in prepared["tasks"]]:
        raise ValueError("every assigned task must be retained in its original order")
    summary = summarize(registry["request"]["options"], conditions)
    if summary["bank_summary"] != coverage["summary"]:
        raise ValueError("candidate readout disagrees with the source bank summary")
    return write_new(
        output,
        dict(
            coverage=ref,
            registry=study["registry"],
            implementation=[
                artifact(Path(__file__)),
                artifact(Path(__file__).with_name("motion2scene_extension_coverage.py")),
            ],
            physics_executed=0,
            summary=summary,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.coverage, args.output)))
