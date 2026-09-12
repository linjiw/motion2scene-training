#!/usr/bin/env python3
"""Measured passage/time opportunity beyond fixed schedules on a development pool."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)


def summarize(option_ids, conditions, selections):
    """Choose the strongest fixed reference on this inspected pool, not on a test set.

    The per-task fastest passing schedule is a privileged reference. This
    calculation alone cannot establish that an observation-conditioned policy
    can realize it. Counts and costs remain separate; no failure cost is invented.
    """
    if not option_ids or len(set(option_ids)) != len(option_ids):
        raise ValueError("distinct ordered schedule identities required")
    tasks = {}
    for c in conditions:
        if (
            c["task_id"] in tasks
            or len(c["passed"]) != len(option_ids)
            or len(c["times_s"]) != len(option_ids)
        ):
            raise ValueError("distinct complete task/schedule table required")
        for passed, time in zip(c["passed"], c["times_s"], strict=True):
            if type(passed) is not bool:
                raise ValueError("known physical success or failure required")
            if (
                passed and (type(time) not in (int, float) or not math.isfinite(time) or time <= 0)
            ) or (not passed and time is not None):
                raise ValueError("positive measured successful time; null failed time required")
        tasks[c["task_id"]] = c
    if not tasks:
        raise ValueError("nonempty task population required")
    ranking = [
        dict(
            option_id=name,
            schedule_index=i,
            passages=sum(c["passed"][i] for c in conditions),
            successful_time_sum_s=math.fsum(c["times_s"][i] for c in conditions if c["passed"][i]),
        )
        for i, name in enumerate(option_ids)
    ]
    best = min(
        ranking, key=lambda r: (-r["passages"], r["successful_time_sum_s"], r["schedule_index"])
    )
    reference = {t: best["schedule_index"] for t in tasks}
    oracle = {
        t: min(
            (i for i, p in enumerate(c["passed"]) if p),
            key=lambda i: (c["times_s"][i], i),
            default=None,
        )
        for t, c in tasks.items()
    }

    def assessment(choices):
        if set(choices) != set(tasks) or any(
            i is not None and (type(i) is not int or not 0 <= i < len(option_ids))
            for i in choices.values()
        ):
            raise ValueError("one supported selection per matched task required")
        successes, differences, excess = [], [], []
        for t, i in choices.items():
            c, b = tasks[t], reference[t]
            if i is None:
                if any(c["passed"]):
                    raise ValueError("missing selection for a solvable task")
                continue
            if c["passed"][i]:
                successes.append(t)
                excess.append(c["times_s"][i] - c["times_s"][oracle[t]])
                if c["passed"][b]:
                    differences.append(c["times_s"][i] - c["times_s"][b])
        return dict(
            assigned_tasks=len(tasks),
            passages=len(successes),
            passage_difference_from_best_fixed=len(successes) - best["passages"],
            mutually_successful_with_best_fixed=len(differences),
            mean_paired_time_minus_best_fixed_s=(
                math.fsum(differences) / len(differences) if differences else None
            ),
            mean_excess_time_on_selected_successes_s=(
                math.fsum(excess) / len(excess) if excess else None
            ),
            selected_schedule_counts=dict(
                Counter(
                    option_ids[i] if i is not None else "bank_unsolvable" for i in choices.values()
                )
            ),
        )

    return dict(
        assigned_tasks=len(tasks),
        fixed_ranking=ranking,
        best_fixed=best,
        bank_capability=sum(i is not None for i in oracle.values()),
        privileged_fastest_passing=assessment(oracle),
        selections={name: assessment(choices) for name, choices in selections.items()},
        interpretation="In-pool fixed reference and privileged per-scene opportunity; not held-out selection or sensor realizability.",
    )


def run(readout, output):
    ref = artifact(readout)
    data = read_checked(ref)
    results, sources = [], {}
    for corpus in data["corpora"]:
        conditions, ids = [], None
        scene_ids = []
        for slot in corpus["validation"]:
            result = read_checked(slot["collection"])
            manifest = read_checked(result["manifest"])
            if (
                manifest["split"] != "development"
                or manifest["scene_definition"] != slot["scene_definition"]
            ):
                raise ValueError("matched development task definition required")
            if any(c["runtime_seed"] != slot["seed"] for c in manifest["cells"]):
                raise ValueError("matched physical seed required")
            option_ids = manifest["option_ids"]
            if ids is not None and ids != option_ids:
                raise ValueError("common ordered schedule bank required")
            ids = option_ids
            by_option = {r["forced_option_id"]: r for r in result["rows"]}
            if len(by_option) != len(result["rows"]) or set(by_option) != set(ids):
                raise ValueError("exact complete forced schedule table required")
            rows = [by_option[i] for i in ids]
            for row in rows:
                if row["outcome"]["task_outcome"] not in ("pass", "failure") or row["pass"] != (
                    row["outcome"]["task_outcome"] == "pass"
                ):
                    raise ValueError(
                        "only complete known physical outcomes may enter this comparison"
                    )
            scene = read_checked(slot["scene_definition"])
            task = slot["geometry_key"]
            scene_ids.append((scene["scene_id"], task))
            conditions.append(
                dict(
                    task_id=task,
                    passed=[r["pass"] for r in rows],
                    times_s=[r["costs"]["passage_time_s"] if r["pass"] else None for r in rows],
                )
            )
            sources[slot["collection"]["path"]] = slot["collection"]
        assessment = corpus["assessment"]
        if [a["scene_id"] for a in assessment] != [s for s, _ in scene_ids]:
            raise ValueError("recorded selections must align exactly with validation tasks")
        choices = {}
        for (_, task), c, a in zip(scene_ids, conditions, assessment, strict=True):
            i = a["selected_schedule"]
            if type(i) is not int or not 0 <= i < len(ids):
                raise ValueError("unsupported recorded selection")
            if (
                a["branch_proxy_passed"] != c["passed"][i]
                or a["branch_proxy_time_s"] != c["times_s"][i]
            ):
                raise ValueError("readout outcome differs from its selected physical branch")
            choices[task] = i
        results.append(
            dict(
                run_id=corpus["run_id"],
                seed=corpus["seed"],
                arm=corpus["arm"],
                summary=summarize(ids, conditions, {"learned_recorded_branch": choices}),
            )
        )
    return write_new(
        output,
        dict(
            readout=ref,
            implementation=artifact(Path(__file__)),
            physical_tables=list(sources.values()),
            corpora=results,
            new_physics_steps=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.readout, args.output)))
