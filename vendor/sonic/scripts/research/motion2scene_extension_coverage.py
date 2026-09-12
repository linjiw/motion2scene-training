#!/usr/bin/env python3
"""Capability of each fixed extension bank, keeping unknown outcomes in the panel."""

import argparse
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


def union_bounds(states):
    """An observed passing member proves capability even if others are unknown."""
    if not states or any(s not in (-1, 0, 1) for s in states):
        raise ValueError("nonempty measured failure/pass or unknown states required")
    return [int(1 in states), int(any(s != 0 for s in states))]


def summarize(option_ids, conditions):
    if not conditions or len({c["task_id"] for c in conditions}) != len(conditions):
        raise ValueError("distinct assigned tasks required")
    if not option_ids or len(set(option_ids)) != len(option_ids) or option_ids[0] != "neutral":
        raise ValueError("distinct option IDs starting with the shared neutral required")
    arms = {
        a: [i for i, name in enumerate(option_ids) if name.startswith(a + "_")]
        for a in ("generated", "authored")
    }
    if not all(arms.values()) or set([0, *arms["generated"], *arms["authored"]]) != set(
        range(len(option_ids))
    ):
        raise ValueError("exact shared neutral and two extension arms required")
    rows, differences = [], []
    for condition in conditions:
        states, times = condition["states"], condition["times_s"]
        if len(states) != len(option_ids) or len(times) != len(states):
            raise ValueError("every assigned schedule requires an explicit outcome slot")
        union_bounds(states)
        for state, time in zip(states, times, strict=True):
            if state == 1 and (time is None or not np.isfinite(time) or time < 0):
                raise ValueError("passing branch needs its measured passage time")
            if state != 1 and time is not None:
                raise ValueError("unknown/failed branches cannot supply successful time")
        row = dict(task_id=condition["task_id"], neutral=union_bounds(states[:1]), arms={})
        extensions = {}
        for arm, ids in arms.items():
            extensions[arm] = union_bounds([states[i] for i in ids])
            selected = [0, *ids]
            known = all(states[i] != -1 for i in selected)
            costs = [times[i] for i in selected if states[i] == 1]
            row["arms"][arm] = dict(
                capability=union_bounds([states[i] for i in selected]),
                incremental_over_neutral=[
                    int(states[0] == 0 and extensions[arm][0] == 1),
                    int(states[0] != 1 and extensions[arm][1] == 1),
                ],
                all_branch_outcomes_known=known,
                fastest_measured_passing_time_s=min(costs) if costs else None,
            )
        g, a = extensions["generated"], extensions["authored"]
        row["generated_only"] = [
            int(states[0] == 0 and g[0] == 1 and a[1] == 0),
            int(states[0] != 1 and g[1] == 1 and a[0] == 0),
        ]
        row["authored_only"] = [
            int(states[0] == 0 and a[0] == 1 and g[1] == 0),
            int(states[0] != 1 and a[1] == 1 and g[0] == 0),
        ]
        row["paired_best_time_difference_s"] = None
        if all(v["all_branch_outcomes_known"] and v["capability"][0] for v in row["arms"].values()):
            delta = (
                row["arms"]["generated"]["fastest_measured_passing_time_s"]
                - row["arms"]["authored"]["fastest_measured_passing_time_s"]
            )
            row["paired_best_time_difference_s"] = delta
            differences.append(delta)
        rows.append(row)

    def total(values):
        return np.sum(values, axis=0).astype(int).tolist()

    return dict(
        assigned_tasks=len(conditions),
        assigned_branches=len(conditions) * len(option_ids),
        measured_passes=sum(c["states"].count(1) for c in conditions),
        measured_failures=sum(c["states"].count(0) for c in conditions),
        unknown_outcomes=sum(c["states"].count(-1) for c in conditions),
        neutral_capability_bounds=total([r["neutral"] for r in rows]),
        arms={
            arm: dict(
                capability_bounds=total([r["arms"][arm]["capability"] for r in rows]),
                incremental_coverage_bounds=total(
                    [r["arms"][arm]["incremental_over_neutral"] for r in rows]
                ),
            )
            for arm in arms
        },
        generated_only_bounds=total([r["generated_only"] for r in rows]),
        authored_only_bounds=total([r["authored_only"] for r in rows]),
        mutually_solvable_complete_tasks=len(differences),
        mean_paired_best_time_difference_s=float(np.mean(differences)) if differences else None,
        interpretation=(
            "finite-bank capability and best measured schedule cost, not learned policy performance; "
            "bounds reflect unknown outcomes, not statistical uncertainty"
        ),
        task_results=rows,
    )


def run(panel, output):
    prepared = read_checked(artifact(panel / "prepared.json"))
    study = read_checked(prepared["study"])
    registry = read_checked(study["registry"])
    option_ids = ["neutral"] + [o["option_id"] for o in registry["request"]["options"]]
    conditions, results = [], []
    for assignment in prepared["tasks"]:
        manifest = read_checked(assignment["collection"])
        if (
            manifest["registry"] != study["registry"]
            or manifest["split"] != "development"
            or manifest["option_ids"] != option_ids
            or len(manifest["cells"]) != len(option_ids)
            or any(c["runtime_seed"] != study["physics_seed"] for c in manifest["cells"])
        ):
            raise ValueError("capability panel bank, task allocation or execution seed differs")
        states, times = [-1] * len(option_ids), [None] * len(option_ids)
        result_path = Path(assignment["collection"]["path"]).parent / "result.json"
        if result_path.exists():
            ref = artifact(result_path)
            result = read_checked(ref)
            if result["manifest"] != assignment["collection"]:
                raise ValueError("result belongs to a different task assignment")
            seen = set()
            for row in result["rows"]:
                option = row["forced_option_id"]
                if option not in option_ids or option in seen:
                    raise ValueError("unexpected or duplicate executed schedule")
                seen.add(option)
                i = option_ids.index(option)
                outcome = row["outcome"]["task_outcome"]
                if outcome not in ("pass", "failure", "unknown"):
                    raise ValueError("unknown outcome category")
                states[i] = {"pass": 1, "failure": 0, "unknown": -1}[outcome]
                if bool(row["pass"]) != (states[i] == 1):
                    raise ValueError("physical passage labels disagree")
                times[i] = row["costs"]["passage_time_s"] if states[i] == 1 else None
            results.append(ref)
        conditions.append(dict(task_id=assignment["task_id"], states=states, times_s=times))
    return write_new(
        output,
        dict(
            prepared=artifact(panel / "prepared.json"),
            results=results,
            implementation=artifact(Path(__file__)),
            conditions=conditions,
            summary=summarize(option_ids, conditions),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.panel, args.output)))
