#!/usr/bin/env python3
"""Summarize complete primary checkpoints without interrupting GPU acquisition."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_acquisition_yield import run as task_yield
from motion2scene_cross_corpus_readout import run as cross_corpus
from motion2scene_decision_study import read_checked
from motion2scene_run_primary_acquisition import acquisition_lock, commit
from motion2scene_timing_diagnostic import artifact, checked


def missing_inputs(plan, budget):
    """Do not start an arm comparison until every assigned corpus has completed."""
    missing = []
    for run in plan["runs"]:
        rows = run["rounds"][: budget + 1]
        if len(rows) != budget + 1:
            raise ValueError("budget is outside the assigned acquisition trajectory")
        for row in rows:
            names = [
                "expected_postupdate_training_result_path",
                "expected_postupdate_policy_path",
                "expected_teacher_result_path",
            ]
            if row["round_index"]:
                names.append("expected_student_result_path")
            missing.extend(str(row[name]) for name in names if not Path(row[name]).is_file())
    return missing


def work(plan_path, out):
    plan_ref = artifact(plan_path)
    plan = read_checked(plan_ref)
    runs = {(r["physics_seed"], r["arm"]) for r in plan["runs"]}
    expected = {
        (seed, arm)
        for seed in (93201, 93202, 93203)
        for arm in ("uniform", "target_only", "analytic_contrast", "observation_curriculum")
    }
    if len(plan["runs"]) != 12 or runs != expected:
        raise ValueError("complete original four-arm, three-seed acquisition required")
    sources = [
        artifact(Path(__file__).with_name(name))
        for name in (
            "motion2scene_watch_primary_readouts.py",
            "motion2scene_acquisition_yield.py",
            "motion2scene_cross_corpus_readout.py",
        )
    ]
    out.mkdir(parents=True, exist_ok=True)
    experiment = commit(
        out / "experiment.json",
        dict(
            plan=plan_ref,
            checkpoints=[2, 4],
            implementations=sources,
            new_physics_steps=0,
            analysis="complete acquired-task yield and cross-seed recorded branch selections",
            interpretation=(
                "the cross-corpus validation pool grows with acquisition budget; "
                "this is not a fixed held-out learning curve"
            ),
        ),
    )
    with acquisition_lock(out / ".worker.lock"):
        previous = None
        while True:
            for source in sources:
                checked(Path(source["path"]), source["sha256"])
            waiting = {}
            for budget in (2, 4):
                target = out / f"M{budget}"
                completion = target / "complete.json"
                if completion.is_file():
                    result = read_checked(artifact(completion))
                    if result["experiment"] != experiment or result["budget"] != budget:
                        raise ValueError("existing readout belongs to another experiment")
                    for name in ("task_yield", "cross_corpus"):
                        read_checked(result[name])
                    continue
                missing = missing_inputs(plan, budget)
                if missing:
                    waiting[budget] = len(missing)
                    continue
                target.mkdir(parents=True, exist_ok=True)
                print(json.dumps(dict(status="running_readouts", budget=budget)), flush=True)
                results = {}
                for name, analyze in (("task_yield", task_yield), ("cross_corpus", cross_corpus)):
                    destination = target / name
                    if destination.exists():
                        raise ValueError(
                            "preserve partial CPU analysis and inspect before continuation"
                        )
                    results[name] = analyze(plan_path, budget, destination)
                commit(completion, dict(experiment=experiment, budget=budget, **results))
                print(
                    json.dumps(dict(status="completed_readouts", budget=budget, **results)),
                    flush=True,
                )
            if not waiting:
                return commit(
                    out / "complete.json", dict(experiment=experiment, checkpoints=[2, 4])
                )
            if waiting != previous:
                print(
                    json.dumps(
                        dict(status="waiting_for_complete_checkpoints", missing_inputs=waiting)
                    ),
                    flush=True,
                )
                previous = waiting
            time.sleep(45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(work(args.plan, args.out)))
