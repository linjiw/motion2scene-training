#!/usr/bin/env python3
"""Fit development replay controls as fixed acquisition checkpoints complete."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_checkpoint_controls import MODES, bound, checkpoint_slots, run
from motion2scene_run_primary_acquisition import acquisition_lock, commit
from motion2scene_timing_diagnostic import artifact, checked


def work(plan_path, validation, out):
    plan_ref, validation_ref = artifact(plan_path), artifact(validation / "result.json")
    plan = bound(plan_ref)
    if plan["schema"] != "motion2scene_expanded_acquisition_v1":
        raise ValueError("expanded checkpoint trajectory required")
    assignments = [
        (r["run_id"], k)
        for k in (2, 4, 8, 16, 32)
        for r in plan["runs"]
        if r["arm"] == "observation_curriculum"
    ]
    out.mkdir(parents=True, exist_ok=True)
    experiment = dict(
        plan=plan_ref,
        validation=validation_ref,
        methods=list(MODES),
        refits=3,
        assignments=assignments,
        new_physics_steps=0,
        selection="all three observation-curriculum corpora at every specified checkpoint",
        implementation=[
            artifact(Path(__file__)),
            artifact(Path(__file__).with_name("motion2scene_checkpoint_controls.py")),
        ],
    )
    # Canonicalize tuples before the idempotent JSON comparison.
    experiment = json.loads(json.dumps(experiment))
    commit(out / "experiment.json", experiment)
    previous = None
    with acquisition_lock(out / ".worker.lock"):
        while True:
            completed, waiting = 0, []
            for run_id, budget in assignments:
                destination = out / f"{run_id}_M{budget}"
                result = destination / "result.json"
                if result.exists():
                    measured = bound(artifact(result))
                    inputs = bound(measured["experiment"])
                    if (
                        inputs["plan"] != plan_ref
                        or inputs["run_id"] != run_id
                        or inputs["encounters"] != budget
                    ):
                        raise ValueError("saved controls differ from the fixed assignment")
                    for method in measured["methods"].values():
                        ref = method["policy"]
                        checked(Path(ref["path"]), ref["sha256"])
                    completed += 1
                elif checkpoint_slots(plan, run_id, budget)[-1]["model"].is_file():
                    result_ref = run(plan_path, run_id, budget, validation, destination)
                    print(
                        json.dumps(
                            dict(
                                status="checkpoint_controls_complete",
                                run_id=run_id,
                                budget=budget,
                                result=result_ref,
                            )
                        ),
                        flush=True,
                    )
                    completed += 1
                else:
                    waiting.append((run_id, budget))
            state = dict(
                completed=completed, assigned=len(assignments), waiting=waiting, new_physics_steps=0
            )
            if state != previous:
                print(json.dumps(state), flush=True)
                previous = state
            if not waiting:
                return commit(out / "completion.json", state)
            time.sleep(45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "validation", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(work(args.plan, args.validation, args.out)))
