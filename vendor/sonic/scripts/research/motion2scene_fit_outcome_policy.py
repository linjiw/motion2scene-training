#!/usr/bin/env python3
"""Export the fixed development outcome learner for closed-loop comparison."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import OutcomeTrees, evaluate, read_checked  # noqa: E402
from motion2scene_outcome_policy import choose_action, export_policy  # noqa: E402
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


def fit(source, out):
    original = read_checked(artifact(source / "result.json"))
    registration = read_checked(original["registration"])
    groups = read_checked(original["teachers"])
    for group in groups:
        collection = read_checked(group["collection"])
        if read_checked(collection["manifest"])["split"] != "development":
            raise ValueError("development teaching only; no reserved outcomes may enter fitting")
    ref = registration["registry"]
    bank = load_verified_registry(ref["path"], ref["sha256"])
    _, _, entries = schedule_layout(bank)
    out.mkdir(parents=True, exist_ok=False)
    experiment = write_new(
        out / "experiment.json",
        dict(
            source=artifact(source / "result.json"),
            teachers=original["teachers"],
            registry=ref,
            learner="existing fixed depth-two OutcomeTrees",
            implementation=[
                artifact(Path(__file__)),
                artifact(ROOT / "scripts/research/motion2scene_decision_study.py"),
                artifact(ROOT / "scripts/research/motion2scene_outcome_policy.py"),
            ],
            threshold=0.5,
            fitting_seed=0,
            parameter_search=False,
            new_physics_steps=0,
        ),
    )
    rows = [row for group in groups for row in group["targets"]]
    learner = OutcomeTrees(rows)
    policy = export_policy(learner, bank)

    def choose(row, model=policy):
        return choose_action(
            model, row["features"], np.asarray(row["legal_mask"]), row["phase_tick"]
        )[0]

    if any(choose(row) != learner.choose(row) for row in rows):
        raise ValueError("portable tree differs on a recorded teaching decision")
    policy_ref = write_new(out / "policy.json", policy)
    training = evaluate(choose, groups, entries)
    holdouts = []
    for excluded, held in enumerate(groups):
        training_rows = [
            row for i, group in enumerate(groups) if i != excluded for row in group["targets"]
        ]
        local = OutcomeTrees(training_rows)
        model = export_policy(local, bank)
        choose_local = lambda row: choose(row, model)  # noqa: E731
        if any(choose_local(row) != local.choose(row) for row in held["targets"]):
            raise ValueError("portable tree differs on a whole-context holdout decision")
        holdouts.extend(evaluate(choose_local, [held], entries))
    return write_new(
        out / "result.json",
        dict(
            experiment=experiment,
            policy=policy_ref,
            training_branch_proxies=training,
            whole_context_holdout_branch_proxies=holdouts,
            exact_exported_training_decisions=len(rows),
            exact_exported_holdout_decisions=len(rows),
            new_physics_steps=0,
            scope="deployable model and existing recorded-branch readouts; not new physical performance",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(fit(args.source, args.out)))
