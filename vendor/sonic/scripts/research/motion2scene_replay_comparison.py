#!/usr/bin/env python3
"""Fixed-data replay comparison; three refits per arm, whole-context holdouts."""

import argparse
import json
from pathlib import Path

from motion2scene_decision_study import evaluate, read_checked, ridge_choice
from motion2scene_replay_controls import (
    current_supervised_error,
    replay_control_weights,
)
from motion2scene_tune_schedule_learner import arrays
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_imitation import (
    physical_regret,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_curriculum import (
    DEFAULT_RULE,
    coverage_key,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (
    fit_timed_schedule_policy,
    schedule_layout,
)


def run(source, out):
    original = json.loads((source / "result.json").read_text())
    registration = read_checked(original["registration"])
    groups = read_checked(original["teachers"])
    ref = registration["registry"]
    bank = load_verified_registry(ref["path"], ref["sha256"])
    _, _, entries = schedule_layout(bank)
    for group in groups:
        collection = read_checked(group["collection"])
        manifest = read_checked(collection["manifest"])
        if manifest["split"] != "development":
            raise ValueError("development data only")
        group["scene"] = read_checked(manifest["scene_definition"])
    out.mkdir(parents=True, exist_ok=False)
    experiment = dict(
        source_teachers=original["teachers"],
        refits=3,
        methods=["uniform", "coverage_only", "uniform_coverage", "current_error"],
        scope="fixed-data recorded-branch proxies; no historical physical-gap comparison",
        new_physics_steps=0,
    )
    (out / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    results = []
    for held in [None, *range(len(groups))]:
        train_groups = [g for i, g in enumerate(groups) if i != held]
        targets = [r for g in train_groups for r in g["targets"]]
        data = arrays(targets)
        _, eligible = physical_regret(
            data["passed"], data["passage_time_s"], data["admitted"], data["legality"]
        )
        records = []
        for group in train_groups:
            for row in group["targets"]:
                records.append(
                    dict(
                        encounter_id=group["scene_id"],
                        phase_tick=row["phase_tick"],
                        supervision_available=bool(eligible[len(records)]),
                        gap=None,
                        coverage_key=coverage_key(bank, group["scene"], row, DEFAULT_RULE),
                    )
                )
        for mode in experiment["methods"]:
            model, _ = fit_timed_schedule_policy(
                bank, feature_names=targets[0]["feature_names"], l2=10, **data
            )
            history = []
            for iteration in range(experiment["refits"]):
                predictions = []
                for row in targets:
                    k = int(np.flatnonzero(model["phase_ticks"] == row["phase_tick"])[0])
                    z = (np.asarray(row["features"]) - model["mean"][k]) / model["std"][k]
                    predictions.append(-(z @ model["weights"][k] + model["bias"][k]))
                error = current_supervised_error(
                    predictions,
                    data["passed"],
                    data["passage_time_s"],
                    data["admitted"],
                    data["legality"],
                )
                control = replay_control_weights(records, mode, supervised_error=error)
                weights = np.asarray(control["weights"])
                # Excluded targets are ignored by the fitter, whose input contract
                # nonetheless requires strictly positive supplied sample weights.
                weights[~eligible] = 1.0
                model, _ = fit_timed_schedule_policy(
                    bank,
                    feature_names=targets[0]["feature_names"],
                    l2=10,
                    sample_weights=weights,
                    **data
                )
                history.append(dict(iteration=iteration, **control))
            assessment = evaluate(
                lambda row: ridge_choice(model, row),
                groups if held is None else [groups[held]],
                entries,
            )
            results.append(
                dict(
                    mode=mode,
                    fold="training" if held is None else groups[held]["scene_id"],
                    assessments=assessment,
                    weight_history=history,
                )
            )
    summary = {}
    for mode in experiment["methods"]:
        rows = [
            a
            for r in results
            if r["mode"] == mode and r["fold"] != "training"
            for a in r["assessments"]
        ]
        summary[mode] = dict(
            passing_branch_proxies=sum(a["branch_proxy_passed"] for a in rows),
            contexts=len(rows),
            mean_recorded_phase_regret=float(
                np.mean(
                    [
                        d["regret"]
                        for a in rows
                        for d in a["phase_decisions"]
                        if d["regret"] is not None
                    ]
                )
            ),
        )
    (out / "result.json").write_text(
        json.dumps(dict(**experiment, summary=summary, results=results), indent=2, allow_nan=False)
        + "\n"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
