#!/usr/bin/env python3
"""Verify saved replay-control readouts; no new fitting or physical evaluation."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_acquisition_curve_dataset_baseline import hashable_coverage  # noqa: E402
from motion2scene_checkpoint_controls import MODES  # noqa: E402
from motion2scene_decision_study import evaluate, read_checked, ridge_choice  # noqa: E402
from motion2scene_replay_controls import replay_control_weights  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    schedule_layout,
)


def summarize(source, checkpoints, out):
    if (
        not checkpoints
        or checkpoints != sorted(set(checkpoints))
        or any(n not in (2, 4, 8, 16, 32) for n in checkpoints)
    ):
        raise ValueError("distinct ordered declared checkpoints required")
    study_ref = artifact(source / "experiment.json")
    study = read_checked(study_ref)
    plan = read_checked(study["plan"])
    bank = load_verified_registry(plan["registry"]["path"], plan["registry"]["sha256"])
    entries = schedule_layout(bank)[2]
    records, rows, common_teachers = [], [], None
    for checkpoint in checkpoints:
        for seed in (93201, 93202, 93203):
            run_id = f"seed{seed}_observation_curriculum"
            if [run_id, checkpoint] not in study["assignments"]:
                raise ValueError("control was not assigned in the original watcher declaration")
            source_ref = artifact(source / f"{run_id}_M{checkpoint}" / "result.json")
            result = read_checked(source_ref)
            experiment = read_checked(result["experiment"])
            if (
                experiment["plan"] != study["plan"]
                or experiment["run_id"] != run_id
                or experiment["encounters"] != checkpoint
                or experiment["l2"] != 10.0
                or experiment["refits"] != 3
                or set(result["methods"]) != set(MODES)
                or result["new_physics_steps"] != 0
            ):
                raise ValueError("complete original common-data control assignment required")
            for implementation in experiment["implementations"]:
                checked(Path(implementation["path"]), implementation["sha256"])
            groups = read_checked(experiment["validation_teachers"])
            if common_teachers is None:
                common_teachers = experiment["validation_teachers"]
            if experiment["validation_teachers"] != common_teachers or len(groups) != 6:
                raise ValueError("the same six inspected development contexts must be retained")
            for group in groups:
                collection = read_checked(group["collection"])
                if read_checked(collection["manifest"])["split"] != "development":
                    raise ValueError("reserved records cannot enter these development controls")
            signals = [
                dict(r, coverage_key=hashable_coverage(r["coverage_key"]))
                for r in experiment["records"]
            ]
            methods = result["methods"]
            baseline = methods["uniform"]["assessments"]
            gate_equal = all(
                a["weights"] == b["weights"]
                for a, b in zip(
                    methods["historical_gated"]["weights"],
                    methods["historical_ungated"]["weights"],
                    strict=True,
                )
            )
            for mode in MODES:
                control = methods[mode]
                if len(control["weights"]) != 3:
                    raise ValueError("every predeclared refit weighting record is required")
                for step, weighting in enumerate(control["weights"]):
                    recomputed = replay_control_weights(
                        signals,
                        mode,
                        supervised_error=weighting["supervised_error"],
                        ungated_gaps=[r["gap"] for r in experiment["ungated"]],
                    )
                    if weighting["step"] != step or any(
                        weighting[k] != v for k, v in recomputed.items()
                    ):
                        raise ValueError(
                            "archived signals do not reproduce the original weight recipe"
                        )
                policy = checked(Path(control["policy"]["path"]), control["policy"]["sha256"])
                with np.load(policy) as model:
                    assessments = evaluate(lambda row: ridge_choice(model, row), groups, entries)
                if (
                    assessments != control["assessments"]
                    or len(assessments) != control["validation_contexts"]
                    or sum(a["branch_proxy_passed"] for a in assessments)
                    != control["passing_branch_proxies"]
                    or [a["scene_id"] for a in assessments] != [a["scene_id"] for a in baseline]
                ):
                    raise ValueError(
                        "saved-policy decisions differ from the recorded control readout"
                    )
                paired = [
                    (a, b)
                    for a, b in zip(assessments, baseline, strict=True)
                    if a["branch_proxy_passed"] and b["branch_proxy_passed"]
                ]
                rows.append(
                    dict(
                        checkpoint=checkpoint,
                        seed=seed,
                        method=mode,
                        passing_recorded_branches=control["passing_branch_proxies"],
                        contexts=len(assessments),
                        passing_branch_difference_vs_uniform=control["passing_branch_proxies"]
                        - methods["uniform"]["passing_branch_proxies"],
                        selected_schedule_changes_vs_uniform=sum(
                            a["selected_schedule"] != b["selected_schedule"]
                            for a, b in zip(assessments, baseline, strict=True)
                        ),
                        mutually_successful_recorded_contexts=len(paired),
                        mean_recorded_time_difference_vs_uniform_s=(
                            float(
                                np.mean(
                                    [
                                        a["branch_proxy_time_s"] - b["branch_proxy_time_s"]
                                        for a, b in paired
                                    ]
                                )
                            )
                            if paired
                            else None
                        ),
                        selected_schedules=[
                            bank.option_ids[a["selected_schedule"]] for a in assessments
                        ],
                        final_weights_differ_from_uniform=control["weights"][-1]["weights"]
                        != methods["uniform"]["weights"][-1]["weights"],
                    )
                )
            records.append(
                dict(
                    checkpoint=checkpoint,
                    seed=seed,
                    result=source_ref,
                    positive_historical_phase_gaps=sum((r["gap"] or 0) > 0 for r in signals),
                    supervised_phase_rows=sum(r["supervision_available"] for r in signals),
                    nonempty_phase_cue_rows=sum(
                        bool(r["deadline"]["current_phase"]["beams"]) for r in signals
                    ),
                    eligible_nonempty_phase_cue_rows=sum(
                        bool(r["deadline"]["current_phase"]["beams"])
                        and r["deadline"]["current_eligible"]
                        for r in signals
                    ),
                    historical_gated_ungated_weights_identical=gate_equal,
                )
            )
    out.mkdir(parents=True, exist_ok=False)
    with (out / "controls.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return write_new(
        out / "result.json",
        dict(
            schema="motion2scene_verified_checkpoint_control_readouts_v1",
            study=study_ref,
            implementation=artifact(Path(__file__)),
            checkpoints=checkpoints,
            validation_teachers=common_teachers,
            sources=records,
            rows=rows,
            models_rechecked=len(rows),
            recorded_context_choices_rechecked=sum(r["contexts"] for r in rows),
            weight_recipes_rechecked=3 * len(rows),
            new_model_fits=0,
            new_physics_steps=0,
            scope="Saved-policy decisions and weight recipes rechecked using archived physical/residual signals; "
            "not a new trajectory audit, fitting run, physical gap, or closed-loop policy evaluation; "
            "six reused development contexts and three training corpora per checkpoint; "
            "no independent-episode inference",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.source, args.checkpoints, args.out)))
