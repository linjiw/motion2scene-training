#!/usr/bin/env python3
"""Fit matched generated/authored selectors with whole-center development holdouts."""

import argparse
import json
from math import fsum
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_extension_teaching import (  # noqa: E402
    arm_view,
    longitudinal_folds,
    outcomes_from_audited_group,
    restricted_teacher,
)
from motion2scene_run_extension_tasks import verify_panel  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402
from motion2scene_train_timed_schedules import audit_collection  # noqa: E402
from motion2scene_tune_schedule_learner import arrays  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    definition_digest,
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    fit_timed_schedule_policy,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    expected_feature_names,
    load_schedule_policy,
    validate_schedule_policy,
)


def select_constant(view, results):
    """Select using training outcomes only: passage, successful time, fixed order."""
    if not results:
        raise ValueError("nonempty training-only physical tables required")
    ranking = []
    for order, option in enumerate(view.option_ids):
        times, passed = [], 0
        for result in results:
            matching = [r for r in result["rows"] if r["forced_option_id"] == option]
            if len(matching) != 1:
                raise ValueError(
                    "exactly one assigned branch per training task and schedule required"
                )
            row = matching[0]
            outcome = row["outcome"]["task_outcome"]
            if outcome not in ("pass", "failure") or bool(row["pass"]) != (outcome == "pass"):
                raise ValueError("constant selection requires known physical training outcomes")
            if outcome == "pass":
                cost = row["costs"]["passage_time_s"]
                if cost is None or not np.isfinite(cost) or cost < 0:
                    raise ValueError("measured successful passage time required")
                times.append(cost)
                passed += 1
        ranking.append(
            dict(option_id=option, order=order, passages=passed, successful_time_sum_s=fsum(times))
        )
    best = min(ranking, key=lambda r: (-r["passages"], r["successful_time_sum_s"], r["order"]))
    return dict(
        selected_option_id=best["option_id"], assigned_training_tasks=len(results), ranking=ranking
    )


def fit_fold(bank, arm, fold, training):
    """Accept only the eight assigned training tasks, not a whole panel to filter."""
    expected = fold["training_task_ids"]
    if (
        len(expected) != 8
        or len(set(expected)) != 8
        or len(fold["evaluation_task_ids"]) != 4
        or set(expected) & set(fold["evaluation_task_ids"])
        or list(training) != expected
    ):
        raise ValueError(
            "exact ordered eight-task training fold without held-center inputs required"
        )
    view, indices = arm_view(bank, arm)
    targets, sources, measured_steps, unknown_steps = [], [], 0, 0
    for task_id, (group, result) in training.items():
        if group["scene_id"] != task_id:
            raise ValueError("training task identity differs from its independently checked scene")
        branches = outcomes_from_audited_group(bank, group, result)
        for target in group["targets"]:
            if not target["available"]:
                continue  # No unrecorded neutral history or input is synthesized.
            targets.append(
                restricted_teacher(
                    bank,
                    arm,
                    branches,
                    target["phase_tick"],
                    target["recorded_history_sha256"],
                    group["physics_seed"],
                    target["feature_names"],
                    target["features"],
                    np.asarray(target["legal_mask"], bool),
                )
            )
        for row in group["branch_assessments"]:
            if row["option_id"] in view.option_ids:
                steps = row["physics_steps"]
                measured_steps += steps or 0
                unknown_steps += steps is None
        sources.append(group["collection"])
    constant = select_constant(view, [item[1] for item in training.values()])
    model, report = fit_timed_schedule_policy(
        view,
        feature_names=expected_feature_names(len(view.option_ids)),
        l2=10.0,
        allow_measured_tie_initialization=True,
        **arrays(targets),
    )
    validate_schedule_policy(model, view)
    if not np.array_equal(model["trained_mask"], model["qualified_mask"]):
        raise ValueError("every available arm action requires a trained phase head")
    return model, dict(
        arm=arm,
        fold=fold,
        training_collections=sources,
        targets=targets,
        fit=report,
        constant=constant,
        original_option_indices=indices.tolist(),
        assigned_training_branches=len(expected) * len(view.option_ids),
        measured_training_physics_steps=measured_steps,
        training_branches_with_unknown_step_count=unknown_steps,
        scope="training-only fit; held-center outcomes and new closed-loop executions not used",
    )


def declare(panel, out):
    prepared = verify_panel(panel)
    source = read_checked(prepared["study"])
    plan = read_checked(source["plan"])
    bank = load_verified_registry(source["registry"]["path"], source["registry"]["sha256"])
    for arm in ("generated", "authored"):
        if len(arm_view(bank, arm)[0].option_ids) != 9:
            raise ValueError("shared neutral and eight schedules per arm required")
    folds = longitudinal_folds(plan["specification"]["independent_capability_tasks"])
    out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__)])):
        snapshot = out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(snapshot)})
    return write_new(
        out / "study.json",
        dict(
            schema="motion2scene_extension_equivalent_teaching_v1",
            prepared=artifact(panel / "prepared.json"),
            registry=source["registry"],
            folds=folds,
            arms=["generated", "authored"],
            l2=10.0,
            allow_measured_tie_initialization=True,
            replay="uniform",
            parameter_search=False,
            source_closure=sources,
            fitting_models=6,
            assigned_future_policy_episodes=24,
            maximum_future_policy_physics_steps=28608,
            policy_evaluation_seed=97001,
            constant_selection="training passage count, successful time sum, original option order",
            scope="whole-center procedure-development folds; not the reserved curriculum benchmark",
            physics_executed=0,
        ),
    )


def validate_design(study, tasks, registry):
    expected = dict(
        schema="motion2scene_extension_equivalent_teaching_v1",
        registry=registry,
        folds=longitudinal_folds(tasks),
        arms=["generated", "authored"],
        l2=10.0,
        allow_measured_tie_initialization=True,
        replay="uniform",
        parameter_search=False,
        fitting_models=6,
        assigned_future_policy_episodes=24,
        maximum_future_policy_physics_steps=28608,
        policy_evaluation_seed=97001,
    )
    if any(study.get(key) != value for key, value in expected.items()):
        raise ValueError("declared folds, common learner, bank or physical allocation changed")


def run(out):
    study_ref = artifact(out / "study.json")
    study = read_checked(study_ref)
    for source in study["source_closure"]:
        checked(Path(source["path"]), source["sha256"])
        checked(Path(source["snapshot"]["path"]), source["sha256"])
    prepared = read_checked(study["prepared"])
    if verify_panel(Path(study["prepared"]["path"]).parent) != prepared:
        raise ValueError("fixed task panel differs from the declared study")
    source = read_checked(prepared["study"])
    plan = read_checked(source["plan"])
    validate_design(
        study, plan["specification"]["independent_capability_tasks"], source["registry"]
    )
    ref = study["registry"]
    bank = load_verified_registry(ref["path"], ref["sha256"])
    assignments = {t["task_id"]: t for t in prepared["tasks"]}
    cache, fitted = {}, []
    for i, fold in enumerate(study["folds"]):
        training = {}
        for task_id in fold["training_task_ids"]:
            assignment = assignments[task_id]
            if task_id not in cache:
                path = Path(assignment["collection"]["path"]).parent / "result.json"
                if not path.is_file():
                    raise FileNotFoundError(
                        "assigned physical task collection is not complete: " + task_id
                    )
                result = read_checked(artifact(path))
                if result["manifest"] != assignment["collection"]:
                    raise ValueError("training result belongs to another assigned collection")
                cache[task_id] = (audit_collection(path, bank, ref), result)
            training[task_id] = cache[task_id]
        for arm in study["arms"]:
            folder = out / f"fold_{i}_{arm}"
            if (folder / "result.json").exists():
                saved = read_checked(artifact(folder / "result.json"))
                if (
                    saved["study"] != study_ref
                    or saved["analysis"]["fold"] != fold
                    or saved["analysis"]["arm"] != arm
                ):
                    raise ValueError("existing model belongs to a different fitting assignment")
                load_schedule_policy(
                    saved["policy"]["path"], saved["policy"]["sha256"], arm_view(bank, arm)[0]
                )
                fitted.append(artifact(folder / "result.json"))
                continue
            folder.mkdir(parents=True, exist_ok=False)
            try:
                model, report = fit_fold(bank, arm, fold, training)
            except ValueError as error:
                write_new(
                    folder / "fit_failure.json",
                    dict(
                        study=study_ref,
                        arm=arm,
                        fold=fold,
                        error=str(error),
                        training_collections=[g["collection"] for g, _ in training.values()],
                        status="no policy produced; not a physical traversal failure",
                        physics_executed=0,
                    ),
                )
                raise
            with (folder / "policy.npz").open("xb") as stream:
                np.savez_compressed(stream, **model)
            policy_ref = artifact(folder / "policy.npz")
            view = arm_view(bank, arm)[0]
            load_schedule_policy(policy_ref["path"], policy_ref["sha256"], view)
            fitted.append(
                write_new(
                    folder / "result.json",
                    dict(
                        study=study_ref,
                        registry=ref,
                        arm=arm,
                        logical_request_digest=definition_digest(view.request),
                        policy=policy_ref,
                        analysis=report,
                        physics_executed=0,
                    ),
                )
            )
            print(json.dumps(dict(fitted_arm=arm, held_center_m=fold["held_center_m"])), flush=True)
    return write_new(
        out / "result.json",
        dict(
            study=study_ref,
            models=fitted,
            physics_executed=0,
            status="six models fitted; held-center policy execution remains separate",
        ),
    )


def watch(out):
    """Wait for the assigned capability panel, then fit once without simulation."""
    study = read_checked(artifact(out / "study.json"))
    prepared = read_checked(study["prepared"])
    coverage_path = Path(study["prepared"]["path"]).parent / "coverage.json"
    previous = None
    while True:
        missing = sum(
            not (Path(t["collection"]["path"]).parent / "result.json").is_file()
            for t in prepared["tasks"]
        )
        ready = coverage_path.is_file()
        state = (missing, ready)
        if state != previous:
            print(
                json.dumps(
                    dict(
                        status="waiting_for_fixed_task_physics",
                        missing_tasks=missing,
                        coverage_ready=ready,
                    )
                ),
                flush=True,
            )
            previous = state
        if not missing and ready:
            coverage = read_checked(artifact(coverage_path))
            if (
                coverage["prepared"] != study["prepared"]
                or coverage["summary"]["assigned_branches"] != 204
                or coverage["summary"]["unknown_outcomes"] != 0
            ):
                raise ValueError("complete assigned capability outcomes required before fitting")
            return run(out)
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("declare", "fit", "watch"))
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "declare" and args.panel is None:
        parser.error("declare requires --panel")
    operation = dict(
        declare=lambda: declare(args.panel, args.out),
        fit=lambda: run(args.out),
        watch=lambda: watch(args.out),
    )
    print(json.dumps(operation[args.operation]()))
