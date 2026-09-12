#!/usr/bin/env python3
"""Held-center policy comparisons with training-selected, arm-restricted baselines."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_development_panel_statistics import (  # noqa: E402
    paired_time,
    summarize as policy_summary,
)
from motion2scene_extension_coverage import summarize as bank_summary  # noqa: E402
from motion2scene_extension_policy_panel import (  # noqa: E402
    bind_models,
    check_prepared,
    verify_study,
)
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)


def summarize(option_ids, conditions, policies):
    """Every arm predicts every task once; folds are not independent corpus replicas."""
    bank_summary(option_ids, conditions)
    tasks = {c["task_id"]: c for c in conditions}
    matrix = {}
    folds = {}
    constants = {}
    for row in policies:
        arm, task, fold = row["arm"], row["task_id"], row["fold_index"]
        key = arm, task
        if arm not in ("generated", "authored") or task not in tasks or key in matrix:
            raise ValueError("distinct declared arm/task policy assignments required")
        if type(fold) is not int or fold < 0:
            raise ValueError("nonnegative held-center fold index required")
        if task in folds and folds[task] != fold:
            raise ValueError("both arms must use the same held-center fold")
        folds[task] = fold
        if (arm, fold) in constants and constants[arm, fold] != row["training_constant_id"]:
            raise ValueError("one training-selected constant per arm and fold required")
        constants[arm, fold] = row["training_constant_id"]
        matrix[key] = row
    if set(matrix) != {(a, t) for a in ("generated", "authored") for t in tasks}:
        raise ValueError("complete two-arm task matrix required, including missing outcomes")

    def measure(policy, task, state, cost):
        return dict(
            policy_id=policy,
            scene_id=task,
            physics_seed=97001,
            status={1: "pass", 0: "failure", -1: "technical_missing"}[state],
            time_s=cost,
        )

    arms, learned = {}, {}
    for arm in ("generated", "authored"):
        ids = [i for i, name in enumerate(option_ids) if i == 0 or name.startswith(arm + "_")]
        own_ids = [option_ids[i] for i in ids]
        rows, learned[arm] = [], {}
        for task, condition in tasks.items():
            pred = matrix[arm, task]
            constant = pred["training_constant_id"]
            if constant not in own_ids:
                raise ValueError("training constant must belong to its own motion bank")
            for i in ids:
                rows.append(
                    measure(option_ids[i], task, condition["states"][i], condition["times_s"][i])
                )
            i = option_ids.index(constant)
            rows.append(
                measure("training_constant", task, condition["states"][i], condition["times_s"][i])
            )
            if pred["state"] not in (-1, 0, 1):
                raise ValueError("policy state must be pass, failure or unknown")
            value = measure("learned", task, pred["state"], pred["time_s"])
            learned[arm][task] = value
            rows.append(value)
        arms[arm] = policy_summary(rows, own_ids, reference="training_constant")

    def comparison(task_ids):
        a = {t: learned["generated"][t] for t in task_ids}
        b = {t: learned["authored"][t] for t in task_ids}
        complete = all(r["status"] != "technical_missing" for r in [*a.values(), *b.values()])
        return dict(
            assigned_tasks=len(task_ids),
            passage_count_difference=(
                (
                    sum(r["status"] == "pass" for r in a.values())
                    - sum(r["status"] == "pass" for r in b.values())
                )
                if complete
                else None
            ),
            **paired_time(a, b),
        )

    return dict(
        assigned_policy_episodes=len(policies),
        arms=arms,
        generated_minus_authored=comparison(sorted(tasks)),
        held_center_comparisons={
            str(f): comparison(sorted(t for t in tasks if folds[t] == f))
            for f in sorted(set(folds.values()))
        },
        interpretation=(
            "Matched descriptive comparisons of one generated and one authored bank; "
            "held-center folds share training tasks and are not independent corpus replicas. "
            "Bank and constant outcomes reuse the fixed-task forced executions. "
            "Time differences use mutually successful tasks only."
        ),
    )


def run(panel, coverage_path, output):
    study_ref, study, fitting = verify_study(panel)
    prepared_ref = artifact(panel / "prepared.json")
    prepared = read_checked(prepared_ref)
    models = bind_models(study, fitting)
    check_prepared(prepared, study_ref, study, models)
    coverage_ref = artifact(coverage_path)
    coverage = read_checked(coverage_ref)
    if coverage["prepared"] != study["capability_prepared"]:
        raise ValueError("forced schedules must use the matched capability task allocation")
    registry = read_checked(study["registry"])
    ids = ["neutral", *[o["option_id"] for o in registry["request"]["options"]]]
    if coverage["summary"] != bank_summary(ids, coverage["conditions"]):
        raise ValueError("capability summary differs from its complete outcome table")
    for ref in coverage["results"]:
        read_checked(ref)
    rows, results = [], []
    for slot in prepared["assignments"]:
        model = read_checked(slot["model_result"])
        manifest = read_checked(slot["episode"])
        if (
            manifest["model_result"] != slot["model_result"]
            or manifest["nominal_template"] != slot["nominal_template"]
        ):
            raise ValueError("policy episode model or matched task differs")
        state, cost = -1, None
        result_path = Path(slot["episode"]["path"]).parent / "result.json"
        if result_path.exists():
            ref = artifact(result_path)
            result = read_checked(ref)
            if result["manifest"] != slot["episode"]:
                raise ValueError("policy outcome belongs to a different physical assignment")
            physical = result["row"]
            outcome = physical["outcome"]["task_outcome"]
            if outcome not in ("pass", "failure", "unknown"):
                raise ValueError("unsupported physical outcome")
            state = {"pass": 1, "failure": 0, "unknown": -1}[outcome]
            if bool(physical["pass"]) != (state == 1):
                raise ValueError("physical outcome labels disagree")
            if state == 1:
                if not result["complete_policy_trace_verified"]:
                    raise ValueError("passing policy needs its complete decision trace")
                cost = physical["costs"]["passage_time_s"]
            results.append(ref)
        rows.append(
            dict(
                arm=slot["arm"],
                task_id=slot["task_id"],
                fold_index=slot["fold_index"],
                training_constant_id=model["analysis"]["constant"]["selected_option_id"],
                state=state,
                time_s=cost,
            )
        )
    return write_new(
        output,
        dict(
            study=study_ref,
            prepared=prepared_ref,
            coverage=coverage_ref,
            results=results,
            implementation=artifact(Path(__file__)),
            conditions=rows,
            summary=summarize(ids, coverage["conditions"], rows),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.panel, args.coverage, args.output)))
