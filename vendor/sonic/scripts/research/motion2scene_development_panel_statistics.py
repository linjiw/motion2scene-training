#!/usr/bin/env python3
"""Separate measured capability, policy selection and paired successful time."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_development_checkpoint_panel import (  # noqa: E402
    PHYSICS_SEED,
    assignments as expected_assignments,
    bind_models,
    check_assignment,
)
from motion2scene_evaluation_statistics import completion_summary, variation  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)

STATUSES = ("pass", "failure", "technical_missing", "not_run")


def paired_time(a, b):
    """Identical condition keys, and only the intersection of measured successes."""
    if set(a) != set(b):
        raise ValueError("paired time requires identical evaluation conditions")
    deltas = [
        a[k]["time_s"] - b[k]["time_s"]
        for k in sorted(a)
        if a[k]["status"] == b[k]["status"] == "pass"
    ]
    return dict(
        mutually_successful_contexts=len(deltas),
        mean_time_difference_s=math.fsum(deltas) / len(deltas) if deltas else None,
    )


def summarize(rows, fixed_ids, reference="script"):
    """Keep unexecuted slots; never turn missing bank outcomes into infeasibility."""
    matrix, conditions = {}, set()
    for row in rows:
        key = (row["scene_id"], row["physics_seed"])
        policy, status, cost = row["policy_id"], row["status"], row["time_s"]
        if (
            status not in STATUSES
            or (
                status == "pass"
                and (type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0)
            )
            or (status != "pass" and cost is not None)
        ):
            raise ValueError("known pass needs positive time; other outcomes need null time")
        if key in matrix.setdefault(policy, {}):
            raise ValueError("duplicate policy and evaluation condition")
        matrix[policy][key] = row
        conditions.add(key)
    if (
        not conditions
        or len(set(fixed_ids)) != len(fixed_ids)
        or not fixed_ids
        or not set(fixed_ids).issubset(matrix)
        or reference not in matrix
        or any(set(values) != conditions for values in matrix.values())
    ):
        raise ValueError(
            "complete rectangular assignment matrix and declared reference bank required"
        )
    capability = {}
    for key in sorted(conditions):
        statuses = [matrix[p][key]["status"] for p in fixed_ids]
        passing = [p for p in fixed_ids if matrix[p][key]["status"] == "pass"]
        complete = all(s in ("pass", "failure") for s in statuses)
        capability[key] = dict(
            scene_id=key[0],
            physics_seed=key[1],
            measured_passing_schedules=passing,
            complete_bank_measurement=complete,
            lower=int(bool(passing)),
            upper=int(bool(passing) or not complete),
            best_passing_time_s=(
                min(matrix[p][key]["time_s"] for p in passing) if complete and passing else None
            ),
        )
    policies = {}
    for policy, values in matrix.items():
        matched = [
            k
            for k in conditions
            if capability[k]["complete_bank_measurement"]
            and values[k]["status"] in ("pass", "failure")
        ]
        selected_failures = [
            k for k in matched if capability[k]["lower"] == 1 and values[k]["status"] == "failure"
        ]
        policy_only = [
            k for k in matched if capability[k]["upper"] == 0 and values[k]["status"] == "pass"
        ]
        policies[policy] = dict(
            **completion_summary(list(values.values())),
            paired_with_reference=paired_time(values, matrix[reference]),
            paired_with_fixed_schedules={p: paired_time(values, matrix[p]) for p in fixed_ids},
            complete_matched_capability_conditions=len(matched),
            selection_failures=len(selected_failures),
            selection_failure_conditions=[list(k) for k in sorted(selected_failures)],
            policy_only_successes=len(policy_only),
            policy_only_success_conditions=[list(k) for k in sorted(policy_only)],
            measured_capability_minus_policy_passages=(
                sum(capability[k]["lower"] - int(values[k]["status"] == "pass") for k in matched)
                if len(matched) == len(conditions)
                else None
            ),
        )
    counts = Counter(r["status"] for r in rows)
    return dict(
        assigned_episodes=len(rows),
        outcome_counts={s: counts[s] for s in STATUSES},
        complete=all(r["status"] in ("pass", "failure") for r in rows),
        capability=list(capability.values()),
        capability_count_lower=sum(c["lower"] for c in capability.values()),
        capability_count_upper=sum(c["upper"] for c in capability.values()),
        policies=policies,
        reference_policy=reference,
        interpretation=(
            "capability bounds reflect unavailable measurements, not confidence intervals; "
            "policy-only successes are reported as matched-repeat disagreements, not clipped away"
        ),
        time_difference_direction="policy minus named comparator; negative is faster",
    )


def measure(row):
    outcome = row["outcome"]["task_outcome"]
    if outcome not in ("pass", "failure", "unknown"):
        raise ValueError("unsupported physical attempt outcome")
    if bool(row["pass"]) != (outcome == "pass"):
        raise ValueError("pass flag differs from measured task outcome")
    return dict(
        status="technical_missing" if outcome == "unknown" else outcome,
        time_s=row["costs"]["passage_time_s"],
        measurement_admitted=row["measurement_admitted"],
        recorded_physics_steps=row["outcome"]["physics_steps_recorded"],
    )


def construction_comparisons(rows, models):
    matrix = {}
    for row in rows:
        matrix.setdefault(row["policy_id"], {})[row["scene_id"], row["physics_seed"]] = row
    names = {(m["arm"], m["seed"]): m["run_id"] for m in models}
    pairs = [
        ("analytic_contrast", "uniform"),
        ("analytic_contrast", "target_only"),
        ("analytic_contrast", "reference_contrast"),
        ("observation_curriculum", "analytic_contrast"),
    ]
    results = {}
    for arm, reference in pairs:
        corpora = []
        for seed in (93201, 93202, 93203):
            a, b = matrix[names[arm, seed]], matrix[names[reference, seed]]
            paired = paired_time(a, b)
            complete = all(r["status"] in ("pass", "failure") for r in [*a.values(), *b.values()])
            difference = (
                (
                    sum(r["status"] == "pass" for r in a.values())
                    - sum(r["status"] == "pass" for r in b.values())
                )
                if complete
                else None
            )
            corpora.append(
                dict(
                    seed=seed,
                    policy=names[arm, seed],
                    reference=names[reference, seed],
                    passage_count_difference=difference,
                    **paired,
                )
            )
        differences = [r["passage_count_difference"] for r in corpora]
        results[arm + "_minus_" + reference] = dict(
            corpora=corpora,
            descriptive_passage_difference_variation=(
                variation(differences) if all(d is not None for d in differences) else None
            ),
        )
    return results


def run(panel, out):
    prepared_ref = artifact(panel / "prepared.json")
    prepared = read_checked(prepared_ref)
    study = read_checked(prepared["study"])
    plan = read_checked(study["expanded_plan"])
    if prepared["models"] != bind_models(study):
        raise ValueError("statistics must use the declared acquired checkpoints")
    bank = read_checked(study["registry"])
    option_ids = ["neutral"] + [r["option_id"] for r in bank["request"]["options"]]
    scenes = {
        r["scene_id"]: dict(scene_id=r["scene_id"], scene_definition=r["scene_definition"])
        for r in study["assignments"]
    }
    expected = expected_assignments(plan["runs"], list(scenes.values()), option_ids)
    if study["assignments"] != expected or len(prepared["assignments"]) != 138:
        raise ValueError("the full predeclared M8 assignment matrix is required")
    rows, sources = [], []
    for assigned, row in zip(expected, prepared["assignments"], strict=True):
        if {k: v for k, v in row.items() if k != "collection"} != assigned:
            raise ValueError("prepared row differs from its declared assignment")
        manifest = read_checked(row["collection"])
        check_assignment(row, manifest, study, prepared["models"])
        path = Path(row["collection"]["path"]).with_name("result.json")
        result = dict(
            status="not_run", time_s=None, measurement_admitted=False, recorded_physics_steps=None
        )
        output = Path(manifest["cells"][0]["output"])
        if (path.parent / "launch.json").exists() or output.exists():
            result["status"] = "technical_missing"
        if path.is_file():
            ref = artifact(path)
            actual = read_checked(ref)
            if actual["manifest"] != row["collection"] or len(actual["rows"]) != 1:
                raise ValueError("one measured result for this assignment required")
            result = measure(actual["rows"][0])
            sources.append(dict(assignment_id=row["assignment_id"], result=ref))
        rows.append(
            dict(
                assignment_id=row["assignment_id"],
                policy_id=row["policy_id"],
                scene_id=row["scene_id"],
                physics_seed=PHYSICS_SEED,
                **result,
            )
        )
    summary = summarize(rows, ["fixed_" + name for name in option_ids])
    finished_ref = None
    if (panel / "result.json").is_file():
        finished_ref = artifact(panel / "result.json")
        finished = read_checked(finished_ref)
        if (
            finished["prepared"] != prepared_ref
            or finished["assignments"] != sources
            or not summary["complete"]
        ):
            raise ValueError("finished panel must contain every assigned known outcome")
    arms = {}
    for arm in sorted({m["arm"] for m in study["models"]}):
        models = sorted([m for m in study["models"] if m["arm"] == arm], key=lambda m: m["seed"])
        rates = [summary["policies"][m["run_id"]]["complete_outcome_mean"] for m in models]
        arms[arm] = dict(
            corpus_seeds=[m["seed"] for m in models],
            passage_fraction_by_corpus=rates,
            descriptive_corpus_variation=(
                variation(rates) if all(r is not None for r in rates) else None
            ),
            interpretation=(
                "three training seeds, shared six development contexts and one execution seed; "
                "not 18 independent layouts"
            ),
        )
    out.mkdir(parents=True, exist_ok=False)
    return write_new(
        out / "result.json",
        dict(
            prepared=prepared_ref,
            completed_execution=finished_ref,
            implementation=artifact(Path(__file__)),
            source_results=sources,
            rows=rows,
            summary=summary,
            construction_arms=arms,
            construction_comparisons=construction_comparisons(rows, study["models"]),
            new_physics_steps=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.panel, args.out)))
