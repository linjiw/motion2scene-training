#!/usr/bin/env python3
"""JSON/NumPy analysis of the expanded reserved nominal assignment.

No simulator, model-weight or robot-package imports. Physical provenance is
established by the execution runner; this reader verifies the full assignment
and frozen metadata and retains every unavailable outcome explicitly.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from motion2scene_evaluation_statistics import (
    bootstrap_weights,
    completion_summary,
    paired_comparison,
    variation,
)
import numpy as np

SCHEMA = "motion2scene_reserved_learning_curve_v1"
CHECKPOINTS = (8, 16, 32)
CORPORA = (93201, 93202, 93203)
ARMS = (
    "uniform",
    "target_only",
    "reference_contrast",
    "analytic_contrast",
    "observation_curriculum",
)
SEEDS = (94301, 94302)


def file_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())


def read_json(ref):
    actual = file_ref(ref["path"])
    if actual["sha256"].removeprefix("sha256:") != ref["sha256"].removeprefix("sha256:"):
        raise ValueError("JSON artifact content differs from its declared digest")
    return json.loads(Path(actual["path"]).read_text())


def validate(receipt, protocol, assignment, inventory):
    if (
        receipt["schema"] != SCHEMA
        or protocol["execution_contract"]["schema"] != SCHEMA
        or protocol["status"] != "ADOPTED_READY_FOR_REGISTERED_EXECUTION"
    ):
        raise ValueError("expanded reserved execution schema required")
    policies = {p["policy_id"]: p for p in protocol["policies"]}
    learned = {f"seed{s}_{a}__M{b}" for a in ARMS for s in CORPORA for b in CHECKPOINTS}
    fixed = ["fixed_" + option for option in protocol["schedules"]["option_ids"]]
    baseline_ids = ["script", *fixed]
    layouts = sorted(s["layout_id"] for s in protocol["scenes"] if s["variant_id"] == "nominal")
    keys = {(p, layout, seed) for p in policies for layout in layouts for seed in SEEDS}
    if (
        set(policies) != learned | set(baseline_ids)
        or len(policies) != len(protocol["policies"])
        or len(fixed) != 7
        or len(set(layouts)) != len(layouts)
        or len(layouts) != 18
        or protocol["physics_seeds"] != list(SEEDS)
        or protocol["enabled_variant_ids"] != ["nominal"]
        or len(assignment) != len(keys)
        or {tuple(r[k] for k in ("policy_id", "layout_id", "physics_seed")) for r in assignment}
        != keys
        or inventory["policies"] != protocol["policies"]
        or set(inventory["models"]) != learned
        or protocol["execution_contract"]["assigned_episodes"] != len(keys)
    ):
        raise ValueError(
            "every original arm/checkpoint/layout/seed and fixed schedule must remain assigned"
        )
    rows = receipt["report"]["rows"]
    if len(rows) != len(assignment) or receipt["report"]["assigned_episodes"] != len(assignment):
        raise ValueError(
            "a partial result must retain explicit rows for all unexecuted assignments"
        )
    scene_metadata = {s["scene_id"]: s for s in protocol["scenes"]}
    for row, expected in zip(rows, assignment, strict=True):
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError("report ordering and complete assignment identities differ")
        p = policies[row["policy_id"]]
        m = inventory["models"].get(row["policy_id"])
        if (
            row["policy_mode"] != p["mode"]
            or row["model"] != p["model"]
            or row["checkpoint"] != p.get("checkpoint")
            or row["training_arm"] != p.get("arm")
            or row["acquisition_seed"] != p.get("seed")
            or row["layout_family"] != scene_metadata[row["scene_id"]]["family"]
            or row["variant_id"] != "nominal"
        ):
            raise ValueError("model, checkpoint or reserved condition metadata differs")
        if m is not None:
            cost = m["acquisition_cost"]["total_recorded_steps"]
            maximum = (7 + 8 * p["checkpoint"]) * 1192
            if (
                type(cost) is not int
                or not 0 < cost <= maximum
                or row["acquisition_actual_physics_steps"] != cost
                or row["acquisition_assigned_maximum_steps"] != maximum
                or m["policy"] != p["model"]
            ):
                raise ValueError("actual acquisition cost must bind the exact frozen model prefix")
        elif (
            row["acquisition_actual_physics_steps"] is not None
            or row["acquisition_assigned_maximum_steps"] is not None
        ):
            raise ValueError("shared baselines cannot acquire imputed training cost")
        status, time, steps = (
            row["status"],
            row["successful_passage_time_s"],
            row["recorded_physics_steps"],
        )
        if (
            status not in ("pass", "failure", "technical_missing", "not_run")
            or (
                status == "pass"
                and (type(time) not in (int, float) or not np.isfinite(time) or time <= 0)
            )
            or (status != "pass" and time is not None)
            or (steps is not None and (type(steps) is not int or not 0 <= steps <= 1192))
            or (status in ("pass", "failure") and (steps is None or steps == 0))
        ):
            raise ValueError(
                "success time and recorded steps must retain their actual measurement definitions"
            )
    return rows, policies, fixed, baseline_ids, layouts


def summarize(receipt, protocol, assignment, inventory, *, draws=20000, seed=202609081822):
    rows, policies, fixed, baselines, layouts = validate(receipt, protocol, assignment, inventory)
    by_policy = {p: [r for r in rows if r["policy_id"] == p] for p in policies}
    by_key = {(r["policy_id"], r["layout_id"], r["physics_seed"]): r for r in rows}
    weights = bootstrap_weights(3, len(layouts), draws=draws, seed=seed)
    summaries = {p: completion_summary(values) for p, values in by_policy.items()}
    points = []
    for arm in ARMS:
        for budget in CHECKPOINTS:
            for corpus in CORPORA:
                policy_id = f"seed{corpus}_{arm}__M{budget}"
                row = by_policy[policy_id][0]
                points.append(
                    dict(
                        policy_id=policy_id,
                        training_arm=arm,
                        acquisition_seed=corpus,
                        checkpoint=budget,
                        model=row["model"],
                        acquisition_actual_physics_steps=row["acquisition_actual_physics_steps"],
                        acquisition_assigned_maximum_steps=row[
                            "acquisition_assigned_maximum_steps"
                        ],
                        **summaries[policy_id],
                    )
                )

    def tensors(name, budget):
        ids = [name] if name in baselines else [f"seed{s}_{name}__M{budget}" for s in CORPORA]
        y = np.full((len(ids), len(layouts), 2), np.nan)
        time = np.full_like(y, np.nan)
        for c, p in enumerate(ids):
            for i, layout in enumerate(layouts):
                for j, execution_seed in enumerate(SEEDS):
                    row = by_key[p, layout, execution_seed]
                    if row["status"] in ("pass", "failure"):
                        y[c, i, j] = int(row["status"] == "pass")
                    if row["status"] == "pass":
                        time[c, i, j] = row["successful_passage_time_s"]
        return (y[0], time[0]) if name in baselines else (y, time)

    comparisons = []

    def compare(left, right, left_budget, right_budget):
        ly, lt = tensors(left, left_budget)
        ry, rt = tensors(right, right_budget)
        result = paired_comparison(ly, ry, lt, rt, weights, shared_right=right in baselines)
        for row, corpus in zip(result["per_corpus"], CORPORA, strict=True):
            row["acquisition_seed"] = corpus
        comparisons.append(
            dict(
                left=left,
                right=right,
                left_checkpoint=left_budget,
                right_checkpoint=right_budget,
                **result,
            )
        )

    pairs = (
        ("analytic_contrast", "uniform"),
        ("analytic_contrast", "target_only"),
        ("analytic_contrast", "reference_contrast"),
        ("observation_curriculum", "analytic_contrast"),
    )
    for budget in CHECKPOINTS:
        for a, b in pairs:
            compare(a, b, budget, budget)
        for arm in ARMS:
            for baseline in baselines:
                compare(arm, baseline, budget, None)
    for arm in ARMS:
        for earlier, later in zip(CHECKPOINTS, CHECKPOINTS[1:]):
            compare(arm, arm, later, earlier)
    capability = []
    for layout in layouts:
        for execution_seed in SEEDS:
            bank = [by_key[p, layout, execution_seed] for p in fixed]
            passing = [r for r in bank if r["status"] == "pass"]
            complete = all(r["status"] in ("pass", "failure") for r in bank)
            capability.append(
                dict(
                    layout_id=layout,
                    physics_seed=execution_seed,
                    complete_bank_measurement=complete,
                    passing_schedules=[r["policy_id"] for r in passing],
                    lower=int(bool(passing)),
                    upper=int(bool(passing) or not complete),
                )
            )
    selection = {}
    for p in policies:
        matched = [
            c
            for c in capability
            if c["complete_bank_measurement"]
            and by_key[p, c["layout_id"], c["physics_seed"]]["status"] in ("pass", "failure")
        ]
        failures = [
            c
            for c in matched
            if c["lower"] == 1
            and by_key[p, c["layout_id"], c["physics_seed"]]["status"] == "failure"
        ]
        policy_only = [
            c
            for c in matched
            if c["upper"] == 0 and by_key[p, c["layout_id"], c["physics_seed"]]["status"] == "pass"
        ]
        selection[p] = dict(
            complete_matched_conditions=len(matched),
            selection_failures=len(failures),
            policy_only_successes=len(policy_only),
        )
    failure_counts = {
        p: dict(
            Counter(
                k for r in values if r["status"] == "failure" for k in set(r["failure_categories"])
            )
        )
        for p, values in by_policy.items()
    }
    strata = {
        family: {
            p: completion_summary([r for r in values if r["layout_family"] == family])
            for p, values in by_policy.items()
        }
        for family in sorted({r["layout_family"] for r in rows})
    }
    return dict(
        schema="motion2scene_reserved_curve_statistics_v1",
        rows=rows,
        assigned_unique_episodes=len(rows),
        learned_curve_points=points,
        baseline_summaries={p: summaries[p] for p in baselines},
        policy_summaries=summaries,
        capability=capability,
        capability_count_lower=sum(c["lower"] for c in capability),
        capability_count_upper=sum(c["upper"] for c in capability),
        selection=selection,
        failure_categories=failure_counts,
        layout_families=strata,
        comparisons=comparisons,
        corpus_variation=[
            dict(
                arm=arm,
                checkpoint=budget,
                completion_lower=variation(
                    [
                        p["completion_lower"]
                        for p in points
                        if p["training_arm"] == arm and p["checkpoint"] == budget
                    ]
                ),
                completion_upper=variation(
                    [
                        p["completion_upper"]
                        for p in points
                        if p["training_arm"] == arm and p["checkpoint"] == budget
                    ]
                ),
            )
            for arm in ARMS
            for budget in CHECKPOINTS
        ],
        evaluation_recorded_steps=sum(r["recorded_physics_steps"] or 0 for r in rows),
        unknown_step_counts=sum(r["recorded_physics_steps"] is None for r in rows),
        bootstrap=dict(
            draws=draws,
            seed=seed,
            corpus_weights_sha256=hashlib.sha256(weights[0].tobytes()).hexdigest(),
            layout_weights_sha256=hashlib.sha256(weights[1].tobytes()).hexdigest(),
        ),
        interpretation=(
            "Descriptive crossed corpus/layout intervals with three acquired corpora; "
            "two execution seeds remain together, shared baselines are not independent replicas. "
            "All tasks remain assigned, including bank-unsolvable cases. "
            "No multiplicity-adjusted or source-transfer claim."
        ),
        contact_and_fall_definitions={
            "verified_environment_contact": (
                "undesired environment normal force >1 N, all captured physics substeps; "
                "foot-floor support excluded"
            ),
            "recorded_fall_or_upright_threshold_failure": (
                "root height below 0.5 m or minus projected gravity Z below 0.5 "
                "during the recorded horizon"
            ),
            "flags": "overlap; other failures are not reclassified as contact or fall",
        },
        new_physics_steps=0,
    )


def run(source, output):
    source_ref = file_ref(source)
    receipt = read_json(source_ref)
    if receipt.get("kind") == "evaluation_status":
        receipt = receipt["receipt"]
    registration = read_json(receipt["registration"])
    protocol = read_json(registration["protocol"])
    inventory = read_json(registration["inventory"])
    assigned = read_json(registration["assignments"])
    result = summarize(receipt, protocol, assigned, inventory)
    result.update(
        input=source_ref,
        implementation=file_ref(__file__),
        registration=receipt["registration"],
        numpy_version=np.__version__,
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    fields = (
        "policy_id",
        "training_arm",
        "acquisition_seed",
        "checkpoint",
        "acquisition_actual_physics_steps",
        "acquisition_assigned_maximum_steps",
        "assigned",
        "pass",
        "completion_lower",
        "completion_upper",
    )
    with (output / "learning_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["learned_curve_points"])
    return dict(
        result=file_ref(output / "result.json"),
        learning_curves=file_ref(output / "learning_curves.csv"),
        new_physics_steps=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.out)))
