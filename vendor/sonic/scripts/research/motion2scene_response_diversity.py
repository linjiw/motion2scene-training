#!/usr/bin/env python3
"""Audit measured acquisition yield and fixed-schedule coverage at a checkpoint.

Reads and re-audits completed acquisition prefixes. No fitting, queue mutation,
reserved data, or new simulation. Task counts refer to constructor-specific
training scenes, never to a common-set policy success rate.
"""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_checkpoint_controls import checkpoint_slots, load_checkpoint  # noqa: E402
from motion2scene_cross_corpus_readout import geometry_key  # noqa: E402
from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)


def response_summary(rows, option_ids):
    """Exact finite-bank coverage; unknowns are bounded, never imputed failures."""
    if not option_ids or len(set(option_ids)) != len(option_ids) or "neutral" not in option_ids:
        raise ValueError("distinct complete schedule identities including neutral required")
    if not rows or any(
        set(row) != set(option_ids) or not set(row.values()) <= {"pass", "failure", "unknown"}
        for row in rows
    ):
        raise ValueError("nonempty rectangular pass/failure/unknown outcome table required")
    complete = all("unknown" not in row.values() for row in rows)
    passing = [{s for s in option_ids if row[s] == "pass"} for row in rows]
    solvable = [i for i, schedules in enumerate(passing) if schedules]
    fixed = {
        s: dict(
            passes=sum(row[s] == "pass" for row in rows),
            failures=sum(row[s] == "failure" for row in rows),
            unknown=sum(row[s] == "unknown" for row in rows),
        )
        for s in option_ids
    }
    best = max(v["passes"] for v in fixed.values())
    best_upper = max(v["passes"] + v["unknown"] for v in fixed.values())
    lower = len(solvable)
    upper = sum(bool(p) or "unknown" in row.values() for p, row in zip(passing, rows, strict=True))
    covers = None
    if complete:
        for size in range(len(option_ids) + 1):
            covers = [
                list(selected)
                for selected in combinations(option_ids, size)
                if all(set(selected) & passing[i] for i in solvable)
            ]
            if covers:
                break
    patterns = Counter(tuple(s for s in option_ids if s in p) for p in passing)
    return dict(
        assigned_tasks=len(rows),
        complete=complete,
        bank_solvable_lower=lower,
        bank_solvable_upper=upper,
        bank_unsolvable=sum(all(v == "failure" for v in r.values()) for r in rows),
        unknown_branch_outcomes=sum(v["unknown"] for v in fixed.values()),
        adaptation_required_lower=sum(
            bool(p) and r["neutral"] == "failure" for p, r in zip(passing, rows, strict=True)
        ),
        fixed_schedules=fixed,
        best_fixed_passages_lower=best,
        best_fixed_passages_upper=best_upper,
        best_fixed_schedule_ids=(
            [s for s in option_ids if fixed[s]["passes"] == best] if complete else None
        ),
        capability_minus_best_fixed_lower=max(0, lower - best_upper),
        capability_minus_best_fixed_upper=upper - best,
        minimum_cover_size=len(covers[0]) if covers else None,
        minimum_schedule_covers=covers,
        one_fixed_covers_every_solvable_task=(best == lower if complete and lower else None),
        passing_set_patterns=(
            [dict(schedules=list(p), tasks=n) for p, n in sorted(patterns.items())]
            if complete
            else None
        ),
        interpretation=(
            "finite measured training-bank coverage, not sensor realizability "
            "or executed policy performance"
        ),
    )


def validate_paired_queues(plan, budget):
    runs = {(r.get("seed", r.get("physics_seed")), r["arm"]): r for r in plan["runs"]}
    expanded = plan["schema"] == "motion2scene_expanded_acquisition_v1"
    fields = (
        ("candidate_id", "scene", "branch_order")
        if expanded
        else ("candidate_id", "scene_definition", "teacher_branch_order")
    )
    for seed, arm in runs:
        if arm != "analytic_contrast":
            continue
        a = runs[seed, arm]["rounds"][: budget + 1]
        b = runs[seed, "observation_curriculum"]["rounds"][: budget + 1]
        if (
            len(a) != budget + 1
            or len(b) != budget + 1
            or any(any(x[k] != y[k] for k in fields) for x, y in zip(a, b, strict=True))
        ):
            raise ValueError(
                "contrast/replay candidate identities, scenes and branch ordering differ"
            )


def measured_collection(ref, scene, seed, option_ids, student=False):
    result = read_checked(ref)
    manifest = read_checked(result["manifest"])
    if (
        manifest["split"] != "development"
        or manifest["scene_definition"] != scene
        or any(c["runtime_seed"] != seed for c in manifest["cells"])
    ):
        raise ValueError("only the assigned development scene and execution seed may be analyzed")
    rows = result["rows"]
    if len(rows) != (1 if student else len(option_ids)):
        raise ValueError("all assigned collection outcomes required")
    if not student and {r["forced_option_id"] for r in rows} != set(option_ids):
        raise ValueError("every complete teacher schedule required exactly once")
    states = [r["outcome"]["task_outcome"] for r in rows]
    if any(s not in ("pass", "failure") for s in states):
        raise ValueError("checkpoint has an unresolved physical outcome")
    if type(result["physics_steps"]) is not int or result["physics_steps"] < 0:
        raise ValueError("measured nonnegative integer physics accounting required")
    return result, manifest


def run(plan_path, budget, out):
    plan_ref = artifact(plan_path)
    plan = read_checked(plan_ref)
    expanded = plan["schema"] == "motion2scene_expanded_acquisition_v1"
    if budget not in ((1, 2, 4, 8, 16, 32) if expanded else (1, 2, 4)):
        raise ValueError("a declared acquisition checkpoint is required")
    validate_paired_queues(plan, budget)
    reports, all_tasks = [], []
    for assignment in plan["runs"]:
        seed = assignment.get("seed", assignment.get("physics_seed"))
        bank, groups, _, checkpoint = load_checkpoint(plan, assignment["run_id"], budget)
        slots = checkpoint_slots(plan, assignment["run_id"], budget)
        option_ids = list(bank.option_ids)
        costs = dict(
            bootstrap_teacher_steps=0, encounter_teacher_steps=0, preupdate_student_steps=0
        )
        tasks = []
        sources = []
        for index, (group, slot) in enumerate(zip(groups, slots, strict=True)):
            teacher, manifest = measured_collection(
                group["collection"], slot["scene"], seed, option_ids
            )
            if teacher["physics_steps"] != group["physics_steps"]:
                raise ValueError("re-audited teacher steps differ from stored accounting")
            sources.append(dict(teacher=group["collection"], student=None))
            if index == 0:
                costs["bootstrap_teacher_steps"] += teacher["physics_steps"]
                continue
            student_ref = artifact(slot["student"])
            student, _ = measured_collection(
                student_ref, slot["scene"], seed, option_ids, student=True
            )
            sources[-1]["student"] = student_ref
            costs["encounter_teacher_steps"] += teacher["physics_steps"]
            costs["preupdate_student_steps"] += student["physics_steps"]
            tasks.append(
                dict(
                    run_id=assignment["run_id"],
                    arm=assignment["arm"],
                    seed=seed,
                    round=index,
                    candidate_id=assignment["rounds"][index]["candidate_id"],
                    stratum=assignment["rounds"][index]["stratum"],
                    geometry_key=geometry_key(read_checked(manifest["scene_definition"])),
                    outcomes={
                        r["forced_option_id"]: r["outcome"]["task_outcome"] for r in teacher["rows"]
                    },
                    student_outcome=student["rows"][0]["outcome"]["task_outcome"],
                    teacher_steps=teacher["physics_steps"],
                    student_steps=student["physics_steps"],
                )
            )
        reports.append(
            dict(
                run_id=assignment["run_id"],
                arm=assignment["arm"],
                seed=seed,
                checkpoint=checkpoint,
                source_results=sources,
                physical_cost=dict(
                    **costs,
                    total_recorded_steps=sum(costs.values()),
                    assigned_episodes=len(option_ids) + (len(option_ids) + 1) * budget,
                    scope=(
                        "completed acquisition captures including own bootstrap; "
                        "unresolved reservations and historical development excluded"
                    ),
                ),
                responses=response_summary([t["outcomes"] for t in tasks], option_ids),
                tasks=tasks,
            )
        )
        all_tasks.extend(tasks)
    arms = {}
    for arm in sorted({r["arm"] for r in reports}):
        local = [r for r in reports if r["arm"] == arm]
        if len(local) != 3 or {r["seed"] for r in local} != {93201, 93202, 93203}:
            raise ValueError("all three corpus seeds required for an arm-level comparison")
        tasks = [t for r in local for t in r["tasks"]]
        arms[arm] = dict(
            pooled_responses=response_summary([t["outcomes"] for t in tasks], option_ids),
            actual_recorded_steps=sum(r["physical_cost"]["total_recorded_steps"] for r in local),
            distinct_geometries=len({t["geometry_key"] for t in tasks}),
            independent_corpora=3,
            interpretation="assigned corpus/task pairs retained; repeated geometries are not independent layouts",
        )
    paired = []
    for seed in (93201, 93202, 93203):
        a = [t for t in all_tasks if t["seed"] == seed and t["arm"] == "analytic_contrast"]
        b = [t for t in all_tasks if t["seed"] == seed and t["arm"] == "observation_curriculum"]
        paired.append(
            dict(
                seed=seed,
                same_candidate_order=[t["candidate_id"] for t in a]
                == [t["candidate_id"] for t in b],
                teacher_outcome_disagreements=[
                    x["round"] for x, y in zip(a, b, strict=True) if x["outcomes"] != y["outcomes"]
                ],
            )
        )
    out.mkdir(parents=True, exist_ok=False)
    return write_new(
        out / "result.json",
        dict(
            schema="motion2scene_response_diversity_v1",
            plan=plan_ref,
            budget=budget,
            implementation=artifact(Path(__file__)),
            corpora=reports,
            arms=arms,
            paired_contrast_replay=paired,
            new_physics_steps=0,
            evidence=(
                "completed controller receipts with independent prefix/teacher re-audit; "
                "no newly executed policies"
            ),
            geometric_cost_sources={
                k: plan[k]
                for k in (
                    "candidate_pools",
                    "reference_pools",
                    "candidate_pool_result",
                    "proposal_accounting",
                )
                if k in plan
            },
            geometric_cost_rule=(
                "shared pool preparation, separate from recorded physics; "
                "no per-arm runtime inferred from shared caching"
            ),
            decision_rule=(
                "inspect at M8; only a new separately budgeted development intervention "
                "may change proposal complementarity; frozen queues unchanged"
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.budget, args.out)))
