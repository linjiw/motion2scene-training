#!/usr/bin/env python3
"""Verify the first early-constraint encounters and acquired response diversity.

This is an interim development mechanism analysis, not an extra checkpoint
performance comparison. The complete audited M2 pool is extended by all three
executed-contrast round-3 encounters. No training, proposal, or physics is run.
"""

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

import motion2scene_build_timed_replay as replay  # noqa: E402
from motion2scene_checkpoint_controls import checkpoint_slots  # noqa: E402
from motion2scene_cross_corpus_readout import geometry_key  # noqa: E402
from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_response_diversity import (  # noqa: E402
    measured_collection,
    response_summary,
    validate_paired_queues,
)
from motion2scene_timing_diagnostic import artifact  # noqa: E402
from motion2scene_train_timed_schedules import audit_collection  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)


def distinct_conditions(tasks):
    unique = {}
    for task in tasks:
        key = task["geometry_key"], task["seed"]
        if key in unique and task["outcomes"] != unique[key]["outcomes"]:
            raise ValueError(
                "repeated physical conditions disagree; retain repeats for separate analysis"
            )
        unique.setdefault(key, task)
    return list(unique.values())


def disjoint_response_pairs(tasks):
    result = []
    for left, right in combinations(tasks, 2):
        if set(left["outcomes"]) != set(right["outcomes"]):
            raise ValueError("all complete schedule identities must match")
        if any(
            v not in ("pass", "failure") for row in (left, right) for v in row["outcomes"].values()
        ):
            continue
        a = {s for s, value in left["outcomes"].items() if value == "pass"}
        b = {s for s, value in right["outcomes"].items() if value == "pass"}
        if a and b and not a & b:
            result.append(
                dict(
                    left=left["candidate_id"],
                    right=right["candidate_id"],
                    left_seed=left["seed"],
                    right_seed=right["seed"],
                    left_passing=sorted(a),
                    right_passing=sorted(b),
                )
            )
    return result


def phase_feature_contrast(left, right):
    """Exact declared 114D observations; no new tolerance or geometric feature."""
    if left["option_ids"] != right["option_ids"]:
        raise ValueError("the same schedule repertoire is required")
    rows = []
    for a, b in zip(left["targets"], right["targets"], strict=True):
        if a["phase_tick"] != b["phase_tick"]:
            raise ValueError("only matched commitment phases can be compared")
        if not a.get("available", True) or not b.get("available", True):
            rows.append(dict(phase_tick=a["phase_tick"], available=False))
            continue
        if a["feature_names"] != b["feature_names"] or len(a["features"]) != 114:
            raise ValueError("exact common 114D observation interface required")
        x, y = np.asarray(a["features"]), np.asarray(b["features"])
        changed = np.flatnonzero(x != y)
        rows.append(
            dict(
                phase_tick=a["phase_tick"],
                available=True,
                exact_full_feature_equality=bool(np.array_equal(x, y)),
                exact_perception_feature_equality=bool(np.array_equal(x[:28], y[:28])),
                differing_features=[
                    dict(
                        index=int(i),
                        name=a["feature_names"][i],
                        left=float(x[i]),
                        right=float(y[i]),
                    )
                    for i in changed
                ],
                information_model=(
                    "exact current 114D features including existing temporal rays; "
                    "no tolerance introduced"
                ),
            )
        )
    return rows


def run(plan_path, previous_path, output):
    plan_ref, previous_ref = artifact(plan_path), artifact(previous_path)
    plan, previous = read_checked(plan_ref), read_checked(previous_ref)
    if previous["plan"] != plan_ref or previous["budget"] != 2 or len(previous["corpora"]) != 12:
        raise ValueError("the complete previously audited M2 comparison must be retained")
    validate_paired_queues(plan, 3)
    bank = load_verified_registry(plan["registry"]["path"], plan["registry"]["sha256"])
    option_ids = list(bank.option_ids)
    runs = [r for r in plan["runs"] if r["arm"] == "analytic_contrast"]
    if {r["physics_seed"] for r in runs} != {93201, 93202, 93203} or len(runs) != 3:
        raise ValueError("all three original executed-contrast corpora are required")
    snapshot = []
    for run in runs:
        slots = checkpoint_slots(plan, run["run_id"], 3)
        last = slots[-1]
        trained_ref = artifact(last["model"])
        trained = read_checked(trained_ref)
        teacher_refs = [artifact(s["teacher"]) for s in slots]
        if (
            trained["status"] != "complete"
            or trained["audit_errors"]
            or [g["collection"] for g in read_checked(trained["teachers"])] != teacher_refs
        ):
            raise ValueError("complete exact M3 teacher prefix required")
        snapshot.append(
            dict(
                run_id=run["run_id"],
                teacher=teacher_refs[-1],
                student=artifact(last["student"]),
                trained=trained_ref,
                previous_model=artifact(slots[-2]["model"]),
            )
        )
    output.mkdir(parents=True, exist_ok=False)
    write_new(
        output / "registration.json",
        dict(
            plan=plan_ref,
            previous_audit=previous_ref,
            new_collections=snapshot,
            implementation=artifact(Path(__file__)),
            scope=__doc__,
            new_physics_steps=0,
            selection=(
                "all three assigned executed-contrast round-3 encounters; "
                "witnesses identified after outcome inspection"
            ),
        ),
    )
    tasks = []
    for corpus in previous["corpora"]:
        for task in corpus["tasks"]:
            source = corpus["source_results"][task["round"]]["teacher"]
            slot = checkpoint_slots(plan, task["run_id"], 2)[task["round"]]
            old, _ = measured_collection(source, slot["scene"], task["seed"], option_ids)
            if source != artifact(slot["teacher"]) or task["outcomes"] != {
                row["forced_option_id"]: row["outcome"]["task_outcome"] for row in old["rows"]
            }:
                raise ValueError(
                    "the prior audited outcome table differs from its assigned receipts"
                )
            tasks.append(dict(**task, collection=source))
    new_groups, new_tasks = {}, []
    for run, item in zip(runs, snapshot, strict=True):
        print(json.dumps(dict(status="auditing_new_capture", run_id=run["run_id"])), flush=True)
        slot = checkpoint_slots(plan, run["run_id"], 3)[-1]
        group = audit_collection(Path(item["teacher"]["path"]), bank, plan["registry"])
        trained = read_checked(item["trained"])
        if group != read_checked(trained["teachers"])[-1]:
            raise ValueError("new raw physical audit differs from the stored complete teacher")
        students, _, unknown = replay.audit_students(
            Path(item["student"]["path"]), bank, plan["registry"]
        )
        if (
            unknown
            or len(students) != 1
            or students[0]["manifest"]["policy"] != read_checked(item["previous_model"])["policy"]
        ):
            raise ValueError("actual pre-update student must retain its generating M2 policy")
        teacher, manifest = measured_collection(
            item["teacher"], slot["scene"], run["physics_seed"], option_ids
        )
        student, _ = measured_collection(
            item["student"], slot["scene"], run["physics_seed"], option_ids, student=True
        )
        row = run["rounds"][3]
        task = dict(
            run_id=run["run_id"],
            arm=run["arm"],
            seed=run["physics_seed"],
            round=3,
            candidate_id=row["candidate_id"],
            stratum=row["stratum"],
            geometry_key=geometry_key(read_checked(slot["scene"])),
            outcomes={r["forced_option_id"]: r["outcome"]["task_outcome"] for r in teacher["rows"]},
            student_outcome=student["rows"][0]["outcome"]["task_outcome"],
            teacher_steps=teacher["physics_steps"],
            student_steps=student["physics_steps"],
            collection=item["teacher"],
            student_collection=item["student"],
            physical_events={
                r["forced_option_id"]: r["outcome"]["physical_events"] for r in teacher["rows"]
            },
        )
        if manifest["scene_definition"] != row["scene_definition"]:
            raise ValueError("registered early-constraint candidate changed")
        new_groups[task["candidate_id"]] = group
        new_tasks.append(task)
        write_new(
            output / (run["run_id"] + ".json"),
            dict(
                task=task,
                teacher_reaudit=group,
                student_generating_model=item["previous_model"],
                student_admitted=students[0]["task_outcome_admitted"],
            ),
        )
        print(
            json.dumps(
                dict(
                    status="capture_verified",
                    run_id=run["run_id"],
                    teacher_steps=task["teacher_steps"],
                    student_steps=task["student_steps"],
                )
            ),
            flush=True,
        )
    unique = distinct_conditions(tasks + new_tasks)
    pairs = disjoint_response_pairs(unique)
    feature_pairs = []
    for pair in pairs:
        if pair["left_seed"] != pair["right_seed"]:
            continue
        selected = [
            next(
                t
                for t in unique
                if t["candidate_id"] == pair[side] and t["seed"] == pair[side + "_seed"]
            )
            for side in ("left", "right")
        ]
        groups = []
        for task in selected:
            if task["candidate_id"] in new_groups:
                groups.append(new_groups[task["candidate_id"]])
            else:
                source = next(c for c in previous["corpora"] if c["run_id"] == task["run_id"])
                group = read_checked(read_checked(source["checkpoint"])["teachers"])[task["round"]]
                if group["collection"] != task["collection"]:
                    raise ValueError("prior audited observation group was substituted")
                groups.append(group)
        feature_pairs.append(dict(**pair, phase_features=phase_feature_contrast(*groups)))
    return write_new(
        output / "result.json",
        dict(
            schema="motion2scene_acquired_response_witness_v1",
            registration=artifact(output / "registration.json"),
            previous_audit=previous_ref,
            new_tasks=new_tasks,
            unique_conditions=unique,
            pooled_responses=response_summary([t["outcomes"] for t in unique], option_ids),
            executed_corpora=[
                dict(
                    run_id=r["run_id"],
                    responses=response_summary(
                        [t["outcomes"] for t in tasks + new_tasks if t["run_id"] == r["run_id"]],
                        option_ids,
                    ),
                )
                for r in runs
            ],
            disjoint_response_pairs=pairs,
            same_seed_information_pairs=feature_pairs,
            newly_reaudited_acquisition_episodes=24,
            newly_reaudited_acquisition_steps=sum(
                t["teacher_steps"] + t["student_steps"] for t in new_tasks
            ),
            prior_audited_acquisition_steps=sum(
                c["physical_cost"]["total_recorded_steps"] for c in previous["corpora"]
            ),
            new_physics_steps=0,
            interpretation=(
                "Interim acquired-pool capability evidence; not a balanced M3 arm comparison, "
                "fixed-test learning curve, or new policy evaluation. "
                "Individual corpus diversity is reported separately."
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--previous-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.previous_audit, args.out)))
