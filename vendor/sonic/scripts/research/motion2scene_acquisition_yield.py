#!/usr/bin/env python3
"""Measured teaching-task yield at a complete primary acquisition checkpoint."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)


def teaching_yield(outcomes):
    if (
        "neutral" not in outcomes
        or len(outcomes) < 2
        or any(state not in ("pass", "failure") for state in outcomes.values())
    ):
        raise ValueError("every assigned teacher outcome must be physically known")
    passes = [name for name, state in outcomes.items() if state == "pass"]
    failures = [name for name, state in outcomes.items() if state == "failure"]
    return dict(
        passing_schedules=passes,
        failing_schedules=failures,
        bank_solvable=bool(passes),
        passage_contrast=bool(passes and failures),
        adaptation_required=bool(passes and outcomes["neutral"] == "failure"),
    )


def run(plan_path, budget, out):
    if budget not in (1, 2, 4):
        raise ValueError("completed primary checkpoint 1, 2 or 4 required")
    plan_ref = artifact(plan_path)
    plan = read_checked(plan_ref)
    rows, refs = [], []
    by_arm = defaultdict(list)
    for assignment in plan["runs"]:
        counts = dict(
            scenes=0,
            bank_solvable=0,
            passage_contrast=0,
            adaptation_required=0,
            preupdate_student_passes=0,
            actual_physics_steps=0,
        )
        for slot in assignment["rounds"][1 : budget + 1]:
            teacher_ref, student_ref = [
                artifact(Path(slot[k]))
                for k in ("expected_teacher_result_path", "expected_student_result_path")
            ]
            teacher, student = read_checked(teacher_ref), read_checked(student_ref)
            for result in (teacher, student):
                manifest = read_checked(result["manifest"])
                if (
                    manifest["scene_definition"] != slot["scene_definition"]
                    or manifest["split"] != "development"
                    or any(c["runtime_seed"] != slot["physics_seed"] for c in manifest["cells"])
                ):
                    raise ValueError(
                        "acquired scene or physics seed differs from the candidate assignment"
                    )
            outcomes = {
                r["forced_option_id"]: r["outcome"]["task_outcome"] for r in teacher["rows"]
            }
            if (
                len(teacher["rows"]) != 7
                or len(outcomes) != 7
                or set(outcomes) != set(slot["teacher_branch_order"])
                or len(student["rows"]) != 1
            ):
                raise ValueError("complete assigned seven-branch teacher and one student required")
            measured = teaching_yield(outcomes)
            student_outcome = student["rows"][0]["outcome"]["task_outcome"]
            if student_outcome not in ("pass", "failure"):
                raise ValueError("assigned student's physical outcome remains unknown")
            record = dict(
                run_id=assignment["run_id"],
                arm=assignment["arm"],
                seed=assignment["physics_seed"],
                round=slot["round_index"],
                candidate_id=slot["candidate_id"],
                stratum=slot["stratum"],
                student_passed=student_outcome == "pass",
                **measured
            )
            rows.append(record)
            counts["scenes"] += 1
            for key in ("bank_solvable", "passage_contrast", "adaptation_required"):
                counts[key] += measured[key]
            counts["preupdate_student_passes"] += student_outcome == "pass"
            counts["actual_physics_steps"] += teacher["physics_steps"] + student["physics_steps"]
            refs.append(dict(teacher=teacher_ref, student=student_ref))
        checkpoint = assignment["rounds"][budget]
        fit_ref = artifact(Path(checkpoint["expected_postupdate_training_result_path"]))
        read_checked(fit_ref)
        policy_ref = artifact(Path(checkpoint["expected_postupdate_policy_path"]))
        checked(Path(policy_ref["path"]), policy_ref["sha256"])
        if counts["scenes"] != budget:
            raise ValueError("incomplete acquisition budget")
        by_arm[assignment["arm"]].append(
            dict(
                seed=assignment["physics_seed"],
                **counts,
                fitted_model=policy_ref,
                training_result=fit_ref
            )
        )
    summaries = {}
    for arm, corpora in by_arm.items():
        if len(corpora) != 3 or len({c["seed"] for c in corpora}) != 3:
            raise ValueError("all three predeclared corpus replicates required")
        summaries[arm] = dict(
            corpora=corpora, totals={k: sum(c[k] for c in corpora) for k in counts}
        )
    out.mkdir(parents=True, exist_ok=False)
    return write_new(
        out / "result.json",
        dict(
            plan=plan_ref,
            implementation=artifact(Path(__file__)),
            budget=budget,
            source_results=refs,
            arms=summaries,
            encounters=rows,
            acquisition_episodes_per_corpus_excluding_bootstrap=8 * budget,
            nominal_cumulative_episodes_per_corpus_including_bootstrap=7 + 8 * budget,
            interpretation=(
                "measured acquisition yield, not held-out policy performance; "
                "student outcomes occur on constructor-specific tasks; "
                "all-passing tasks may still teach measured cost distinctions"
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
