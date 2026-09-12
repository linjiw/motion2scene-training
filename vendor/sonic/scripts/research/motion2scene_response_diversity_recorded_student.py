#!/usr/bin/env python3
"""Audit the unchanged acquisition while retaining one reconciled unknown student.

The original reader requires resolved student outcomes even when their cost is
fully measured. This version permits only the explicitly reconciled recording;
teacher outcomes and all other source checks still use the original reader.
"""

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

import motion2scene_response_diversity as original  # noqa: E402
import motion2scene_resume_recorded_student as recovery  # noqa: E402


def read_reconciled(ref, scene, seed, declaration, out, audited_row):
    if ref != original.artifact(out / "result.json"):
        raise ValueError("reader exception cannot admit another student recording")
    result = original.read_checked(ref)
    manifest = original.read_checked(result["manifest"])
    if (
        result["manifest"] != declaration["manifest"]
        or manifest["split"] != "development"
        or manifest["scene_definition"] != scene
        or len(manifest["cells"]) != 1
        or manifest["cells"][0]["runtime_seed"] != seed
        or result["rows"] != [audited_row]
        or audited_row["outcome"]["task_outcome"] != "unknown"
        or audited_row["task_outcome_admitted"] is not False
        or type(result["physics_steps"]) is not int
        or result["physics_steps"] != audited_row["physics_steps"]
        or result["physics_steps"] != 1192
    ):
        raise ValueError(
            "reconciled student must retain its assigned scene, seed, unknown and cost"
        )
    return result, manifest


@contextmanager
def retained_unknown_student(declaration_path):
    declaration, out, audited_row = recovery.audit(declaration_path)
    expected = original.artifact(out / "result.json")
    original_reader, original_writer = original.measured_collection, original.write_new

    def reader(ref, scene, seed, option_ids, student=False):
        if not student or ref != expected:
            return original_reader(ref, scene, seed, option_ids, student=student)
        return read_reconciled(ref, scene, seed, declaration, out, audited_row)

    def writer(path, value):
        # Bind the effective reader and enumerate unknowns without changing rows.
        if value["schema"] != "motion2scene_response_diversity_v1":
            raise ValueError("unexpected publication from the original audit")
        unknowns = [
            dict(run_id=corpus["run_id"], round=task["round"], steps=task["student_steps"])
            for corpus in value["corpora"]
            for task in corpus["tasks"]
            if task["student_outcome"] == "unknown"
        ]
        if len(unknowns) > 1 or any(
            row["run_id"] != declaration["run_id"]
            or row["round"] != declaration["round_index"]
            or row["steps"] != 1192
            for row in unknowns
        ):
            raise ValueError("unknown student list differs from the declared recording")
        return original_writer(
            path,
            dict(
                value,
                implementation=original.artifact(Path(__file__)),
                original_implementation=value["implementation"],
                recorded_student_reconciliation=original.artifact(declaration_path),
                retained_unknown_students=unknowns,
                student_outcome_scope=(
                    "Unknown is neither failure nor passage; its measured physical "
                    "cost and assigned encounter remain included. Teacher response "
                    "statistics use the unchanged complete teacher branches."
                ),
            ),
        )

    original.measured_collection, original.write_new = reader, writer
    try:
        yield
    finally:
        original.measured_collection, original.write_new = original_reader, original_writer


def run(plan_path, budget, out, declaration_path):
    declaration, _, _ = recovery.audit(declaration_path)
    if declaration["plan"] != original.artifact(plan_path):
        raise ValueError("reconciliation belongs to a different acquisition plan")
    with retained_unknown_student(declaration_path):
        return original.run(plan_path, budget, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=(8, 16, 32), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.plan, args.budget, args.out, args.reconciliation))
