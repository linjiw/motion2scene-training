#!/usr/bin/env python3
"""Continue completed development corpora through 8/16/32 distinct encounters.

Uses the frozen collector and runtime, with a separate acquisition plan and
training implementation. Original M0--M4 files remain in their original locations.
Every new student runs before the current seven teacher branches. This runner
never opens reserved task outcomes or launches reserved evaluation.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

import motion2scene_build_timed_replay as replay  # noqa: E402
from motion2scene_run_primary_acquisition import (  # noqa: E402
    Controller,
    LocalBackend,
    Paused,
    acquisition_lock,
    commit,
    load_context,
)
from motion2scene_storage import deduplicate_inventories  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402
from motion2scene_train_timed_schedules import audit_collection, weighted_targets  # noqa: E402
from motion2scene_tune_schedule_learner import arrays  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_curriculum import (  # noqa: E402
    best_complete_teacher,
    coverage_key,
    deadline_eligibility,
    encounter_replay_weights,
    verified_decision_gap,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    fit_timed_schedule_policy,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    validate_schedule_policy,
)


def bound(ref):
    return replay.read_bound(ref)


def available_disk_gib(root):
    while not root.exists():
        root = root.parent
    return shutil.disk_usage(root).free / 1024**3


def validate_training_prefix(registration, earlier_collections, registry):
    if registration["collections"] != earlier_collections or registration["registry"] != registry:
        raise ValueError("pre-update fitting must contain exactly the earlier encounter prefix")


def priority_records(bank, groups, students, rule):
    """Physical gap on each historical generating policy's measured encounter."""
    records = []
    for group, student in zip(groups, students, strict=True):
        result = bound(group["collection"])
        manifest = bound(result["manifest"])
        scene = bound(manifest["scene_definition"])
        neutral = next(r for r in result["rows"] if r["forced_option_id"] == "neutral")
        for target in group["targets"]:
            teacher = best_complete_teacher(target)
            continuation = None if teacher is None else teacher["continuation_option_index"]
            entry = (
                None
                if not continuation
                else bank.request["options"][continuation - 1]["entry_tick"]
            )
            deadline = deadline_eligibility(
                neutral.get("observation_timing", []),
                scene["beam_collision_enabled"],
                target["phase_tick"],
                entry,
                rule,
            )
            matched = None
            if student is not None:
                matched = dict(student, **student["phase_records"].get(target["phase_tick"], {}))
                matched["source_admitted"] = student["source_admitted"] and (
                    replay.dependency_identities(manifest)
                    == replay.dependency_identities(student["manifest"])
                )
                matched["matched_history"] = matched.get("recorded_history_sha256") == target.get(
                    "recorded_history_sha256"
                )
            records.append(
                dict(
                    encounter_id=group["scene_id"],
                    phase_tick=target["phase_tick"],
                    supervision_available=teacher is not None,
                    coverage_key=coverage_key(bank, scene, target, rule),
                    **verified_decision_gap(target, matched, deadline),
                    teacher_collection=group["collection"],
                    student_collection=None if student is None else student["collection"],
                    generating_model=None if student is None else student["manifest"]["policy"],
                    deadline=deadline,
                )
            )
    return records


class ExpandedController(Controller):
    """Reuse tested attempt execution while extending the number of encounters."""

    def __init__(self, context, run, inherited_ledger):
        super().__init__(context, LocalBackend())
        self.expanded_run = run
        self.inherited_ledger = inherited_ledger

    def accounting(self):
        report = super().accounting()
        inherited = self.inherited_ledger or {}
        for key in (
            "actual_recorded_physics_steps",
            "unknown_attempt_reserved_steps",
            "conservative_charged_steps",
            "unknown_outcome_tail_reserved_steps",
        ):
            report[key] += inherited.get(key, 0)
        report["budget_remaining_after_reservations"] -= inherited.get(
            "conservative_charged_steps", 0
        )
        report["within_budget"] = report["budget_remaining_after_reservations"] >= 0
        source_slots = 39 if self.inherited_ledger else 0
        report["unlaunched_assigned_episode_slots"] = 263 - source_slots - len(report["attempts"])
        report["reserved_future_assigned_steps"] = (
            report["unlaunched_assigned_episode_slots"] * 1192
        )
        report["inherited_assigned_episode_slots"] = source_slots
        report["new_recorded_physics_steps"] = report[
            "actual_recorded_physics_steps"
        ] - inherited.get("actual_recorded_physics_steps", 0)
        report["inherited_recorded_captures"] = inherited.get(
            "distinct_original_recorded_captures", 0
        )
        report["distinct_original_recorded_captures"] += report["inherited_recorded_captures"]
        report["completed_cpu_fit_receipts"] = len(
            list((self.context["run_root"] / "models").glob("model_*/result.json"))
        )
        report["inherited_completed_cpu_fit_receipts"] = inherited.get(
            "completed_cpu_fit_receipts", 0
        )
        return report

    def collect(self, row, kind, model=None):
        if available_disk_gib(self.context["execution_root"]) < 30:
            raise Paused("fewer than 30 GiB free; remaining physics stays unlaunched")
        return super().collect(row, kind, model)

    def fit_expanded(self, groups, students, out):
        collections = [g["collection"] for g in groups]
        if (out / "result.json").exists():
            saved = bound(artifact(out / "result.json"))
            validate_training_prefix(
                bound(saved["registration"]), collections, self.context["plan"]["registry"]
            )
            if bound(saved["teachers"]) != groups:
                raise ValueError("cached fit teacher values changed")
            return saved
        if out.exists():
            raise Paused(f"incomplete fit retained for recovery: {out}")
        bank_ref = self.context["plan"]["registry"]
        bank = load_verified_registry(bank_ref["path"], bank_ref["sha256"])
        weights, evidence = None, None
        if self.expanded_run["arm"] == "observation_curriculum":
            records = priority_records(
                bank, groups, students, bound(self.context["runtime"]["replay_rule"])
            )
            evidence = encounter_replay_weights(records)
            weights = evidence["weights"]
        rows, sample_weights = weighted_targets(groups, weights)
        model, fitting = fit_timed_schedule_policy(
            bank,
            feature_names=rows[0]["feature_names"],
            l2=10.0,
            sample_weights=sample_weights,
            allow_measured_tie_initialization=True,
            **arrays(rows),
        )
        validate_schedule_policy(model, bank)
        out.mkdir(parents=True)
        registration = write_new(
            out / "registration.json",
            dict(
                schema="motion2scene_expanded_fit_v1",
                plan=self.context["expanded_plan_ref"],
                collections=collections,
                registry=bank_ref,
                l2=10.0,
                allow_measured_tie_initialization=True,
                implementation=artifact(Path(__file__)),
                weighting=(
                    "historical_observation_gap" if weights is not None else "uniform_per_phase"
                ),
            ),
        )
        teachers = write_new(out / "teachers.json", groups)
        if evidence is not None:
            write_new(
                out / "replay.json",
                dict(
                    records=records,
                    weights=evidence,
                    signal="historical measured generating-policy outcomes",
                ),
            )
        np.savez_compressed(out / "policy.npz", **model)
        result = dict(
            status="complete",
            policy=artifact(out / "policy.npz"),
            registration=registration,
            teachers=teachers,
            fit=fitting,
            source_physics_steps=sum(g["physics_steps"] for g in groups),
            audit_errors=[],
            new_physics_steps=0,
        )
        write_new(out / "result.json", result)
        return result

    def acquire_expanded(self, stop):
        groups, students, previous, previous_result = [], [], None, None
        bank_ref = self.context["plan"]["registry"]
        bank = load_verified_registry(bank_ref["path"], bank_ref["sha256"])
        for slot in self.expanded_run["rounds"][: stop + 1]:
            index = slot["index"]
            if slot["inherited_model"]:
                previous_result = artifact(slot["inherited_model"])
                previous = bound(previous_result)
                registered = bound(previous["registration"])
                imported_groups = bound(previous["teachers"])
                if imported_groups[:-1] != groups:
                    raise ValueError("inherited models do not form one teaching prefix")
                if registered["collections"] != [g["collection"] for g in imported_groups]:
                    raise ValueError("inherited teaching order differs")
                groups = imported_groups
                if index:
                    actual, _, unknown = replay.audit_students(
                        Path(slot["inherited_student"]), bank, bank_ref
                    )
                    if unknown or len(actual) != 1:
                        raise Paused("inherited student measurement unavailable")
                    students.append(actual[0])
                else:
                    students.append(None)
                continue
            row = dict(
                round_index=index,
                scene_definition=slot["scene"],
                physics_seed=self.expanded_run["seed"],
                teacher_branch_order=slot["branch_order"],
                teacher_directory=slot["teacher_directory"],
                student_directory=slot["student_directory"],
            )
            current = Path(slot["teacher_directory"]).parent
            student = None
            if index:
                binding_path = current / "preupdate.json"
                expected = dict(
                    model=previous["policy"],
                    training_result=previous_result,
                    earlier_collections=[g["collection"] for g in groups],
                    round_index=index,
                    scene=slot["scene"],
                    expanded_plan=self.context["expanded_plan_ref"],
                )
                if not binding_path.exists() and (
                    Path(slot["teacher_directory"]).exists()
                    or Path(slot["student_directory"]).exists()
                ):
                    raise Paused("current collection exists without pre-update model commitment")
                validate_training_prefix(
                    bound(previous["registration"]), expected["earlier_collections"], bank_ref
                )
                commit(binding_path, expected)
                student_ref = self.collect(row, "student", previous["policy"])
                release_path = current / "student_complete.json"
                if not release_path.exists() and Path(slot["teacher_directory"]).exists():
                    raise Paused("teachers exist without a completed-student release")
                commit(release_path, dict(preupdate=artifact(binding_path), student=student_ref))
                actual, _, unknown = replay.audit_students(
                    Path(student_ref["path"]), bank, bank_ref
                )
                if (
                    unknown
                    or len(actual) != 1
                    or actual[0]["manifest"]["policy"] != previous["policy"]
                ):
                    raise Paused("actual student does not bind the pre-update policy")
                student = actual[0]
            teacher_ref = self.collect(row, "teacher")
            group = audit_collection(Path(teacher_ref["path"]), bank, bank_ref)
            if (
                group["scene_id"] != bound(slot["scene"])["scene_id"]
                or group["physics_seed"] != self.expanded_run["seed"]
            ):
                raise ValueError("measured encounter differs from the assigned scene and seed")
            groups.append(group)
            students.append(student)
            storage_path = current / "storage.json"
            if not storage_path.exists():
                collections = [Path(slot["teacher_directory"])]
                if index:
                    collections.append(Path(slot["student_directory"]))
                write_new(storage_path, deduplicate_inventories(collections))
            previous = self.fit_expanded(groups, students, Path(slot["model_directory"]))
            previous_result = artifact(Path(slot["model_directory"]) / "result.json")
        output = dict(
            expanded_plan=self.context["expanded_plan_ref"],
            run_id=self.expanded_run["run_id"],
            completed_through_round=stop,
            model=previous["policy"],
            training_result=previous_result,
            accounting=self.accounting(),
            reserved_evaluation_started=False,
        )
        return commit(self.folder / f"complete_{stop:03d}.json", output)


def run(plan_path, adoption_path, stop):
    plan_ref = artifact(plan_path)
    plan = bound(plan_ref)
    if plan.get("schema") != "motion2scene_expanded_acquisition_v1" or stop not in (8, 16, 32):
        raise ValueError("expanded acquisition plan and declared budget required")
    for source in plan["implementation"]:
        checked(Path(source["path"]), source["sha256"])
    predecessor = bound(plan["predecessor"])
    for inherited in predecessor["runs"]:
        completion = (
            Path(predecessor["intended_execution_root"])
            / inherited["run_id"]
            / "controller/completion.json"
        )
        if not completion.exists():
            return dict(status="waiting_for_primary_M4", missing_run=inherited["run_id"])
        completed = bound(artifact(completion))
        if (
            completed["status"] != "complete"
            or completed["completed_through_round"] != 4
            or completed["plan"] != plan["predecessor"]
        ):
            raise ValueError("all inherited corpora must finish the exact primary plan")
        bound(completed["final_model_training_result"])
        checked(Path(completed["final_model"]["path"]), completed["final_model"]["sha256"])
    # Interleave every constructor at each encounter budget. Original four-arm
    # prefixes are imported; reference construction acquires its own prefix.
    for boundary in range(stop + 1):
        for assignment in plan["runs"]:
            folder = Path(plan["execution_root"]) / assignment["run_id"]
            done = folder / "controller" / f"complete_{boundary:03d}.json"
            if done.exists():
                completed = bound(artifact(done))
                if completed["expanded_plan"] != plan_ref:
                    raise ValueError("completed boundary belongs to a different expansion plan")
                bound(completed["training_result"])
                continue
            inherited = assignment["arm"] != "reference_contrast"
            if inherited and boundary < 4:
                continue
            source_id = (
                assignment["run_id"] if inherited else f"seed{assignment['seed']}_analytic_contrast"
            )
            context = load_context(Path(plan["predecessor"]["path"]), adoption_path, source_id)
            inherited_ledger = Controller(context).accounting() if inherited else None
            source_root = context["execution_root"]
            context = dict(
                context,
                plan={
                    **context["plan"],
                    "budget": {"budget_physics_steps": plan["maximum_charged_steps_per_corpus"]},
                },
                expanded_plan_ref=plan_ref,
                run=assignment,
                run_root=folder,
                execution_root=Path(plan["execution_root"]),
                reused_bootstrap=None,
            )
            controller = ExpandedController(context, assignment, inherited_ledger)
            try:
                with acquisition_lock(source_root / ".expanded_acquisition.lock"):
                    completed = controller.acquire_expanded(boundary)
                print(
                    json.dumps(
                        dict(
                            status="completed_boundary",
                            run_id=assignment["run_id"],
                            boundary=boundary,
                            result=completed,
                        )
                    ),
                    flush=True,
                )
            except Paused as error:
                return dict(
                    status="paused",
                    reason=str(error),
                    run_id=assignment["run_id"],
                    boundary=boundary,
                )
    return dict(status="complete", encounters_per_corpus=stop, corpora=len(plan["runs"]))


def watch(plan_path, adoption_path, stop):
    """Wait for prerequisites/capacity; never retry an unknown physical outcome."""
    previous = None
    while True:
        result = run(plan_path, adoption_path, stop)
        if result != previous:
            print(json.dumps(result), flush=True)
            previous = result
        can_wait = result["status"] == "waiting_for_primary_M4" or (
            result["status"] == "paused"
            and result["reason"]
            in (
                "resource preflight paused before intent; assigned cell remains unlaunched",
                "fewer than 30 GiB free; remaining physics stays unlaunched",
            )
        )
        if not can_wait:
            return result
        time.sleep(45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--until-budget", type=int, choices=(8, 16, 32), default=32)
    parser.add_argument(
        "--watch", action="store_true", help="wait for primary M4 and free capacity"
    )
    args = parser.parse_args()
    result = (watch if args.watch else run)(args.plan, args.adoption, args.until_budget)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "complete" else 75)
