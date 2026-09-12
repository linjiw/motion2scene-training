#!/usr/bin/env python3
"""Fit controlled replay alternatives to one completed development checkpoint.

Historical scores retain their actual generating policies. Only prediction
residuals are refreshed. Saved policies use the existing closed-loop interface;
recorded validation branches remain explicitly labeled as offline proxies.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import evaluate, ridge_choice  # noqa: E402
from motion2scene_expanded_acquisition import (  # noqa: E402
    bound,
    priority_records,
    replay,
    validate_training_prefix,
)
from motion2scene_replay_controls import replay_control_weights  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402
from motion2scene_train_timed_schedules import audit_collection, weighted_targets  # noqa: E402
from motion2scene_tune_schedule_learner import arrays  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_imitation import (  # noqa: E402
    physical_regret,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    definition_digest,
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_curriculum import (  # noqa: E402
    DEFAULT_RULE,
    verified_decision_gap,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    fit_timed_schedule_policy,
    schedule_layout,
)

MODES = (
    "uniform",
    "coverage_only",
    "uniform_coverage",
    "historical_gated",
    "historical_ungated",
    "current_error",
    "historical_current_error",
)


def checkpoint_slots(plan, run_id, budget):
    run = next(r for r in plan["runs"] if r["run_id"] == run_id)
    if type(budget) is not int or not 0 <= budget < len(run["rounds"]):
        raise ValueError("checkpoint must belong to the specified acquisition trajectory")
    slots = []
    for index, row in enumerate(run["rounds"][: budget + 1]):
        if plan["schema"] == "motion2scene_expanded_acquisition_v1":
            model = row["inherited_model"] or str(Path(row["model_directory"]) / "result.json")
            teacher = row["inherited_teacher"] or str(
                Path(row["teacher_directory"]) / "result.json"
            )
            student = (
                None
                if index == 0
                else (
                    row["inherited_student"] or str(Path(row["student_directory"]) / "result.json")
                )
            )
            scene = row["scene"]
        else:
            model = row["expected_postupdate_training_result_path"]
            teacher = row["expected_teacher_result_path"]
            student = row["expected_student_result_path"]
            scene = row["scene_definition"]
        slots.append(
            dict(
                model=Path(model),
                teacher=Path(teacher),
                student=None if student is None else Path(student),
                scene=scene,
            )
        )
    return slots


def load_checkpoint(plan, run_id, budget):
    slots = checkpoint_slots(plan, run_id, budget)
    if any(not slot["model"].is_file() for slot in slots):
        raise FileNotFoundError("checkpoint has not completed fitting")
    bank_ref = plan["registry"]
    bank = load_verified_registry(bank_ref["path"], bank_ref["sha256"])
    final = bound(artifact(slots[-1]["model"]))
    stored = bound(final["teachers"])
    if [g["collection"] for g in stored] != [artifact(s["teacher"]) for s in slots]:
        raise ValueError("checkpoint teachers must be exactly the assigned prefix")
    groups, students, prior = [], [], None
    for index, (slot, expected) in enumerate(zip(slots, stored, strict=True)):
        group = audit_collection(slot["teacher"], bank, bank_ref)
        if group != expected:
            raise ValueError("stored targets differ from measured schedule branches")
        collection = bound(group["collection"])
        manifest = bound(collection["manifest"])
        if manifest["split"] != "development" or manifest["scene_definition"] != slot["scene"]:
            raise ValueError("assigned development scene required")
        student = None
        if index:
            actual, _, unknown = replay.audit_students(slot["student"], bank, bank_ref)
            if unknown or len(actual) != 1:
                raise ValueError("one measured pre-update student required")
            student = actual[0]
            identity = replay.group_identity(
                bank_ref, manifest["scene_definition"], group["physics_seed"]
            )
            if (
                student["encounter_id"] != definition_digest(identity)
                or student["manifest"]["policy"] != prior["policy"]
            ):
                raise ValueError(
                    "historical score must bind its actual pre-update policy and encounter"
                )
            validate_training_prefix(
                bound(prior["registration"]), [g["collection"] for g in groups], bank_ref
            )
        groups.append(group)
        students.append(student)
        prior = bound(artifact(slot["model"]))
        validate_training_prefix(
            bound(prior["registration"]), [g["collection"] for g in groups], bank_ref
        )
        if (
            prior["status"] != "complete"
            or prior.get("audit_errors")
            or bound(prior["teachers"]) != groups
        ):
            raise ValueError("each generating model must have the exact measured earlier targets")
        checked(Path(prior["policy"]["path"]), prior["policy"]["sha256"])
    return bank, groups, students, artifact(slots[-1]["model"])


def ungated_scores(groups, students):
    """Remove only the current-cue condition, retaining all physical/history tests."""
    scores = []
    for group, student in zip(groups, students, strict=True):
        manifest = bound(bound(group["collection"])["manifest"])
        for target in group["targets"]:
            matched = None
            if student is not None:
                matched = {**student, **student["phase_records"].get(target["phase_tick"], {})}
                matched["source_admitted"] = student["source_admitted"] and (
                    replay.dependency_identities(manifest)
                    == replay.dependency_identities(student["manifest"])
                )
                matched["matched_history"] = matched.get("recorded_history_sha256") == target.get(
                    "recorded_history_sha256"
                )
            scores.append(verified_decision_gap(target, matched, {"current_eligible": True}))
    return scores


def prediction_errors(model, targets, records):
    """Residuals on complete feasible tables, including measured zero-regret ties.

    No residual is assigned to incomplete or all-failed supervision. The common
    fitter still excludes ties when a phase has consequential targets.
    """
    errors = np.full(len(targets), np.nan)
    for i, (target, record) in enumerate(zip(targets, records, strict=True)):
        if not record["supervision_available"]:
            continue
        data = arrays([target])
        regrets, _ = physical_regret(
            data["passed"], data["passage_time_s"], data["admitted"], data["legality"]
        )
        legal = data["legality"][0]
        if not data["admitted"][0, legal].all() or not data["passed"][0, legal].any():
            raise ValueError("priority requires a complete feasible physical teacher")
        phase = int(np.flatnonzero(model["phase_ticks"] == target["phase_tick"])[0])
        z = (data["features"][0] - model["mean"][phase]) / model["std"][phase]
        predicted = -(z @ model["weights"][phase] + model["bias"][phase])
        errors[i] = float(np.mean((predicted[legal] - regrets[0, legal]) ** 2))
    return errors


def fit_controls(bank, groups, records, ungated, refits=3):
    if type(refits) is not int or refits < 1:
        raise ValueError("positive fixed refit count required")
    targets = [r for g in groups for r in g["targets"]]

    def fit(weights=None):
        rows, selected = weighted_targets(groups, weights)
        return fit_timed_schedule_policy(
            bank,
            feature_names=rows[0]["feature_names"],
            l2=10.0,
            sample_weights=selected,
            allow_measured_tie_initialization=True,
            **arrays(rows),
        )

    initial, _ = fit()
    results = {}
    for mode in MODES:
        model, history = initial, []
        for step in range(refits):
            errors = prediction_errors(model, targets, records)
            control = replay_control_weights(
                records, mode, supervised_error=errors, ungated_gaps=ungated
            )
            model, fitting = fit(control["weights"])
            history.append(
                dict(
                    step=step,
                    **control,
                    supervised_error=[float(x) if np.isfinite(x) else None for x in errors],
                )
            )
        results[mode] = dict(model=model, fit=fitting, weights=history)
    return results


def run(plan_path, run_id, budget, validation, out):
    plan = bound(artifact(plan_path))
    bank, groups, students, checkpoint = load_checkpoint(plan, run_id, budget)
    records = priority_records(bank, groups, students, DEFAULT_RULE)
    ungated = ungated_scores(groups, students)
    reference = bound(artifact(validation / "result.json"))
    if bound(reference["registration"])["registry"] != plan["registry"]:
        raise ValueError("validation must use the same qualified schedule bank")
    validation_groups = bound(reference["teachers"])
    for group in validation_groups:
        if bound(bound(group["collection"])["manifest"])["split"] != "development":
            raise ValueError("reserved outcomes cannot select a development learner")
    fitted = fit_controls(bank, groups, records, [r["gap"] for r in ungated])
    _, _, entries = schedule_layout(bank)
    out.mkdir(parents=True, exist_ok=False)
    experiment = write_new(
        out / "experiment.json",
        dict(
            plan=artifact(plan_path),
            run_id=run_id,
            checkpoint=checkpoint,
            encounters=budget,
            validation_teachers=reference["teachers"],
            implementations=[
                artifact(Path(__file__)),
                artifact(ROOT / "scripts/research/motion2scene_replay_controls.py"),
            ],
            methods=list(MODES),
            refits=3,
            l2=10.0,
            records=records,
            ungated=ungated,
            source_teacher_steps=sum(g["physics_steps"] for g in groups),
            new_physics_steps=0,
            historical_signal="measured pre-update generating policies, not refreshed control-policy outcomes",
            validation_scope=(
                "previously inspected six-context development branch proxies; "
                "no new physical evaluation"
            ),
        ),
    )
    results = {}
    for mode, result in fitted.items():
        folder = out / mode
        folder.mkdir()
        np.savez_compressed(folder / "policy.npz", **result.pop("model"))
        with np.load(folder / "policy.npz") as model:
            assessment = evaluate(lambda row: ridge_choice(model, row), validation_groups, entries)
        results[mode] = dict(
            policy=artifact(folder / "policy.npz"),
            **result,
            assessments=assessment,
            passing_branch_proxies=sum(r["branch_proxy_passed"] for r in assessment),
            validation_contexts=len(assessment),
        )
    return write_new(
        out / "result.json", dict(experiment=experiment, methods=results, new_physics_steps=0)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "validation", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.run_id, args.budget, args.validation, args.out)))
