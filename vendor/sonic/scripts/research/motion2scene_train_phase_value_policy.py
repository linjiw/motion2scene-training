#!/usr/bin/env python3
"""Register and fit fixed phase heads on unchanged physical schedule teachers."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
from motion2scene_train_value_policy import evaluate_model  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_imitation import (  # noqa: E402
    physical_regret,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_policy import (  # noqa: E402
    load_multi_policy,
    load_option_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_phase_value_imitation import (  # noqa: E402
    OPTION_IDS,
    PHASES,
    QUALIFIED,
    RIDGE_L2,
    choose_phase_option,
    fit_phase_value_policy,
)


def register(args):
    prior_path = args.source_run / "registration.json"
    prior = json.loads(prior_path.read_text())
    if prior["l2"] != RIDGE_L2 or tuple(prior["option_ids"]) != OPTION_IDS:
        raise ValueError(
            "source must use the frozen five-option/lambda1e-6 physical-regret configuration"
        )
    checked(Path(prior["teacher_table"]["path"]), prior["teacher_table"]["sha256"])
    registry_ref = artifact(args.option_registry)
    if registry_ref["sha256"] != prior["registry_sha256"]:
        raise ValueError("source teacher registry differs from the supplied qualified registry")
    registry = load_option_registry(registry_ref["path"], registry_ref["sha256"])
    for phase_index, phase in enumerate(PHASES):
        qualified = [True] + [
            any(abs(phase - tick) < 1e-8 for tick in entry["qualified_entry_times_s"])
            for entry in registry["references"][1:]
        ]
        if not np.array_equal(qualified, QUALIFIED[phase_index]):
            raise ValueError("phase support differs from physical option qualification")
    interface = prior["feature_interface"]
    names = load_multi_policy(interface["path"], interface["sha256"], list(OPTION_IDS))[
        "feature_names"
    ]
    if len(names) != 110:
        raise ValueError("this experiment requires the exact existing 110-feature interface")
    baseline = artifact(args.source_run / "policy.npz")
    load_multi_policy(baseline["path"], baseline["sha256"], list(OPTION_IDS))
    if prior["sample_weights"] is not None:
        checked(Path(prior["sample_weights"]["path"]), prior["sample_weights"]["sha256"])
    args.out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__)])):
        snapshot = args.out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(snapshot)})
    write_new(
        args.out / "registration.json",
        {
            "schema": "motion2scene_phase_value_development_v1",
            "teacher_table": prior["teacher_table"],
            "source_registration": artifact(prior_path),
            "feature_interface": interface,
            "baseline_linear_policy": baseline,
            "option_ids": list(OPTION_IDS),
            "option_registry": registry_ref,
            "registry_sha256": registry_ref["sha256"],
            "phases_s": PHASES.tolist(),
            "qualified_mask": QUALIFIED.tolist(),
            "l2": RIDGE_L2,
            "sample_weights": prior["sample_weights"],
            "implementation": sources,
            "objective": (
                "same measured physical regret; one weighted ridge fit per phase/observed legal option; "
                "unpenalized intercept"
            ),
            "normalization": "per-phase complete consequential row mean/std, same0.05 floor",
            "missing_targets": (
                "full-table completeness checked before fitting; illegal/unobserved option values absent"
            ),
            "student_inputs": (
                "exact110 sensor/state/controller features; phase comes from existing controller clock; "
                "no scene labels"
            ),
            "fixed_design": "only phases .2/.3/.4 and lambda1e-6; no model or hyperparameter search",
            "runtime_contract": "single-legal-action hold and mandatory guarded recovery bypass learned values",
            "scope": (
                "offline development after the known aggregated linear cost regression; finite teacher table only"
            ),
        },
    )
    print(json.dumps({"registration": artifact(args.out / "registration.json")}), flush=True)


def phase_readout(model, teachers, regret, supervised):
    rows = []
    for index in np.flatnonzero(supervised):
        teacher = teachers[index]
        legal = np.array(teacher["legal_mask"], dtype=bool)
        action, logits = choose_phase_option(
            model, tuple(model["feature_names"]), teacher["features"], legal, teacher["phase_s"]
        )
        rows.append(
            {
                "teacher_index": int(index),
                "group": teacher["group"],
                "phase_s": teacher["phase_s"],
                "selected_action": action,
                "teacher_tiebroken_action": teacher["teacher_action"],
                "cost_optimal_immediate_action": bool(
                    regret[index, action] <= regret[index, legal].min() + 1e-12
                ),
                "measured_continuation_regret": float(regret[index, action]),
                "verified_passing_continuation_exists": teacher["pass_labels"][action],
                "selected_continuation_time_s": teacher["passage_time_s"][action],
                "continuation_branch": teacher["continuation_branch_ids"][action],
                "legal_mask": legal.tolist(),
                "masked_logits": logits,
            }
        )
    return {
        "complete_consequential_decisions": len(rows),
        "cost_optimal_immediate_actions": sum(row["cost_optimal_immediate_action"] for row in rows),
        "mean_selected_continuation_regret": float(
            np.mean([row["measured_continuation_regret"] for row in rows])
        ),
        "decisions": rows,
        "scope": "finite recorded teacher-table readout; actual student execution is still required",
    }


def fit(out):
    registration = json.loads((out / "registration.json").read_text())
    if (out / "fit_attempt.json").exists() or (out / "policy.npz").exists():
        raise ValueError("one fit per registration; preserve prior attempts")
    for ref in registration["implementation"]:
        checked(Path(ref["path"]), ref["sha256"])
    for key in (
        "source_registration",
        "feature_interface",
        "baseline_linear_policy",
        "option_registry",
    ):
        ref = registration[key]
        checked(Path(ref["path"]), ref["sha256"])
    ref = registration["teacher_table"]
    teachers = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    interface = registration["feature_interface"]
    names = list(
        load_multi_policy(interface["path"], interface["sha256"], list(OPTION_IDS))["feature_names"]
    )
    passed = np.array([row["pass_labels"] for row in teachers], dtype=bool)
    times = np.array(
        [
            [np.nan if value is None else value for value in row["passage_time_s"]]
            for row in teachers
        ]
    )
    admitted = np.array([row["admitted"] for row in teachers], dtype=bool)
    legal = np.array([row["legal_mask"] for row in teachers], dtype=bool)
    regret, supervised = physical_regret(passed, times, admitted, legal)
    weights = registration["sample_weights"]
    if weights is not None:
        weights = json.loads(checked(Path(weights["path"]), weights["sha256"]).read_text())
    write_new(out / "fit_attempt.json", {"registration": artifact(out / "registration.json")})
    started = time.monotonic()
    model, report = fit_phase_value_policy(
        [row["features"] for row in teachers],
        names,
        OPTION_IDS,
        [row["phase_s"] for row in teachers],
        passed,
        times,
        admitted,
        legal,
        sample_weights=weights,
    )
    fit_seconds = time.monotonic() - started
    path = out / "policy.npz"
    np.savez_compressed(path, **model)
    model_ref = artifact(path)
    model = load_multi_policy(path, model_ref["sha256"], list(OPTION_IDS))
    baseline = registration["baseline_linear_policy"]
    old_model = load_multi_policy(baseline["path"], baseline["sha256"], list(OPTION_IDS))
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "model": model_ref,
            "fit_seconds": fit_seconds,
            "fit": report,
            "teacher_table_readout": phase_readout(model, teachers, regret, supervised),
            "baseline_linear_readout": evaluate_model(old_model, teachers, regret, supervised),
            "new_physics_steps": 0,
            "scope": "fixed-design finite-table learner development; no traversal gain or generalization claim",
        },
    )
    print(json.dumps({"model": model_ref, "fit_seconds": fit_seconds}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "fit"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--option-registry", type=Path)
    args = parser.parse_args()
    if args.stage == "register":
        register(args)
    else:
        fit(args.out)
