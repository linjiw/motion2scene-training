#!/usr/bin/env python3
"""Register and fit an offline ridge-value student on fixed schedule teacher tables."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_imitation import (  # noqa: E402
    physical_regret,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_policy import (  # noqa: E402
    load_multi_policy,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_value_imitation import (  # noqa: E402
    fit_value_policy,
)


def register(args):
    original_path = args.teacher_run / "registration.json"
    original = json.loads(original_path.read_text())
    teacher = original["teacher_table"]
    checked(Path(teacher["path"]), teacher["sha256"])
    interface = artifact(args.teacher_run / "policy.npz")
    load_multi_policy(interface["path"], interface["sha256"], original["option_ids"])
    if not np.isfinite(args.l2) or args.l2 < 0:
        raise ValueError("finite nonnegative ridge regularization required")
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
            "schema": "motion2scene_value_imitation_development_v1",
            "teacher_table": teacher,
            "teacher_registration": artifact(original_path),
            "feature_interface": interface,
            "option_ids": original["option_ids"],
            "registry_sha256": original["registry_sha256"],
            "baseline_policies": [artifact(p) for p in args.baseline_policies],
            "l2": args.l2,
            "sample_weights": (
                None if args.sample_weights is None else artifact(args.sample_weights)
            ),
            "implementation": sources,
            "objective": (
                "one fixed per-option ridge fit of measured physical regret; complete consequential "
                "legal targets only; unpenalized intercept; no synthetic missing-branch labels"
            ),
            "normalization": "same feature set; supervised-row mean/std with std floor0.05",
            "metrics": [
                "mean selected-action regret on the fixed teacher table",
                "number of cost-optimal immediate actions on that table",
                "legal-target value fitting error",
            ],
            "scope": (
                "offline learner development after known softmax readout cost errors; "
                "same recorded teacher data, no new physics or held-out evaluation"
            ),
        },
    )
    print(json.dumps({"registered": str(args.out)}), flush=True)


def evaluate_model(model, teachers, regret, supervised):
    features = np.array([row["features"] for row in teachers])
    legal = np.array([row["legal_mask"] for row in teachers], dtype=bool)
    logits = ((features - model["mean"]) / model["std"]) @ model["weights"] + model["bias"]
    actions = np.where(legal, logits, -np.inf).argmax(axis=1)
    rows = []
    for index in np.flatnonzero(supervised):
        teacher, action = teachers[index], int(actions[index])
        best = float(regret[index, legal[index]].min())
        rows.append(
            {
                "teacher_index": int(index),
                "group": teacher["group"],
                "phase_s": teacher["phase_s"],
                "selected_action": action,
                "teacher_tiebroken_action": teacher["teacher_action"],
                "cost_optimal_immediate_action": bool(regret[index, action] <= best + 1e-12),
                "measured_continuation_regret": float(regret[index, action]),
                "verified_passing_continuation_exists": teacher["pass_labels"][action],
                "selected_continuation_time_s": teacher["passage_time_s"][action],
                "continuation_branch": teacher["continuation_branch_ids"][action],
                "legal_mask": legal[index].tolist(),
                "masked_logits": [
                    float(logit) if allowed else None
                    for logit, allowed in zip(logits[index], legal[index], strict=True)
                ],
            }
        )
    return {
        "complete_consequential_decisions": len(rows),
        "cost_optimal_immediate_actions": sum(r["cost_optimal_immediate_action"] for r in rows),
        "mean_selected_continuation_regret": float(
            np.mean([row["measured_continuation_regret"] for row in rows])
        ),
        "decisions": rows,
        "scope": "teacher-table action values only; student continuation requires actual execution",
    }


def fit(out):
    registration = json.loads((out / "registration.json").read_text())
    if (out / "policy.npz").exists() or (out / "result.json").exists():
        raise ValueError("fit output already exists; preserve it and register a fresh experiment")
    for ref in registration["implementation"]:
        checked(Path(ref["path"]), ref["sha256"])
    table = registration["teacher_table"]
    teachers = json.loads(checked(Path(table["path"]), table["sha256"]).read_text())
    interface = registration["feature_interface"]
    old_model = load_multi_policy(
        interface["path"], interface["sha256"], registration["option_ids"]
    )
    names = list(old_model["feature_names"])
    passed = np.array([row["pass_labels"] for row in teachers], dtype=bool)
    times = [[np.nan if v is None else v for v in row["passage_time_s"]] for row in teachers]
    admitted = np.array([row["admitted"] for row in teachers], dtype=bool)
    legal = np.array([row["legal_mask"] for row in teachers], dtype=bool)
    regret, supervised = physical_regret(passed, times, admitted, legal)
    weights = registration["sample_weights"]
    if weights is not None:
        weights = json.loads(checked(Path(weights["path"]), weights["sha256"]).read_text())
    started = time.monotonic()
    model, report = fit_value_policy(
        [row["features"] for row in teachers],
        names,
        registration["option_ids"],
        passed,
        times,
        admitted,
        legal,
        l2=registration["l2"],
        sample_weights=weights,
    )
    fit_seconds = time.monotonic() - started
    path = out / "policy.npz"
    np.savez_compressed(path, **model)
    model_ref = artifact(path)
    load_multi_policy(path, model_ref["sha256"], registration["option_ids"])
    baselines = []
    for ref in registration["baseline_policies"]:
        baseline = load_multi_policy(ref["path"], ref["sha256"], registration["option_ids"])
        if list(baseline["feature_names"]) != names:
            raise ValueError("comparison requires the exact same feature names")
        baselines.append(
            {
                "model": ref,
                "teacher_table_readout": evaluate_model(baseline, teachers, regret, supervised),
            }
        )
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "model": model_ref,
            "fit_seconds": fit_seconds,
            "fit": report,
            "teacher_table_readout": evaluate_model(model, teachers, regret, supervised),
            "baseline_readouts": baselines,
            "new_physics_steps": 0,
            "scope": "offline learner development; no curriculum advantage or robot-performance claim",
        },
    )
    print(json.dumps({"model": model_ref, "fit_seconds": fit_seconds}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "fit"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher-run", type=Path)
    parser.add_argument("--baseline-policies", type=Path, nargs="+", default=[])
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--sample-weights", type=Path)
    args = parser.parse_args()
    if args.stage == "register":
        register(args)
    else:
        fit(args.out)
