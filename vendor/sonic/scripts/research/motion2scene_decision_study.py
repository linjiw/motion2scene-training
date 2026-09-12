#!/usr/bin/env python3
"""Compare recorded decision learning and finite observation-consistent teaching.

Development-only CPU study. A branch-table rollout is a proxy, not a new physical
execution. Whole encounters stay together in every leave-one-context-out fold.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_observation_teacher import observation_consistent_teacher  # noqa: E402
from motion2scene_tune_schedule_learner import arrays  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_imitation import (  # noqa: E402
    physical_regret,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    fit_timed_schedule_policy,
    schedule_layout,
)


def read_checked(ref):
    content = Path(ref["path"]).read_bytes()
    if "sha256:" + hashlib.sha256(content).hexdigest() != ref["sha256"]:
        raise ValueError(f"changed input: {ref['path']}")
    return json.loads(content)


def feature_keys(features):
    """Exact current student-vector equality; no privileged prefix/state hashes."""
    x = np.asarray(features, dtype="<f8").copy()
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("finite encounter-by-phase student features required")
    x[x == 0] = 0  # +0 and -0 are equal observations.
    return np.asarray([[hashlib.sha256(row.tobytes()).hexdigest() for row in group] for group in x])


def complete_schedules(groups, entries):
    """Recover measured schedule outcomes from their immediate commitment row."""
    passed, times, admitted = [], [], []
    for group in groups:
        targets = {row["phase_tick"]: row for row in group["targets"]}
        p, t, a = [], [], []
        for option, tick in entries.items():
            row = targets[max(targets) if tick is None else tick]
            p.append(row["pass_labels"][option])
            t.append(row["passage_time_s"][option])
            a.append(row["admitted"][option])
        passed.append(p)
        times.append(t)
        admitted.append(a)
    return np.asarray(passed, bool), np.asarray(times, float), np.asarray(admitted, bool)


class OutcomeTrees:
    """Small separate feasibility/time trees with a fixed, untuned depth of two.

    Time is fitted only on measured successful branches. Predicted probability
    >= 0.5 defines candidate actions; use minimum predicted successful time.
    If none qualifies, maximize estimated feasibility, then minimize time.
    This threshold is an estimate, not a safety certificate.
    """

    def __init__(self, rows):
        self.heads = {}
        for tick in sorted({row["phase_tick"] for row in rows}):
            local = [row for row in rows if row["phase_tick"] == tick]
            data = arrays(local)
            for action in np.flatnonzero(data["legality"].any(0)):
                measured = data["admitted"][:, action] & data["legality"][:, action]
                if not measured.any():
                    raise ValueError("legal action lacks measured feasibility targets")
                success = measured & data["passed"][:, action]
                classifier = DecisionTreeClassifier(max_depth=2, random_state=0).fit(
                    data["features"][measured], data["passed"][measured, action]
                )
                regressor = None
                if success.any():
                    regressor = DecisionTreeRegressor(max_depth=2, random_state=0).fit(
                        data["features"][success], data["passage_time_s"][success, action]
                    )
                self.heads[tick, int(action)] = classifier, regressor

    def choose(self, row):
        predictions = []
        x = np.asarray(row["features"])[None]
        for action in np.flatnonzero(row["legal_mask"]):
            classifier, regressor = self.heads[row["phase_tick"], int(action)]
            index = np.flatnonzero(classifier.classes_ == True)  # noqa: E712
            probability = float(classifier.predict_proba(x)[0, index[0]]) if len(index) else 0.0
            time = float(regressor.predict(x)[0]) if regressor is not None else np.inf
            predictions.append((int(action), probability, time))
        feasible = [v for v in predictions if v[1] >= 0.5 and np.isfinite(v[2])]
        if feasible:
            return min(feasible, key=lambda v: (v[2], -v[1], v[0]))[0]
        return min(predictions, key=lambda v: (-v[1], v[2], v[0]))[0]


def ridge_choice(model, row):
    phase = int(np.flatnonzero(model["phase_ticks"] == row["phase_tick"])[0])
    x = (np.asarray(row["features"]) - model["mean"][phase]) / model["std"][phase]
    values = x @ model["weights"][phase] + model["bias"][phase]
    return int(np.where(row["legal_mask"], values, -np.inf).argmax())


def evaluate(choose, groups, entries):
    reports = []
    passed, times, admitted = complete_schedules(groups, entries)
    for i, group in enumerate(groups):
        rows = sorted(group["targets"], key=lambda r: r["phase_tick"])
        regret, eligible = physical_regret(
            **{
                k: v
                for k, v in arrays(rows).items()
                if k in ("passed", "passage_time_s", "admitted", "legality")
            }
        )
        choices = [choose(row) for row in rows]
        selected = next((action for action in choices if action), 0)
        if not admitted[i, selected]:
            raise ValueError("selected full schedule has an unknown outcome")
        # Following neutral prefixes until first commitment selects one measured
        # schedule. Subsequent hypothetical neutral decisions are not visited.
        decision_rows = [
            dict(
                phase_tick=row["phase_tick"],
                action=action,
                regret=float(regret[j, action]) if eligible[j] else None,
            )
            for j, (row, action) in enumerate(zip(rows, choices, strict=True))
        ]
        reports.append(
            dict(
                scene_id=group["scene_id"],
                selected_schedule=selected,
                branch_proxy_passed=bool(passed[i, selected]),
                branch_proxy_time_s=float(times[i, selected]) if passed[i, selected] else None,
                phase_decisions=decision_rows,
            )
        )
    return reports


def run(source, out):
    result_path = source / "result.json"
    original = json.loads(result_path.read_text())
    registration = read_checked(original["registration"])
    groups = read_checked(original["teachers"])
    for ref in registration["collections"]:
        collection = read_checked(ref)
        manifest = read_checked(collection["manifest"])
        if manifest["split"] != "development":
            raise ValueError("development data only")
    bank_ref = registration["registry"]
    bank = load_verified_registry(Path(bank_ref["path"]), bank_ref["sha256"])
    phases, _, entries = schedule_layout(bank)
    if len({g["scene_id"] for g in groups}) != len(groups):
        raise ValueError("distinct development contexts required")
    for group in groups:
        group["targets"].sort(key=lambda r: r["phase_tick"])
        if [r["phase_tick"] for r in group["targets"]] != phases.tolist():
            raise ValueError("every recorded neutral phase required")
    keys = feature_keys([[r["features"] for r in g["targets"]] for g in groups])
    passed, times, admitted = complete_schedules(groups, entries)
    teacher = observation_consistent_teacher(
        keys, phases, [entries[i] for i in range(len(entries))], passed, times, admitted
    )
    out.mkdir(parents=True, exist_ok=False)
    experiment = dict(
        source_result=dict(
            path=str(result_path),
            sha256="sha256:" + hashlib.sha256(result_path.read_bytes()).hexdigest(),
        ),
        source_teachers=original["teachers"],
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        methods=["regret_ridge_l2_10", "separate_outcome_trees_depth_2"],
        comparison="fixed choices, no hyperparameter search; objective and model change jointly",
        fold_rule="fit five contexts, test the sixth; separate six-context training readout",
        interpretation="recorded-branch proxy only; new physical rollouts = 0",
    )
    (out / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    assessments = []
    for held in [None, *range(len(groups))]:
        training = [r for i, g in enumerate(groups) if i != held for r in g["targets"]]
        testing = groups if held is None else [groups[held]]
        ridge, _ = fit_timed_schedule_policy(
            bank, feature_names=training[0]["feature_names"], l2=10.0, **arrays(training)
        )
        trees = OutcomeTrees(training)
        assessments.append(
            dict(
                fold="training_readout" if held is None else groups[held]["scene_id"],
                ridge=evaluate(lambda row: ridge_choice(ridge, row), testing, entries),
                outcome_trees=evaluate(trees.choose, testing, entries),
            )
        )
    summary = {}
    for name in ("ridge", "outcome_trees"):
        for split, folds in (("training", assessments[:1]), ("leave_context_out", assessments[1:])):
            rows = [r for fold in folds for r in fold[name]]
            decisions = [
                d["regret"] for r in rows for d in r["phase_decisions"] if d["regret"] is not None
            ]
            summary[f"{name}_{split}"] = dict(
                passing_branch_proxies=sum(r["branch_proxy_passed"] for r in rows),
                contexts=len(rows),
                mean_recorded_phase_regret=float(np.mean(decisions)),
            )
    report = dict(
        **experiment,
        summary=summary,
        folds=assessments,
        observation_teacher=teacher,
        exact_groups_per_phase=[len(set(keys[:, k])) for k in range(len(phases))],
        information_conflicts=sum(r["information_conflict"] for r in teacher),
        source_physics_steps=original["source_physics_steps"],
        new_physics_steps=0,
    )
    (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({"summary": summary, "information_conflicts": report["information_conflicts"]})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
