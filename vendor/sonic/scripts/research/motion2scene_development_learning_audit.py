#!/usr/bin/env python3
"""Test information consistency and learner regressions on released development data."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_kernel_outcomes import fit_kernel_outcomes, predict_kernel_outcomes  # noqa: E402
from motion2scene_observation_teacher import observation_consistent_teacher  # noqa: E402
from motion2scene_schedule_dataset_baseline import (  # noqa: E402
    arrays,
    fit_targets,
    inspect_package,
    read_json,
    sha256,
)

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    schedule_layout,
)


def numeric_key(values):
    """Exact numeric equality, including equal signed zeros, without scene identity."""
    values = np.array(values, dtype="<f8", copy=True)
    if not np.isfinite(values).all():
        raise ValueError("finite observed histories required")
    values[values == 0] = 0
    return hashlib.sha256(values.tobytes()).hexdigest()


def fitted(bank, rows, kind):
    if kind == "ridge":
        model, report = fit_targets(bank, rows, 10.0, allow_measured_tie_initialization=True)
        if model is None:
            raise ValueError(str(report))
        return model
    return fit_kernel_outcomes(
        [r["features"] for r in rows],
        [r["phase_tick"] for r in rows],
        [r["pass_labels"] for r in rows],
        [r["passage_time_s"] for r in rows],
        [r["admitted"] for r in rows],
        [r["legal_mask"] for r in rows],
    )


def predict(model, row, kind):
    if kind == "kernel":
        return predict_kernel_outcomes(
            model, row["features"], row["phase_tick"], row["legal_mask"]
        )["action"]
    k = model["phase_ticks"].tolist().index(row["phase_tick"])
    values = ((np.asarray(row["features"]) - model["mean"][k]) / model["std"][k]) @ model[
        "weights"
    ][k] + model["bias"][k]
    legal = np.asarray(row["legal_mask"], bool)
    if np.any(legal & ~model["trained_mask"][k]):
        raise ValueError("legal prediction lacks a fitted column")
    return int(np.where(legal, values, -np.inf).argmax())


def evaluate(model, rows, kind, teacher, episodes, bank):
    decisions = [dict(phase_tick=r["phase_tick"], action=predict(model, r, kind)) for r in rows]
    selected = next((d["action"] for d in decisions if d["action"]), 0)
    episode = episodes[teacher["episode_ids"][selected]]
    assessment = episode["assessment"]
    passed = bool(assessment["pass"])
    cost = assessment["costs"]["passage_time_s"] if passed else None
    successful_times = [
        episodes[e]["assessment"]["costs"]["passage_time_s"]
        for e in teacher["episode_ids"]
        if episodes[e]["assessment"]["pass"]
    ]
    best = min(successful_times) if successful_times else None
    return dict(
        scene_id=teacher["scene_id"],
        selected_option=bank.option_ids[selected],
        selected_option_index=selected,
        recorded_branch_pass=passed,
        recorded_branch_time_s=cost,
        best_bank_time_s=best,
        excess_time_s=None if cost is None else cost - best,
        all_neutral_phase_readouts=decisions,
    )


def run(dataset, out):
    out.mkdir(parents=True, exist_ok=False)
    registration = dict(
        dataset=str(dataset.resolve()),
        dataset_manifest_sha256=sha256(dataset / "manifest.json"),
        models={
            "ridge": {"l2": 10.0},
            "kernel": {
                "l2": 0.001,
                "bandwidth": "median positive training mean-squared distance",
                "feasibility_threshold": 0.5,
            },
        },
        comparisons=["five-context refit", "six-context refit", "leave-one-development-scene-out"],
        equivalence="exact equality of the cumulative recorded 114D input sequence; no rounding or privileged scene/state identifiers",
        selection="No deployed learner selection from this offline audit alone.",
        scope="Previously inspected development data. All traversal quantities are recorded-branch proxies, not new closed-loop outcomes.",
        new_physics_steps=0,
    )
    sources = [
        Path(__file__),
        Path(__file__).with_name("motion2scene_observation_teacher.py"),
        Path(__file__).with_name("motion2scene_kernel_outcomes.py"),
    ]
    snapshot = out / "source"
    snapshot.mkdir()
    for source in sources:
        (snapshot / source.name).write_bytes(source.read_bytes())
    registration["source_sha256"] = {p.name: sha256(p) for p in sources}
    (out / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    bank, targets, inspection = inspect_package(dataset)
    manifest = read_json(dataset / "manifest.json")
    episodes = {e["episode_id"]: e for e in manifest["episodes"]}
    teachers = read_json(dataset / "teachers.json")
    phases, _, entry_map = schedule_layout(bank)
    groups = [[r for r in targets if r["group"] == t["recorded_history_group"]] for t in teachers]
    pass_table, time_table, admitted_table, history_keys = [], [], [], []
    current_collisions = []
    for t in teachers:
        records = [episodes[e] for e in t["episode_ids"]]
        neutral = arrays(dataset / records[0]["directory"] / "student_inputs.npz")["features"]
        history_keys.append([numeric_key(neutral[:tick]) for tick in phases])
        pass_table.append([e["assessment"]["pass"] for e in records])
        time_table.append([e["assessment"]["costs"]["passage_time_s"] for e in records])
        admitted_table.append([e["assessment"]["measurement_admitted"] for e in records])
    for k, tick in enumerate(phases.tolist()):
        by_key = {}
        for i, group in enumerate(groups):
            by_key.setdefault(numeric_key(group[k]["features"]), []).append(i)
        for ids in by_key.values():
            if len(ids) < 2:
                continue
            passing = [
                set(
                    np.flatnonzero(
                        np.array(groups[i][k]["legal_mask"]) & np.array(groups[i][k]["pass_labels"])
                    ).tolist()
                )
                for i in ids
            ]
            current_collisions.append(
                dict(
                    phase_tick=tick,
                    scenes=[teachers[i]["scene_id"] for i in ids],
                    teacher_actions=[groups[i][k]["teacher_action"] for i in ids],
                    common_scene_wise_passing_actions=sorted(set.intersection(*passing)),
                )
            )
    tree = observation_consistent_teacher(
        history_keys,
        phases,
        [entry_map[i] for i in range(len(bank.option_ids))],
        pass_table,
        time_table,
        admitted_table,
    )
    comparisons = []
    for kind in ("ridge", "kernel"):
        for training_count in (5, 6):
            model = fitted(bank, [r for g in groups[:training_count] for r in g], kind)
            rows = [
                evaluate(model, g, kind, t, episodes, bank)
                for g, t in zip(groups, teachers, strict=True)
            ]
            comparisons.append(
                dict(
                    model=kind,
                    training_contexts=training_count,
                    evaluation="all six inspected contexts",
                    pass_count=sum(r["recorded_branch_pass"] for r in rows),
                    rows=rows,
                )
            )
        folds = []
        for held, (group, teacher) in enumerate(zip(groups, teachers, strict=True)):
            model = fitted(bank, [r for i, g in enumerate(groups) if i != held for r in g], kind)
            folds.append(evaluate(model, group, kind, teacher, episodes, bank))
        comparisons.append(
            dict(
                model=kind,
                training_contexts=5,
                evaluation="leave one inspected development scene out",
                pass_count=sum(r["recorded_branch_pass"] for r in folds),
                rows=folds,
            )
        )
    result = dict(
        status="complete_offline_development_audit",
        inspection=inspection,
        current_feature_collisions=current_collisions,
        observation_consistent_tree=tree,
        history_collision_groups=sum(len(r["encounter_indices"]) > 1 for r in tree),
        information_conflict_groups=sum(r["information_conflict"] for r in tree),
        comparisons=comparisons,
        scope=registration["scope"],
        new_physics_steps=0,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Development teacher and learner experiment",
        "",
        registration["scope"],
        "",
        f"Exact history collisions: {result['history_collision_groups']} groups; incompatible successful continuations: {result['information_conflict_groups']} groups.",
        "",
        "| Learner | Training contexts | Readout | Passing recorded branches |",
        "|---|---:|---|---:|",
    ]
    lines += [
        f"| {r['model']} | {r['training_contexts']} | {r['evaluation']} | {r['pass_count']}/6 |"
        for r in comparisons
    ]
    lines += [
        "",
        "No new held-out or closed-loop performance is claimed. The nonlinear model must be evaluated in actual executions before selecting it for acquisition.",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "history_collision_groups", "information_conflict_groups")
            }
        )
    )
    print(
        json.dumps(
            [
                dict(
                    model=r["model"],
                    training_contexts=r["training_contexts"],
                    evaluation=r["evaluation"],
                    pass_count=r["pass_count"],
                )
                for r in comparisons
            ]
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.dataset, args.out)
