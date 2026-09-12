#!/usr/bin/env python3
"""Evaluate passage-first observation-consistent targets on recorded schedules."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import complete_schedules, feature_keys, read_checked  # noqa: E402
from motion2scene_information_ablation import (  # noqa: E402
    mask_scene_features,
    maximum_causal_passages,
)
from motion2scene_information_policy import (  # noqa: E402
    evaluate_finite_policy,
    passage_first_information_teacher,
)
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    schedule_layout,
)


def run(source, out):
    original = read_checked(artifact(source / "result.json"))
    registration = read_checked(original["registration"])
    groups = read_checked(original["teachers"])
    bank_ref = registration["registry"]
    bank = load_verified_registry(bank_ref["path"], bank_ref["sha256"])
    phases, _, layout = schedule_layout(bank)
    entries = [layout[i] for i in range(len(layout))]
    for group in groups:
        manifest = read_checked(read_checked(group["collection"])["manifest"])
        if manifest["split"] != "development":
            raise ValueError("recorded development groups only")
        group["targets"].sort(key=lambda r: r["phase_tick"])
        if [r["phase_tick"] for r in group["targets"]] != phases.tolist():
            raise ValueError("complete recorded neutral phases required")
    features = np.array([[r["features"] for r in g["targets"]] for g in groups])
    names = groups[0]["targets"][0]["feature_names"]
    passed, times, admitted = complete_schedules(groups, layout)
    out.mkdir(parents=True, exist_ok=False)
    experiment = write_new(
        out / "experiment.json",
        dict(
            source=artifact(source / "result.json"),
            teachers=original["teachers"],
            implementation=[
                artifact(Path(__file__)),
                artifact(ROOT / "scripts/research/motion2scene_information_policy.py"),
            ],
            objective="maximize empirical passage count, then minimize successful-time sum at equal counts",
            distribution="uniform over the six recorded development contexts",
            intervention="same four corridor-summary reveal conditions as the existing information ablation",
            reveal_ticks=phases.tolist() + [None],
            new_physics_steps=0,
            interpretation=(
                "finite shared-policy optimization on measured schedule tables; "
                "not a fitted student, sensor-latency rollout or held-out result"
            ),
        ),
    )
    conditions = []
    for reveal in [*phases.tolist(), None]:
        keys = feature_keys(mask_scene_features(features, names, phases, reveal))
        teacher = passage_first_information_teacher(keys, phases, entries, passed, times, admitted)
        evaluation = evaluate_finite_policy(teacher, keys, phases, passed, times)
        if evaluation["passage_count"] != maximum_causal_passages(
            keys, phases, entries, passed, times, admitted
        ):
            raise ValueError("constructed policy does not realize the independent count optimum")
        selected = evaluation["selected_schedules"]
        condition = dict(
            reveal_tick=reveal,
            teacher=teacher,
            evaluation=evaluation,
            initial_actions=[r["teacher_action"] for r in teacher if r["phase_tick"] == phases[0]],
            per_context=[
                dict(
                    scene_id=g["scene_id"],
                    selected_schedule=bank.option_ids[selected[i]],
                    passed=bool(passed[i, selected[i]]),
                    passage_time_s=float(times[i, selected[i]]) if passed[i, selected[i]] else None,
                )
                for i, g in enumerate(groups)
            ],
        )
        baseline = conditions[0] if conditions else condition
        differences = [
            r["passage_time_s"] - b["passage_time_s"]
            for r, b in zip(condition["per_context"], baseline["per_context"], strict=True)
            if r["passed"] and b["passed"]
        ]
        condition["mutually_successful_with_earliest_reveal"] = len(differences)
        condition["mean_paired_time_difference_from_earliest_reveal_s"] = (
            float(np.mean(differences)) if differences else None
        )
        conditions.append(condition)
    return write_new(
        out / "result.json",
        dict(
            experiment=experiment,
            conditions=conditions,
            finite_bank_capability=int(passed.any(axis=1).sum()),
            scope="causally realizable finite teacher targets; downstream learner unchanged",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.out)))
