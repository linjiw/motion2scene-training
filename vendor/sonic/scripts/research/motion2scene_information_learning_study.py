#!/usr/bin/env python3
"""Same ridge learner, scene-wise versus observation-consistent teaching targets."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import (  # noqa: E402
    complete_schedules,
    feature_keys,
    read_checked,
    ridge_choice,
)
from motion2scene_information_ablation import mask_scene_features  # noqa: E402
from motion2scene_information_policy import passage_first_information_teacher  # noqa: E402
from motion2scene_information_regret import expand_group_targets, fit_explicit_regret  # noqa: E402
from motion2scene_timing_diagnostic import artifact  # noqa: E402
from motion2scene_tune_schedule_learner import arrays  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
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


def follow(model, x, legal, phases, passed, times, ids):
    reports = []
    for i, scene_id in enumerate(ids):
        selected = 0
        for k, tick in enumerate(phases):
            selected = ridge_choice(
                model, dict(phase_tick=tick, features=x[i, k], legal_mask=legal[i, k])
            )
            if selected:
                break
        reports.append(
            dict(
                scene_id=scene_id,
                selected_schedule=int(selected),
                branch_proxy_passed=bool(passed[i, selected]),
                branch_proxy_time_s=float(times[i, selected]) if passed[i, selected] else None,
            )
        )
    return reports


def run(source, out):
    original = read_checked(artifact(source / "result.json"))
    registration = read_checked(original["registration"])
    groups = read_checked(original["teachers"])
    for group in groups:
        if read_checked(read_checked(group["collection"])["manifest"])["split"] != "development":
            raise ValueError("development data only")
        group["targets"].sort(key=lambda r: r["phase_tick"])
    ref = registration["registry"]
    bank = load_verified_registry(ref["path"], ref["sha256"])
    phases, _, layout = schedule_layout(bank)
    entries = [layout[i] for i in range(len(layout))]
    records = [r for g in groups for r in g["targets"]]
    data = arrays(records)
    names = records[0]["feature_names"]
    x = data["features"].reshape(len(groups), len(phases), len(names))
    passed, times, admitted = complete_schedules(groups, layout)
    old_regret, _ = physical_regret(
        data["passed"], data["passage_time_s"], data["admitted"], data["legality"]
    )
    out.mkdir(parents=True, exist_ok=False)
    experiment = write_new(
        out / "experiment.json",
        dict(
            source=artifact(source / "result.json"),
            teachers=original["teachers"],
            implementations=[
                artifact(Path(__file__)),
                artifact(ROOT / "scripts/research/motion2scene_information_policy.py"),
                artifact(ROOT / "scripts/research/motion2scene_information_regret.py"),
            ],
            conditions=phases.tolist() + [None],
            l2=10.0,
            control=(
                "same recorded inputs, phase normalization, ridge penalty and equations; "
                "different continuation targets"
            ),
            weights="uniform per encounter within each phase, as in the original development learner",
            scope=(
                "recorded training readouts under a controlled information ablation; "
                "not new physical evaluations"
            ),
            new_physics_steps=0,
        ),
    )
    reports = []
    for reveal in [*phases.tolist(), None]:
        values = mask_scene_features(x, names, phases, reveal)
        keys = feature_keys(values)
        teacher = passage_first_information_teacher(keys, phases, entries, passed, times, admitted)
        targets, legal = expand_group_targets(teacher, values, phases, len(bank.option_ids))
        flat = values.reshape(data["features"].shape)
        baseline, _ = fit_timed_schedule_policy(
            bank, feature_names=names, l2=10.0, **dict(data, features=flat)
        )
        model = fit_explicit_regret(
            bank,
            flat,
            names,
            data["phase_ticks"],
            targets.reshape(old_regret.shape),
            legal.reshape(data["legality"].shape),
            l2=10.0,
        )
        label = "never" if reveal is None else str(reveal)
        np.savez(out / f"scene_wise_{label}.npz", **baseline)
        np.savez(out / f"observation_consistent_{label}.npz", **model)
        old = follow(
            baseline, values, legal, phases, passed, times, [g["scene_id"] for g in groups]
        )
        new = follow(model, values, legal, phases, passed, times, [g["scene_id"] for g in groups])
        differences = [
            b["branch_proxy_time_s"] - a["branch_proxy_time_s"]
            for a, b in zip(old, new, strict=True)
            if a["branch_proxy_passed"] and b["branch_proxy_passed"]
        ]
        if reveal == int(phases[0]):
            finite = np.isfinite(targets.reshape(old_regret.shape)) & data["legality"]
            if not np.array_equal(targets.reshape(old_regret.shape)[finite], old_regret[finite]):
                raise ValueError(
                    "singleton targets differ from the existing physical-regret teacher"
                )
            maximum = max(
                float(np.max(np.abs(baseline[k] - model[k])))
                for k in ("mean", "std", "weights", "bias")
            )
            if maximum > 1e-12:
                raise ValueError("nominal singleton control changed the common learner")
        else:
            maximum = None
        reports.append(
            dict(
                reveal_tick=reveal,
                scene_wise=old,
                observation_consistent=new,
                scene_wise_proxy_passages=sum(r["branch_proxy_passed"] for r in old),
                observation_consistent_proxy_passages=sum(r["branch_proxy_passed"] for r in new),
                mutually_successful_contexts=len(differences),
                mean_paired_proxy_time_difference_s=(
                    float(np.mean(differences)) if differences else None
                ),
                nominal_coefficient_maximum_difference=maximum,
                model=artifact(out / f"observation_consistent_{label}.npz"),
                baseline=artifact(out / f"scene_wise_{label}.npz"),
            )
        )
    return write_new(
        out / "result.json", dict(experiment=experiment, conditions=reports, new_physics_steps=0)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.out)))
