#!/usr/bin/env python3
"""Read acquired models on other acquisition seeds' measured teacher branches."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_checkpoint_controls import load_checkpoint  # noqa: E402
from motion2scene_decision_study import (  # noqa: E402
    complete_schedules,
    evaluate,
    read_checked,
    ridge_choice,
)
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    definition_digest,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    schedule_layout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    load_schedule_policy,
)

VALIDATION_ARMS = ("uniform", "target_only", "analytic_contrast")


def geometry_key(scene):
    """Physical beam fields, independent of scene names and construction metadata."""
    fields = ("center_xy_m", "yaw_rad", "length_m", "width_m", "thickness_m", "underside_m")
    return definition_digest(
        [
            {k: beam[k] for k in fields}
            for beam, enabled in zip(scene["beams"], scene["beam_collision_enabled"], strict=True)
            if enabled
        ]
    )


def select_validation(candidates, training_seed, training_geometries):
    """Same seed-wise evaluation set for every construction arm; no outcome filters."""
    selected, omitted, seen = [], [], set()
    for row in candidates:
        reason = None
        if row["arm"] not in VALIDATION_ARMS:
            reason = "paired replay arm is not an independent scene source"
        elif row["seed"] == training_seed:
            reason = "same acquisition seed"
        elif row["geometry_key"] in training_geometries:
            reason = "geometry occurs in a training-arm prefix for this seed"
        elif (row["seed"], row["geometry_key"]) in seen:
            reason = "duplicate geometry and execution seed"
        if reason:
            omitted.append(dict(run_id=row["run_id"], round=row["round"], reason=reason))
        else:
            selected.append(row)
            seen.add((row["seed"], row["geometry_key"]))
    return selected, omitted


def paired_summary(rows, baseline):
    if len(rows) != len(baseline) or [r["scene_id"] for r in rows] != [
        r["scene_id"] for r in baseline
    ]:
        raise ValueError("identical ordered validation encounters required")
    differences = [
        a["branch_proxy_time_s"] - b["branch_proxy_time_s"]
        for a, b in zip(rows, baseline, strict=True)
        if a["branch_proxy_passed"] and b["branch_proxy_passed"]
    ]
    return dict(
        contexts=len(rows),
        passing_branch_proxies=sum(r["branch_proxy_passed"] for r in rows),
        constant_prior_passages=sum(r["branch_proxy_passed"] for r in baseline),
        mutually_successful_contexts=len(differences),
        mean_paired_proxy_time_difference_s=(
            sum(differences) / len(differences) if differences else None
        ),
    )


def run(plan_path, budget, out):
    if budget not in (1, 2, 4):
        raise ValueError("primary acquisition budget 1, 2 or 4 required")
    plan_ref = artifact(plan_path)
    plan = read_checked(plan_ref)
    loaded, candidates, geometry_by_seed = {}, [], defaultdict(set)
    for assignment in plan["runs"]:
        bank, groups, _, checkpoint = load_checkpoint(plan, assignment["run_id"], budget)
        loaded[assignment["run_id"]] = (groups, checkpoint)
        for index, group in enumerate(groups[1:], 1):
            manifest = read_checked(read_checked(group["collection"])["manifest"])
            scene = read_checked(manifest["scene_definition"])
            key = geometry_key(scene)
            geometry_by_seed[assignment["physics_seed"]].add(key)
            candidates.append(
                dict(
                    run_id=assignment["run_id"],
                    arm=assignment["arm"],
                    seed=assignment["physics_seed"],
                    round=index,
                    geometry_key=key,
                    teacher=group,
                    scene_definition=manifest["scene_definition"],
                )
            )
    _, _, entries = schedule_layout(bank)
    prior = [
        i
        for i, option in enumerate(bank.request["options"], 1)
        if option["reference_id"] == "prior_splice"
    ]
    if not prior:
        raise ValueError("qualified prior baseline required")
    constant = min(prior, key=lambda i: entries[i])
    out.mkdir(parents=True, exist_ok=False)
    experiment = write_new(
        out / "experiment.json",
        dict(
            plan=plan_ref,
            budget=budget,
            implementation=artifact(Path(__file__)),
            validation_source_arms=list(VALIDATION_ARMS),
            selection="other acquisition seeds, all assigned nonempty encounters through the same budget",
            exclusions=(
                "paired replay duplicate source, same seed, shared training geometry "
                "and duplicate validation geometry"
            ),
            matching="all four arms for a training seed share the same validation encounters",
            interpretation=(
                "exploratory cross-corpus recorded-branch readout; "
                "not new closed-loop or reserved evaluation"
            ),
            new_physics_steps=0,
        ),
    )
    reports = []
    for assignment in plan["runs"]:
        _, checkpoint = loaded[assignment["run_id"]]
        source = read_checked(checkpoint)
        ref = source["policy"]
        model = load_schedule_policy(ref["path"], ref["sha256"], bank)
        selected, omitted = select_validation(
            candidates, assignment["physics_seed"], geometry_by_seed[assignment["physics_seed"]]
        )
        if not selected:
            raise ValueError("no independent validation encounters remain")
        teachers = [r["teacher"] for r in selected]
        passed, _, admitted = complete_schedules(teachers, entries)
        if not admitted.all():
            raise ValueError("complete validation schedule outcomes required")
        outcomes = evaluate(lambda row: ridge_choice(model, row), teachers, entries)
        baseline = evaluate(
            lambda row: constant if row["legal_mask"][constant] else 0, teachers, entries
        )
        reports.append(
            dict(
                run_id=assignment["run_id"],
                arm=assignment["arm"],
                seed=assignment["physics_seed"],
                checkpoint=checkpoint,
                policy=ref,
                validation=[
                    {k: v for k, v in r.items() if k != "teacher"}
                    | {"collection": r["teacher"]["collection"]}
                    for r in selected
                ],
                omitted=omitted,
                bank_capability=int(passed.any(1).sum()),
                assessment=outcomes,
                constant_prior=baseline,
                **paired_summary(outcomes, baseline),
            )
        )
    return write_new(
        out / "result.json", dict(experiment=experiment, corpora=reports, new_physics_steps=0)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.budget, args.out)))
