#!/usr/bin/env python3
"""Serial matched execution of all six equivalent-teaching models on held centers."""

import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_development_checkpoint_panel import resource_ready  # noqa: E402
import motion2scene_extension_episode as episode  # noqa: E402
from motion2scene_extension_policy import load_extension_policy  # noqa: E402
from motion2scene_fit_extension_teaching import validate_design  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)

ORDER_SEED = 202609091850


def assignments(fitting, study, prepared):
    tasks = {t["task_id"]: t for t in prepared["tasks"]}
    if (
        len(tasks) != 12
        or len(study["folds"]) != 3
        or study["arms"] != ["generated", "authored"]
        or study["policy_evaluation_seed"] != 97001
    ):
        raise ValueError("three complete folds, both arms and the common physical seed required")
    rows = []
    seen = set()
    for i, fold in enumerate(study["folds"]):
        train, held = fold["training_task_ids"], fold["evaluation_task_ids"]
        if (
            len(train) != 8
            or len(set(train)) != 8
            or len(held) != 4
            or len(set(held)) != 4
            or set(train) & set(held)
            or set(train) | set(held) != set(tasks)
        ):
            raise ValueError("each fold must partition the twelve fixed tasks")
        for arm in study["arms"]:
            for task_id in held:
                if (arm, task_id) in seen:
                    raise ValueError("each arm/task must have exactly one held-center prediction")
                seen.add((arm, task_id))
                rows.append(
                    dict(
                        arm=arm,
                        fold_index=i,
                        task_id=task_id,
                        physics_seed=97001,
                        expected_model_result=str(fitting / f"fold_{i}_{arm}" / "result.json"),
                        nominal_template=tasks[task_id]["collection"],
                    )
                )
    random.Random(ORDER_SEED).shuffle(rows)
    return [dict(row, assignment_id=f"episode_{i:03d}") for i, row in enumerate(rows)]


def declare(fitting, out):
    fitting = fitting.resolve()
    study_ref = artifact(fitting / "study.json")
    study = read_checked(study_ref)
    prepared = read_checked(study["prepared"])
    capability_study = read_checked(prepared["study"])
    proposal = read_checked(capability_study["plan"])
    validate_design(
        study,
        proposal["specification"]["independent_capability_tasks"],
        capability_study["registry"],
    )
    rows = assignments(fitting, study, prepared)
    out.mkdir(parents=True, exist_ok=False)
    sources = []
    seeds = [Path(__file__), ROOT / "scripts/research/motion2scene_extension_execution.py"]
    for path in sorted(closure(seeds)):
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources.append(dict(**artifact(path), snapshot=artifact(target)))
    return write_new(
        out / "study.json",
        dict(
            schema="motion2scene_extension_policy_panel_v1",
            fitting_study=study_ref,
            registry=study["registry"],
            capability_prepared=study["prepared"],
            assignments=rows,
            assigned_episodes=24,
            maximum_physics_steps=28608,
            order_seed=ORDER_SEED,
            source_closure=sources,
            physics_executed=0,
            comparison=(
                "both arms on every held-center task; training-selected constants and bank "
                "references use matched fixed-task executions"
            ),
        ),
    )


def verify_study(panel):
    ref = artifact(panel / "study.json")
    study = read_checked(ref)
    fitting = read_checked(study["fitting_study"])
    prepared = read_checked(study["capability_prepared"])
    expected = assignments(Path(study["fitting_study"]["path"]).parent, fitting, prepared)
    if (
        study["schema"] != "motion2scene_extension_policy_panel_v1"
        or study["registry"] != fitting["registry"]
        or study["capability_prepared"] != fitting["prepared"]
        or study["assignments"] != expected
        or study["assigned_episodes"] != 24
        or study["maximum_physics_steps"] != 28608
        or study["order_seed"] != ORDER_SEED
    ):
        raise ValueError("complete matched policy allocation differs from the declaration")
    for source in study["source_closure"]:
        checked(Path(source["path"]), source["sha256"])
        checked(Path(source["snapshot"]["path"]), source["sha256"])
    return ref, study, fitting


def bind_models(study, fitting):
    bank = load_verified_registry(study["registry"]["path"], study["registry"]["sha256"])
    root = Path(study["fitting_study"]["path"]).parent
    completed = read_checked(artifact(root / "result.json"))
    expected = {r["expected_model_result"] for r in study["assignments"]}
    if (
        completed["study"] != study["fitting_study"]
        or len(completed["models"]) != 6
        or {r["path"] for r in completed["models"]} != expected
    ):
        raise ValueError(
            "all six assigned models must be fitted before preparing physical episodes"
        )
    bindings = {}
    for ref in completed["models"]:
        _, result = load_extension_policy(ref["path"], ref["sha256"], bank)
        slots = [r for r in study["assignments"] if r["expected_model_result"] == ref["path"]]
        if (
            result["study"] != study["fitting_study"]
            or len(slots) != 4
            or any(
                result["arm"] != s["arm"]
                or result["analysis"]["fold"] != fitting["folds"][s["fold_index"]]
                for s in slots
            )
        ):
            raise ValueError("fitted model differs from its assigned arm and held-center fold")
        bindings[ref["path"]] = ref
    return bindings


def prepare(panel):
    ref, study, fitting = verify_study(panel)
    models = bind_models(study, fitting)
    rows = []
    for row in study["assignments"]:
        out = panel / "episodes" / row["assignment_id"]
        model = models[row["expected_model_result"]]
        manifest = episode.prepare(Path(row["nominal_template"]["path"]), Path(model["path"]), out)
        rows.append(dict(row, model_result=model, episode=manifest))
    return write_new(panel / "prepared.json", dict(study=ref, models=models, assignments=rows))


def check_prepared(prepared, study_ref, study, models):
    expected = [
        dict(row, model_result=models[row["expected_model_result"]]) for row in study["assignments"]
    ]
    rows = prepared["assignments"]
    if (
        prepared["study"] != study_ref
        or prepared["models"] != models
        or len(rows) != 24
        or [{k: v for k, v in r.items() if k != "episode"} for r in rows] != expected
    ):
        raise ValueError(
            "prepared episodes do not match every original assignment and fitted model"
        )


def run(panel):
    study_ref, study, fitting = verify_study(panel)
    prepared = read_checked(artifact(panel / "prepared.json"))
    models = bind_models(study, fitting)
    check_prepared(prepared, study_ref, study, models)
    results, steps, traces = [], 0, 0
    for row in prepared["assignments"]:
        out = Path(row["episode"]["path"]).parent
        checked(Path(row["episode"]["path"]), row["episode"]["sha256"])
        manifest, _, _, _ = episode.verify(out, execution=True)
        if (
            manifest["model_result"] != row["model_result"]
            or manifest["nominal_template"] != row["nominal_template"]
        ):
            raise ValueError("episode model or matched physical task changed")
        result_path = out / "result.json"
        if not result_path.exists():
            while not resource_ready(panel):
                print(
                    json.dumps(
                        dict(status="waiting_for_native_capacity", assignment=row["assignment_id"])
                    ),
                    flush=True,
                )
                time.sleep(30)
            episode.run(out)
        result_ref = artifact(result_path)
        result = read_checked(result_ref)
        if result["manifest"] != row["episode"]:
            raise ValueError("physical result belongs to a different assigned episode")
        outcome = result["row"]["outcome"]["task_outcome"]
        if outcome not in ("pass", "failure"):
            raise RuntimeError(
                "unresolved physical outcome retained; inspect before further episodes"
            )
        if outcome == "pass" and not result["complete_policy_trace_verified"]:
            raise ValueError("passing policy requires its complete verified decision trace")
        measured_steps = result["row"]["physics_steps"]
        if type(measured_steps) is not int or measured_steps < 1:
            raise ValueError("assigned physical episode lacks a measured step count")
        steps += measured_steps
        traces += result["complete_policy_trace_verified"]
        results.append(dict(assignment_id=row["assignment_id"], result=result_ref))
        print(
            json.dumps(
                dict(
                    completed=row["assignment_id"],
                    task=row["task_id"],
                    arm=row["arm"],
                    outcome=outcome,
                )
            ),
            flush=True,
        )
    return write_new(
        panel / "result.json",
        dict(
            study=study_ref,
            prepared=artifact(panel / "prepared.json"),
            results=results,
            assigned_episodes=24,
            measured_physics_steps=steps,
            complete_policy_traces=traces,
            status="all assigned physical outcomes measured; paired comparative analysis separate",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("declare", "prepare", "run"))
    parser.add_argument("--fitting", type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    args = parser.parse_args()
    if args.operation == "declare":
        if args.fitting is None:
            parser.error("declare requires --fitting")
        result = declare(args.fitting, args.panel)
    else:
        result = globals()[args.operation](args.panel)
    print(json.dumps(result))
