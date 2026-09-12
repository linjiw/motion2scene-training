#!/usr/bin/env python3
"""Prepare and execute eligible assignments from the unchanged M8 development panel."""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
import motion2scene_development_checkpoint_panel as panel_api  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)


def write_once(path, value):
    if path.exists():
        ref = artifact(path)
        if read_checked(ref) != value:
            raise ValueError("existing preparation or completion differs; retained for inspection")
        return ref
    return write_new(path, value)


def declare(panel):
    study_ref = artifact(panel / "study.json")
    study = read_checked(study_ref)
    checked(Path(study["implementation"]["path"]), study["implementation"]["sha256"])
    plan = read_checked(study["expanded_plan"])
    bank = panel_api.load_verified_registry(study["registry"]["path"], study["registry"]["sha256"])
    scenes = {
        r["scene_id"]: dict(scene_id=r["scene_id"], scene_definition=r["scene_definition"])
        for r in study["assignments"]
    }
    expected = panel_api.assignments(plan["runs"], list(scenes.values()), bank.option_ids)
    if study["assignments"] != expected or study["assigned_episodes"] != 138:
        raise ValueError("the unchanged complete 138-assignment development panel is required")
    if (panel / "prepared.json").exists() or any((panel / "episodes").glob("*/result.json")):
        raise ValueError("declare readiness scheduling before panel preparation or outcomes")
    sources = [
        Path(__file__),
        Path(panel_api.__file__),
        Path(panel_api.collection.__file__),
        ROOT / "scripts/research/motion2scene_run_extension_tasks.py",
    ]
    return write_once(
        panel / "readiness.json",
        dict(
            schema="motion2scene_development_readiness_v1",
            study=study_ref,
            implementation=[artifact(p) for p in sources],
            scheduling="original manifest order restricted to currently eligible assignments",
            reason="execute the 48 assigned comparators before all learned M8 models exist",
            assigned_episodes=138,
            fixed_comparator_assignments=48,
            unchanged="all scene/seed/model/script/schedule assignments, collector and scoring",
            completion="only the full original panel can produce the final panel result",
            failure_policy="one charged attempt per assignment; retain failures and unknowns",
        ),
    )


def verify(panel):
    ref = artifact(panel / "readiness.json")
    declaration = read_checked(ref)
    if declaration["study"] != artifact(panel / "study.json"):
        raise ValueError("frozen development study changed")
    for source in declaration["implementation"]:
        checked(Path(source["path"]), source["sha256"])
    return ref, read_checked(declaration["study"])


def available_models(study):
    plan = read_checked(study["expanded_plan"])
    models = {}
    for model in study["models"]:
        boundary = (
            Path(plan["execution_root"])
            / model["run_id"]
            / "controller"
            / f"complete_{study['checkpoint']:03d}.json"
        )
        if not boundary.is_file():
            continue
        complete = read_checked(artifact(boundary))
        if complete["expanded_plan"] != study["expanded_plan"] or complete[
            "training_result"
        ] != artifact(Path(model["expected_training_result"])):
            raise ValueError("completed model boundary differs from its assigned checkpoint")
        models.update(panel_api.bind_models(dict(study, models=[model])))
    return models


def eligible_assignments(study, models, comparators_only=False):
    return [
        r
        for r in study["assignments"]
        if r["mode"] != "learned" or (not comparators_only and r["run_id"] in models)
    ]


def prepare_one(panel, declaration, study, row, models):
    target = panel / "episodes" / row["assignment_id"]
    binding = {row["run_id"]: models[row["run_id"]]} if row["mode"] == "learned" else {}
    policy = binding[row["run_id"]]["policy"] if binding else None
    if not target.exists():
        panel_api.collection.prepare(
            SimpleNamespace(
                registry=Path(study["registry"]["path"]),
                request=Path(study["request"]["path"]),
                template=Path(study["template"]["path"]),
                cell="neutral",
                scene_definition=Path(row["scene_definition"]["path"]),
                policy_mode=row["mode"],
                policy=None if policy is None else Path(policy["path"]),
                script_parameters=(
                    Path(study["script"]["path"]) if row["mode"] == "scripted_multi" else None
                ),
                forced_option_ids=[row["option_id"]] if row["mode"] == "forced" else None,
                preferred_option_id="neutral",
                preferred_reference_id="sustained",
                seed=panel_api.PHYSICS_SEED,
                out=target,
            )
        )
    if not (target / "manifest.json").is_file():
        raise ValueError("partial preparation retained; complete assigned manifest required")
    collection_ref = artifact(target / "manifest.json")
    common, _, _ = panel_api.collection.verify_manifest(target)
    panel_api.check_assignment(row, common, study, binding)
    receipt = dict(readiness=declaration, assignment=row, models=binding, collection=collection_ref)
    write_once(panel / "ready" / f"{row['assignment_id']}.json", receipt)
    return dict(row, collection=collection_ref)


def prepare_available(panel, comparators_only=False):
    declaration, study = verify(panel)
    models = {} if comparators_only else available_models(study)
    (panel / "ready").mkdir(exist_ok=True)
    rows = [
        prepare_one(panel, declaration, study, row, models)
        for row in eligible_assignments(study, models, comparators_only)
    ]
    return study, models, rows


def completed_result(row):
    path = Path(row["collection"]["path"]).parent / "result.json"
    if not path.exists():
        return None
    ref = artifact(path)
    result = read_checked(ref)
    if result["manifest"] != row["collection"] or len(result["rows"]) != 1:
        raise ValueError("physical result belongs to another assignment")
    outcome = result["rows"][0]["outcome"]["task_outcome"]
    if outcome == "unknown":
        raise RuntimeError("unknown assigned outcome retained; no automatic retry")
    if outcome not in ("pass", "failure"):
        raise ValueError("explicit physical outcome required")
    return dict(assignment_id=row["assignment_id"], result=ref)


def run_available(panel, comparators_only=False):
    study, models, rows = prepare_available(panel, comparators_only)
    results = []
    for row in rows:
        out = Path(row["collection"]["path"]).parent
        complete = completed_result(row)
        if complete is None:
            while not panel_api.resource_ready(panel):
                print(
                    json.dumps(
                        dict(status="waiting_for_native_capacity", assignment=row["assignment_id"])
                    ),
                    flush=True,
                )
                time.sleep(30)
            common, bank, scene = panel_api.collection.verify_manifest(out, execution=True)
            cell = panel_api.check_assignment(row, common, study, models)
            panel_api.execute_cell(cell, common, bank, scene, out / "launch.json")
            panel_api.collection.analyze(out)
            write_once(out / "storage.json", panel_api.deduplicate_inventories([out]))
            complete = completed_result(row)
        results.append(complete)
        print(
            json.dumps(
                dict(
                    status="assigned_development_episode_complete", assignment=row["assignment_id"]
                )
            ),
            flush=True,
        )
    value = dict(
        readiness=artifact(panel / "readiness.json"),
        assignments=results,
        complete_panel=False,
        assigned_panel_episodes=138,
    )
    if comparators_only:
        if len(results) != 48:
            raise ValueError("all 48 fixed-comparator assignments are required")
        return write_once(panel / "comparators_complete.json", value)
    return value


def finalize(panel):
    study, models, rows = prepare_available(panel)
    if len(models) != len(study["models"]) or len(rows) != len(study["assignments"]):
        raise ValueError("full M8 model inventory required before finalizing the original panel")
    if models != panel_api.bind_models(study):
        raise ValueError("ready models differ from the original full-panel bindings")
    return write_once(
        panel / "prepared.json",
        dict(study=artifact(panel / "study.json"), models=models, assignments=rows),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "prepare", "run", "finalize"))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--comparators-only", action="store_true")
    args = parser.parse_args()
    if args.action == "declare":
        result = declare(args.panel)
    elif args.action == "prepare":
        _, models, rows = prepare_available(args.panel, args.comparators_only)
        result = dict(
            eligible_models=len(models), prepared_assignments=len(rows), complete_panel=False
        )
    elif args.action == "run":
        result = run_available(args.panel, args.comparators_only)
    else:
        result = finalize(args.panel)
    print(json.dumps(result))
