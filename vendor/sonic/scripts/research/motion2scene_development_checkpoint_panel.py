#!/usr/bin/env python3
"""Matched M8 policy and finite-bank executions on fixed development contexts."""

import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_checkpoint_controls import checkpoint_slots  # noqa: E402
import motion2scene_collect_timed_schedules as collection  # noqa: E402
from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_expanded_acquisition import validate_training_prefix  # noqa: E402
from motion2scene_run_extension_tasks import execute_cell  # noqa: E402
from motion2scene_storage import deduplicate_inventories  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    load_schedule_policy,
    load_script_parameters,
)

BUDGET = 8
PHYSICS_SEED = 8732
ORDER_SEED = 202609091734
CONTEXTS = {
    "long_neutral_empty",
    "timed_development_0187",
    "timed_development_0247",
    "early_prior_development_0284",
    "early_prior_two_beam_development_v1",
    "complementary_late_development_0100",
}
ARMS = {
    "uniform",
    "target_only",
    "analytic_contrast",
    "observation_curriculum",
    "reference_contrast",
}


def assignments(runs, scenes, option_ids):
    identities = {(r["seed"], r["arm"]) for r in runs}
    if len(runs) != 15 or identities != {(s, a) for s in (93201, 93202, 93203) for a in ARMS}:
        raise ValueError("all five construction arms and three seeds are required")
    if len(scenes) != 6 or {s["scene_id"] for s in scenes} != CONTEXTS:
        raise ValueError("all six fixed development contexts are required")
    if len(option_ids) != 7 or len(set(option_ids)) != 7 or option_ids[0] != "neutral":
        raise ValueError("the full seven-schedule repertoire is required")
    policies = [
        dict(policy_id=r["run_id"], mode="learned", run_id=r["run_id"])
        for r in sorted(runs, key=lambda x: x["run_id"])
    ]
    policies += [dict(policy_id="script", mode="scripted_multi")]
    policies += [
        dict(policy_id="fixed_" + option, mode="forced", option_id=option) for option in option_ids
    ]
    rows = [
        dict(
            policy,
            scene_id=scene["scene_id"],
            scene_definition=scene["scene_definition"],
            physics_seed=PHYSICS_SEED,
        )
        for policy in policies
        for scene in sorted(scenes, key=lambda x: x["scene_id"])
    ]
    random.Random(ORDER_SEED).shuffle(rows)
    return [dict(row, assignment_id=f"episode_{i:03d}") for i, row in enumerate(rows)]


def declare(plan_path, development, script_path, out):
    plan_ref, source_ref = artifact(plan_path), artifact(development / "result.json")
    plan, source = read_checked(plan_ref), read_checked(source_ref)
    if plan["schema"] != "motion2scene_expanded_acquisition_v1":
        raise ValueError("the expanded acquisition plan is required")
    bank_ref = plan["registry"]
    bank = load_verified_registry(bank_ref["path"], bank_ref["sha256"])
    script_ref = artifact(script_path)
    load_script_parameters(script_ref["path"], script_ref["sha256"], bank)
    scenes, common = [], None
    for group in read_checked(source["teachers"]):
        manifest = read_checked(read_checked(group["collection"])["manifest"])
        if manifest["split"] != "development" or manifest["registry"] != bank_ref:
            raise ValueError("only existing development contexts under the same bank may be used")
        this = {k: manifest[k] for k in ("request", "template")}
        if common is not None and this != common:
            raise ValueError("common qualified template and request required")
        common = this
        scenes.append(
            dict(scene_id=group["scene_id"], scene_definition=manifest["scene_definition"])
        )
    rows = assignments(plan["runs"], scenes, bank.option_ids)
    models = [
        dict(
            run_id=r["run_id"],
            arm=r["arm"],
            seed=r["seed"],
            expected_training_result=str(checkpoint_slots(plan, r["run_id"], BUDGET)[-1]["model"]),
        )
        for r in plan["runs"]
    ]
    out.mkdir(parents=True, exist_ok=False)
    return write_new(
        out / "study.json",
        dict(
            expanded_plan=plan_ref,
            development_source=source_ref,
            registry=bank_ref,
            **common,
            script=script_ref,
            models=models,
            assignments=rows,
            checkpoint=BUDGET,
            assigned_episodes=len(rows),
            maximum_physics_steps=1192 * len(rows),
            implementation=artifact(Path(__file__)),
            scope="fixed development-policy comparison; no reserved layouts or outcomes",
            comparison="15 M8 acquired policies, unchanged strong script, all seven fixed schedules",
            matching="same six geometries, physics seed, bank, ideal observations and physical scorer",
            selection="all assigned models; no checkpoint or task selection using this panel",
            metrics=[
                "passages over all assignments",
                "matched finite-bank capability",
                "selection failures",
                "time on mutually successful matched tasks",
            ],
            continuation="after M8 acquisition, before the independent extension-capability study",
        ),
    )


def bind_models(study):
    plan = read_checked(study["expanded_plan"])
    bank = load_verified_registry(study["registry"]["path"], study["registry"]["sha256"])
    bindings = {}
    for model in study["models"]:
        ref = artifact(Path(model["expected_training_result"]))
        result = read_checked(ref)
        slots = checkpoint_slots(plan, model["run_id"], BUDGET)
        teachers = [artifact(slot["teacher"]) for slot in slots]
        if result["status"] != "complete" or result["audit_errors"]:
            raise ValueError("complete acquired checkpoint required before physical evaluation")
        validate_training_prefix(read_checked(result["registration"]), teachers, study["registry"])
        if [g["collection"] for g in read_checked(result["teachers"])] != teachers:
            raise ValueError("checkpoint does not contain the exact assigned M8 teacher prefix")
        if (
            Path(result["policy"]["path"]).resolve()
            != Path(ref["path"]).with_name("policy.npz").resolve()
        ):
            raise ValueError("checkpoint's policy path differs from its acquisition assignment")
        load_schedule_policy(result["policy"]["path"], result["policy"]["sha256"], bank)
        bindings[model["run_id"]] = dict(training_result=ref, policy=result["policy"])
    return bindings


def prepare(panel):
    study_ref = artifact(panel / "study.json")
    study = read_checked(study_ref)
    checked(Path(study["implementation"]["path"]), study["implementation"]["sha256"])
    models = bind_models(study)
    prepared = []
    for row in study["assignments"]:
        target = panel / "episodes" / row["assignment_id"]
        policy = models[row["run_id"]]["policy"] if row["mode"] == "learned" else None
        args = SimpleNamespace(
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
            seed=PHYSICS_SEED,
            out=target,
        )
        if target.exists():
            raise ValueError("partial preparation retained; inspect before continuing")
        collection.prepare(args)
        prepared.append(dict(**row, collection=artifact(target / "manifest.json")))
    return write_new(
        panel / "prepared.json", dict(study=study_ref, models=models, assignments=prepared)
    )


def check_assignment(row, common, study, models):
    if (
        common["split"] != "development"
        or common["registry"] != study["registry"]
        or common["scene_definition"] != row["scene_definition"]
        or len(common["cells"]) != 1
        or common["expected_physics_steps"] != 1192
    ):
        raise ValueError("physical evaluation assignment changed")
    cell = common["cells"][0]
    if cell["runtime_seed"] != PHYSICS_SEED or cell["timed_schedule_mode"] != row["mode"]:
        raise ValueError("policy mode or paired execution seed differs")
    expected = models[row["run_id"]]["policy"] if row["mode"] == "learned" else None
    if common["policy"] != expected:
        raise ValueError("acquired policy differs from the frozen checkpoint")
    script = study["script"] if row["mode"] == "scripted_multi" else None
    if common["script_parameters"] != script:
        raise ValueError("strong script parameters changed")
    if row["mode"] == "forced" and cell["forced_option_id"] != row["option_id"]:
        raise ValueError("fixed reference schedule changed")
    return cell


def resource_ready(panel):
    drivers = subprocess.run(
        ["pgrep", "-f", "python.*gear_sonic/eval_agent_trl.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    if drivers.returncode not in (0, 1):
        raise RuntimeError("cannot determine native simulator occupancy")
    return (
        not drivers.stdout.strip()
        and collection.free_gpu_mib() >= 7500
        and shutil.disk_usage(panel).free >= 30 * 1024**3
    )


def run(panel):
    prepared = read_checked(artifact(panel / "prepared.json"))
    study = read_checked(prepared["study"])
    checked(Path(study["implementation"]["path"]), study["implementation"]["sha256"])
    if prepared["models"] != bind_models(study):
        raise ValueError("prepared policies differ from the acquired M8 checkpoints")
    if [
        dict((k, v) for k, v in row.items() if k != "collection") for row in prepared["assignments"]
    ] != study["assignments"]:
        raise ValueError("prepared assignments differ from the fixed full panel")
    results = []
    for row in prepared["assignments"]:
        out = Path(row["collection"]["path"]).parent
        check_assignment(row, read_checked(row["collection"]), study, prepared["models"])
        if not (out / "result.json").exists():
            while not resource_ready(panel):
                print(
                    json.dumps(
                        dict(status="waiting_for_native_capacity", assignment=row["assignment_id"])
                    ),
                    flush=True,
                )
                time.sleep(45)
            common, bank, scene = collection.verify_manifest(out, execution=True)
            cell = check_assignment(row, common, study, prepared["models"])
            execute_cell(cell, common, bank, scene, out / "launch.json")
            collection.analyze(out)
            write_new(out / "storage.json", deduplicate_inventories([out]))
        ref = artifact(out / "result.json")
        result = read_checked(ref)
        if result["manifest"] != row["collection"] or len(result["rows"]) != 1:
            raise ValueError("physical result belongs to another assignment")
        outcome = result["rows"][0]["outcome"]["task_outcome"]
        results.append(dict(assignment_id=row["assignment_id"], result=ref))
        print(
            json.dumps(
                dict(
                    status="development_episode_complete",
                    assignment=row["assignment_id"],
                    outcome=outcome,
                )
            ),
            flush=True,
        )
        if outcome == "unknown":
            raise RuntimeError("unknown assigned outcome retained; no automatic retry")
    return write_new(
        panel / "result.json", dict(prepared=artifact(panel / "prepared.json"), assignments=results)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "prepare", "run"))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--script", type=Path)
    args = parser.parse_args()
    if args.action == "declare":
        result = declare(args.plan, args.development, args.script, args.panel)
    else:
        result = {"prepare": prepare, "run": run}[args.action](args.panel)
    print(json.dumps(result))
