#!/usr/bin/env python3
"""Materialize the pre-generation fixed task panel for the qualified extensions."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

import motion2scene_collect_timed_schedules as collection  # noqa: E402
from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_course import (  # noqa: E402
    author_course,
    validate_beams,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)


def fixed_tasks(plan):
    tasks = plan["specification"]["independent_capability_tasks"]
    if len(tasks) != 12 or len({t["task_id"] for t in tasks}) != 12:
        raise ValueError("all twelve pre-generation tasks required")
    beams = [{k: v for k, v in task.items() if k != "task_id"} for task in tasks]
    validate_beams(beams)
    keys = [(b["center_xy_m"][0], b["length_m"], b["underside_m"]) for b in beams]
    expected = [
        (x, length, z) for x in (1.7, 2.25, 2.8) for length in (0.1, 0.75) for z in (1.24, 1.30)
    ]
    if Counter(keys) != Counter(expected) or any(
        b["center_xy_m"][1] != -0.1
        or b["yaw_rad"] != 0
        or b["width_m"] != 1.2
        or b["thickness_m"] != 0.1
        for b in beams
    ):
        raise ValueError("fixed geometry panel changed")
    return [(task["task_id"], beam) for task, beam in zip(tasks, beams, strict=True)]


def prepare(study, qualification, template, out):
    plan_ref, result_ref = artifact(study / "plan.json"), artifact(
        qualification / "environment_result.json"
    )
    plan, qualified = read_checked(plan_ref), read_checked(result_ref)
    manifest = read_checked(qualified["manifest"])
    if manifest["study"] != plan_ref or not all(r["qualified"] for r in qualified["rows"]):
        raise ValueError("this extension panel requires the complete qualified candidate bank")
    bank_ref = qualified["registry"]
    bank = load_verified_registry(bank_ref["path"], bank_ref["sha256"])
    if len(bank.option_ids) != 17:
        raise ValueError("shared neutral plus all sixteen assigned schedules required")
    room_ref = manifest["cells"][0]["scene"]
    room = checked(Path(room_ref["path"]), room_ref["sha256"]).read_text()
    tasks = fixed_tasks(plan)
    out.mkdir(parents=True, exist_ok=False)
    declaration = write_new(
        out / "study.json",
        dict(
            plan=plan_ref,
            qualification=result_ref,
            registry=bank_ref,
            implementation=artifact(Path(__file__)),
            template=artifact(template),
            assigned_tasks=len(tasks),
            assigned_schedules_per_task=17,
            assigned_episodes=204,
            charged_maximum_physics_steps=243168,
            physics_seed=plan["specification"]["task_physics_seed"],
            policy="all qualified candidates retained; no scene screening or outcome-dependent replacement",
            comparison=(
                "generated bank plus neutral versus authored bank plus the same neutral; "
                "full bank loaded in every episode"
            ),
            scope=(
                "fixed procedure-development tasks declared before candidate generation; "
                "not reserved primary evaluation"
            ),
            metrics=[
                "incremental task coverage over neutral",
                "per-candidate task coverage",
                "time on mutually solvable tasks",
            ],
        ),
    )
    prepared = []
    for task_id, beam in tasks:
        scene_dir = out / "scenes"
        scene_dir.mkdir(exist_ok=True)
        path = scene_dir / (task_id + ".usda")
        path.write_text(author_course(room, [beam], course_id=task_id))
        definition = write_new(
            scene_dir / (task_id + ".json"),
            dict(
                schema="motion2scene_timed_schedule_scene_v1",
                split="development",
                scene_id=task_id,
                scene=artifact(path),
                beams=[beam],
                beam_collision_enabled=[True],
                provenance=dict(fixed_pre_generation_plan=plan_ref, task_id=task_id),
            ),
        )
        target = out / "collections" / task_id
        collection.prepare(
            SimpleNamespace(
                registry=Path(bank_ref["path"]),
                request=Path(manifest["request"]["path"]),
                scene_definition=Path(definition["path"]),
                template=template,
                cell="neutral",
                policy_mode="forced",
                preferred_option_id="neutral",
                preferred_reference_id="generated_00",
                policy=None,
                seed=plan["specification"]["task_physics_seed"],
                out=target,
            )
        )
        collection.verify_manifest(target, execution=True)
        prepared.append(
            dict(task_id=task_id, scene=definition, collection=artifact(target / "manifest.json"))
        )
    return write_new(
        out / "prepared.json", dict(study=declaration, tasks=prepared, physics_executed=0)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.study, args.qualification, args.template, args.out)))
