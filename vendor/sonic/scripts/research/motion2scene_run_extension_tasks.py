#!/usr/bin/env python3
"""Execute every fixed extension capability task, retaining each charged attempt."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

import motion2scene_collect_timed_schedules as collection  # noqa: E402
from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_extension_coverage import run as summarize_panel  # noqa: E402
from motion2scene_extension_tasks import fixed_tasks  # noqa: E402
from motion2scene_storage import deduplicate_inventories  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)


def verify_panel(panel):
    prepared = read_checked(artifact(panel / "prepared.json"))
    study = read_checked(prepared["study"])
    tasks = dict(fixed_tasks(read_checked(study["plan"])))
    if (
        len(prepared["tasks"]) != 12
        or {t["task_id"] for t in prepared["tasks"]} != set(tasks)
        or study["assigned_episodes"] != 204
        or study["charged_maximum_physics_steps"] != 243168
        or study["physics_seed"] != 97001
    ):
        raise ValueError("the complete fixed extension task allocation is required")
    for task in prepared["tasks"]:
        manifest = read_checked(task["collection"])
        scene = read_checked(task["scene"])
        if (
            manifest["registry"] != study["registry"]
            or manifest["scene_definition"] != task["scene"]
            or scene["scene_id"] != task["task_id"]
            or scene["split"] != "development"
            or scene["beams"] != [tasks[task["task_id"]]]
            or scene["beam_collision_enabled"] != [True]
            or manifest["expected_physics_steps"] != 1192
            or len(manifest["cells"]) != 17
            or any(
                c["runtime_seed"] != 97001 or c["timed_schedule_mode"] != "forced"
                for c in manifest["cells"]
            )
        ):
            raise ValueError("fixed task geometry, bank or physical allocation differs")
    return prepared


def execute_cell(cell, common, bank, scene, launch):
    folder = Path(cell["output"])
    attempt_path = folder / "attempt.json"
    if attempt_path.exists():
        previous = json.loads(attempt_path.read_text())
        if previous["command"] != cell["command"]:
            raise ValueError("retained attempt differs from its assignment")
        return previous
    if folder.exists() or launch.exists():
        raise RuntimeError("unfinished attempt retained; no automatic retry")
    collection.validate_collection_context(cell, common, bank, scene)
    if collection.free_gpu_mib() < common["limits"]["minimum_free_gpu_mib"]:
        raise RuntimeError("resource wait before launch; no attempt charged")
    write_new(launch, dict(command=cell["command"], charged_maximum_physics_steps=1192))
    started, interrupted = time.monotonic(), None
    try:
        status = collection.run_with_process_group(
            cell["command"], timeout=common["limits"]["timeout_s"]
        )
    except subprocess.TimeoutExpired:
        status = 124
    except BaseException as error:
        status, interrupted = -1, error
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(
        command=cell["command"], exit_status=status, wall_seconds=time.monotonic() - started
    )
    write_new(attempt_path, result)
    if interrupted is not None:
        raise interrupted
    return result


def run(panel):
    prepared = verify_panel(panel)
    implementation = panel / "capability_execution.json"
    if implementation.exists():
        declaration = read_checked(artifact(implementation))
        if declaration["prepared"] != artifact(panel / "prepared.json"):
            raise ValueError("execution declaration belongs to another panel")
        for source in declaration["implementation"]:
            checked(Path(source["path"]), source["sha256"])
    else:
        sources = [
            Path(__file__),
            ROOT / "scripts/research/motion2scene_storage.py",
            ROOT / "scripts/research/motion2scene_extension_coverage.py",
            ROOT / "scripts/research/motion2scene_extension_tasks.py",
        ]
        write_new(
            implementation,
            dict(
                prepared=artifact(panel / "prepared.json"),
                implementation=[artifact(p) for p in sources],
                failure_policy="one charged attempt per assignment; retain partial failures and unknowns",
            ),
        )
    for task in prepared["tasks"]:
        out = Path(task["collection"]["path"]).parent
        if (out / "result.json").exists():
            completed = read_checked(artifact(out / "result.json"))
            if completed["manifest"] != task["collection"]:
                raise ValueError("existing result belongs to another task")
            if any(r["outcome"]["task_outcome"] == "unknown" for r in completed["rows"]):
                raise RuntimeError(
                    "unknown task outcome retained; inspect before continuing the panel"
                )
            continue
        if shutil.disk_usage(panel).free < 30 * 1024**3:
            raise RuntimeError("resource wait before launch; fewer than 30 GiB free")
        common, bank, scene = collection.verify_manifest(out, execution=True)
        for cell in common["cells"]:
            attempt = execute_cell(
                cell, common, bank, scene, out / (cell["cell_id"] + "_launch.json")
            )
            print(
                json.dumps(
                    dict(
                        task=task["task_id"],
                        cell=cell["cell_id"],
                        exit_status=attempt["exit_status"],
                    )
                ),
                flush=True,
            )
            if attempt["exit_status"] != 0:
                partial = collection.analyze_incomplete(
                    cell, common, bank, scene, attempt, "nonzero_process_exit"
                )
                if partial["outcome"]["task_outcome"] == "unknown":
                    raise RuntimeError(
                        "unknown partial outcome retained; inspect before further physical execution"
                    )
        collection.analyze(out)
        write_new(out / "storage.json", deduplicate_inventories([out]))
        report = read_checked(artifact(out / "result.json"))
        if any(r["outcome"]["task_outcome"] == "unknown" for r in report["rows"]):
            raise RuntimeError("unknown task outcome retained; inspect before continuing the panel")
        print(
            json.dumps(
                dict(
                    completed_task=task["task_id"],
                    passing_schedules=sum(r["pass"] for r in report["rows"]),
                )
            ),
            flush=True,
        )
    output = panel / "coverage.json"
    if output.exists():
        ref = artifact(output)
        if read_checked(ref)["prepared"] != artifact(panel / "prepared.json"):
            raise ValueError("coverage result belongs to another panel")
        return ref
    return summarize_panel(panel, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.panel)))
