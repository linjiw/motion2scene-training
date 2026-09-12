#!/usr/bin/env python3
"""Register bounded, serial batches of complete timed teacher or policy episodes."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from hallucination.run_approved_manifest import free_gpu_mib  # noqa: E402
from motion2scene_collect_timed_schedules import (  # noqa: E402
    prepare as prepare_collection,
    run as run_collection,
    verify_manifest,
)
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)

SCHEMA = "motion2scene_timed_schedule_batch_v1"


def prepare(spec_path, out):
    spec = json.loads(spec_path.read_text())
    if spec["schema"] != SCHEMA or not spec["collections"]:
        raise ValueError("nonempty explicit timed batch specification required")
    for key in ("registry", "request", "template"):
        checked(Path(spec[key]["path"]), spec[key]["sha256"])
    bank = load_verified_registry(spec["registry"]["path"], spec["registry"]["sha256"])
    episodes = 0
    for item in spec["collections"]:
        ref = item["scene_definition"]
        checked(Path(ref["path"]), ref["sha256"])
        for key in ("policy", "script_parameters", "runtime_assets"):
            if item.get(key) is not None:
                ref = item[key]
                checked(Path(ref["path"]), ref["sha256"])
        subset = item.get("forced_option_ids")
        episodes += (
            (len(bank.option_ids) if subset is None else len(subset))
            if item["policy_mode"] == "forced"
            else 1
        )
    maximum_steps = episodes * 4 * (bank.frame_count - 1)
    budget = spec["maximum_physics_steps"]
    if type(budget) is not int or budget < 1 or maximum_steps > budget:
        raise ValueError("all declared branches must fit the explicit maximum physics budget")
    out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__)])):
        dest = out / "source_snapshot" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(dest)})
    write_new(
        out / "plan.json",
        dict(
            schema=SCHEMA,
            specification=artifact(spec_path),
            expected_episodes=episodes,
            planned_maximum_physics_steps=maximum_steps,
            budget_physics_steps=budget,
            implementation=sources,
            serial=True,
            scope=spec["scope"],
        ),
    )
    children = []
    for index, item in enumerate(spec["collections"]):
        folder = out / f"collection_{index:03d}"
        prepare_collection(
            SimpleNamespace(
                out=folder,
                registry=Path(spec["registry"]["path"]),
                request=Path(spec["request"]["path"]),
                template=Path(spec["template"]["path"]),
                cell="neutral",
                scene_definition=Path(item["scene_definition"]["path"]),
                policy_mode=item["policy_mode"],
                policy=Path(item["policy"]["path"]) if item.get("policy") else None,
                policy_id=item.get("policy_id"),
                script_parameters=(
                    Path(item["script_parameters"]["path"])
                    if item.get("script_parameters")
                    else None
                ),
                runtime_assets=(
                    Path(item["runtime_assets"]["path"]) if item.get("runtime_assets") else None
                ),
                seed=item["seed"],
                forced_option_ids=item.get("forced_option_ids"),
                preferred_option_id=item.get("preferred_option_id", "neutral"),
                preferred_reference_id=item.get("preferred_reference_id", "sustained"),
            )
        )
        children.append(dict(index=index, manifest=artifact(folder / "manifest.json")))
    write_new(
        out / "registration.json",
        dict(
            plan=artifact(out / "plan.json"),
            children=children,
            minimum_free_gpu_mib=7500,
            failure_policy="retain all physical/technical failures; never automatically repeat attempted cells",
        ),
    )
    print(json.dumps({"registered_collections": len(children), "episodes": episodes}), flush=True)


def _run_locked(out):
    if (out / "result.json").exists():
        raise ValueError("completed immutable batch must not be rerun")
    registration = json.loads((out / "registration.json").read_text())
    ref = registration["plan"]
    plan = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    for ref in plan["implementation"]:
        checked(Path(ref["path"]), ref["sha256"])
    ref = plan["specification"]
    checked(Path(ref["path"]), ref["sha256"])
    groups = []
    for child in registration["children"]:
        ref = child["manifest"]
        folder = checked(Path(ref["path"]), ref["sha256"]).parent
        manifest, _, _ = verify_manifest(folder, execution=not (folder / "result.json").exists())
        if not (folder / "result.json").exists():
            pending_physics = any(
                not (Path(cell["output"]) / "attempt.json").exists() for cell in manifest["cells"]
            )
            if pending_physics and free_gpu_mib() < registration["minimum_free_gpu_mib"]:
                print(json.dumps({"status": "resource_pause", "next": str(folder)}), flush=True)
                return
            run_collection(folder)
        result = json.loads((folder / "result.json").read_text())
        if (
            result["manifest"] != child["manifest"]
            or len(result["rows"]) != len(manifest["cells"])
            or {row["cell_id"] for row in result["rows"]}
            != {cell["cell_id"] for cell in manifest["cells"]}
        ):
            raise ValueError("result does not retain every assigned physical attempt")
        groups.append(
            dict(
                **child,
                result=artifact(folder / "result.json"),
                episodes=len(result["rows"]),
                physics_steps=result["physics_steps"],
                unmeasured_failed_attempts=result["unmeasured_failed_attempts"],
            )
        )
        print(
            json.dumps(
                {"completed_collections": len(groups), "total": len(registration["children"])}
            ),
            flush=True,
        )
    measured_steps = sum(group["physics_steps"] for group in groups)
    unknown = sum(group["unmeasured_failed_attempts"] for group in groups)
    if measured_steps > plan["budget_physics_steps"]:
        raise ValueError("measured integration steps exceed the declared acquisition budget")
    write_new(
        out / "result.json",
        dict(
            registration=artifact(out / "registration.json"),
            groups=groups,
            episodes=sum(group["episodes"] for group in groups),
            measured_physics_steps=measured_steps,
            unmeasured_failed_attempts=unknown,
            budget_accounting_complete=unknown == 0,
            scope=plan["scope"],
        ),
    )


def run(out):
    with (out / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another controller already owns this batch") from error
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        _run_locked(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--specification", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.specification is None:
            parser.error("prepare requires --specification")
        prepare(args.specification, args.out)
    else:
        run(args.out)
