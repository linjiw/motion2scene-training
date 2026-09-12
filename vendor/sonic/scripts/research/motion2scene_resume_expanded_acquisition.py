#!/usr/bin/env python3
"""Resume the frozen expansion with a declared string-to-Path bookkeeping repair.

The original driver passes JSON ``inherited_model`` strings to a Path-only
artifact reader. Normalize those arguments in this module's import of the
driver; preserve the original on-disk driver, plan, native runtime and checks.
Every activation requires a bound repair declaration in addition to the plan.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

import motion2scene_expanded_acquisition as expansion  # noqa: E402
import motion2scene_timing_diagnostic as artifacts  # noqa: E402

REPAIR = "normalize_inherited_model_string_to_Path_v1"


@contextmanager
def normalized_driver_artifacts():
    original = expansion.artifact
    if original is not artifacts.artifact:
        raise ValueError("unexpected existing expanded-driver artifact override")

    def normalized(path):
        return original(Path(path) if isinstance(path, str) else path)

    expansion.artifact = normalized
    try:
        yield
    finally:
        expansion.artifact = original


def validate_repair(repair_path, plan_path, adoption_path):
    declaration = json.loads(repair_path.read_text())
    expected = dict(
        repair=REPAIR,
        plan=artifacts.artifact(plan_path),
        adoption=artifacts.artifact(adoption_path),
        adapter=artifacts.artifact(Path(__file__)),
        original_driver=artifacts.artifact(Path(expansion.__file__)),
        original_artifact_reader=artifacts.artifact(Path(artifacts.__file__)),
    )
    for key, value in expected.items():
        if declaration.get(key) != value:
            raise ValueError(f"repair declaration differs in {key}")
    plan = json.loads(plan_path.read_text())
    if expected["original_driver"] not in plan["implementation"]:
        raise ValueError("repair requires the exact original plan-bound driver")
    for source in plan["implementation"]:
        artifacts.checked(Path(source["path"]), source["sha256"])
    return artifacts.artifact(repair_path)


def resume(plan_path, adoption_path, stop, repair_path, *, watch=False):
    repair = validate_repair(repair_path, plan_path, adoption_path)
    print(json.dumps(dict(active_bookkeeping_repair=repair)), flush=True)
    with normalized_driver_artifacts():
        runner = expansion.watch if watch else expansion.run
        return runner(plan_path, adoption_path, stop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--until-budget", type=int, choices=(8, 16, 32), default=32)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    result = resume(args.plan, args.adoption, args.until_budget, args.repair, watch=args.watch)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "complete" else 75)
