#!/usr/bin/env python3
"""Register and execute common-scene timed policies with all simple baselines."""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from hallucination.run_approved_manifest import free_gpu_mib  # noqa: E402
from motion2scene_collect_timed_history import (  # noqa: E402
    prepare as prepare_collection,
    run as run_collection,
)
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402

MODES = ("learned", "scripted_sustained", "constant_short", "constant_sustained", "always_walk")


def prepare(args):
    inputs = []
    for path in args.collections:
        result = json.loads(path.read_text())
        ref = result["manifest"]
        manifest = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        if manifest["split"] != "development":
            raise ValueError("this comparator accepts development layouts only")
        inputs.append((path, manifest))
    if (
        not inputs
        or len({str(p.resolve()) for p, _ in inputs}) != len(inputs)
        or args.seed < 0
        or len({m["registry"]["sha256"] for _, m in inputs}) != 1
    ):
        raise ValueError("distinct common-registry development inputs and explicit seed required")
    args.out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__)])):
        dest = args.out / "source_snapshot" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(dest)})
    write_new(
        args.out / "plan.json",
        dict(
            schema="motion2scene_timed_policy_comparison_v1",
            split="development",
            collections=[artifact(p) for p, _ in inputs],
            policy=artifact(args.policy),
            modes=list(MODES),
            physics_seed=args.seed,
            implementation=sources,
            expected_episodes=len(inputs) * len(MODES),
            expected_physics_steps=sum(m["expected_physics_steps"] for _, m in inputs) * len(MODES),
            controls=(
                "identical scene, reference registry, sensor, seed, "
                "legal entry/return and scoring across policies"
            ),
            scope=(
                "reused development layouts with explicit common physics seed; "
                "no held-out layout or curriculum advantage inferred"
            ),
        ),
    )
    children = []
    for i, (_, manifest) in enumerate(inputs):
        for mode in MODES:
            folder = args.out / f"scene_{i:02d}" / mode
            prepare_collection(
                SimpleNamespace(
                    out=folder,
                    registry=Path(manifest["registry"]["path"]),
                    request=Path(manifest["request"]["path"]),
                    template=Path(manifest["template"]["path"]),
                    cell="neutral",
                    scene_definition=Path(manifest["scene_definition"]["path"]),
                    policy_mode=mode,
                    policy=args.policy if mode == "learned" else None,
                    seed=args.seed,
                )
            )
            child = json.loads((folder / "manifest.json").read_text())
            children.append(
                dict(
                    scene_index=i,
                    mode=mode,
                    manifest=artifact(folder / "manifest.json"),
                    scene_definition=child["scene_definition"],
                )
            )
    write_new(
        args.out / "registration.json",
        dict(
            plan=artifact(args.out / "plan.json"),
            children=children,
            serial=True,
            minimum_free_gpu_mib=7500,
            failure_policy="retain attempted cells, never repeat physics automatically",
        ),
    )
    print(json.dumps({"registered": len(children), "path": str(args.out)}), flush=True)


def run(out):
    registration = json.loads((out / "registration.json").read_text())
    ref = registration["plan"]
    plan = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    for ref in plan["implementation"]:
        checked(Path(ref["path"]), ref["sha256"])
    checked(Path(plan["policy"]["path"]), plan["policy"]["sha256"])
    results = []
    for child in registration["children"]:
        ref = child["manifest"]
        folder = checked(Path(ref["path"]), ref["sha256"]).parent
        if not (folder / "result.json").exists():
            if free_gpu_mib() < registration["minimum_free_gpu_mib"]:
                print(
                    json.dumps({"status": "resource_pause", "unlaunched": str(folder)}), flush=True
                )
                return
            started = time.monotonic()
            run_collection(folder)
            write_new(
                folder / "comparison_completion.json",
                dict(
                    wall_seconds=time.monotonic() - started, result=artifact(folder / "result.json")
                ),
            )
        result = json.loads((folder / "result.json").read_text())
        results.append(
            dict(
                **child,
                result=artifact(folder / "result.json"),
                rows=result["rows"],
                physics_steps=result["physics_steps"],
                unmeasured_failed_attempts=result["unmeasured_failed_attempts"],
            )
        )
        print(
            json.dumps(
                {
                    "completed": len(results),
                    "total": len(registration["children"]),
                    "mode": child["mode"],
                    "scene_index": child["scene_index"],
                    "pass": [r["pass"] for r in result["rows"]],
                }
            ),
            flush=True,
        )
    write_new(
        out / "result.json",
        dict(
            registration=artifact(out / "registration.json"),
            groups=results,
            episodes=sum(len(r["rows"]) for r in results),
            physics_steps=sum(r["physics_steps"] for r in results),
            unmeasured_failed_attempts=sum(r["unmeasured_failed_attempts"] for r in results),
            scope=plan["scope"],
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collections", type=Path, nargs="+")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--seed", type=int, default=8732)
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.collections or args.policy is None:
            parser.error("prepare requires --collections and --policy")
        prepare(args)
    else:
        run(args.out)
