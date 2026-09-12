#!/usr/bin/env python3
"""Separate development episodes for the outcome-tree decision learner."""

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
import motion2scene_collect_timed_schedules as nominal  # noqa: E402
from motion2scene_outcome_policy import choose_schedule, load_policy  # noqa: E402
from motion2scene_sensor_episode import analyze_complete, replace_overrides  # noqa: E402
from motion2scene_sensor_perturbations import SensorPerturbation  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)

SCHEMA = "motion2scene_outcome_episode_v1"
RUNTIME = "scripts.research.motion2scene_outcome_execution"
SENSOR_KEYS = {
    "dropout_probability": "sensor_dropout_probability",
    "range_noise_std_m": "sensor_range_noise_std_m",
    "latency_s": "sensor_latency_s",
    "seed": "sensor_corruption_seed",
}
TARGETS = {
    "manager_env._target_": ("OutcomeScheduleEnvCfg", "TimedScheduleEnvCfg"),
    "manager_env.recorders.trajectory._target_": (
        "OutcomeScheduleRecorderCfg",
        "TimedScheduleRecorderCfg",
    ),
}


def expected_cell(original, policy_ref, settings, out):
    cell = copy.deepcopy(original)
    if cell["timed_schedule_mode"] != "learned":
        raise ValueError("matched learned-policy template required")
    cell["output"] = str(out.resolve() / "rollout")
    cell["command"][cell["command"].index("--out") + 1] = cell["output"]
    updates = {key: RUNTIME + "." + names[0] for key, names in TARGETS.items()}
    updates.update({"manager_env.config." + SENSOR_KEYS[k]: v for k, v in asdict(settings).items()})
    updates.update(
        {
            "manager_env.config.learner_family": "outcome_tree",
            "manager_env.config.timed_policy_path": policy_ref["path"],
            "manager_env.config.timed_policy_sha256": policy_ref["sha256"],
        }
    )
    return replace_overrides(cell, updates)


def validate_context(manifest, common, bank, scene, *, actual=False):
    if manifest["schema"] != SCHEMA or scene["split"] != "development":
        raise ValueError("outcome learner comparison is development-only")
    settings = SensorPerturbation(**manifest["sensor_settings"])
    cell = manifest["cell"]
    source = next(c for c in common["cells"] if c["cell_id"] == cell["cell_id"])
    expected = expected_cell(source, manifest["policy"], settings, Path(cell["output"]).parent)
    if cell != expected:
        raise ValueError("outcome invocation changes an undeclared nominal component")
    model = load_policy(manifest["policy"]["path"], manifest["policy"]["sha256"], bank)
    view = replace_overrides(
        cell,
        {key: nominal.RUNTIME + "." + names[1] for key, names in TARGETS.items()},
        remove=["manager_env.config." + key for key in SENSOR_KEYS.values()]
        + ["manager_env.config.learner_family"],
    )
    nominal.validate_collection_context(view, common, bank, scene, actual=False)
    if actual:
        folder = Path(cell["output"])
        attempt = json.loads((folder / "attempt.json").read_text())
        if attempt["command"] != cell["command"] or attempt["exit_status"] != 0:
            raise ValueError("outcome policy attempt differs or is incomplete")
        runtime = json.loads((folder / "success_manifest.json").read_text())
        context = runtime["capture_context"]
        if (
            context["scene_id"] != scene["scene_id"]
            or context["scene"]["hash"] != scene["scene"]["sha256"]
            or Path(context["scene"]["resolved"]).resolve()
            != Path(scene["scene"]["path"]).resolve()
        ):
            raise ValueError("actual outcome evaluation scene differs")
        interface = json.loads((folder / "trajectories/reactive_interface.json").read_text())
        if (
            interface.get("learner_family") != "outcome_tree"
            or interface.get("outcome_policy_schema") != model["schema"]
        ):
            raise ValueError("captured learner is not the declared outcome policy")
        options = {o["option_id"]: o for o in bank.request["options"]}
        for row in interface["observations"]:
            active = bank.option_ids.index(row["active_before"])
            mandatory = (
                0
                if active and row["tick"] == options[row["active_before"]]["return_tick"]
                else None
            )
            choice, predictions = choose_schedule(
                model,
                interface["feature_names"],
                row["features"],
                np.asarray(row["legal_mask"], dtype=bool),
                active,
                row["tick"],
                mandatory,
                bank,
            )
            if (
                choice != row["selected_option_id"]
                or predictions != row["outcome_predictions"]
                or row["policy_values"] is not None
            ):
                raise ValueError(
                    "captured policy decisions differ from the exported outcome learner"
                )
    return settings


def prepare(template, cell_id, policy, out):
    common, bank, scene = nominal.verify_manifest(template.parent, execution=False)
    policy_ref = artifact(policy)
    load_policy(policy_ref["path"], policy_ref["sha256"], bank)
    settings = SensorPerturbation(seed=95001)
    source = next(c for c in common["cells"] if c["cell_id"] == cell_id)
    cell = expected_cell(source, policy_ref, settings, out)
    common["policy"] = policy_ref
    manifest = dict(
        schema=SCHEMA,
        nominal_template=artifact(template),
        policy=policy_ref,
        cell=cell,
        sensor_settings=asdict(settings),
        limits=common["limits"],
        comparison=(
            "same acquired teachers, geometry, physics seed, observation and physical criteria; "
            "changed learner"
        ),
        failure_policy="retain assigned failures and incomplete attempts; no automatic retries",
    )
    validate_context(manifest, common, bank, scene)
    sources = closure([Path(__file__), ROOT / (RUNTIME.replace(".", "/") + ".py")]) | {
        ROOT / "scripts/research/run_kimodo_sonic_rollout.sh",
        ROOT / "gear_sonic/eval_agent_trl.py",
    }
    out.mkdir(parents=True, exist_ok=False)
    manifest["implementation"] = []
    for path in sorted(sources):
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest["implementation"].append(dict(**artifact(path), snapshot=artifact(target)))
    return write_new(out / "manifest.json", manifest)


def verify(out, *, execution=False):
    manifest = json.loads((out / "manifest.json").read_text())
    ref = manifest["nominal_template"]
    checked(Path(ref["path"]), ref["sha256"])
    common, bank, scene = nominal.verify_manifest(Path(ref["path"]).parent, execution=False)
    common["policy"] = manifest["policy"]
    for ref in manifest["implementation"]:
        selected = ref if execution else ref["snapshot"]
        checked(Path(selected["path"]), selected["sha256"])
    validate_context(manifest, common, bank, scene)
    return manifest, common, bank, scene


def analyze(out):
    manifest, common, bank, scene = verify(out)
    cell = manifest["cell"]
    attempt = json.loads((Path(cell["output"]) / "attempt.json").read_text())
    if attempt["command"] != cell["command"]:
        raise ValueError("recorded outcome attempt differs")
    try:
        result = analyze_complete(
            manifest, common, bank, scene, execution_validator=validate_context
        )
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        result = nominal.analyze_incomplete(cell, common, bank, scene, attempt, str(error))
    return write_new(
        out / "result.json",
        dict(
            schema=SCHEMA,
            manifest=artifact(out / "manifest.json"),
            row=result,
            learner_family="outcome_tree",
        ),
    )


def run(out):
    manifest, common, _, _ = verify(out, execution=True)
    cell = manifest["cell"]
    folder = Path(cell["output"])
    if folder.exists() or (out / "launch.json").exists():
        raise ValueError("previous attempt retained; no automatic retry")
    if nominal.free_gpu_mib() < manifest["limits"]["minimum_free_gpu_mib"]:
        raise RuntimeError("insufficient capacity; outcome evaluation remains unlaunched")
    write_new(
        out / "launch.json",
        dict(
            command=cell["command"], charged_maximum_physics_steps=common["expected_physics_steps"]
        ),
    )
    started, interrupted = time.monotonic(), None
    try:
        status = nominal.run_with_process_group(
            cell["command"], timeout=manifest["limits"]["timeout_s"]
        )
    except subprocess.TimeoutExpired:
        status = 124
    except BaseException as error:
        status, interrupted = -1, error
    folder.mkdir(parents=True, exist_ok=True)
    write_new(
        folder / "attempt.json",
        dict(command=cell["command"], exit_status=status, wall_seconds=time.monotonic() - started),
    )
    result = analyze(out)
    if interrupted is not None:
        raise interrupted
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run", "analyze"))
    parser.add_argument("--template", type=Path)
    parser.add_argument("--cell", default="learned_neutral")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.template, args.cell, args.policy, args.out)
    elif args.command == "verify":
        verify(args.out, execution=True)
        result = dict(verified=True)
    else:
        result = globals()[args.command](args.out)
    print(json.dumps(result))
