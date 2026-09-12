#!/usr/bin/env python3
"""Prepare and score assigned held-center episodes for equivalent-teaching policies."""

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
from motion2scene_extension_policy import load_extension_policy, read  # noqa: E402
from motion2scene_extension_teaching import arm_view, choose_arm_schedule  # noqa: E402
from motion2scene_sensor_episode import analyze_complete, replace_overrides  # noqa: E402
from motion2scene_sensor_perturbations import SensorPerturbation  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)

SCHEMA = "motion2scene_extension_arm_episode_v1"
RUNTIME = "scripts.research.motion2scene_extension_execution"
TARGETS = {
    "manager_env._target_": ("ExtensionScheduleEnvCfg", "TimedScheduleEnvCfg"),
    "manager_env.recorders.trajectory._target_": (
        "ExtensionScheduleRecorderCfg",
        "TimedScheduleRecorderCfg",
    ),
}
EXTRA_KEYS = ("learner_family", "extension_result_path", "extension_result_sha256")


def expected_cell(original, result_ref, result, out):
    if original["timed_schedule_mode"] != "forced" or original["forced_option_id"] != "neutral":
        raise ValueError("the assigned full-bank forced-neutral task template is required")
    cell = copy.deepcopy(original)
    cell.update(
        cell_id="extension_arm_learned",
        role="repeated_schedule_policy",
        timed_schedule_mode="learned",
        policy_id="extension_" + result["arm"],
    )
    cell["output"] = str(out.resolve() / "rollout")
    cell["command"][cell["command"].index("--out") + 1] = cell["output"]
    updates = {key: RUNTIME + "." + names[0] for key, names in TARGETS.items()}
    updates.update(
        {
            "manager_env.config.timed_schedule_mode": "learned",
            "manager_env.config.timed_policy_path": result["policy"]["path"],
            "manager_env.config.timed_policy_sha256": result["policy"]["sha256"],
            "manager_env.config.learner_family": "extension_arm_ridge",
            "manager_env.config.extension_result_path": result_ref["path"],
            "manager_env.config.extension_result_sha256": result_ref["sha256"],
        }
    )
    return replace_overrides(cell, updates)


def verify_policy_trace(interface, model, result, result_ref, bank):
    arm = result["arm"]
    view, _ = arm_view(bank, arm)
    expected = dict(
        learner_family="extension_arm_ridge",
        extension_arm=arm,
        extension_result_sha256=result_ref["sha256"],
        logical_option_ids=list(view.option_ids),
        logical_request_digest=result["logical_request_digest"],
        logical_feature_names=model["feature_names"].tolist(),
    )
    if any(interface.get(k) != v for k, v in expected.items()):
        raise ValueError("captured arm policy or logical interface differs from the assigned model")
    options = {o["option_id"]: o for o in bank.request["options"]}
    for row in interface["observations"]:
        active = bank.option_ids.index(row["active_before"])
        mandatory = (
            0 if active and row["tick"] == options[row["active_before"]]["return_tick"] else None
        )
        selected, projected = choose_arm_schedule(
            bank,
            arm,
            model,
            interface["feature_names"],
            row["features"],
            np.asarray(row["legal_mask"], bool),
            active,
            row["tick"],
            mandatory,
        )
        saved = row["arm_policy"]
        if (
            selected != row["selected_option_id"]
            or row["policy_values"] is not None
            or set(saved) != set(projected)
            or any(saved[k] != projected[k] for k in projected if k != "policy_values")
        ):
            raise ValueError("recorded decision or projected input differs from actual arm policy")
        values, expected_values = saved["policy_values"], projected["policy_values"]
        if expected_values is None:
            equal = values is None
        else:
            equal = (
                values is not None
                and np.shape(values) == np.shape(expected_values)
                and np.allclose(values, expected_values, rtol=0, atol=1e-12)
            )
        if not equal:
            raise ValueError("recorded ridge values differ; absolute readout tolerance is 1e-12")
    return len(interface["observations"])


def validate_context(manifest, common, bank, scene, *, actual=False):
    if manifest["schema"] != SCHEMA or scene["split"] != "development":
        raise ValueError("equivalent-teaching episodes are development-only")
    if manifest["sensor_settings"] != asdict(SensorPerturbation(seed=95001)):
        raise ValueError("equivalent teaching retains the declared nominal sensor settings")
    ref = manifest["model_result"]
    model, result = load_extension_policy(ref["path"], ref["sha256"], bank)
    study = read(result["study"])
    prepared = read(study["prepared"])
    assignment = next((t for t in prepared["tasks"] if t["task_id"] == scene["scene_id"]), None)
    if (
        scene["scene_id"] not in result["analysis"]["fold"]["evaluation_task_ids"]
        or assignment is None
        or assignment["collection"] != manifest["nominal_template"]
        or manifest["cell"]["runtime_seed"] != study["policy_evaluation_seed"]
        or result["registry"] != common["registry"]
        or manifest["policy"] != result["policy"]
    ):
        raise ValueError("assigned held-center scene, common seed and fitted model required")
    original = next(c for c in common["cells"] if c["forced_option_id"] == "neutral")
    cell = manifest["cell"]
    if cell != expected_cell(original, ref, result, Path(cell["output"]).parent):
        raise ValueError("invocation changes an undeclared component of the assigned task")
    normalized = replace_overrides(
        cell,
        {key: nominal.RUNTIME + "." + names[1] for key, names in TARGETS.items()},
        remove=["manager_env.config." + key for key in EXTRA_KEYS],
    )
    nominal.validate_collection_context(normalized, common, bank, scene, actual=False)
    if actual:
        folder = Path(cell["output"])
        attempt = json.loads((folder / "attempt.json").read_text())
        if attempt["command"] != cell["command"] or attempt["exit_status"] != 0:
            raise ValueError("assigned arm-policy invocation is incomplete or differs")
        context = json.loads((folder / "success_manifest.json").read_text())["capture_context"]
        if (
            context["scene_id"] != scene["scene_id"]
            or context["scene"]["hash"] != scene["scene"]["sha256"]
            or Path(context["scene"]["resolved"]).resolve()
            != Path(scene["scene"]["path"]).resolve()
        ):
            raise ValueError("actual native scene differs from the assigned held-center task")
        interface = json.loads((folder / "trajectories/reactive_interface.json").read_text())
        verify_policy_trace(interface, model, result, ref, bank)
    return SensorPerturbation(seed=95001)


def prepare(template, model_result, out):
    common, bank, scene = nominal.verify_manifest(template.parent, execution=False)
    ref = artifact(model_result)
    _, result = load_extension_policy(ref["path"], ref["sha256"], bank)
    original = next(c for c in common["cells"] if c["forced_option_id"] == "neutral")
    manifest = dict(
        schema=SCHEMA,
        nominal_template=artifact(template),
        model_result=ref,
        policy=result["policy"],
        cell=expected_cell(original, ref, result, out),
        limits=common["limits"],
        sensor_settings=asdict(SensorPerturbation(seed=95001)),
        comparison="assigned held-center task and seed; full bank, nominal sensing and original physical criteria",
        failure_policy="retain every charged attempt; no automatic retry or policy substitution",
    )
    common["policy"] = result["policy"]  # Validation-only view; original template stays unchanged.
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
    manifest = read(artifact(out / "manifest.json"))
    ref = manifest["nominal_template"]
    checked(Path(ref["path"]), ref["sha256"])
    common, bank, scene = nominal.verify_manifest(Path(ref["path"]).parent, execution=False)
    common["policy"] = manifest["policy"]
    for source in manifest["implementation"]:
        selected = source if execution else source["snapshot"]
        checked(Path(selected["path"]), selected["sha256"])
    validate_context(manifest, common, bank, scene)
    return manifest, common, bank, scene


def analyze(out):
    manifest, common, bank, scene = verify(out)
    cell = manifest["cell"]
    attempt = json.loads((Path(cell["output"]) / "attempt.json").read_text())
    if attempt["command"] != cell["command"]:
        raise ValueError("recorded attempt differs from assigned arm-policy invocation")
    try:
        row = analyze_complete(manifest, common, bank, scene, execution_validator=validate_context)
        trace_verified = True
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        row = nominal.analyze_incomplete(cell, common, bank, scene, attempt, str(error))
        trace_verified = False
    return write_new(
        out / "result.json",
        dict(
            schema=SCHEMA,
            manifest=artifact(out / "manifest.json"),
            row=row,
            learner_family="extension_arm_ridge",
            complete_policy_trace_verified=trace_verified,
        ),
    )


def run(out):
    manifest, common, _, _ = verify(out, execution=True)
    cell = manifest["cell"]
    folder = Path(cell["output"])
    if folder.exists() or (out / "launch.json").exists():
        raise ValueError("previous attempt retained; no automatic retry")
    if nominal.free_gpu_mib() < manifest["limits"]["minimum_free_gpu_mib"]:
        raise RuntimeError("insufficient capacity; episode remains unlaunched")
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
    parser.add_argument("operation", choices=("prepare", "verify", "run", "analyze"))
    parser.add_argument("--template", type=Path)
    parser.add_argument("--model-result", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.operation == "prepare":
        if args.template is None or args.model_result is None:
            parser.error("prepare requires --template and --model-result")
        result = prepare(args.template, args.model_result, args.out)
    elif args.operation == "verify":
        verify(args.out, execution=True)
        result = dict(verified=True)
    else:
        result = globals()[args.operation](args.out)
    print(json.dumps(result))
