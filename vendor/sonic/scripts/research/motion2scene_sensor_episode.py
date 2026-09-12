#!/usr/bin/env python3
"""Prepare and score separate development-only sensor sensitivity episodes.

Common geometry, schedule and contact criteria are shared with nominal traversal.
The new sensor schema verifies corruption and causal history from raw captures;
it never represents missing channels as nominal 65-ray observations.
"""

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
from motion2scene_sensor_perturbations import (  # noqa: E402
    PerturbedObservationStream,
    SensorPerturbation,
)
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import (  # noqa: E402
    SensorRay,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    schedule_history_features,
)

SCHEMA = "motion2scene_sensor_episode_v1"
SENSOR_SCHEMA = "motion2scene_sensor_sensitivity_v1"
RUNTIME = "scripts.research.motion2scene_perturbed_execution"
TARGETS = {
    "manager_env._target_": ("PerturbedScheduleEnvCfg", "TimedScheduleEnvCfg"),
    "manager_env.recorders.trajectory._target_": (
        "PerturbedScheduleRecorderCfg",
        "TimedScheduleRecorderCfg",
    ),
}
SENSOR_KEYS = {
    "dropout_probability": "sensor_dropout_probability",
    "range_noise_std_m": "sensor_range_noise_std_m",
    "latency_s": "sensor_latency_s",
    "seed": "sensor_corruption_seed",
}


def replace_overrides(cell, updates, remove=()):
    cell = copy.deepcopy(cell)
    cell["hydra_overrides"] = [
        value
        for value in cell["hydra_overrides"]
        if value.lstrip("+").split("=", 1)[0] not in set(updates) | set(remove)
    ] + [f"++{key}={value}" for key, value in updates.items()]
    command = cell["command"]
    if command.count("--extra") != 1:
        raise ValueError("one explicit override argument required")
    command[command.index("--extra") + 1] = " ".join(cell["hydra_overrides"])
    return cell


def validate_context(manifest, common, bank, scene, *, actual=False):
    if manifest["schema"] != SCHEMA or scene["split"] != "development":
        raise ValueError("this sensitivity implementation is development-only")
    settings = SensorPerturbation(**manifest["sensor_settings"])
    cell = manifest["cell"]
    command = cell["command"]
    if command.count("--extra") != 1 or command[command.index("--extra") + 1] != " ".join(
        cell["hydra_overrides"]
    ):
        raise ValueError("actual sensor command differs from declared overrides")
    effective = {}
    for item in cell["hydra_overrides"]:
        key, value = item.lstrip("+").split("=", 1)
        if key in effective:
            raise ValueError("duplicate effective override")
        effective[key] = value
    expected = {key: RUNTIME + "." + names[0] for key, names in TARGETS.items()}
    expected.update(
        {
            "manager_env.config." + SENSOR_KEYS[key]: str(value)
            for key, value in asdict(settings).items()
        }
    )
    if any(effective.get(key) != value for key, value in expected.items()):
        raise ValueError("sensor settings or runtime target differ from registration")
    # Factor only the shared execution checks through the nominal validator.
    # This view is never written or executed; actual commands keep the new schema.
    common_view = replace_overrides(
        cell,
        {key: nominal.RUNTIME + "." + names[1] for key, names in TARGETS.items()},
        remove=["manager_env.config." + key for key in SENSOR_KEYS.values()],
    )
    nominal.validate_collection_context(common_view, common, bank, scene, actual=False)
    original = next(c for c in common["cells"] if c["cell_id"] == cell["cell_id"])
    original = copy.deepcopy(original)
    original["output"] = cell["output"]
    original["command"][original["command"].index("--out") + 1] = cell["output"]

    # The new runtime and sensor settings are the only permitted intervention.
    # Compare effective mappings because replacement changes override ordering.
    def mapping(value):
        return dict(item.lstrip("+").split("=", 1) for item in value["hydra_overrides"])

    if mapping(common_view) != mapping(original):
        raise ValueError("sensor episode changes a non-sensor nominal override")
    for name in set(original) - {"command", "hydra_overrides"}:
        if common_view.get(name) != original[name]:
            raise ValueError("sensor episode changes nominal cell metadata: " + name)
    normalized = copy.deepcopy(common_view["command"])
    normalized[normalized.index("--extra") + 1] = " ".join(original["hydra_overrides"])
    if normalized != original["command"]:
        raise ValueError("sensor episode changes a non-sensor command argument")
    if actual:
        folder = Path(cell["output"])
        attempt = json.loads((folder / "attempt.json").read_text())
        if attempt["command"] != command:
            raise ValueError("attempted command differs from sensor registration")
        runtime = json.loads((folder / "success_manifest.json").read_text())
        capture = runtime["capture_context"]
        if (
            capture["scene_id"] != scene["scene_id"]
            or capture["scene"]["hash"] != scene["scene"]["sha256"]
            or Path(capture["scene"]["resolved"]).resolve()
            != Path(scene["scene"]["path"]).resolve()
        ):
            raise ValueError("captured scene differs from sensitivity registration")
    return settings


def prepare(template, cell_id, out, settings):
    common, bank, scene = nominal.verify_manifest(template.parent, execution=False)
    if template.name != "manifest.json" or scene["split"] != "development":
        raise ValueError("a verified nominal development template is required")
    cell = copy.deepcopy(next(c for c in common["cells"] if c["cell_id"] == cell_id))
    cell["output"] = str(out.resolve() / "rollout")
    cell["command"][cell["command"].index("--out") + 1] = cell["output"]
    updates = {key: RUNTIME + "." + names[0] for key, names in TARGETS.items()}
    updates.update(
        {"manager_env.config." + SENSOR_KEYS[key]: value for key, value in asdict(settings).items()}
    )
    cell = replace_overrides(cell, updates)
    sources = closure([Path(__file__), ROOT / (RUNTIME.replace(".", "/") + ".py")]) | {
        ROOT / "scripts/research/run_kimodo_sonic_rollout.sh",
        ROOT / "gear_sonic/eval_agent_trl.py",
    }
    manifest = dict(
        schema=SCHEMA,
        nominal_template=artifact(template),
        cell=cell,
        sensor_settings=asdict(settings),
        implementation=[artifact(p) for p in sorted(sources)],
        scope="development sensitivity; same task, bank, policy, physics seed and physical scorer",
        failure_policy="retain every attempt and partial capture; never automatically retry",
        limits=common["limits"],
    )
    validate_context(manifest, common, bank, scene)
    out.mkdir(parents=True, exist_ok=False)
    for ref in manifest["implementation"]:
        path = Path(ref["path"])
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        ref["snapshot"] = artifact(target)
    return write_new(out / "manifest.json", manifest)


def verify(out, *, execution=False):
    manifest = json.loads((out / "manifest.json").read_text())
    ref = manifest["nominal_template"]
    checked(Path(ref["path"]), ref["sha256"])
    common, bank, scene = nominal.verify_manifest(Path(ref["path"]).parent, execution=False)
    for ref in manifest["implementation"]:
        chosen = ref if execution else ref["snapshot"]
        checked(Path(chosen["path"]), chosen["sha256"])
    validate_context(manifest, common, bank, scene)
    return manifest, common, bank, scene


def verify_sensor_capture(interface, bank, settings):
    """Reconstruct every policy input from corrupted, causally delivered packets."""
    if interface.get("sensor_schema") != SENSOR_SCHEMA or interface.get(
        "sensor_settings"
    ) != asdict(settings):
        raise ValueError("captured sensor declaration differs")
    stream = PerturbedObservationStream(settings)
    for row in interface["observations"]:
        if len(row["raw_measurements"]) != 65:
            raise ValueError("raw verification fan must retain all 65 source channels")
        cache, _ = stream.push([SensorRay(**r) for r in row["raw_measurements"]], row["tick"] / 50)
        # Recorder JSON represents typed ray tuples as arrays.
        for key, value in json.loads(json.dumps(cache, allow_nan=False)).items():
            if row.get(key) != value:
                raise ValueError("sensor reconstruction differs: " + key)
        observation = stream.history.snapshot(
            row["root_pos_w"], row["root_quat_w"], cache["capture_elapsed_s"]
        )
        names, features = schedule_history_features(
            observation,
            stream.history.grid,
            row["state"],
            row["tick"],
            bank.option_ids.index(row["active_before"]),
            np.asarray(row["legal_mask"], dtype=bool),
            cache["observation_age_s"],
        )
        if tuple(interface["feature_names"]) != tuple(names) or not np.array_equal(
            features, row["features"]
        ):
            raise ValueError("recorded student features differ from causal sensor reconstruction")
    return dict(reconstructed_control_rows=len(interface["observations"]), exact_features=True)


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def analyze_complete(manifest, common, bank, scene, *, execution_validator=validate_context):
    """Common physical scorer; each opt-in learner validates its own execution binding."""
    settings = execution_validator(manifest, common, bank, scene, actual=True)
    cell = manifest["cell"]
    folder = Path(cell["output"]) / "trajectories"
    paths = list(folder.glob("*.trajectory.pkl"))
    if len(paths) != 1:
        raise ValueError("one complete physical capture required")
    payload = nominal.load_reset_capture(paths[0])
    interface = json.loads((folder / "reactive_interface.json").read_text())
    expected = dict(
        timed_schedule_mode=cell["timed_schedule_mode"],
        timed_policy_sha256=common["policy"]["sha256"] if common["policy"] else "",
        script_parameters_sha256=(
            common["script_parameters"]["sha256"] if common.get("script_parameters") else ""
        ),
        feature_schema=nominal.SCHEMA,
        option_ids=list(bank.option_ids),
        feature_names=list(nominal.expected_feature_names(len(bank.option_ids))),
        timed_request_digest=nominal.definition_digest(bank.request),
        timed_registry_sha256=common["registry"]["sha256"],
        phase_ticks=common["phase_ticks"],
        history_max_age_s=nominal.HISTORY_SECONDS,
        history_max_frames=nominal.HISTORY_FRAMES,
        qualification_only=False,
    )
    if any(interface.get(key) != value for key, value in expected.items()):
        raise ValueError("captured policy or schedule interface differs")
    sensor = verify_sensor_capture(interface, bank, settings)
    observations = interface["observations"]
    archive = arrays(folder / "timed_schedule_features.npz")
    if str(archive["schema_version"]) != nominal.SCHEMA or tuple(archive["feature_names"]) != tuple(
        expected["feature_names"]
    ):
        raise ValueError("feature archive schema differs")
    for key, field in dict(
        features="features",
        active_before="active_before",
        active="active",
        command_ticks="tick",
        legal_mask="legal_mask",
    ).items():
        if not np.array_equal(archive[key], [row[field] for row in observations]):
            raise ValueError("feature archive differs: " + key)
    alignment = nominal.audit_sensor_alignment(
        payload, observations, reference_frames=bank.frame_count
    )
    contacts = arrays(folder / "all_body_contacts.npz")
    pairs = arrays(folder / "environment_pair_contacts.npz")
    mapping = json.loads((folder / "environment_contact_mapping.json").read_text())
    contact = nominal.audit_environment_contacts(
        pairs,
        contacts,
        mapping,
        common["expected_physics_steps"],
        neutral_self_pairs=common.get("declared_neutral_self_pairs", []),
        beam_paths=nominal.scene_beam_paths(scene),
    )
    forces, sync = nominal.synchronize_pair_beam_forces(
        pairs, mapping, nominal.scene_beam_paths(scene), contacts["control_steps"]
    )
    inventory = json.loads((folder / "native_collision_inventory.json").read_text())
    geometry = nominal.audit_scene_geometry(inventory["shapes"], scene)
    runtime = json.loads((folder.parent / "success_manifest.json").read_text())
    references = json.loads((folder / "loaded_reference_bank.json").read_text())
    source_bound = (
        runtime["checkpoint_hash"] == bank.request["controller"]["sha256"]
        and runtime["capture_context"]["motion"]["hash"]
        == bank.request["references"][0]["motion"]["sha256"]
        and runtime["capture_context"]["use_encoder"] == "g1"
        and max(references["source_route_max_error_m"]) <= 1e-4
        and interface["loaded_reference_ids"]
        == [v["reference_id"] for v in bank.request["references"]]
        and interface["loaded_reference_frames"]
        == [bank.frame_count] * len(bank.request["references"])
    )
    admitted = bool(
        source_bound
        and all(alignment["packet_eligible"])
        and len(observations) == bank.frame_count - 1
        and contact["complete_synchronized_streams"]
        and np.array_equal(contacts["control_steps"], [r["physics_step"] for r in observations])
    )
    stable = bool(
        (
            (np.asarray(payload["root_pos_w"])[:, 2] >= 0.5)
            & (-np.asarray(payload["projected_gravity_b"])[:, 2] >= 0.5)
        ).all()
    )
    schedule = nominal.audit_executed_schedule(
        bank,
        observations,
        interface["switches"],
        cell["timed_schedule_mode"],
        cell["forced_option_id"],
    )
    passage = nominal.score_scene(payload, forces, scene, bank, commands_valid=schedule["valid"])
    returned = observations[-1]["active"] == "neutral"
    passed = bool(
        admitted
        and passage["pass"]
        and stable
        and returned
        and schedule["no_refusals"]
        and contact["no_undesired_measured_contact"]
        and schedule["valid"]
    )
    outcome = nominal.classify_timed_attempt(
        payload,
        observations,
        reference_frames=bank.frame_count,
        phase_ticks=common["phase_ticks"],
        exit_status=0,
        measurement_admitted=admitted,
        passage=passage,
        contact_audit=contact,
        schedule_audit=schedule,
        physics_steps=pairs["physics_steps"],
    )
    if passed != outcome["complete_pass"]:
        raise ValueError("physical passage scorers disagree")
    finish = passage["passage_finish_frame_exclusive"]
    return dict(
        cell_id=cell["cell_id"],
        mode=cell["timed_schedule_mode"],
        **{"pass": passed},
        outcome=outcome,
        measurement_admitted=admitted,
        task_outcome_admitted=outcome["task_outcome"] != "unknown",
        passage=passage,
        whole_horizon_stable=stable,
        returned_to_neutral=returned,
        physics_steps=contact["recorded_physics_steps"],
        sensor_reconstruction=sensor,
        sensor_alignment=alignment,
        contact_audit=contact,
        schedule_audit=schedule,
        imported_geometry_audit=geometry,
        beam_contact_synchronization=sync,
        trajectory=artifact(paths[0]),
        sensor=artifact(folder / "reactive_interface.json"),
        environment_pairs=artifact(folder / "environment_pair_contacts.npz"),
        decisions=[
            dict(
                tick=r["tick"],
                selected_option_id=r["selected_option_id"],
                observation_age_s=r["observation_age_s"],
            )
            for r in observations
            if r["tick"] in common["phase_ticks"]
        ],
        costs=dict(
            passage_time_s=(finish - 1) / 50 if passed else None,
            whole_episode_time_s=len(observations) / 50,
            switch_count=len(interface["switches"]),
            positive_mechanical_work_j=None,
        ),
    )


def analyze(out):
    manifest, common, bank, scene = verify(out)
    cell = manifest["cell"]
    attempt = json.loads((Path(cell["output"]) / "attempt.json").read_text())
    if attempt["command"] != cell["command"]:
        raise ValueError("attempt command differs")
    try:
        if attempt["exit_status"] != 0:
            raise ValueError("native runtime exited unsuccessfully")
        result = analyze_complete(manifest, common, bank, scene)
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        result = nominal.analyze_incomplete(cell, common, bank, scene, attempt, str(error))
    return write_new(
        out / "result.json",
        dict(
            schema=SCHEMA,
            manifest=artifact(out / "manifest.json"),
            attempt=artifact(Path(cell["output"]) / "attempt.json"),
            row=result,
            sensor_settings=manifest["sensor_settings"],
        ),
    )


def run(out):
    """One charged attempt; an interrupted or unknown outcome is not retried."""
    manifest, common, _, _ = verify(out, execution=True)
    cell = manifest["cell"]
    folder = Path(cell["output"])
    if folder.exists() or (out / "launch.json").exists():
        raise ValueError("previously launched sensor episode retained; automatic retry forbidden")
    if nominal.free_gpu_mib() < manifest["limits"]["minimum_free_gpu_mib"]:
        raise RuntimeError("insufficient GPU capacity; episode remains unlaunched")
    write_new(
        out / "launch.json",
        dict(
            manifest=artifact(out / "manifest.json"),
            command=cell["command"],
            charged_maximum_physics_steps=common["expected_physics_steps"],
            status="assigned_before_native_launch",
        ),
    )
    started = time.monotonic()
    interrupted = None
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
        dict(exit_status=status, wall_seconds=time.monotonic() - started, command=cell["command"]),
    )
    result = analyze(out)
    if interrupted is not None:
        raise interrupted
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--template", type=Path, required=True)
    prep.add_argument("--cell", required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--dropout", type=float, default=0.0)
    prep.add_argument("--range-noise", type=float, default=0.0)
    prep.add_argument("--latency", type=float, default=0.0)
    prep.add_argument("--sensor-seed", type=int, default=95001)
    for name in ("verify", "analyze", "run"):
        child = commands.add_parser(name)
        child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.template,
            args.cell,
            args.out,
            SensorPerturbation(args.dropout, args.range_noise, args.latency, args.sensor_seed),
        )
    elif args.command == "verify":
        verify(args.out, execution=True)
        result = dict(verified=True)
    elif args.command == "run":
        result = run(args.out)
    else:
        result = analyze(args.out)
    print(json.dumps(result))
