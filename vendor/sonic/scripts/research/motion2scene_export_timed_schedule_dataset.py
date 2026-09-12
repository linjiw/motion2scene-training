#!/usr/bin/env python3
"""Portable repeated-schedule data with causal inputs and verified WAIT continuations."""

import argparse
import copy
import json
from pathlib import Path
import pickle
import sys
import tarfile

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/research"))

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_collect_timed_schedules import (  # noqa: E402
    analyze_cell,
    audit_scene_geometry,
    scene_beam_paths,
    score_scene,
    verify_manifest,
)
from motion2scene_export_option_dataset import copy_npz, save_pickle_free  # noqa: E402
from motion2scene_export_timed_dataset import (  # noqa: E402
    COMMAND_KEYS,
    FORBIDDEN_SENSOR_KEYS,
    array_file,
    causal_packets,
    executed_schedule,
    read_json,
    ref,
    safe_path,
)
from motion2scene_export_traversal_dataset import checked, digest, write_json  # noqa: E402
from motion2scene_train_timed_schedules import (  # noqa: E402
    ARTIFACT_KEYS,
    audit_collection,
    history_id,
    validate_invocation,
)
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_environment_contacts import (  # noqa: E402
    audit_environment_contacts,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_sensor_alignment import (  # noqa: E402
    audit_sensor_alignment,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_course_measurements import (  # noqa: E402
    synchronize_pair_beam_forces,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    TimedOptionBank,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_outcome import (  # noqa: E402
    classify_timed_attempt,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    TimedScheduledOutcome,
    schedule_layout,
    timed_schedule_teacher,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    SCHEMA as INPUT_SCHEMA,
    audit_executed_schedule,
    expected_feature_names,
)

SCHEMA = "motion2scene_timed_schedule_portable_dataset_v1"
INPUT_KEYS = {
    "schema_version",
    "feature_names",
    "option_ids",
    "features",
    "command_ticks",
    "capture_elapsed_s",
    "legal_mask",
    "active_before",
}
CONTACT_NAMES = (
    "all_body_contacts.npz",
    "physics_beam_contacts.npz",
    "beam_contacts.npz",
    "option_mechanical_work.npz",
    "environment_pair_contacts.npz",
)


def export_inputs(source, destination, interface):
    values = copy_npz(source, destination, allowed=INPUT_KEYS)
    count = len(interface["option_ids"])
    observations = interface["observations"]
    if (
        set(values) != INPUT_KEYS
        or str(values["schema_version"]) != INPUT_SCHEMA
        or tuple(values["feature_names"]) != expected_feature_names(count)
        or values["features"].shape != (len(observations), 100 + 2 * count)
        or list(values["option_ids"]) != interface["option_ids"]
        or not np.isfinite(values["features"]).all()
        or not np.array_equal(values["features"], [r["features"] for r in observations])
        or not np.array_equal(values["command_ticks"], [r["tick"] for r in observations])
        or not np.array_equal(values["active_before"], [r["active_before"] for r in observations])
        or not np.array_equal(values["legal_mask"], [r["legal_mask"] for r in observations])
    ):
        raise ValueError("exact named100+2K preaction input archive required")
    return values


def portable_payload(folder):
    arrays = array_file(folder / "trajectory.npz")

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"npz_array"}:
                return arrays[value["npz_array"]]
            return {key: decode(child) for key, child in value.items()}
        return value

    return decode(read_json(folder / "trajectory_metadata.json"))


def portable_interface(folder, bank):
    packets = read_json(folder / "sensor_history.json")
    inputs = array_file(folder / "student_inputs.npz")
    commands = read_json(folder / "commands.json")
    if (
        set(inputs) != INPUT_KEYS
        or str(inputs["schema_version"]) != INPUT_SCHEMA
        or tuple(inputs["option_ids"]) != bank.option_ids
        or tuple(inputs["feature_names"]) != expected_feature_names(len(bank.option_ids))
        or inputs["features"].shape != (len(packets), 100 + 2 * len(bank.option_ids))
        or len(commands["records"]) != len(packets)
    ):
        raise ValueError("portable input/command/sensor schema differs")

    def guard(value):
        if isinstance(value, dict):
            if FORBIDDEN_SENSOR_KEYS.intersection(value):
                raise ValueError("privileged geometry/labels in causal sensor packet")
            for child in value.values():
                guard(child)
        elif isinstance(value, list):
            for child in value:
                guard(child)

    guard(packets)
    observations = []
    for index, (packet, command) in enumerate(zip(packets, commands["records"], strict=True)):
        if (
            packet["tick"] != command["tick"]
            or packet["tick"] != inputs["command_ticks"][index]
            or command["active_before"] != inputs["active_before"][index]
            or not np.array_equal(command["legal_mask"], inputs["legal_mask"][index])
        ):
            raise ValueError("portable preaction identity or legal mask differs")
        observations.append({**packet, **command, "features": inputs["features"][index].tolist()})
    return dict(
        observations=observations,
        switches=commands["switches"],
        feature_names=inputs["feature_names"].tolist(),
        option_ids=list(bank.option_ids),
    )


def verify_teacher_group(teacher, episode_map, bank):
    """Recompute every WAIT and immediate target solely from the portable archive."""
    ids = teacher["episode_ids"]
    if len(ids) != len(bank.option_ids) or len(set(ids)) != len(ids):
        raise ValueError("teacher requires one distinct episode per complete schedule")
    captures = [episode_map[identifier] for identifier in ids]
    phases, _, entries = schedule_layout(bank)
    group = teacher["recorded_history_group"]
    branches = []
    for index, (episode, payload, interface) in enumerate(captures):
        if (
            episode["configured_forced_option_id"] != bank.option_ids[index]
            or episode["physics_seed"] != teacher["physics_seed"]
            or episode["collection_sha256"] != teacher["collection"]["sha256"]
        ):
            raise ValueError("teacher branch identities differ from actual complete episodes")
        hashes = {
            tick: history_id(group, payload, interface, tick)
            for tick in phases.tolist()
            if entries[index] is None or entries[index] >= tick
        }
        assessment = episode["assessment"]
        branches.append(
            TimedScheduledOutcome(
                episode["original_cell_id"],
                index,
                episode["physics_seed"],
                assessment["pass"],
                assessment["costs"]["passage_time_s"],
                assessment["measurement_admitted"],
                hashes,
                episode["physics_steps"],
            )
        )
    if len(teacher["targets"]) != len(phases):
        raise ValueError("one actual neutral target per registered phase required")
    for tick, stored in zip(phases.tolist(), teacher["targets"], strict=True):
        packet = captures[0][2]["observations"][tick - 1]
        value = timed_schedule_teacher(
            bank,
            branches,
            tick,
            branches[0].prefix_hash_by_tick[tick],
            teacher["physics_seed"],
            np.asarray(packet["legal_mask"], bool),
        )
        for key, expected in value.items():
            if stored[key] != expected:
                raise ValueError(
                    "portable WAIT/phase target differs from actual continuation: " + key
                )
        expected_ids = [None if i is None else ids[i] for i in value["continuation_option_indices"]]
        if (
            stored["continuation_episode_ids"] != expected_ids
            or stored["recorded_history_sha256"] != branches[0].prefix_hash_by_tick[tick]
            or stored["features"] != packet["features"]
        ):
            raise ValueError("teacher continuation episode or recorded sensor input differs")
    return len(phases)


def audit(out):
    manifest = read_json(out / "manifest.json")
    if manifest["schema"] != SCHEMA:
        raise ValueError("unsupported portable schedule schema")
    actual_files = {
        str(p.relative_to(out))
        for p in out.rglob("*")
        if p.is_file() and p not in (out / "manifest.json", out / "audit.json")
    }
    if actual_files != {r["path"] for r in manifest["files"]}:
        raise ValueError("portable file inventory changed")
    for item in manifest["files"]:
        path = safe_path(out, item["path"])
        if digest(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise ValueError("portable artifact changed: " + item["path"])
        if (
            path.suffix in (".pkl", ".pt", ".onnx", ".safetensors")
            or path.name == "loaded_reference_bank.npz"
        ):
            raise ValueError("raw motion/model redistribution is forbidden in this archive")
        if path.suffix == ".npz" and any(
            a.dtype.kind not in "biufcUS" for a in array_file(path).values()
        ):
            raise ValueError("object arrays are forbidden")
    banks = {
        key: TimedOptionBank(read_json(safe_path(out, value["portable"])), True)
        for key, value in manifest["registries"].items()
    }
    episode_map, identities = {}, set()
    total_frames = total_steps = eligible = 0
    for episode in manifest["episodes"]:
        identity = episode["source_trajectory"]["path"]
        if identity in identities:
            raise ValueError("same actual capture counted twice")
        identities.add(identity)
        folder = safe_path(out, episode["directory"])
        bank = banks[episode["registry_id"]]
        payload = portable_payload(folder)
        interface = portable_interface(folder, bank)
        observations = interface["observations"]
        alignment = audit_sensor_alignment(payload, observations, reference_frames=bank.frame_count)
        if alignment != read_json(folder / "sensor_alignment.json"):
            raise ValueError("portable sensor alignment differs")
        supervision = folder / "supervision"
        scene = read_json(supervision / "scene_definition.json")
        mapping = read_json(supervision / "environment_contact_mapping.json")
        net = array_file(supervision / "all_body_contacts.npz")
        pairs = array_file(supervision / "environment_pair_contacts.npz")
        contact = audit_environment_contacts(
            pairs,
            net,
            mapping,
            episode["physics_steps"],
            neutral_self_pairs=episode["declared_neutral_self_pairs"],
            beam_paths=scene_beam_paths(scene),
        )
        forces, sync = synchronize_pair_beam_forces(
            pairs, mapping, scene_beam_paths(scene), net["control_steps"]
        )
        if (
            contact != episode["assessment"]["contact_audit"]
            or sync != episode["assessment"]["beam_contact_synchronization"]
        ):
            raise ValueError("portable native counterpart ordering/contact outcome differs")
        shapes = read_json(supervision / "native_collision_inventory.json")["shapes"]
        if audit_scene_geometry(shapes, scene) != episode["assessment"]["imported_geometry_audit"]:
            raise ValueError("portable imported geometry differs")
        schedule = audit_executed_schedule(
            bank,
            observations,
            interface["switches"],
            episode["mode"],
            episode["configured_forced_option_id"],
        )
        passage = score_scene(payload, forces, scene, bank, commands_valid=schedule["valid"])
        raw_interface = read_json(supervision / "raw_interface.json")
        runtime = read_json(supervision / "success_manifest.json")
        source_bank = read_json(supervision / "loaded_reference_bank.json")
        source_bound = (
            runtime["checkpoint_hash"] == bank.request["controller"]["sha256"]
            and runtime["capture_context"]["motion"]["hash"]
            == bank.request["references"][0]["motion"]["sha256"]
            and runtime["capture_context"]["use_encoder"] == "g1"
            and max(source_bank["source_route_max_error_m"]) <= 1e-4
            and raw_interface["loaded_reference_ids"]
            == [r["reference_id"] for r in bank.request["references"]]
            and raw_interface["loaded_reference_frames"]
            == [bank.frame_count] * len(bank.request["references"])
        )
        unchanged = all(
            all(
                r["transition"][key]
                for key in ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
            )
            for r in observations
        )
        admitted = bool(
            source_bound
            and (unchanged or "outcome" in episode["assessment"])
            and all(alignment["packet_eligible"])
            and len(observations) == bank.frame_count - 1
            and contact["complete_synchronized_streams"]
            and np.array_equal(net["control_steps"], [r["physics_step"] for r in observations])
        )
        if admitted != episode["assessment"]["measurement_admitted"]:
            raise ValueError("portable measurement admission differs from recorded clocks/bindings")
        stable = bool(
            (np.asarray(payload["root_pos_w"])[:, 2] >= 0.5).all()
            and (-np.asarray(payload["projected_gravity_b"])[:, 2] >= 0.5).all()
        )
        passed = bool(
            admitted
            and passage["pass"]
            and stable
            and schedule["valid"]
            and contact["no_undesired_measured_contact"]
        )
        finish = passage["passage_finish_frame_exclusive"]
        if (
            schedule != episode["assessment"]["schedule_audit"]
            or passage != episode["assessment"]["passage"]
            or passed != episode["assessment"]["pass"]
            or ((finish - 1) / 50 if passed else None)
            != episode["assessment"]["costs"]["passage_time_s"]
        ):
            raise ValueError("portable physical passage/schedule label or cost differs")
        if "outcome" in episode["assessment"]:
            outcome = classify_timed_attempt(
                payload,
                observations,
                reference_frames=bank.frame_count,
                phase_ticks=schedule_layout(bank)[0].tolist(),
                exit_status=0,
                measurement_admitted=admitted,
                passage=passage,
                contact_audit=contact,
                schedule_audit=schedule,
                physics_steps=pairs["physics_steps"],
            )
            if outcome != episode["assessment"]["outcome"]:
                raise ValueError(
                    "portable task/measurement/phase-availability classification differs"
                )
        episode_map[episode["episode_id"]] = (episode, payload, interface)
        total_frames += len(observations)
        total_steps += episode["physics_steps"]
        eligible += sum(alignment["packet_eligible"])
    decisions = sum(
        verify_teacher_group(t, episode_map, banks[t["registry_id"]])
        for t in read_json(out / "teachers.json")
    )
    counts = dict(
        episodes=len(episode_map),
        physical_control_frames=total_frames,
        physics_steps=total_steps
        + sum(f["assessment"].get("physics_steps") or 0 for f in manifest["failed_attempts"]),
        complete_episode_physics_steps=total_steps,
        failed_attempt_physics_steps=sum(
            f["assessment"].get("physics_steps") or 0 for f in manifest["failed_attempts"]
        ),
        failed_attempts_with_unknown_steps=sum(
            f["assessment"].get("physics_steps") is None for f in manifest["failed_attempts"]
        ),
        eligible_sensor_packets=eligible,
        teacher_decisions=decisions,
        failed_attempts=len(manifest["failed_attempts"]),
    )
    if counts != manifest["counts"]:
        raise ValueError("portable aggregate counts differ")
    return dict(
        schema=SCHEMA,
        **counts,
        all_hashes_valid=True,
        pickle_free=True,
        teacher_continuations_recomputed=True,
        beam_counterparts_recomputed=True,
    )


def export(out, collections, teachers_path=None, archive=False):
    if out.exists():
        raise FileExistsError("new immutable dataset version required")
    if not collections or len(set(p.resolve() for p in collections)) != len(collections):
        raise ValueError("distinct nonempty collection paths required")
    out.mkdir(parents=True)
    provenance = out / "provenance"
    provenance.mkdir()
    registries, episodes, teachers, failures, identities = {}, [], [], [], set()
    supplied = read_json(teachers_path) if teachers_path is not None else None
    source_results = []
    for study_index, result_path in enumerate(collections):
        result = read_json(result_path)
        manifest_path = checked(result["manifest"])
        manifest, bank, scene = verify_manifest(manifest_path.parent, execution=False)
        source_results.append(ref(result_path))
        registry_id = manifest["registry"]["sha256"]
        if registry_id not in registries:
            target = provenance / f"registry_{len(registries):02d}.json"
            write_json(target, bank.definition)
            registries[registry_id] = dict(
                source=manifest["registry"], portable=str(target.relative_to(out))
            )
        study_folder = provenance / f"study_{study_index:03d}"
        study_folder.mkdir()
        write_json(study_folder / "manifest.json", manifest)
        write_json(study_folder / "result.json", result)
        if manifest.get("script_parameters"):
            write_json(
                study_folder / "script_parameters.json",
                read_json(checked(manifest["script_parameters"])),
            )
        for dependency in manifest["dependencies"]:
            original = checked(dependency["snapshot"])
            target = study_folder / "source" / Path(dependency["path"]).relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original.read_bytes())
        episode_ids = {}
        cell_map = {cell["cell_id"]: cell for cell in manifest["cells"]}
        if (
            len(cell_map) != len(manifest["cells"])
            or len(result["rows"]) != len(cell_map)
            or {r["cell_id"] for r in result["rows"]} != set(cell_map)
        ):
            raise ValueError("one result assessment per registered cell required")
        for recorded in result["rows"]:
            cell = cell_map[recorded["cell_id"]]
            if recorded.get("status") in ("failed_attempt", "invalid_or_partial_measurement"):
                partial_folder = study_folder / f'failed_{cell["cell_id"]}'
                partial_folder.mkdir()
                write_json(partial_folder / "assessment.json", recorded)
                raw_folder = Path(cell["output"]) / "trajectories"
                raw_recording = raw_folder / "aborted_raw_recording.pkl"
                if raw_recording.exists():
                    source_ref = ref(raw_recording)
                    with checked(source_ref).open("rb") as handle:
                        raw_frames = pickle.load(
                            handle
                        )  # noqa: S301 - local registered recorder output
                    if raw_frames is not None:
                        metadata, excluded = save_pickle_free(
                            partial_folder / "aborted_raw_recording.npz", raw_frames
                        )
                        if excluded:
                            raise ValueError("unhandled partial physical arrays: " + str(excluded))
                        write_json(partial_folder / "aborted_raw_recording_metadata.json", metadata)
                    write_json(partial_folder / "aborted_source_binding.json", source_ref)
                for source in raw_folder.glob("*.npz"):
                    if source.name != "loaded_reference_bank.npz":
                        copy_npz(source, partial_folder / source.name)
                for source in raw_folder.glob("*.trajectory.pkl"):
                    with checked(ref(source)).open("rb") as handle:
                        raw_payload = pickle.load(handle)  # noqa: S301 - trusted bound capture.
                    metadata, excluded = save_pickle_free(
                        partial_folder / "partial_trajectory.npz", raw_payload
                    )
                    if excluded:
                        raise ValueError("unhandled partial physical fields")
                    write_json(partial_folder / "partial_trajectory_metadata.json", metadata)
                    write_json(partial_folder / "partial_trajectory_source.json", ref(source))
                for source in raw_folder.glob("*environment_contact_mapping.json"):
                    write_json(partial_folder / source.name, read_json(source))
                if recorded.get("partial"):
                    write_json(
                        partial_folder / "partial_interface.json",
                        read_json(checked(recorded["partial"])),
                    )
                failures.append(
                    dict(assessment=recorded, directory=str(partial_folder.relative_to(out)))
                )
                continue
            validate_invocation(cell, recorded, scene)
            verified, payload, interface = analyze_cell(cell, manifest, bank, scene)
            for key in ARTIFACT_KEYS + ("pass", "measurement_admitted", "costs", "schedule_audit"):
                if recorded[key] != verified[key]:
                    raise ValueError("source result differs from physical re-audit: " + key)
            trajectory = checked(verified["trajectory"])
            identity = str(trajectory.resolve())
            if identity in identities:
                raise ValueError("same actual recording supplied in multiple collections")
            identities.add(identity)
            folder = out / "episodes" / f"episode_{len(episodes):04d}"
            folder.mkdir(parents=True)
            metadata, excluded = save_pickle_free(folder / "trajectory.npz", payload)
            if excluded:
                raise ValueError("unhandled physical recording arrays: " + str(excluded))
            write_json(folder / "trajectory_metadata.json", metadata)
            write_json(folder / "sensor_history.json", causal_packets(interface))
            write_json(folder / "sensor_alignment.json", verified["sensor_alignment"])
            source_folder = trajectory.parent
            export_inputs(
                source_folder / "timed_schedule_features.npz",
                folder / "student_inputs.npz",
                interface,
            )
            write_json(
                folder / "commands.json",
                {
                    **executed_schedule(interface),
                    "records": [
                        {key: value for key, value in row.items() if key in COMMAND_KEYS}
                        for row in interface["observations"]
                    ],
                },
            )
            supervision = folder / "supervision"
            supervision.mkdir()
            write_json(supervision / "assessment.json", verified)
            write_json(supervision / "scene_definition.json", scene)
            write_json(supervision / "raw_interface.json", interface)
            for name in ("attempt.json", "success_manifest.json"):
                write_json(supervision / name, read_json(source_folder.parent / name))
            for name in CONTACT_NAMES:
                copy_npz(source_folder / name, supervision / name)
            for name in (
                "environment_contact_mapping.json",
                "native_collision_inventory.json",
                "loaded_reference_bank.json",
            ):
                write_json(supervision / name, read_json(source_folder / name))
            write_json(
                supervision / "source_bindings.json", {key: verified[key] for key in ARTIFACT_KEYS}
            )
            episode = dict(
                episode_id=folder.name,
                directory=str(folder.relative_to(out)),
                original_cell_id=cell["cell_id"],
                configured_forced_option_id=cell["forced_option_id"],
                mode=cell["timed_schedule_mode"],
                physics_seed=cell["runtime_seed"],
                registry_id=registry_id,
                source_trajectory=ref(trajectory),
                source_sensor=verified["sensor"],
                source_features=verified["features"],
                collection_sha256=ref(result_path)["sha256"],
                split=manifest["split"],
                beam_count=len(scene["beams"]),
                beam_paths=scene_beam_paths(scene),
                loaded_reference_frames=interface["loaded_reference_frames"],
                physical_control_frames=len(interface["observations"]),
                physics_steps=verified["physics_steps"],
                eligible_sensor_packets=sum(verified["sensor_alignment"]["packet_eligible"]),
                declared_neutral_self_pairs=manifest["declared_neutral_self_pairs"],
                assessment=verified,
            )
            episodes.append(episode)
            episode_ids[cell["forced_option_id"]] = folder.name
        if (
            set(episode_ids) == set(bank.option_ids)
            and all(c["timed_schedule_mode"] == "forced" for c in manifest["cells"])
            and manifest["split"] == "development"
        ):
            teacher = audit_collection(result_path, bank, manifest["registry"])
            if supplied is not None:
                matches = [
                    group for group in supplied if group["collection"] == teacher["collection"]
                ]
                if len(matches) != 1 or matches[0] != teacher:
                    raise ValueError(
                        "supplied phase teacher differs from independent actual re-audit"
                    )
            ids = [episode_ids[option] for option in bank.option_ids]
            teacher = copy.deepcopy(teacher)
            teacher.update(
                episode_ids=ids,
                registry_id=registry_id,
                recorded_history_group=[
                    registry_id,
                    manifest["scene_definition"]["sha256"],
                    teacher["physics_seed"],
                ],
            )
            for target in teacher["targets"]:
                target["continuation_episode_ids"] = [
                    None if i is None else ids[i] for i in target["continuation_option_indices"]
                ]
            teachers.append(teacher)
    write_json(out / "teachers.json", teachers)
    for source in sorted(closure([Path(__file__).resolve()])):
        target = provenance / "exporter_source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    counts = dict(
        episodes=len(episodes),
        physical_control_frames=sum(e["physical_control_frames"] for e in episodes),
        physics_steps=sum(e["physics_steps"] for e in episodes)
        + sum(f["assessment"].get("physics_steps") or 0 for f in failures),
        complete_episode_physics_steps=sum(e["physics_steps"] for e in episodes),
        failed_attempt_physics_steps=sum(
            f["assessment"].get("physics_steps") or 0 for f in failures
        ),
        failed_attempts_with_unknown_steps=sum(
            f["assessment"].get("physics_steps") is None for f in failures
        ),
        eligible_sensor_packets=sum(e["eligible_sensor_packets"] for e in episodes),
        teacher_decisions=sum(len(t["targets"]) for t in teachers),
        failed_attempts=len(failures),
    )
    (out / "DATA_CARD.md").write_text(
        "# Repeated timed-schedule humanoid recordings\n\n"
        "Actual finite reference executions, causal65-ray sensor histories and exact100+2K inputs. "
        "For seven schedules the input is114D, with2s/101-frame history. "
        "Geometry, contacts, outcome labels and WAIT continuation identities are separate supervision. "
        "Each WAIT target names the best verified complete future schedule from the matched recorded prefix. "
        "Missing branches never become failure labels. One or two beam counterparts are all recorded at200Hz. "
        "Full captures and failed passages remain. The archive excludes pretrained weights "
        "and raw full prior/reference banks; "
        "recorded reference commands remain in trajectory data. Original hashes and "
        "immutable source snapshots are provenance. "
        "Portable audit recomputes clocks, contact ordering, passage, complete schedules "
        "and WAIT targets without external assets. "
        "This is not new physical qualification, hidden-state snapshot evidence, "
        "hardware transfer or repeated adaptation.\n"
    )
    files = [
        {**ref(p), "path": str(p.relative_to(out))} for p in sorted(out.rglob("*")) if p.is_file()
    ]
    write_json(
        out / "manifest.json",
        dict(
            schema=SCHEMA,
            registries=registries,
            episodes=episodes,
            failed_attempts=failures,
            counts=counts,
            source_results=source_results,
            supplied_teachers=ref(teachers_path) if teachers_path else None,
            files=files,
        ),
    )
    report = audit(out)
    write_json(out / "audit.json", report)
    if archive:
        destination = out.with_suffix(".tar.gz")
        if destination.exists():
            raise FileExistsError("preserve existing portable archive")
        with tarfile.open(destination, "w:gz") as handle:
            handle.add(out, arcname=out.name)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "audit"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collections", type=Path, nargs="+")
    parser.add_argument("--teachers", type=Path)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if args.action == "export":
        export(args.out, args.collections, args.teachers, args.archive)
    else:
        print(json.dumps(audit(args.out)))
