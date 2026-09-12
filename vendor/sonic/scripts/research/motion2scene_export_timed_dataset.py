#!/usr/bin/env python3
"""Export six-second development recordings with explicit contact criteria and clocks."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gear_sonic.dataset_generation.hallucination.motion2scene_environment_contacts import (  # noqa: E402
    audit_environment_contacts,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_sensor_alignment import (  # noqa: E402
    audit_sensor_alignment,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_history_policy import (  # noqa: E402
    SCHEMA as TIMED_SCHEMA,
    expected_feature_names,
)
from scripts.research.motion2scene_export_option_dataset import (  # noqa: E402
    copy_npz,
    save_pickle_free,
    sensor_packets,
)
from scripts.research.motion2scene_export_traversal_dataset import (  # noqa: E402
    checked,
    digest,
    write_json,
)

SCHEMA = "motion2scene_six_second_development_dataset_v1"
DATA = ROOT.parent / "research-data/groot-wbc"
DEFAULT_RESULTS = [
    DATA / "m2s-long-schedule-qualification-v3/result.json",
    DATA / "m2s-neutral-contact-counterpart-diagnostic-v1/result.json",
    DATA / "m2s-environment-contact-qualification-v1/environment_result.json",
]
CONTACT_ARRAYS = (
    "all_body_contacts.npz",
    "physics_beam_contacts.npz",
    "beam_contacts.npz",
    "option_mechanical_work.npz",
    "environment_pair_contacts.npz",
    "pair_resolved_contacts.npz",
)
SUPERVISION_JSON = (
    "qualification_evidence.json",
    "environment_qualification_evidence.json",
    "environment_contact_mapping.json",
    "contact_counterpart_mapping.json",
    "environment_contact_audit.json",
    "all_contact_audit.json",
    "prefix_audit.json",
    "clock_audit.json",
    "geometry_audit.json",
    "native_collision_inventory.json",
    "scene_metadata.json",
    "loaded_reference_bank.json",
)
COMMAND_KEYS = {
    "tick",
    "time_s",
    "phase_s",
    "physics_step",
    "active_before",
    "active",
    "forced_option_id",
    "transition",
    "selected_option_id",
    "policy_values",
    "legal_mask",
}
FEATURE_KEYS = {
    "schema_version",
    "feature_names",
    "option_ids",
    "features",
    "command_ticks",
    "capture_elapsed_s",
    "legal_mask",
}
FORBIDDEN_SENSOR_KEYS = {
    "scene",
    "beam",
    "object_id",
    "prim_path",
    "rigid_body",
    "collision",
    "requested",
    "selected_option_id",
    "pass",
    "features",
    "policy_values",
    "occupied",
}


def ref(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": digest(path), "size_bytes": path.stat().st_size}


def read_json(path):
    return json.loads(Path(path).read_text())


def array_file(path):
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k].copy() for k in values.files}


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Portable path escapes package")
    return path


def source_identity(trajectory):
    """Actual capture identity is its resolved source path, not deterministic byte content."""
    return str(Path(trajectory).resolve())


def collect_sources(paths):
    records, studies, pending = {}, {}, list(paths)
    while pending:
        result_path = Path(pending.pop(0)).resolve()
        if str(result_path) in studies:
            continue
        result = read_json(result_path)
        manifest_path = checked(result["manifest"])
        manifest = read_json(manifest_path)
        studies[str(result_path)] = {
            "result": result,
            "manifest": manifest,
            "result_ref": ref(result_path),
            "manifest_ref": ref(manifest_path),
        }
        alias = result.get("old_strict_application_to_new_captures")
        if alias:
            pending.append(checked(alias))
        if result["schema"] == "motion2scene_contact_counterpart_diagnostic_result_v1":
            rows = [
                {
                    "cell_id": manifest["cells"][0]["cell_id"],
                    "trajectory": result["raw"]["000000.trajectory.pkl"],
                    "diagnostic_result": result,
                    "physics_steps": result["physics_steps"],
                }
            ]
        else:
            rows = result["rows"]
        for row in rows:
            cells = [cell for cell in manifest["cells"] if cell["cell_id"] == row["cell_id"]]
            if len(cells) != 1:
                raise ValueError("Result row requires one exact registered cell")
            cell = cells[0]
            if row.get("status") == "failed_attempt" and "trajectory" not in row:
                # Preserve infrastructure/partial attempts in study records; do not invent a rollout label.
                studies[str(result_path)].setdefault("infrastructure_attempts", []).append(row)
                continue
            evidence = read_json(checked(row["evidence"])) if "evidence" in row else None
            trajectory_ref = row.get("trajectory") or evidence["artifacts"]["trajectory"]
            trajectory = checked(trajectory_ref)
            if trajectory.parent != (Path(cell["output"]) / "trajectories").resolve():
                raise ValueError("Trajectory is not from the registered cell output")
            interface_ref = (
                row.get("sensor")
                or (evidence or {}).get("artifacts", {}).get("interface")
                or result.get("raw", {}).get("reactive_interface.json")
            )
            interface_path = (
                checked(interface_ref)
                if interface_ref
                else trajectory.parent / "reactive_interface.json"
            )
            if interface_path.resolve().parent != trajectory.parent:
                raise ValueError("Interface source differs from trajectory source")
            # All direct row/evidence measurement hashes are checked before parsing arrays.
            for item in (evidence or {}).get("artifacts", {}).values():
                if isinstance(item, dict) and {"path", "sha256"}.issubset(item):
                    checked(item)
            for item in row.values():
                if isinstance(item, dict) and {"path", "sha256"}.issubset(item):
                    checked(item)
            identity = source_identity(trajectory)
            assessment = {
                "result_ref": ref(result_path),
                "result_schema": result["schema"],
                "row": row,
            }
            if identity in records:
                record = records[identity]
                if (
                    record["trajectory_ref"]["sha256"] != trajectory_ref["sha256"]
                    or record["cell"] != cell
                ):
                    raise ValueError("Conflicting source identity or registered cell")
                record["assessments"].append(assessment)
            else:
                records[identity] = {
                    "trajectory_ref": ref(trajectory),
                    "interface_ref": ref(interface_path),
                    "cell": cell,
                    "manifest_ref": ref(manifest_path),
                    "manifest": manifest,
                    "assessments": [assessment],
                    "primary_result": str(result_path),
                }
    return list(records.values()), studies


def causal_packets(interface):
    packets = sensor_packets(interface)
    for packet, original in zip(packets, interface["observations"], strict=True):
        for key in ("tick", "phase_s", "physics_step", "joint_names", "normal_known_mask"):
            if key in original:
                packet[key] = original[key]
        if "normal_known_mask" in packet:
            measurements = packet.get("measurements", [])
            if len(packet["normal_known_mask"]) != len(measurements):
                raise ValueError("Normal mask and measurements disagree")
            expected = [ray.get("hit_normal_w") is not None for ray in measurements]
            if packet["normal_known_mask"] != expected:
                raise ValueError("Normal-known mask differs from measured returns")
    return packets


def executed_schedule(interface):
    records = interface["observations"]
    switches = interface.get("switches", [])
    state, entered = "neutral", []
    by_tick = {}
    for event in switches:
        if event["tick"] in by_tick or event["from"] != state:
            raise ValueError("Duplicate or disconnected actual switch log")
        by_tick[event["tick"]] = event
        state = event["to"]
        if state != "neutral":
            entered.append(state)
    state = "neutral"
    for record in records:
        if record["active_before"] != state:
            raise ValueError("Actual state before update differs from switch history")
        if record["tick"] in by_tick:
            state = by_tick[record["tick"]]["to"]
        if record["active"] != state:
            raise ValueError("Actual state after update differs from switch history")
    return {
        "configured_forced_option_id": interface.get("forced_option_id"),
        "loaded_reference_ids": interface.get("loaded_reference_ids"),
        "registered_option_ids": interface.get("option_ids"),
        "actual_entered_option_ids": entered,
        "no_entry_neutral": not entered,
        "actual_final_option_id": state,
        "switches": switches,
        "scope": "Actual action derives from switch/state logs; configured preference is separate",
    }


def audit_contact_streams(folder, payload, expected_steps, neutral_self_pairs=()):
    net = array_file(folder / "all_body_contacts.npz")
    beam = array_file(folder / "physics_beam_contacts.npz")
    control_count = len(payload["motion_time_s"])
    clock = np.arange(1, expected_steps + 1)
    if not (
        np.array_equal(net["physics_steps"], clock)
        and np.array_equal(beam["physics_steps"], clock)
        and np.array_equal(net["control_steps"], np.arange(4, expected_steps + 1, 4))
        and np.array_equal(beam["control_steps"], net["control_steps"])
        and len(net["control_steps"]) == control_count
        and float(net["physics_dt_s"]) == 0.005
        and float(beam["physics_dt"]) == 0.005
    ):
        raise ValueError("Physical step and control clocks are incomplete/inconsistent")
    if net["net_force_w"].shape != (expected_steps, 30, 3) or beam["force_w"].shape != (
        expected_steps,
        30,
        3,
    ):
        raise ValueError("Expected measured200Hz thirty-body streams")
    if not np.array_equal(net["body_names"], payload["contact_body_names"]):
        raise ValueError("Native body order differs from trajectory contact order")
    if not np.array_equal(net["net_force_w"][3::4], payload["robot_contact_force_w"]):
        raise ValueError("Recorded physical and control contact samples disagree")
    report = {
        "physics_steps": expected_steps,
        "control_frames": control_count,
        "physical_and_control_contact_clocks_match": True,
        "environment": None,
        "counterpart_diagnostic": None,
    }
    pairs_path = folder / "environment_pair_contacts.npz"
    if pairs_path.exists():
        pairs = array_file(pairs_path)
        mapping = read_json(folder / "environment_contact_mapping.json")
        audit = audit_environment_contacts(
            pairs, net, mapping, expected_steps, neutral_self_pairs=list(neutral_self_pairs)
        )
        if not audit["complete_synchronized_streams"]:
            raise ValueError("Invalid thirty-body environment counterpart mapping or streams")
        report["environment"] = audit
    diagnostic = folder / "pair_resolved_contacts.npz"
    if diagnostic.exists():
        pairs = array_file(diagnostic)
        mapping = read_json(folder / "contact_counterpart_mapping.json")
        if (
            not np.array_equal(pairs["physics_steps"], clock)
            or float(pairs["physics_dt_s"]) != 0.005
        ):
            raise ValueError("Diagnostic counterpart clock mismatch")
        subjects = [k for k in pairs if k not in ("physics_steps", "physics_dt_s")]
        native = mapping["all_body_sensor"]
        if (
            native["body_names"] != list(net["body_names"])
            or native["native_body_paths"]
            != [f"/World/envs/env_0/Robot/{body}" for body in net["body_names"]]
            or len(native["robot_articulation_body_names"]) != 30
            or set(native["robot_articulation_body_names"]) != set(net["body_names"])
        ):
            raise ValueError("Diagnostic native articulation/net body mapping differs")
        rows = []
        for name in subjects:
            actual = mapping[name]
            expected_paths = [
                f"/World/envs/env_0/Robot/{body}"
                for body in mapping["all_body_sensor"]["robot_articulation_body_names"]
            ] + [
                f"/World/ground/terrain/Structure/{name}"
                for name in ["Floor", "WallWest", "WallEast", "WallSouth", "WallNorth"]
            ]
            if (
                actual["sensor_body_names"] != [name]
                or actual["native_filter_count"] != 35
                or actual["filter_paths_in_matrix_column_order"] != expected_paths
                or pairs[name].shape != (expected_steps, 35, 3)
            ):
                raise ValueError(
                    "Diagnostic subject/counterpart columns differ from native mapping"
                )
            index = list(net["body_names"]).index(name)
            error = float(abs(pairs[name].sum(axis=1) - net["net_force_w"][:, index]).max())
            if error > 0.001:
                raise ValueError("Diagnostic counterpart sum does not reconstruct native net")
            rows.append(
                {"subject": name, "maximum_component_residual_n": error, "counterparts": 35}
            )
        report["counterpart_diagnostic"] = rows
    return report


def export_features(folder, source_folder, interface, alignment):
    source = source_folder / "timed_history_features.npz"
    if not source.exists():
        return {
            "schema": "legacy_sparse_rays_and_measured_state_only",
            "dimension": None,
            "scope": "No214D,100D,110D or106D learned-policy feature vector is invented",
        }
    values = copy_npz(source, folder / "student_inputs.npz", allowed=FEATURE_KEYS)
    names = tuple(values["feature_names"])
    expected = expected_feature_names()
    if (
        str(values["schema_version"]) != TIMED_SCHEMA
        or names != expected
        or len(names) != 106
        or tuple(interface["feature_names"]) != expected
        or interface["feature_schema"] != TIMED_SCHEMA
        or values["features"].shape != (len(interface["observations"]), 106)
        or not np.array_equal(
            values["features"], [row["features"] for row in interface["observations"]]
        )
        or not np.array_equal(
            values["command_ticks"], [row["tick"] for row in interface["observations"]]
        )
        or not np.array_equal(
            values["legal_mask"], [row["legal_mask"] for row in interface["observations"]]
        )
        or list(values["option_ids"]) != interface["option_ids"]
        or not np.isfinite(values["features"]).all()
    ):
        raise ValueError("Exact106D causal feature schema/recording mismatch")
    return {
        "schema": TIMED_SCHEMA,
        "dimension": 106,
        "rows": len(values["features"]),
        "eligible_rows": sum(alignment["packet_eligible"]),
        "source": ref(source),
        "option_ids": values["option_ids"].tolist(),
        "scope": "Sensor history, measured robot state, tick/active and legal mask; no scene geometry or labels",
    }


def prepare_record(record):
    source_folder = checked(record["trajectory_ref"]).parent
    payload = load_reset_capture(checked(record["trajectory_ref"]))
    interface = read_json(checked(record["interface_ref"]))
    horizons = interface["loaded_reference_frames"]
    if len(set(horizons)) != 1 or type(horizons[0]) is not int:
        raise ValueError("One measured common loaded horizon required")
    alignment = audit_sensor_alignment(
        payload, interface["observations"], reference_frames=horizons[0]
    )
    counts = {assessment["row"]["physics_steps"] for assessment in record["assessments"]}
    if len(counts) != 1:
        raise ValueError("Same physical recording has conflicting step counts")
    steps = counts.pop()
    contacts = audit_contact_streams(
        source_folder, payload, steps, record["manifest"].get("declared_neutral_self_pairs", [])
    )
    return source_folder, payload, interface, alignment, contacts


def audit(out):
    manifest = read_json(out / "manifest.json")
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported timed portable schema")
    for item in manifest["files"]:
        path = safe_path(out, item["path"])
        if digest(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise ValueError("Portable artifact changed: " + item["path"])
        if path.suffix == ".npz":
            for array in array_file(path).values():
                if array.dtype.kind not in "biufcUS":
                    raise ValueError("Object serialization is forbidden")
        if (
            path.suffix in (".pkl", ".pt", ".safetensors")
            or path.name == "loaded_reference_bank.npz"
        ):
            raise ValueError("Raw model/reference artifact must not be redistributed")
    ids = [episode["source_trajectory"]["path"] for episode in manifest["episodes"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Actual source capture identity duplicated")
    total_frames, total_steps, total_eligible = 0, 0, 0
    for episode in manifest["episodes"]:
        folder = safe_path(out, episode["directory"])
        arrays = array_file(folder / "trajectory.npz")
        alignment = read_json(folder / "sensor_alignment.json")
        packets = read_json(folder / "sensor_history.json")
        if len(packets) != len(arrays["motion_time_s"]) or len(alignment["packet_eligible"]) != len(
            packets
        ):
            raise ValueError("Portable sensor/physical/alignment counts differ")

        def check_input(value):
            if isinstance(value, dict):
                if FORBIDDEN_SENSOR_KEYS.intersection(value):
                    raise ValueError("Privileged labels/identities in causal sensors")
                for child in value.values():
                    check_input(child)
            elif isinstance(value, list):
                for child in value:
                    check_input(child)

        check_input(packets)
        metadata = read_json(folder / "trajectory_metadata.json")
        payload = {**arrays, "fps": metadata["fps"]}
        actual_alignment = audit_sensor_alignment(
            payload, packets, reference_frames=episode["loaded_reference_frames"][0]
        )
        if actual_alignment != alignment:
            raise ValueError("Portable sensor alignment does not reproduce from recorded values")
        contact = audit_contact_streams(folder / "supervision", payload, episode["physics_steps"])
        if not contact["physical_and_control_contact_clocks_match"]:
            raise ValueError("Portable physical contact alignment failed")
        if episode["student"]["dimension"] == 106:
            inputs = array_file(folder / "student_inputs.npz")
            if (
                set(inputs) != FEATURE_KEYS
                or tuple(inputs["feature_names"]) != expected_feature_names()
            ):
                raise ValueError("Portable106D input contract changed")
            if inputs["features"].shape != (len(packets), 106):
                raise ValueError("Portable input rows differ")
        total_frames += len(packets)
        total_steps += episode["physics_steps"]
        total_eligible += sum(alignment["packet_eligible"])
    episode_map = {episode["episode_id"]: episode for episode in manifest["episodes"]}
    for teacher in read_json(out / "teachers.json"):
        if len(teacher["episode_ids"]) != len(teacher["option_ids"]):
            raise ValueError("Portable teacher branches incomplete")
        reference_features = None
        for i, identifier in enumerate(teacher["episode_ids"]):
            episode = episode_map[identifier]
            folder = safe_path(out, episode["directory"])
            values = array_file(folder / "student_inputs.npz")
            indices = np.flatnonzero(values["command_ticks"] == teacher["entry_tick"])
            if len(indices) != 1:
                raise ValueError("Portable teacher must bind a unique real entry input")
            index = int(indices[0])
            eligible = read_json(folder / "sensor_alignment.json")["packet_eligible"][index]
            if teacher["admitted"][i] and not eligible:
                raise ValueError("Teacher labels cannot admit an ineligible observation")
            if teacher["admitted"][i] and not np.array_equal(
                values["features"][index], teacher["features"]
            ):
                raise ValueError("Admitted teacher target is not its recorded input")
            if (
                teacher["exact_shared_entry_observation"]
                and reference_features is not None
                and not np.array_equal(reference_features, values["features"][index])
            ):
                raise ValueError("Teacher alternatives have different actual student inputs")
            reference_features = values["features"][index]
            if not np.array_equal(values["legal_mask"][index], teacher["legality"]):
                raise ValueError("Teacher legal mask is not its recorded mask")
            labels = [
                c for c in episode["criteria"] if c["criterion"] == "timed_closed_loop_passage"
            ]
            if len(labels) != 1 or labels[0]["pass"] != teacher["passed"][i]:
                raise ValueError("Teacher passage label differs from actual complete episode")
            if labels[0]["costs"]["passage_time_s"] != teacher["passage_time_s"][i]:
                raise ValueError("Teacher cost differs from measured episode cost")
    expected = manifest["counts"]
    if (
        total_frames != expected["physical_control_frames"]
        or total_steps != expected["physics_steps"]
        or total_eligible != expected["eligible_sensor_packets"]
    ):
        raise ValueError("Portable aggregate accounting differs")
    return {
        "schema": SCHEMA,
        "episodes": len(ids),
        **expected,
        "hashed_files": len(manifest["files"]),
        "all_hashes_valid": True,
        "pickle_free": True,
        "source_identity_unique": True,
    }


def export_comparison_suite(result_path, studies, target, out):
    """Bundle an exact completed parent comparison without counting its rows twice."""
    suite = read_json(result_path)
    registration = read_json(checked(suite["registration"]))
    plan = read_json(checked(registration["plan"]))
    if plan.get("schema") != "motion2scene_timed_policy_comparison_v1":
        raise ValueError("Unsupported explicit comparison suite")
    groups = suite["groups"]
    registered = {str(checked(c["manifest"])): c for c in registration["children"]}
    manifests = [str(checked(g["manifest"])) for g in groups]
    result_paths = [str(checked(g["result"])) for g in groups]
    if (
        len(registered) != len(registration["children"])
        or len(manifests) != len(set(manifests))
        or set(manifests) != set(registered)
        or len(result_paths) != len(set(result_paths))
        or set(result_paths) != set(studies)
        or suite["episodes"] != plan["expected_episodes"]
        or suite["physics_steps"] != plan["expected_physics_steps"]
    ):
        raise ValueError("Complete registered comparison and exported child set must match")
    for group, manifest_path, child_path in zip(groups, manifests, result_paths):
        child = studies[child_path]
        if (
            str(checked(child["result"]["manifest"])) != manifest_path
            or group["rows"] != child["result"]["rows"]
            or group["physics_steps"] != child["result"]["physics_steps"]
            or group["mode"] != registered[manifest_path]["mode"]
            or group["scene_index"] != registered[manifest_path]["scene_index"]
            or group["scene_definition"] != registered[manifest_path]["scene_definition"]
        ):
            raise ValueError("Parent comparison differs from registered actual child evidence")
    if sum(g["physics_steps"] for g in groups) != suite["physics_steps"]:
        raise ValueError("Parent comparison physical accounting differs from children")
    target.mkdir(parents=True)
    write_json(target / "result.json", suite)
    write_json(target / "registration.json", registration)
    write_json(target / "plan.json", plan)
    sources = []
    for dependency in plan["implementation"]:
        source = checked(dependency["snapshot"])
        if digest(source) != dependency["sha256"]:
            raise ValueError("Comparison implementation snapshot differs")
        relative = Path(dependency["path"]).relative_to(ROOT)
        destination = target / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        sources.append({"original": dependency, "portable": str(destination.relative_to(out))})
    write_json(target / "source_mapping.json", sources)
    return {
        "result": ref(result_path),
        "registration": suite["registration"],
        "plan": registration["plan"],
        "portable": str(target.relative_to(out)),
        "counted_as_additional_episodes": False,
    }


def export(out, paths, make_archive, suite_results=()):
    records, studies = collect_sources(paths)
    if out.exists():
        raise FileExistsError("Preserve immutable previous dataset; use a fresh version")
    out.mkdir(parents=True)
    provenance = out / "provenance"
    provenance.mkdir()
    episodes = []
    from scripts.research.bundle_motion2scene_sources import closure

    for path in sorted(closure([Path(__file__).resolve()])):
        target = provenance / "exporter_source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    for index, record in enumerate(records):
        source_folder, payload, interface, alignment, contact_audit = prepare_record(record)
        folder = out / "episodes" / f"episode_{index:04d}"
        folder.mkdir(parents=True)
        metadata, excluded = save_pickle_free(folder / "trajectory.npz", payload)
        if excluded:
            raise ValueError(
                "Unsupported physical arrays must be explicitly handled: " + str(excluded)
            )
        write_json(folder / "trajectory_metadata.json", metadata)
        packets = causal_packets(interface)
        write_json(folder / "sensor_history.json", packets)
        write_json(folder / "sensor_alignment.json", alignment)
        np.savez_compressed(
            folder / "sensor_alignment.npz",
            packet_eligible=alignment["packet_eligible"],
            trajectory_phase_s=payload["motion_time_s"],
            sensor_phase_s=[row["time_s"] for row in interface["observations"]],
            confidence=np.asarray([row["confidence"] for row in alignment["packets"]]),
        )
        write_json(
            folder / "sensor_configuration.json",
            {
                key: interface[key]
                for key in (
                    "sensor",
                    "sensor_model",
                    "history_grid",
                    "feature_schema",
                    "feature_names",
                )
                if key in interface
            },
        )
        actual = executed_schedule(interface)
        write_json(
            folder / "commands.json",
            {
                **actual,
                "records": [
                    {key: value for key, value in row.items() if key in COMMAND_KEYS}
                    for row in interface["observations"]
                ],
            },
        )
        supervision = folder / "supervision"
        supervision.mkdir()
        write_json(supervision / "assessments.json", record["assessments"])
        write_json(supervision / "contact_stream_audit.json", contact_audit)
        originals = [record["trajectory_ref"], record["interface_ref"]]
        for name in CONTACT_ARRAYS:
            source = source_folder / name
            if source.exists():
                copy_npz(source, supervision / name)
                originals.append(ref(source))
        for name in SUPERVISION_JSON:
            source = source_folder / name
            if source.exists():
                write_json(supervision / name, read_json(source))
                originals.append(ref(source))
        for name in ("attempt.json", "success_manifest.json"):
            source = source_folder.parent / name
            if source.exists():
                write_json(supervision / name, read_json(source))
                originals.append(ref(source))
        write_json(
            supervision / "source_bindings.json",
            {
                "recorded_artifacts": originals,
                "manifest": record["manifest_ref"],
                "registered_cell": record["cell"],
            },
        )
        student = export_features(folder, source_folder, interface, alignment)
        criteria = []
        for assessment in record["assessments"]:
            row, schema = assessment["row"], assessment["result_schema"]
            criterion = (
                "environment_contact_qualification"
                if schema == "motion2scene_environment_contact_qualification_result_v1"
                else (
                    "strict_net_nonfoot_qualification"
                    if schema == "motion2scene_long_schedule_qualification_v1"
                    else (
                        "timed_closed_loop_passage"
                        if schema == "motion2scene_timed_history_collection_v1"
                        else "counterpart_diagnostic_only"
                    )
                )
            )
            criteria.append(
                {
                    "criterion": criterion,
                    "qualified": row.get("qualified"),
                    "pass": row.get("pass"),
                    "measurement_admitted": row.get("measurement_admitted"),
                    "checks": row.get("checks"),
                    "costs": row.get("costs"),
                    "source_result": assessment["result_ref"],
                }
            )
        episodes.append(
            {
                "episode_id": folder.name,
                "directory": str(folder.relative_to(out)),
                "split": "development_only",
                "source_trajectory": record["trajectory_ref"],
                "source_manifest": record["manifest_ref"],
                "original_cell_id": record["cell"]["cell_id"],
                "source_id": record["cell"].get("generation_seed"),
                "physics_seed": record["cell"].get("runtime_seed"),
                "physical_control_frames": len(payload["motion_time_s"]),
                "physics_steps": contact_audit["physics_steps"],
                "eligible_sensor_packets": sum(alignment["packet_eligible"]),
                "first_episode_frames": alignment["physical_first_episode_frames"],
                "loaded_reference_frames": interface["loaded_reference_frames"],
                "configured_forced_option_id": record["cell"].get("forced_option_id"),
                "configured_policy": {
                    "mode": record["cell"].get("timed_history_mode"),
                    "forced_option_id": record["cell"].get("forced_option_id"),
                    "model": record["manifest"].get("policy"),
                    "scope": "Configuration only; actual action is in executed_schedule",
                },
                "executed_schedule": actual,
                "criteria": criteria,
                "student": student,
                "mechanical_work_j": None,
                "work_scope": (
                    "Estimated implicit PD torque is retained as an estimate; "
                    "measured actuator work unavailable"
                ),
            }
        )
    teacher_rows = []
    for index, study in enumerate(studies.values()):
        write_json(provenance / f"study_{index:03d}_manifest.json", study["manifest"])
        write_json(provenance / f"study_{index:03d}_result.json", study["result"])
        source_rows = []
        for dependency in study["manifest"].get("dependencies", []):
            if "snapshot" not in dependency:
                source_rows.append({"original": dependency, "bundled": False})
                continue
            source = checked(dependency["snapshot"])
            if source.suffix not in (".py", ".sh", ".yaml", ".yml", ".json"):
                source_rows.append({"original": dependency, "bundled": False})
                continue
            relative = Path(dependency["path"]).relative_to(ROOT)
            target = provenance / f"study_{index:03d}_source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            source_rows.append(
                {"original": dependency, "bundled": True, "portable": str(target.relative_to(out))}
            )
        write_json(provenance / f"study_{index:03d}_source_mapping.json", source_rows)
        analysis_sources = []
        for dependency in study["result"].get("analysis_implementation", []):
            source = checked(dependency["snapshot"])
            if dependency["sha256"] != dependency["snapshot"]["sha256"]:
                raise ValueError("Analysis source and frozen snapshot hashes differ")
            target = (
                provenance
                / f"study_{index:03d}_analysis_source"
                / Path(dependency["path"]).relative_to(ROOT)
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            analysis_sources.append(
                {"source": dependency, "portable": str(target.relative_to(out))}
            )
        write_json(provenance / f"study_{index:03d}_analysis_source_mapping.json", analysis_sources)
        for attempt_index, attempt in enumerate(study.get("infrastructure_attempts", [])):
            target = provenance / f"study_{index:03d}_failed_attempt_{attempt_index:03d}"
            target.mkdir()
            write_json(target / "assessment.json", attempt)
            if attempt.get("attempt"):
                write_json(target / "attempt.json", read_json(checked(attempt["attempt"])))
            if attempt.get("partial"):
                partial = read_json(checked(attempt["partial"]))
                observations = partial.pop("observations", [])
                write_json(target / "partial_metadata.json", partial)
                if observations:
                    write_json(
                        target / "partial_sensor_packets.json",
                        causal_packets({"observations": observations}),
                    )
                    write_json(
                        target / "alignment_status.json",
                        {
                            "eligible": [False] * len(observations),
                            "reason": (
                                "Partial attempt lacks a validated paired physical trajectory; "
                                "retain packets without training admission"
                            ),
                        },
                    )
        if study["result"].get("teacher") is not None:
            teacher = copy.deepcopy(study["result"]["teacher"])
            mapping = {}
            for row in study["result"]["rows"]:
                if row.get("trajectory"):
                    matches = [
                        e
                        for e in episodes
                        if source_identity(e["source_trajectory"]["path"])
                        == source_identity(row["trajectory"]["path"])
                    ]
                    if len(matches) != 1:
                        raise ValueError("Teacher branch requires one actual exported episode")
                    mapping[row["forced_option_id"]] = matches[0]["episode_id"]
            if set(mapping) != set(teacher["option_ids"]):
                raise ValueError("Incomplete timed teacher branch mapping")
            teacher_rows.append(
                {
                    **teacher,
                    "episode_ids": [mapping[key] for key in teacher["option_ids"]],
                    "source_result": study["result_ref"],
                    "split": "development_only",
                }
            )
    suite_provenance = []
    seen_suites = set()
    for study in studies.values():
        parent = Path(study["result_ref"]["path"]).parent.parent
        correction = parent / "analysis_correction_registration.json"
        suite_result = parent / "result.json"
        if correction.exists() and suite_result.exists() and parent not in seen_suites:
            suite = read_json(suite_result)
            if suite.get("schema") != "motion2scene_timed_duration_teacher_suite_result_v1":
                # The registered suite schema may evolve; require the explicit child and correction bindings.
                if not {"children", "analysis_correction", "registration"}.issubset(suite):
                    raise ValueError("Unrecognized parent analysis correction receipt")
            checked(suite["analysis_correction"])
            checked(suite["registration"])
            for child in suite["children"]:
                checked(child["result"])
            receipt_folder = provenance / f"suite_{len(seen_suites):03d}"
            receipt_folder.mkdir()
            write_json(receipt_folder / "result.json", suite)
            write_json(
                receipt_folder / "analysis_correction_registration.json", read_json(correction)
            )
            write_json(
                receipt_folder / "registration.json", read_json(checked(suite["registration"]))
            )
            suite_provenance.append(
                {
                    "result": ref(suite_result),
                    "correction": ref(correction),
                    "portable": str(receipt_folder.relative_to(out)),
                    "counted_as_additional_episodes": False,
                }
            )
            seen_suites.add(parent)
    for index, suite_result in enumerate(suite_results):
        suite_provenance.append(
            export_comparison_suite(
                suite_result, studies, provenance / f"comparison_suite_{index:03d}", out
            )
        )
    write_json(out / "teachers.json", teacher_rows)
    counts = {
        "physical_control_frames": sum(e["physical_control_frames"] for e in episodes),
        "physics_steps": sum(e["physics_steps"] for e in episodes),
        "raw_sensor_packets": sum(e["physical_control_frames"] for e in episodes),
        "eligible_sensor_packets": sum(e["eligible_sensor_packets"] for e in episodes),
        "teacher_decisions": len(teacher_rows),
        "criteria_counts": dict(Counter(c["criterion"] for e in episodes for c in e["criteria"])),
        "feature_schema_counts": dict(Counter(e["student"]["schema"] for e in episodes)),
    }
    card = f"""# Six-second humanoid development recordings

This local release contains {len(episodes)} distinct actual captures,
{counts['physics_steps']} measured physics steps, and
{counts['eligible_sensor_packets']} eligible sensor packets. All are development evidence.

Each episode keeps the full physical trajectory as pickle-free NPZ, causal sensor measurements
with identity fields removed, explicit per-packet alignment, actual switch/state logs, and
separate contact/scene/outcome supervision. A capture may have both strict-net and environment
assessments; it counts once. Distinct deterministic captures with identical bytes keep distinct IDs.

The original strict criterion rejects nonfoot net contacts, including self-contact. The separately
registered environment criterion permits foot-floor support, measures every 30-by-36 normal-force
counterpart at 200 Hz, and reports self-contact separately. Complete contact streams can record
failed passages. Diagnostic-only episodes do not imply qualification.

The sparse-ray qualification captures have no invented learned input vector. Later timed sensor
branches store exact 106D inputs: 65 ideal PhysX rays and hit-normal masks, 0.5 s/26-frame causal
history, robot state and legality. This schema differs from earlier 214D/100D/110D releases.
Teachers refer to complete forced schedules at entry tick 15; no variable timing/wait target is
implied. A policy-comparison increment can contain no teacher targets: configured mode/preference
and actual executed switches remain distinct. Failed finite-horizon holds keep a failure label
and null success time, separately from instability or contact. Scene geometry and outcome targets
are excluded from student_inputs and sensor_history.

Physics recording precedes command-cursor advance: command reference phase is one 50 Hz tick later.
Use the alignment mask to select eligible sensor/physics pairs. All physical rows and failures
remain. Estimated PD effort is not measured actuator work or battery energy.

No model weights, raw full-skeleton priors or whole reference banks are redistributed. Logged
reference commands remain identified in trajectory data. Original source hashes, manifests,
attempts and criteria are in provenance/supervision. Upstream asset licenses are not relicensed.
There is no held-out, hardware, generic navigation or repeated-transition claim.
"""
    (out / "DATA_CARD.md").write_text(card)
    files = [
        {**ref(path), "path": str(path.relative_to(out))}
        for path in sorted(out.rglob("*"))
        if path.is_file()
    ]
    write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            "episodes": episodes,
            "counts": counts,
            "files": files,
            "source_results": [s["result_ref"] for s in studies.values()],
            "suite_analysis_provenance": suite_provenance,
            "infrastructure_attempts": [
                a for s in studies.values() for a in s.get("infrastructure_attempts", [])
            ],
        },
    )
    summary = audit(out)
    if make_archive:
        archive = out.with_suffix(".tar.gz")
        if archive.exists():
            raise FileExistsError("Archive already exists")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(out, arcname=out.name)
        receipt = {
            "manifest": ref(out / "manifest.json"),
            "archive": ref(archive),
            "audit": summary,
        }
        write_json(out.with_suffix(".receipt.json"), receipt)
        summary["archive"] = receipt["archive"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "export", "audit"))
    parser.add_argument("--results", type=Path, nargs="+", default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--suite-results", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    if args.mode == "inspect":
        records, studies = collect_sources(args.results)
        print(
            json.dumps(
                {
                    "actual_episodes": len(records),
                    "assessment_sources": len(studies),
                    "physics_steps": sum(
                        prepare_record(row)[-1]["physics_steps"] for row in records
                    ),
                }
            )
        )
    else:
        if args.out is None:
            parser.error("--out required")
        print(
            json.dumps(
                audit(args.out)
                if args.mode == "audit"
                else export(args.out, args.results, args.archive, args.suite_results)
            )
        )


if __name__ == "__main__":
    main()
