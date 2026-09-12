#!/usr/bin/env python3
"""Package new recorded motion-option and sensor-history development episodes.

Only completed result files produce episodes. All failures and reset-spanning
arrays remain; scene/outcome supervision is separate from causal student inputs.
The resulting local package contains derived recordings, not inherited weights
or reference banks. Run ``audit --out PACKAGE`` to verify the portable artifacts.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_sensor_alignment import (  # noqa: E402
    audit_sensor_alignment,
)
from scripts.research.motion2scene_export_traversal_dataset import (  # noqa: E402
    checked,
    digest,
    write_json,
)

DATA = ROOT.parent / "research-data/groot-wbc"
DEFAULT_STUDIES = (
    "m2s-option-qualification-v2",
    "m2s-option-lowbeam-v1",
    "m2s-option-contrast-v1",
    "m2s-option-entry-contrast-v1",
    "m2s-history-d085-acquisition-v1",
    "m2s-option-robustness-lower-v1",
    "m2s-option-robustness-nominal-v1",
    "m2s-option-robustness-higher-v1",
    "m2s-history-d085-policy-evaluation-v2",
)
DEFAULT_ATTEMPTS = (
    DATA
    / "m2s-history-d085-policy-evaluation-v1/rollouts/absent_source41002_d085_learned_t030/attempt.json",
)
SCHEMA = "motion2scene_executed_options_dataset_v1"
MULTI_SCHEMA = "motion2scene_history_multioption_v1"
MEASUREMENT_KEYS = {
    "distance",
    "position",
    "normal",
    "direction",
    "origin",
    "range_m",
    "hit",
    "lower_rays",
    "origin_w",
    "direction_w",
    "hit_distance_m",
    "hit_normal_w",
}
PACKET_KEYS = {
    "time_s",
    "capture_elapsed_s",
    "delivered_capture_elapsed_s",
    "observation_age_s",
    "root_pos_w",
    "root_quat_w",
    "origin",
    "capture_frame",
}
STATE_KEYS = {"projected_gravity_b", "root_lin_vel_w", "root_ang_vel_w", "dof_pos", "dof_vel"}
STUDENT_HISTORY_KEYS = {
    "schema_version",
    "feature_names",
    "features",
    "phase_s",
    "capture_elapsed_s",
    "active_before",
    "legal_mask",
    "joint_names",
    "option_ids",
    "classes",
}
FORBIDDEN_INPUT_KEYS = {
    "path",
    "object_id",
    "beam",
    "scene",
    "source",
    "source_id",
    "layout_id",
    "pass",
    "pass_labels",
    "teacher_action",
    "requested",
    "active_after",
    "policy_logits",
}


def measurement_only(value):
    """Retain legacy and SensorRay measurements; discard identity and annotations."""
    if isinstance(value, dict):
        return {k: measurement_only(v) for k, v in value.items() if k in MEASUREMENT_KEYS}
    if isinstance(value, list):
        return [measurement_only(v) for v in value]
    return value


def sensor_packets(document):
    """Causal measurement/pose packets; commands, logits and labels go elsewhere."""
    packets = []
    for observation in document["observations"]:
        packet = {k: observation[k] for k in PACKET_KEYS if k in observation}
        if "state" in observation:
            packet["state"] = {k: v for k, v in observation["state"].items() if k in STATE_KEYS}
        for field in ("measurements", "rays"):
            if field in observation:
                packet[field] = measurement_only(observation[field])
        if observation.get("delivered") is not None:
            packet["delivered"] = sensor_packets({"observations": [observation["delivered"]]})[0]
        if "measurements" not in packet and "rays" not in packet:
            raise ValueError(
                "sensor packet contains neither legacy rays nor SensorRay measurements"
            )
        packets.append(packet)
    return packets


def save_pickle_free(path, payload):
    """Retain nested arrays, all frames and metadata without object serialization."""
    arrays, metadata, excluded = {}, {}, []

    def encode(key, value):
        if isinstance(value, dict):
            return {
                str(child): encode(
                    key + "/" + str(child).replace("~", "~0").replace("/", "~1"), item
                )
                for child, item in value.items()
            }
        if isinstance(value, (list, tuple, np.ndarray)):
            try:
                array = np.asarray(value)
            except (TypeError, ValueError):
                excluded.append(key)
                return {"unexported_type": type(value).__name__}
            if array.dtype.kind in "biufcUS":
                arrays[key] = array
                return {"npz_array": key}
            else:
                excluded.append(key)
                return {"unexported_dtype": str(array.dtype)}
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        excluded.append(key)
        return {"unexported_type": type(value).__name__}

    for key, value in payload.items():
        metadata[key] = encode(str(key).replace("~", "~0").replace("/", "~1"), value)
    np.savez_compressed(path, **arrays)
    return metadata, excluded


def copy_npz(source, destination, allowed=None):
    with np.load(source, allow_pickle=False) as archive:
        values = {k: archive[k].copy() for k in archive.files if allowed is None or k in allowed}
    if any(v.dtype.kind not in "biufcUS" for v in values.values()):
        raise ValueError(f"nonportable array in {source}")
    np.savez_compressed(destination, **values)
    return values


def executed_options(option_ids, records, switches, first_episode_frames):
    """Derive executed options from observed changes, never configured preferences."""
    count = len(option_ids)
    if count < 2 or len(set(option_ids)) != count or not 0 < first_episode_frames <= len(records):
        raise ValueError("registered option IDs and complete first-episode records required")
    events = []
    for frame, record in enumerate(records):
        before, after = record["active_before"], record["active"]
        if any(type(value) is not int or not 0 <= value < count for value in (before, after)):
            raise ValueError("recorded active option outside registry")
        if before != after:
            transition = record["transition"]
            if (
                not transition["attempted"]
                or not transition["allowed"]
                or transition["requested"] != after
            ):
                raise ValueError("active option change lacks an admitted command transition")
            events.append(
                {
                    "control_frame": frame,
                    "phase_s": record["time_s"],
                    "capture_elapsed_s": record["capture_elapsed_s"],
                    "from": before,
                    "to": after,
                    "from_id": option_ids[before],
                    "to_id": option_ids[after],
                }
            )
    if len(events) != len(switches) or any(
        event["from"] != switch["from"]
        or event["to"] != switch["to"]
        or abs(event["phase_s"] - switch["time_s"]) > 1e-8
        for event, switch in zip(events, switches, strict=False)
    ):
        raise ValueError("recorded option changes and switch log disagree")
    first = [event for event in events if event["control_frame"] < first_episode_frames]
    entries = [event for event in first if event["to"] != 0]
    returns = [event for event in first if event["to"] == 0]
    neutral = all(
        record["active_before"] == record["active"] == 0
        for record in records[:first_episode_frames]
    )
    index = entries[0]["to"] if len(entries) == 1 else (0 if neutral else None)
    return {
        "option_index": index,
        "option_id": option_ids[index] if index is not None else None,
        "entry_time_s": entries[0]["phase_s"] if len(entries) == 1 else None,
        "return_time_s": returns[0]["phase_s"] if len(returns) == 1 else None,
        "stayed_neutral": neutral,
        "first_episode_option_sequence": [records[0]["active_before"]]
        + [event["to"] for event in first],
        "first_episode_final_option_index": records[first_episode_frames - 1]["active"],
        "all_recorded_switch_events": events,
        "scope": "first-episode action summary from logged transitions; all reset-spanning events retained",
    }


def export_student_inputs(folder, row, sensor, provenance):
    """Export exact recorded schema; never append outcome, source or scene fields."""
    if "features" in row:
        source = checked(row["features"])
        provenance["features"] = row["features"]
        path = folder / "student_history.npz"
        arrays = copy_npz(source, path, STUDENT_HISTORY_KEYS)
        names, features = arrays["feature_names"], arrays["features"]
        if (
            features.ndim != 2
            or names.ndim != 1
            or features.shape[1] != len(names)
            or not np.isfinite(features).all()
            or len(set(names.tolist())) != len(names)
            or arrays["phase_s"].shape != features.shape[:1]
            or arrays["capture_elapsed_s"].shape != features.shape[:1]
            or arrays["legal_mask"].ndim != 2
            or arrays["legal_mask"].shape[0] != len(features)
            or arrays["active_before"].shape != features.shape[:1]
        ):
            raise ValueError("invalid recorded history feature schema")
        if len(sensor["observations"]) != len(features):
            raise ValueError("sensor history and feature frame counts differ")
        if "feature_names" in sensor and sensor["feature_names"] != names.tolist():
            raise ValueError("sensor and feature archive names disagree")
        recorded = np.asarray([o["features"] for o in sensor["observations"]], dtype=features.dtype)
        if not np.array_equal(recorded, features):
            raise ValueError("feature archive and recorded policy inputs disagree")
        for archive_key, packet_key in (
            ("phase_s", "time_s"),
            ("capture_elapsed_s", "capture_elapsed_s"),
        ):
            recorded = np.asarray(
                [o[packet_key] for o in sensor["observations"]], dtype=arrays[archive_key].dtype
            )
            if not np.array_equal(recorded, arrays[archive_key]):
                raise ValueError("feature and sensor clocks disagree")
        if str(arrays["schema_version"]) == MULTI_SCHEMA:
            option_ids = arrays["option_ids"].tolist()
            if (
                sensor["feature_schema"] != MULTI_SCHEMA
                or sensor["option_names"] != option_ids
                or not np.array_equal(arrays["classes"], np.arange(len(option_ids)))
                or arrays["legal_mask"].shape[1] != len(option_ids)
                or features.shape[1] != 100 + 2 * len(option_ids)
            ):
                raise ValueError("multi-option registry or exact feature schema disagrees")
            for key in ("active_before", "legal_mask"):
                if not np.array_equal(
                    arrays[key], np.asarray([o[key] for o in sensor["observations"]])
                ):
                    raise ValueError("multi-option command legality and student archive disagree")
        return {
            "file": path.name,
            "kind": "causal_history",
            "recorded_schema": str(arrays["schema_version"]),
            "feature_dimension": features.shape[1],
            "feature_rows": features.shape[0],
            "action_dimension": arrays["legal_mask"].shape[1],
            "feature_names": names.tolist(),
            "option_ids": arrays["option_ids"].tolist() if "option_ids" in arrays else None,
        }
    if "decision" in row:
        source = checked(row["decision"])
        provenance["decision"] = row["decision"]
        decision = json.loads(source.read_text())
        features = np.asarray(decision["features"], dtype=np.float32)
        if features.ndim != 1 or not np.isfinite(features).all():
            raise ValueError("invalid legacy decision features")
        path = folder / "student_decision.npz"
        np.savez_compressed(
            path,
            schema_version=f"legacy_decision_{len(features)}d",
            features=features,
            phase_s=float(decision["phase_s"]),
            capture_elapsed_s=float(decision["capture_elapsed_s"]),
            active_before=int(decision["active_before"]),
        )
        return {
            "file": path.name,
            "kind": "legacy_single_decision",
            "recorded_schema": f"legacy_decision_{len(features)}d",
            "feature_dimension": len(features),
            "feature_rows": 1,
            "feature_names": None,
        }
    raise ValueError("completed episode lacks a pinned student-input artifact")


def portable_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("package path escapes its root")
    return path


def export_schedule_targets(out, paths, episodes, provenance, related_datasets=()):
    """Make finite WAIT/adapt targets portable and bind exact recorded input rows."""
    indexed = {episode["episode_id"]: episode for episode in episodes}
    by_path = {}
    external_by_path, related = {}, []
    for package in related_datasets:
        audit(package)
        manifest_hash = digest(package / "manifest.json")
        relative = Path("related_datasets") / manifest_hash.removeprefix("sha256:")
        destination = out / relative
        destination.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.json", "episodes.json", "provenance.json"):
            (destination / name).write_bytes((package / name).read_bytes())
        related.append({"directory": str(relative), "manifest_sha256": manifest_hash})
        external_episodes = {
            row["episode_id"]: row for row in json.loads((package / "episodes.json").read_text())
        }
        for record in json.loads((package / "provenance.json").read_text()):
            ref = record["artifacts"]["trajectory"]
            external_by_path.setdefault(str(Path(ref["path"]).resolve()), []).append(
                (package, manifest_hash, external_episodes[record["episode_id"]], ref)
            )
    for record in provenance:
        ref = record["artifacts"]["trajectory"]
        by_path.setdefault(str(Path(ref["path"]).resolve()), []).append(
            (record["episode_id"], ref["sha256"])
        )

    def episode_for(ref):
        path = ref if isinstance(ref, str) else ref["path"]
        matches = by_path.get(str(Path(path).resolve()), [])
        if not isinstance(ref, str):
            matches = [match for match in matches if match[1] == ref["sha256"]]
        if len(matches) != 1:
            raise ValueError("schedule target requires one uniquely packaged physical branch")
        return matches[0][0]

    sources, targets = [], []
    for path in paths:
        source = {"path": str(path), "sha256": digest(path)}
        registration_path = path.parent / "registration.json"
        if registration_path.exists():
            registration = json.loads(registration_path.read_text())
            if checked(registration["teacher_table"]).resolve() != path.resolve():
                raise ValueError("schedule teacher table differs from fitted registration")
            source["registration"] = {
                "path": str(registration_path),
                "sha256": digest(registration_path),
            }
            source["registration_contents"] = registration
        sources.append(source)
        for target in json.loads(path.read_text()):
            item = {
                key: value
                for key, value in target.items()
                if key
                not in ("features", "matching_audit", "continuation_branch_ids", "student_visit")
            }
            item.update(
                schedule_teacher_id=f"schedule_{len(targets):03d}",
                source_table_index=len(sources) - 1,
            )
            item["continuation_episode_ids"] = [
                episode_for(ref) if ref is not None else None
                for ref in target["continuation_branch_ids"]
            ]
            item["matching_audit"] = [
                {
                    **{key: value for key, value in check.items() if key != "branch"},
                    "episode_id": episode_for(check["branch"]),
                }
                for check in target["matching_audit"]
            ]
            item["student_visit"] = dict(target.get("student_visit", {}))
            if "student_branch" in item["student_visit"]:
                ref = item["student_visit"].pop("student_branch")
                source_path = str(Path(ref["path"]).resolve())
                if source_path in by_path:
                    item["student_visit"]["student_episode_id"] = episode_for(ref)
                else:
                    matches = [r for r in external_by_path.get(source_path, []) if r[3] == ref]
                    if len(matches) != 1:
                        raise ValueError(
                            "student visit requires one pinned internal or related episode"
                        )
                    package, manifest_hash, episode, trajectory = matches[0]
                    folder = package / episode["directory"]
                    alignment = json.loads((folder / "sensor_alignment.json").read_text())
                    with np.load(
                        folder / episode["student_input"]["file"], allow_pickle=False
                    ) as arrays:
                        indices = np.flatnonzero(
                            np.abs(arrays["phase_s"] - target["phase_s"]) < 1e-8
                        )
                        if len(indices) != 1:
                            raise ValueError("external student visit has no unique decision phase")
                        frame = int(indices[0])
                        if (
                            not alignment["packet_eligible"][frame]
                            or arrays["active_before"][frame] != 0
                            or not np.array_equal(
                                arrays["features"][frame],
                                np.asarray(target["features"], dtype=arrays["features"].dtype),
                            )
                        ):
                            raise ValueError(
                                "external student visit does not match eligible causal input"
                            )
                    item["student_visit"]["external_student_episode"] = {
                        "dataset_manifest_sha256": manifest_hash,
                        "episode_id": episode["episode_id"],
                        "trajectory": trajectory,
                        "row_index": frame,
                        "phase_s": target["phase_s"],
                        "scope": "actual student visit in related immutable package; no episode duplicated",
                    }
            for check in item["matching_audit"]:
                if not check["matched"]:
                    continue
                episode = indexed[check["episode_id"]]
                relative = str(Path(episode["directory"]) / episode["student_input"]["file"])
                alignment = json.loads(
                    (out / episode["directory"] / "sensor_alignment.json").read_text()
                )
                with np.load(out / relative, allow_pickle=False) as arrays:
                    matches = np.flatnonzero(np.abs(arrays["phase_s"] - target["phase_s"]) < 1e-8)
                    if len(matches) != 1:
                        continue
                    frame = int(matches[0])
                    if (
                        not alignment["packet_eligible"][frame]
                        or arrays["active_before"][frame] != 0
                        or not np.array_equal(
                            arrays["features"][frame],
                            np.asarray(target["features"], dtype=arrays["features"].dtype),
                        )
                    ):
                        continue
                item["student_input"] = {
                    "episode_id": episode["episode_id"],
                    "file": relative,
                    "row_index": frame,
                    "phase_s": target["phase_s"],
                }
                break
            if "student_input" not in item:
                raise ValueError("schedule target lacks an exact packaged causal student input")
            item["action_semantics"] = (
                "immediate option selection; zero keeps neutral now with its recorded "
                "future continuation, not a commitment to keep walking"
            )
            targets.append(item)
        if digest(path) != source["sha256"]:
            raise ValueError("schedule targets changed during export")
    write_json(out / "schedule_teachers.json", targets)
    write_json(out / "schedule_teacher_sources.json", sources)
    write_json(out / "related_datasets.json", related)
    return len(targets)


def audit(out):
    """Verify every packaged hash, numeric archive, input separation and episode count."""
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("unsupported dataset schema")
    for relative, expected in manifest["files"].items():
        path = portable_path(out, relative)
        if digest(path) != expected:
            raise ValueError(f"package hash mismatch: {relative}")
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    if archive[key].dtype.kind not in "biufcUS":
                        raise ValueError("object array in portable package")
    episodes = json.loads((out / "episodes.json").read_text())
    if len(episodes) != manifest["episodes"] or len({r["episode_id"] for r in episodes}) != len(
        episodes
    ):
        raise ValueError("invalid episode count or repeated identifier")
    packets = 0

    def check_input_keys(value):
        if isinstance(value, dict):
            if set(value) & FORBIDDEN_INPUT_KEYS:
                raise ValueError("privileged identity, action or outcome in sensor input")
            for child in value.values():
                check_input_keys(child)
        elif isinstance(value, list):
            for child in value:
                check_input_keys(child)

    for episode in episodes:
        folder = portable_path(out, episode["directory"])
        observations = json.loads((folder / "sensor_history.json").read_text())
        check_input_keys(observations)
        packets += len(observations)
        with np.load(folder / episode["student_input"]["file"], allow_pickle=False) as arrays:
            if set(arrays.files) & FORBIDDEN_INPUT_KEYS:
                raise ValueError("outcome or action leakage in student arrays")
            if not np.isfinite(arrays["features"]).all():
                raise ValueError("nonfinite student input")
        with np.load(folder / "trajectory.npz", allow_pickle=False) as arrays:
            n = len(arrays["motion_time_s"])
            if (
                n != episode["recorded_control_frames"]
                or not 0 < episode["first_episode_frames"] <= n
            ):
                raise ValueError("invalid retained first-episode boundary")
            if episode.get("sensor_alignment") is not None:
                expected = json.loads((folder / "sensor_alignment.json").read_text())
                payload = {key: arrays[key] for key in arrays.files}
                if "fps" not in payload:
                    payload["fps"] = json.loads((folder / "trajectory_metadata.json").read_text())[
                        "fields"
                    ]["fps"]
                actual = audit_sensor_alignment(
                    payload, observations, reference_frames=expected["reference_frames"]
                )
                if actual != expected:
                    raise ValueError("packaged sensor alignment does not reproduce")
                with np.load(folder / "sensor_alignment.npz", allow_pickle=False) as masks:
                    if not np.array_equal(masks["packet_eligible"], expected["packet_eligible"]):
                        raise ValueError("packaged sensor eligibility masks disagree")
        if episode.get("registered_option_ids") is not None:
            commands = json.loads((folder / "commands.json").read_text())
            actual = executed_options(
                episode["registered_option_ids"],
                commands["records"],
                commands["switches"],
                episode["first_episode_frames"],
            )
            if actual != episode["executed_option"]:
                raise ValueError("packaged actual option summary disagrees with command log")
            with np.load(folder / episode["student_input"]["file"], allow_pickle=False) as arrays:
                if arrays["option_ids"].tolist() != episode["registered_option_ids"]:
                    raise ValueError("student action classes differ from registered IDs")
    if "schedule_teacher_decisions" in manifest:
        targets = json.loads((out / "schedule_teachers.json").read_text())
        known = {episode["episode_id"] for episode in episodes}
        external_provenance = {}
        related_path = out / "related_datasets.json"
        for related in json.loads(related_path.read_text()) if related_path.exists() else []:
            folder = portable_path(out, related["directory"])
            if digest(folder / "manifest.json") != related["manifest_sha256"]:
                raise ValueError("related dataset manifest pin differs")
            source_manifest = json.loads((folder / "manifest.json").read_text())
            for name in ("episodes.json", "provenance.json"):
                if digest(folder / name) != source_manifest["files"][name]:
                    raise ValueError("related episode/provenance does not match original package")
            external_provenance[related["manifest_sha256"]] = {
                r["episode_id"]: r["artifacts"]["trajectory"]
                for r in json.loads((folder / "provenance.json").read_text())
            }
        if len(targets) != manifest["schedule_teacher_decisions"]:
            raise ValueError("incorrect schedule teacher count")
        for target in targets:
            external = target.get("student_visit", {}).get("external_student_episode")
            if external is not None:
                linked = external_provenance.get(external["dataset_manifest_sha256"], {})
                if linked.get(external["episode_id"]) != external["trajectory"]:
                    raise ValueError(
                        "external student visit differs from pinned related provenance"
                    )
                if external["phase_s"] != target["phase_s"] or external["row_index"] < 0:
                    raise ValueError("external student visit phase differs from target")
            if any(
                identifier not in known
                for identifier in target["continuation_episode_ids"]
                if identifier is not None
            ):
                raise ValueError("schedule target references a missing physical continuation")
            ref = target["student_input"]
            with np.load(portable_path(out, ref["file"]), allow_pickle=False) as arrays:
                if abs(float(arrays["phase_s"][ref["row_index"]]) - ref["phase_s"]) > 1e-8:
                    raise ValueError("schedule teacher input points to a different decision phase")
            aligned = json.loads(
                (portable_path(out, ref["file"]).parent / "sensor_alignment.json").read_text()
            )
            if not aligned["packet_eligible"][ref["row_index"]]:
                raise ValueError("schedule target points to an ineligible sensor packet")
    return {
        "episodes": len(episodes),
        "file_hashes_checked": len(manifest["files"]),
        "sensor_packets": packets,
        "pickle_free": True,
        "student_input_separation": True,
    }


def export(
    out, results, *, allow_missing=False, attempts=(), schedule_teachers=(), related_datasets=()
):
    inputs, missing = [], []
    for path in results:
        if not path.exists():
            if not allow_missing:
                raise FileNotFoundError(f"requested completed result unavailable: {path}")
            missing.append(str(path))
            continue
        result = json.loads(path.read_text())
        if not isinstance(result.get("rows"), list) or not result["rows"]:
            raise ValueError(f"expected completed physical episode rows: {path}")
        manifest_path = checked(result["manifest"])
        manifest = json.loads(manifest_path.read_text())
        if len(result["rows"]) != len(manifest["cells"]):
            raise ValueError("refuse to package an incomplete physical result as completed")
        row_ids = [row["cell_id"] for row in result["rows"]]
        if len(set(row_ids)) != len(row_ids) or set(row_ids) != {
            cell["cell_id"] for cell in manifest["cells"]
        }:
            raise ValueError("result rows must cover each registered cell exactly once")
        inputs.append((path, result, manifest, digest(path)))
    if not inputs or len({p.parent.name for p, _, _, _ in inputs}) != len(inputs):
        raise ValueError("nonempty uniquely named study results required")
    out.mkdir(parents=True, exist_ok=False)
    (out / "episodes").mkdir()
    episodes, studies, provenance, groups, teachers = [], [], [], {}, []
    for result_path, result, manifest, result_hash in inputs:
        study_id = result_path.parent.name
        option_ids, registry = manifest.get("option_ids"), None
        if option_ids is not None:
            registry = json.loads(checked(manifest["registry"]).read_text())
            if option_ids != [reference["name"] for reference in registry["references"]]:
                raise ValueError("study option IDs differ from registered motion names")
            if manifest["implementation"]["checkpoint"] != registry["controller"]:
                raise ValueError("multi-option study controller differs from registry")
        cells = {c["cell_id"]: c for c in manifest["cells"]}
        if len(cells) != len(manifest["cells"]):
            raise ValueError("duplicate source manifest cell identifier")
        mapping = {}
        prefix_records = {p["cell_id"]: p for p in result.get("paired_prefixes", [])}
        for row in result["rows"]:
            cell = cells[row["cell_id"]]
            source = row.get("source", cell["generation_seed"])
            physics_seed = row.get("physics_seed", cell["runtime_seed"])
            if source != cell["generation_seed"] or physics_seed != cell["runtime_seed"]:
                raise ValueError("result source/seed disagrees with executed manifest")
            for key in ("beam", "condition", "option"):
                if key in row and key in cell and row[key] != cell[key]:
                    raise ValueError(f"result {key} disagrees with executed manifest")
            episode_id = f"episode_{len(episodes):04d}"
            mapping[row["cell_id"]] = episode_id
            folder = out / "episodes" / episode_id
            folder.mkdir()
            original = {"trajectory": row["trajectory"], "sensor": row["sensor"]}
            trajectory_path = checked(row["trajectory"])
            payload = load_reset_capture(trajectory_path)
            meta, excluded = save_pickle_free(folder / "trajectory.npz", payload)
            write_json(
                folder / "trajectory_metadata.json",
                {"fields": meta, "excluded_nonportable_fields": excluded},
            )
            sensor = json.loads(checked(row["sensor"]).read_text())
            bank_ref = row.get("bank")
            bank_path = (
                checked(bank_ref)
                if bank_ref
                else trajectory_path.parent / "loaded_reference_bank.npz"
            )
            with np.load(bank_path, allow_pickle=False) as bank:
                reference_frames = int(bank["root_xyz"].shape[1])
            original["bank"] = bank_ref or {
                "path": str(bank_path),
                "sha256": digest(bank_path),
                "pin_scope": "export-time reference horizon metadata; reference arrays not redistributed",
            }
            alignment = audit_sensor_alignment(
                payload, sensor["observations"], reference_frames=reference_frames
            )
            if alignment["physical_first_episode_frames"] != row["first_episode_frames"]:
                raise ValueError("physical first-episode boundary disagrees with recorded clock")
            write_json(folder / "sensor_alignment.json", alignment)
            np.savez_compressed(
                folder / "sensor_alignment.npz",
                packet_eligible=np.asarray(alignment["packet_eligible"], dtype=bool),
                packet_confidence=np.asarray(
                    [packet["confidence"] for packet in alignment["packets"]]
                ),
                sensor_phase_s=np.asarray([packet["time_s"] for packet in sensor["observations"]]),
                trajectory_phase_s=np.asarray(payload["motion_time_s"]),
            )
            actual_option, configured = None, None
            if option_ids is not None:
                preferred = row["option_index"]
                if (
                    type(preferred) is not int
                    or not 0 <= preferred < len(option_ids)
                    or preferred != cell["option_index"]
                    or row["mode"] != cell["multi_option_mode"]
                    or row["mode"] != sensor["mode"]
                    or row["decision_time_s"] != cell["decision_time_s"]
                    or type(row["measurement_admitted"]) is not bool
                    or source != registry["source"]
                    or cell["motion"]["sha256"] != registry["references"][0]["motion"]["sha256"]
                    or sensor["option_names"] != option_ids
                    or sensor["option_registry_sha256"] != manifest["registry"]["sha256"]
                    or row["switches"] != sensor["switches"]
                ):
                    raise ValueError("multi-option row, registry and recorded interface disagree")
                configured = {
                    "option_index": preferred,
                    "option_id": option_ids[preferred],
                    "entry_parameter_s": row["decision_time_s"],
                    "scope": "configured command/preference; not proof of the option actually selected",
                }
                actual_option = executed_options(
                    option_ids,
                    sensor["observations"],
                    sensor["switches"],
                    row["first_episode_frames"],
                )
            write_json(folder / "sensor_history.json", sensor_packets(sensor))
            write_json(
                folder / "sensor_configuration.json",
                {
                    k: sensor[k]
                    for k in ("sensor", "history_grid", "feature_schema", "feature_names")
                    if k in sensor
                },
            )
            command_records = [
                {
                    k: o[k]
                    for k in (
                        "time_s",
                        "capture_elapsed_s",
                        "active_before",
                        "active",
                        "legal_mask",
                        "transition",
                        "policy_logits",
                    )
                    if k in o
                }
                for o in sensor["observations"]
            ]
            write_json(
                folder / "commands.json",
                {
                    "mode": sensor.get("mode"),
                    "policy_sha256": sensor.get("policy_sha256"),
                    "registered_option_ids": option_ids,
                    "configured_preference": configured,
                    "switches": sensor.get("switches", []),
                    "records": command_records,
                },
            )
            student = export_student_inputs(folder, row, sensor, original)
            for name, key in (
                ("physics_beam_contacts.npz", "physics"),
                ("option_mechanical_work.npz", "work"),
            ):
                ref = row.get(key)
                source_path = checked(ref) if ref else trajectory_path.parent / name
                if not source_path.exists():
                    raise ValueError(f"missing physical recording: {source_path}")
                original[key] = ref or {
                    "path": str(source_path),
                    "sha256": digest(source_path),
                    "pin_scope": "export-time snapshot; not a hash originally stored in the result row",
                }
                copy_npz(source_path, folder / name)
            beam = row.get("beam", cell.get("beam"))
            condition = row.get("condition", cell.get("condition"))
            group_key = json.dumps(
                [study_id, source, physics_seed, condition, beam], sort_keys=True
            )
            if group_key not in groups:
                groups[group_key] = {
                    "group_id": f"context_{len(groups):03d}",
                    "study_id": study_id,
                    "episode_ids": [],
                    "scope": "shared scene/source/seed; matched teacher status is recorded separately",
                }
            groups[group_key]["episode_ids"].append(episode_id)
            episode = {
                "episode_id": episode_id,
                "study_id": study_id,
                "original_cell_id": row["cell_id"],
                "directory": str(folder.relative_to(out)),
                "group_id": groups[group_key]["group_id"],
                "split": "development_only",
                "source": source,
                "physics_seed": physics_seed,
                "condition": condition,
                "beam": beam,
                "option": row.get("option"),
                "available_adaptation": cell.get("option") if option_ids is None else None,
                "policy_mode": row.get("mode", sensor.get("mode")),
                "registered_option_ids": option_ids,
                "configured_preference": configured,
                "executed_option": actual_option,
                "measurement_admitted": row.get("measurement_admitted"),
                "decision_time_s": row.get(
                    "decision_time_s", row.get("option", {}).get("entry_time_s")
                ),
                "decision_time_scope": (
                    "configuration_parameter; actual entry is in executed_option"
                    if option_ids is not None
                    else "legacy configured decision"
                ),
                "pass": bool(row["pass"]),
                "qualified": row.get("qualified"),
                "predicates": row.get("predicates"),
                "costs": row.get("costs", {}),
                "sensor_summary": row.get("sensor_summary"),
                "first_episode_frames": int(row["first_episode_frames"]),
                "recorded_control_frames": len(payload["motion_time_s"]),
                "reset_count": int(row["reset_count"]),
                "continuation_verified": False,
                "failure_flags": {
                    k: row[k]
                    for k in (
                        "observed_beam_contact",
                        "fall_observed",
                        "incomplete_crossing",
                        "stabilization_failed",
                    )
                    if k in row
                },
                "maximum_beam_force_n": row.get("maximum_beam_normal_force_n_through_passage"),
                "passage_finish_frame_exclusive": row.get("passage_finish_frame_exclusive"),
                "acquisition": row.get("acquisition", {}),
                "student_input": student,
                "sensor_alignment": {
                    "metadata": "sensor_alignment.json",
                    "mask": "sensor_alignment.npz",
                    "eligible_packets": alignment["eligible_first_episode_packets"],
                    "total_packets": len(alignment["packet_eligible"]),
                    "pose_unavailable_packets": alignment["pose_unavailable_packets"],
                    "expected_sensor_phase_offset_s": alignment["expected_sensor_phase_offset_s"],
                },
                "paired_prefix": prefix_records.get(row["cell_id"]),
            }
            episodes.append(episode)
            provenance.append(
                {
                    "episode_id": episode_id,
                    "artifacts": original,
                    "reference_provenance": {
                        k: cell[k]
                        for k in (
                            "motion",
                            "alternate_motion",
                            "reference",
                            "options",
                            "hydra_overrides",
                        )
                        if k in cell
                    },
                    "result": {"path": str(result_path), "sha256": result_hash},
                    "option_registry": manifest.get("registry"),
                }
            )
        for target in result.get("teachers", []):
            teacher = {k: v for k, v in target.items() if k != "features"}
            teacher["study_id"] = study_id
            teacher["episode_ids"] = [mapping[k] for k in target["branch_cell_ids"]]
            teachers.append(teacher)
        studies.append(
            {
                "study_id": study_id,
                "result_schema": result.get("schema"),
                "episodes": len(result["rows"]),
                "source_result": {"path": str(result_path), "sha256": result_hash},
                "source_manifest": result["manifest"],
                "implementation": manifest.get("implementation"),
                "unsupported": manifest.get("unsupported"),
                "recorded_physics_steps": result.get("physics_steps"),
                "registered_option_ids": option_ids,
                "option_registry": manifest.get("registry"),
                "option_registry_contents": registry,
            }
        )
        if digest(result_path) != result_hash:
            raise ValueError("source result changed while packaging")
    write_json(out / "episodes.json", episodes)
    write_json(out / "groups.json", list(groups.values()))
    write_json(out / "teachers.json", teachers)
    write_json(out / "provenance.json", provenance)
    write_json(out / "studies.json", studies)
    schedule_count = export_schedule_targets(
        out, schedule_teachers, episodes, provenance, related_datasets
    )
    infrastructure = []
    for path in attempts:
        attempt = json.loads(path.read_text())
        if attempt["exit_status"] == 0:
            raise ValueError("non-episode attempt must be an unsuccessful infrastructure run")
        if list(path.parent.rglob("*.trajectory.pkl")) or list(
            path.parent.rglob("reactive_interface.json")
        ):
            raise ValueError("attempt has recordings; preserve and review them as episodes instead")
        log_path = path.parent / "rollout.log"
        infrastructure.append(
            {
                "attempt_id": f"infrastructure_{len(infrastructure):03d}",
                "source": {"path": str(path), "sha256": digest(path)},
                "source_log": (
                    {"path": str(log_path), "sha256": digest(log_path)}
                    if log_path.exists()
                    else None
                ),
                "record": attempt,
                "recorded_control_frames": 0,
                "scored_physical_episode": False,
                "physics_steps": None,
                "sensor_queries": None,
                "accounting_scope": (
                    "failed before recorded traversal; startup physics/query counts are unrecorded"
                ),
            }
        )
    write_json(out / "infrastructure_attempts.json", infrastructure)
    (out / "DATASET_CARD.md").write_text(
        "# Motion2Scene executed options and sensor-history development dataset\n\n"
        f"{len(episodes)} new recorded simulation branches across {len(studies)} studies. "
        "These are recorded option qualification, capability, timing, sensing and policy "
        "development episodes, separate from the legacy72-group package. No held-out or "
        "hardware claim is made.\n\n"
        "Read `episodes.json` for outcomes, costs, scene parameters and references to "
        "relative episode paths. `groups.json` groups common scenes; it does not certify "
        "matched decisions. `teachers.json` preserves explicit matched teacher admissions. "
        "Do not use scene, source IDs, outcomes or teacher metadata as student inputs.\n\n"
        "`student_decision.npz` stores legacy single-decision features (typically214D); "
        "`student_history.npz` stores repeated causal sensor/state inputs (100D binary "
        "integration or the explicitly named K-option schema). Read feature dimensions, "
        "names and legal-mask shape: these schemas must not be concatenated or silently "
        "substituted. Requested actions/logits/active-after records live in `commands.json`, "
        "not student-input arrays.\n\n"
        "For multi-option episodes, `configured_preference` is a configuration parameter. "
        "`executed_option` is derived from logged switches and observed command changes; "
        "a preferred option4 learner may execute option2 or stay neutral. Registered "
        "option IDs/classes and the exact110D five-option schema are preserved. "
        "`measurement_admitted` retains physical recording admission.\n\n"
        "`schedule_teachers.json` maps every finite future continuation to a packaged "
        "episode and points to an exact recorded student-input row. Action0 here means "
        "WAIT now with the selected future continuation; a forced neutral episode "
        "instead commits to walking. Missing schedules are not failed-action labels. "
        "An incremental package can preserve the actual student visit through "
        "`external_student_episode`: manifest SHA256, original trajectory hash and "
        "episode/row identifiers bind a related immutable package. Its manifest, "
        "episode index and provenance are included under `related_datasets/`; "
        "that student's physical rollout is not duplicated. The local target input "
        "is an exact matched teacher-prefix row, while the original student remains "
        "explicitly identified in the related package. "
        "Teacher fields are supervision, never policy inputs.\n\n"
        "`sensor_history.json` contains only measurement/pose fields and no simulator "
        "object identity. New SensorRay packets preserve ideal query normals; old "
        "ray packets do not gain inferred normals. `sensor_configuration.json` identifies "
        "the exact recorded sensor description and map configuration. No sensor noise, "
        "real camera or complete reconstruction guarantee is implied.\n\n"
        "Use `sensor_alignment.npz:packet_eligible` before pairing sensor inputs with "
        "physical outcomes. Raw packets remain untouched in the export: the final "
        "packet may follow a command reset after the last valid physics row, and "
        "mid-run reset packets are excluded from first-episode learning. Equal row "
        "counts do not establish alignment. `sensor_alignment.json` records exact "
        "state checks where available and explicitly weaker phase-only legacy checks. "
        "The recorded sensor reference phase is one control tick ahead of the "
        "physical recording phase; no measured camera latency is inferred.\n\n"
        "All arrays load with `np.load(path,allow_pickle=False)`. Failures and complete "
        "nested tracking-error arrays are retained with slash-separated archive keys; "
        "`trajectory_metadata.json` records the original field structure. Complete "
        "reset-spanning trajectories are retained; score only `first_episode_frames`. "
        "Never let later reset recovery rescue an earlier failure. Missing work remains "
        "null. `estimated_torque_nm` is an implicit-actuator estimate, not measured "
        "applied torque or battery energy. Physics-step clocks and availability flags "
        "are retained. No continuous two-obstacle course is verified here.\n\n"
        "Keep each paired context, motion ancestry and reused layout together in any "
        "exploratory split. Evaluation-role studies here still use development layouts. "
        "These records support option-outcome/time learning, causal sensor-memory "
        "reconstruction and perceptive-selection development. Navigation/route planning "
        "and course transfer require further data.\n\n"
        "`infrastructure_attempts.json` preserves unsuccessful startup attempts and "
        "their wall time separately. These have no recorded traversal and are not "
        "physical failure labels. Unrecorded startup physics/query counts remain null.\n\n"
        "The package contains derived recordings, not inherited controller weights or "
        "reference banks. `provenance.json` retains local source paths/hashes for audit; "
        "local paths are provenance, not required data paths. No new redistribution "
        "license is assigned to inherited assets. Verify every packaged file with "
        "`motion2scene_export_option_dataset.py audit --out PACKAGE`.\n"
    )
    (out / "exporter_source.py").write_bytes(Path(__file__).read_bytes())
    (out / "sensor_alignment_source.py").write_bytes(
        Path(audit_sensor_alignment.__globals__["__file__"]).read_bytes()
    )
    files = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()}
    dimensions = Counter(str(e["student_input"]["feature_dimension"]) for e in episodes)
    manifest = {
        "schema": SCHEMA,
        "exporter_format_version": 2,
        "sensor_alignment_schema": "motion2scene_sensor_physics_alignment_v1",
        "sensor_alignment_implementation_sha256": digest(
            Path(audit_sensor_alignment.__globals__["__file__"])
        ),
        "eligible_sensor_packets": sum(e["sensor_alignment"]["eligible_packets"] for e in episodes),
        "ineligible_sensor_packets": sum(
            e["sensor_alignment"]["total_packets"] - e["sensor_alignment"]["eligible_packets"]
            for e in episodes
        ),
        "role": "local development dataset; no untouched test or navigation claim",
        "episodes": len(episodes),
        "contexts": len(groups),
        "studies": len(studies),
        "outcomes": {
            "pass": sum(e["pass"] for e in episodes),
            "fail": sum(not e["pass"] for e in episodes),
        },
        "feature_dimension_episode_counts": dict(dimensions),
        "source_ids": sorted({e["source"] for e in episodes}),
        "physics_steps": sum(e["acquisition"].get("physics_steps", 0) for e in episodes),
        "all_result_rows_and_reset_histories_retained": True,
        "sensor_object_identity_stripped": True,
        "teacher_groups": len(teachers),
        "schedule_teacher_decisions": schedule_count,
        "course_continuation_verified": False,
        "missing_requested_results": missing,
        "infrastructure_attempts": len(infrastructure),
        "infrastructure_wall_seconds": sum(a["record"]["wall_seconds"] for a in infrastructure),
        "exporter_sha256": digest(Path(__file__)),
        "files": files,
    }
    write_json(out / "manifest.json", manifest)
    report = audit(out)
    archive = out.with_suffix(".tar.gz")
    with tarfile.open(archive, "x:gz") as handle:
        handle.add(out, arcname=out.name)
    return {
        **report,
        "manifest": str(out / "manifest.json"),
        "manifest_sha256": digest(out / "manifest.json"),
        "archive": str(archive),
        "archive_sha256": digest(archive),
        "missing_requested_results": missing,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "audit"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+")
    parser.add_argument("--attempts", type=Path, nargs="*", default=DEFAULT_ATTEMPTS)
    parser.add_argument("--schedule-teachers", type=Path, nargs="*", default=[])
    parser.add_argument("--related-datasets", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="record unavailable studies explicitly; never synthesize episodes",
    )
    args = parser.parse_args()
    paths = args.results or [DATA / name / "result.json" for name in DEFAULT_STUDIES]
    print(
        json.dumps(
            (
                export(
                    args.out,
                    paths,
                    allow_missing=args.allow_missing,
                    attempts=args.attempts,
                    schedule_teachers=args.schedule_teachers,
                    related_datasets=args.related_datasets,
                )
                if args.mode == "export"
                else audit(args.out)
            ),
            indent=2,
        )
    )
