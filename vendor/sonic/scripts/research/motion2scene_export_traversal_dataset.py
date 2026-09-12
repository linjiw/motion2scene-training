#!/usr/bin/env python3
"""Export a portable, pickle-free development dataset from audited physical pairs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import paired_prefix
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_sensor_alignment import (
    audit_sensor_alignment,
)

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(ref):
    path = Path(ref["path"])
    if digest(path) != ref["sha256"]:
        raise ValueError(f"artifact changed: {path}")
    return path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def measurement_only(value):
    """Whitelist range measurements and geometric tests; strip simulator identity."""
    keys = {
        "distance",
        "position",
        "normal",
        "direction",
        "origin",
        "range_m",
        "hit",
        "lower_rays",
        "upper_candidate",
        "overhang",
    }
    if isinstance(value, dict):
        return {k: measurement_only(v) for k, v in value.items() if k in keys}
    if isinstance(value, list):
        return [measurement_only(v) for v in value]
    return value


def history_alignment(payload, observations, full):
    """Bind clock limits to measured loaded banks; unknown history stays masked."""
    bank_ref = full.get("bank")
    if bank_ref is not None:
        with np.load(checked(bank_ref), allow_pickle=False) as bank:
            roots, joints = bank["root_xyz"], bank["joint_pos"]
            fps = float(bank["fps"])
            if (
                roots.ndim != 3
                or joints.ndim != 3
                or roots.shape[:2] != joints.shape[:2]
                or roots.shape[2] != 3
                or roots.shape[1] < 2
                or not np.isfinite(roots).all()
                or not np.isfinite(joints).all()
                or fps != float(payload["fps"])
            ):
                raise ValueError("invalid or incompatible measured reference bank")
            frames = int(roots.shape[1])
        result = audit_sensor_alignment(payload, observations, reference_frames=frames)
        result["reference_evidence"] = {
            "kind": "measured_loaded_reference_bank",
            "artifact": bank_ref,
            "frames": frames,
            "fps": fps,
        }
        return result
    # Do not estimate an unobserved bank endpoint from rollout length, which may
    # contain resets. The separately audited first decision remains available.
    times = np.asarray(payload["motion_time_s"])
    resets = np.flatnonzero(np.diff(times) <= 0) + 1
    first_end = int(resets[0]) if len(resets) else len(times)
    return {
        "schema": "motion2scene_sensor_physics_alignment_v1",
        "reference_evidence": {"kind": "unknown_loaded_reference"},
        "reference_frames": None,
        "physical_capture_frames": len(times),
        "physical_first_episode_frames": first_end,
        "expected_sensor_phase_offset_s": 1 / float(payload["fps"]),
        "packet_eligible": [False] * len(observations),
        "eligible_first_episode_packets": 0,
        "pose_unavailable_packets": len(observations),
        "packets": [
            {
                "index": index,
                "eligible": False,
                "confidence": "unknown_reference_unverified_history",
                "checked_state_fields": [],
                "mismatched_state_fields": [],
                "reasons": ["loaded_reference_evidence_unavailable"],
            }
            for index in range(len(observations))
        ],
        "scope": "history unavailable for training; retain separately audited first-decision input",
    }


def decision_alignment(decision, payload, full, alignment):
    """Retain the original 214D decision with its existing physical audit."""
    index = int(decision["capture_frame"])
    phase = float(decision["phase_s"])
    features = np.asarray(decision["features"])
    within_first = 0 <= index < alignment["physical_first_episode_frames"]
    phase_valid = within_first and bool(
        np.isclose(
            phase,
            payload["motion_time_s"][index] + 1 / float(payload["fps"]),
            atol=1e-8,
            rtol=0,
        )
    )
    required = ("decision_contract", "features_exact", "state_bound", "ray_origin_bound")
    predicates = full.get("predicates", {})
    original_audit = all(predicates.get(key, False) for key in required)
    return {
        "capture_frame": index,
        "phase_s": phase,
        "first_episode": within_first,
        "phase_aligned": phase_valid,
        "original_physical_audit_verified": original_audit,
        "required_original_predicates": {key: predicates.get(key) for key in required},
        "history_packet_eligible": (
            alignment["packet_eligible"][index] if 0 <= index < len(alignment["packets"]) else False
        ),
        "student_input_eligible": bool(
            within_first
            and phase_valid
            and original_audit
            and features.shape == (214,)
            and np.isfinite(features).all()
        ),
    }


def audit_export(out, reference_out=None):
    """Check portable masks and verify raw recordings against a prior release."""
    manifest = json.loads((out / "manifest.json").read_text())
    groups = json.loads((out / "groups.json").read_text())
    if digest(out / "groups.json") != manifest["groups_sha256"]:
        raise ValueError("group manifest changed")
    previous = (
        json.loads((reference_out / "groups.json").read_text())
        if reference_out is not None
        else None
    )
    if previous is not None and len(previous) != len(groups):
        raise ValueError("legacy encounter count changed")
    counts = Counter()
    for i, group in enumerate(groups):
        if previous is not None and group["outcomes"] != previous[i]["outcomes"]:
            raise ValueError("legacy paired outcomes changed")
        for action, branch in enumerate(group["branches"]):
            folder = out / branch["directory"]
            for name, sha in branch["files"].items():
                if digest(folder / name) != sha:
                    raise ValueError("portable episode artifact changed")
            packets = json.loads((folder / "sensor_history.json").read_text())
            alignment = json.loads((folder / "sensor_alignment.json").read_text())
            decision = json.loads((folder / "decision_alignment.json").read_text())
            with np.load(folder / "sensor_alignment.npz", allow_pickle=False) as arrays:
                mask = arrays["packet_eligible"].copy()
                confidence = arrays["confidence"].tolist()
            if (
                mask.dtype != np.bool_
                or mask.shape != (len(packets),)
                or mask.tolist() != alignment["packet_eligible"]
                or confidence != [p["confidence"] for p in alignment["packets"]]
                or len(packets) != branch["recorded_control_frames"]
                or alignment["physical_first_episode_frames"] != branch["first_episode_frames"]
                or mask[branch["first_episode_frames"] :].any()
                or int(mask.sum()) != branch["eligible_sensor_packets"]
                or decision["student_input_eligible"] != branch["student_input_eligible"]
                or (not branch["pass"] and branch["passage_time_s"] is not None)
            ):
                raise ValueError("portable sensor, decision or failure-cost contract violated")
            with np.load(folder / "student_input.npz", allow_pickle=False) as values:
                features = values["features"].copy()
            if features.shape != (214,) or not np.isfinite(features).all():
                raise ValueError("invalid portable decision input")
            if previous is not None:
                prior = previous[i]["branches"][action]
                old = reference_out / prior["directory"]
                if branch["pass"] != prior["pass"]:
                    raise ValueError("legacy physical label changed")
                for name in ("trajectory.npz", "student_input.npz"):
                    with np.load(folder / name, allow_pickle=False) as new_arrays:
                        with np.load(old / name, allow_pickle=False) as old_arrays:
                            if set(new_arrays.files) != set(old_arrays.files) or any(
                                not np.array_equal(new_arrays[key], old_arrays[key])
                                for key in new_arrays.files
                            ):
                                raise ValueError("legacy physical or 214D decision arrays changed")
                if digest(folder / "physics_contacts.npz") != digest(old / "physics_contacts.npz"):
                    raise ValueError("legacy physics contacts changed")
                if packets != json.loads((old / "sensor_history.json").read_text()):
                    raise ValueError("raw legacy measurement packets changed")
                counts["corrected_nonnull_failure_times"] += int(
                    not prior["pass"] and prior["passage_time_s"] is not None
                )
            counts["physical_branches"] += 1
            counts["recorded_physical_rows"] += branch["recorded_control_frames"]
            counts["raw_sensor_packets"] += len(packets)
            counts["eligible_sensor_packets"] += int(mask.sum())
            counts["ineligible_sensor_packets"] += int((~mask).sum())
            counts["audited_214d_inputs"] += int(branch["student_input_eligible"])
            counts["pose_unavailable_packets"] += alignment["pose_unavailable_packets"]
            counts["branches_with_unknown_reference"] += int(alignment["reference_frames"] is None)
    for key in ("physical_branches", "eligible_sensor_packets", "audited_214d_inputs"):
        if manifest[key] != counts[key]:
            raise ValueError("dataset aggregate counts disagree")
    return {
        "schema": "motion2scene_traversal_aligned_export_audit_v1",
        "passed": True,
        "groups": len(groups),
        **counts,
        "prior_release_compared": str(reference_out) if reference_out is not None else None,
        "physical_outcomes_arrays_and_measurements_preserved": previous is not None,
        "per_packet_alignment_not_tail_truncation": True,
        "scope": "legacy history is phase-audited with explicit pose-unavailable confidence",
    }


def export(out, analysis_path, reference_out=None):
    """All selected pairs are retained, including both-fail and reset trajectories."""
    analysis = json.loads(analysis_path.read_text())
    indexed = {r["cell_id"]: r for r in analysis["seed_level_rows"]}
    groups = []
    for condition in analysis["command_lookup"]["conditions"]:
        rows = [indexed[sorted(ids)[0]] for ids in condition["row_ids_by_action"]]
        groups.append(("regression", condition, rows))
    training = {}
    for row in indexed.values():
        if row["cell_id"].startswith("nom_") and row["layout"] is None:
            training.setdefault(row["group_id"], {})[row["action"]] = row
    for key, pair in sorted(training.items()):
        if set(pair) != {0, 1}:
            raise ValueError(f"incomplete nominal acquisition pair: {key}")
        rows = [pair[0], pair[1]]
        result = json.loads(checked(rows[0]["result"]).read_text())
        full = next(r for r in result["rows"] if r["cell_id"] == rows[0]["cell_id"])
        groups.append(
            (
                "acquisition",
                {
                    "source": rows[0]["source"],
                    "physics_seed": rows[0]["physics_seed"],
                    "layout": None,
                    "beam": full["beam"],
                    "outcomes": [r["pass"] for r in rows],
                },
                rows,
            )
        )
    out.mkdir(parents=True, exist_ok=False)
    (out / "episodes").mkdir()
    records, provenance = [], []
    for index, (role, condition, rows) in enumerate(groups):
        payloads = [load_reset_capture(checked(r["trajectory"])) for r in rows]
        prefix = paired_prefix(*payloads, 0.3)
        if not prefix["exact_match"]:
            raise ValueError("unmatched physical branches")
        decisions = [json.loads(checked(r["decision"]).read_text()) for r in rows]
        if not np.array_equal(decisions[0]["features"], decisions[1]["features"]):
            raise ValueError("paired student inputs differ")
        group_id = f"development_{index:04d}"
        branches = []
        for action, (row, payload, decision) in enumerate(zip(rows, payloads, decisions)):
            folder = out / "episodes" / f"{group_id}_a{action}"
            folder.mkdir()
            arrays = {}
            metadata = {}
            for key, value in payload.items():
                array = np.asarray(value) if isinstance(value, (list, tuple, np.ndarray)) else None
                if array is not None and array.dtype.kind in "biufcUS":
                    arrays[key] = array
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    metadata[key] = value
            np.savez_compressed(folder / "trajectory.npz", **arrays)
            write_json(folder / "trajectory_metadata.json", metadata)
            shutil.copyfile(checked(row["physics"]), folder / "physics_contacts.npz")
            # Sensor packets are exported separately from privileged labels/scene geometry.
            result = json.loads(checked(row["result"]).read_text())
            full = next(r for r in result["rows"] if r["cell_id"] == row["cell_id"])
            sensor_path = checked(full["sensor"])
            sensor = json.loads(sensor_path.read_text())
            alignment = history_alignment(payload, sensor["observations"], full)
            decision_audit = decision_alignment(decision, payload, full, alignment)
            write_json(folder / "sensor_alignment.json", alignment)
            write_json(folder / "decision_alignment.json", decision_audit)
            np.savez_compressed(
                folder / "sensor_alignment.npz",
                packet_eligible=np.asarray(alignment["packet_eligible"], dtype=bool),
                confidence=np.asarray([p["confidence"] for p in alignment["packets"]]),
            )
            packets = []
            for observation in sensor["observations"]:
                packets.append(
                    {
                        k: measurement_only(observation[k]) if k == "rays" else observation[k]
                        for k in ("time_s", "origin", "rays", "active", "observation_age_s")
                        if k in observation
                    }
                )
            write_json(folder / "sensor_history.json", packets)
            np.savez_compressed(
                folder / "student_input.npz", features=np.array(decision["features"], np.float32)
            )
            first_n = int(full["first_episode_frames"])
            times = np.asarray(payload["motion_time_s"])
            finish = full["passage_finish_frame_exclusive"]
            branch = {
                "option_id": "walk" if action == 0 else "d040_t030",
                "directory": str(folder.relative_to(out)),
                "pass": bool(row["pass"]),
                "command_executed": row["command_executed"],
                "first_episode_frames": first_n,
                "recorded_control_frames": len(times),
                "passage_time_s": (
                    float(times[finish - 1]) if row["pass"] and finish is not None else None
                ),
                "geometric_crossing_finish_frame_exclusive": finish,
                "first_episode_duration_s": float(times[first_n - 1] - times[0]),
                "mechanical_work_j": None,
                "switches": len(sensor["switches"]),
                "reset_count": row["reset_count"],
                "max_beam_force_n": row["maximum_beam_normal_force_n_through_passage"],
                "continuation_verified": False,
                "eligible_sensor_packets": alignment["eligible_first_episode_packets"],
                "loaded_reference_frames": alignment["reference_frames"],
                "sensor_pose_verification": "phase-only when historical root/joint state is absent",
                "student_input_eligible": decision_audit["student_input_eligible"],
                "files": {p.name: digest(p) for p in sorted(folder.iterdir())},
            }
            branches.append(branch)
            provenance.append({"group_id": group_id, "action": action, "original": row})
        records.append(
            {
                "group_id": group_id,
                "role": role,
                "split": "development_only",
                **condition,
                "matched_prefix": prefix,
                "branches": branches,
            }
        )
    write_json(out / "groups.json", records)
    write_json(out / "provenance.json", provenance)
    counts = Counter("".join(str(int(v)) for v in r["outcomes"]) for r in records)
    manifest = {
        "schema": "motion2scene_traversal_dataset_v3_aligned",
        "role": "development dataset; no untouched test or navigation claim",
        "analysis_sha256": digest(analysis_path),
        "exporter_sha256": digest(Path(__file__)),
        "alignment_helper_sha256": digest(
            ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_sensor_alignment.py"
        ),
        "groups": len(records),
        "physical_branches": 2 * len(records),
        "source_ids": sorted({r["source"] for r in records}),
        "outcome_counts_walk_d040": dict(counts),
        "groups_sha256": digest(out / "groups.json"),
        "all_failures_retained": True,
        "mechanical_work_available": False,
        "sensor_object_identity_stripped": True,
        "all_raw_sensor_packets_retained": True,
        "eligible_sensor_packets": sum(
            b["eligible_sensor_packets"] for r in records for b in r["branches"]
        ),
        "audited_214d_inputs": sum(
            b["student_input_eligible"] for r in records for b in r["branches"]
        ),
        "branches_with_unknown_reference": sum(
            b["loaded_reference_frames"] is None for r in records for b in r["branches"]
        ),
        "physics": "Isaac Lab / PhysX, frozen SONIC; not hardware",
        "sensor": "ideal collision rays; no measured normals in historical packets",
        "student_inputs": "214 causal features; no scene parameters or physical labels",
        "scope": "fixed approach, single overhead encounter; source ancestry has development overlap",
    }
    write_json(out / "manifest.json", manifest)
    (out / "DATASET_CARD.md").write_text(
        "# Motion2Scene humanoid traversal development data\n\n"
        f"{len(records)} paired encounters; {2 * len(records)} physical simulation branches. "
        "Use `groups.json` to iterate paired options and `np.load(..., allow_pickle=False)` "
        "for trajectory and student arrays. All paths inside groups are relative.\n\n"
        "`student_input.npz` contains deployable decision inputs. `sensor_history.json` "
        "contains ideal ray measurements, while `groups.json` contains privileged "
        "scene geometry and supervision. Do not concatenate scene/outcome metadata into "
        "the student input. Historical rays have no normals; front-face hits cannot "
        "be relabeled as measured ceilings.\n\n"
        "All raw sensor packets remain in their original order. Read "
        "`sensor_alignment.npz` for the per-packet `packet_eligible` mask and "
        "confidence strings; `sensor_alignment.json` records reasons and the "
        "SHA-bound measured reference-bank clock. Sensor capture follows the "
        "physical recorder and command update, so a final post-reset sensor "
        "packet can be ineligible while its same-index physical row remains valid. "
        "Legacy history without recorded root/joint state receives phase-only "
        "confidence, not invented pose verification. `decision_alignment.json` "
        "preserves the separately audited 214D decision and its eligibility. "
        "If loaded reference evidence is unavailable, all history packets are "
        "masked and only an independently audited decision may remain usable.\n\n"
        "All data are development-only. Keep each paired encounter, carrier ancestry, "
        "and repeated layout together for exploratory splits. Such splits do not "
        "retroactively establish a clean held-out test. There are no goal-directed "
        "steering labels, course continuation labels, torque measurements, hardware "
        "trials, or real sensor noise in this release. The data support option-outcome "
        "learning, sensor-memory development, and regression checks for overhead traversal.\n\n"
        "Failures and reset-spanning arrays are retained. Score only the first episode; "
        "a later recovery must not rescue a failed run. Passage time is null for failures. "
        "Measured force maxima and switch counts are separate from time. Missing work "
        "is unknown, never zero.\n\n"
        "`provenance.json` records original local paths, controller hashes, and audit "
        "records for reproducibility. This is a local research package; no new license "
        "is assigned to inherited assets, which are not bundled.\n"
    )
    audit = audit_export(out, reference_out)
    write_json(out / "audit.json", audit)
    archive = Path(str(out) + ".tar.gz")
    with tarfile.open(archive, "x:gz") as handle:
        handle.add(out, arcname=out.name)
    print(json.dumps({**manifest, "archive": str(archive), "archive_sha256": digest(archive)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-release", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--analysis",
        type=Path,
        default=ROOT / "docs/motion2scene/submission/evidence/analysis.json",
    )
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit_export(args.out, args.reference_release)))
    else:
        export(args.out, args.analysis, args.reference_release)
