#!/usr/bin/env python3
"""Fit a multi-option policy with physically verified wait/adapt continuations."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from bundle_motion2scene_sources import closure
from motion2scene_timing_diagnostic import artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import paired_prefix
from gear_sonic.dataset_generation.hallucination.motion2scene_curriculum import (
    curriculum_probabilities,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_imitation import (
    fit_multi_option_policy,
    physical_regret,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_teacher import (
    ScheduledOutcome,
    schedule_teacher,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_sensor_alignment import (
    audit_sensor_alignment,
)


def decision_packet(row, phase):
    document = json.loads(checked(Path(row["sensor"]["path"]), row["sensor"]["sha256"]).read_text())
    packets = [
        packet
        for index, packet in enumerate(document["observations"][: row["first_episode_frames"]])
        if abs(packet["time_s"] - phase) < 1e-8
        and ("_sensor_alignment" not in row or row["_sensor_alignment"]["packet_eligible"][index])
    ]
    return (document["feature_names"], packets[0]) if len(packets) == 1 else (None, None)


def group_key(row):
    # Supervision/provenance identity only, never appended to student features.
    return json.dumps(
        [row["source"], row["physics_seed"], row["condition"], row["beam"]], sort_keys=True
    )


def full_episode_pass(row):
    """A teacher continuation includes the recorded return, not just beam crossing."""
    if not row["pass"] or row["fall_observed"] or row["reset_count"]:
        return False
    switches = row["switches"]
    if not switches:
        return True
    return bool(
        len(switches) == 2
        and switches[0]["from"] == 0
        and switches[0]["to"] > 0
        and 0.2 <= switches[0]["time_s"] <= 0.4
        and switches[1]["from"] == switches[0]["to"]
        and switches[1]["to"] == 0
        and 3.3 <= switches[1]["time_s"] <= 3.5
    )


def validate_result_row(row, manifest):
    cells = [cell for cell in manifest["cells"] if cell["cell_id"] == row["cell_id"]]
    if len(cells) != 1:
        raise ValueError("result row must identify exactly one registered acquisition cell")
    cell = cells[0]
    for row_key, cell_key in (
        ("source", "generation_seed"),
        ("physics_seed", "runtime_seed"),
        ("condition", "condition"),
        ("beam", "beam"),
        ("mode", "multi_option_mode"),
        ("option_index", "option_index"),
        ("decision_time_s", "decision_time_s"),
    ):
        if row[row_key] != cell[cell_key]:
            raise ValueError(f"result row {row_key} differs from registered cell")
    recorded = Path(cell["output"]) / "multi_option_row.json"
    if json.loads(recorded.read_text()) != row:
        raise ValueError("aggregated result differs from recorded physical branch row")
    folder = Path(cell["output"]) / "trajectories"
    if (
        Path(row["trajectory"]["path"]).parent.resolve() != folder.resolve()
        or Path(row["sensor"]["path"]).resolve() != (folder / "reactive_interface.json").resolve()
    ):
        raise ValueError("result trajectory or sensor is outside registered branch")


def validate_manifest_registry(manifest, registry):
    if manifest["implementation"]["checkpoint"] != registry["controller"]:
        raise ValueError("acquisition checkpoint differs from qualified registry")
    neutral = registry["references"][0]["motion"]
    for cell in manifest["cells"]:
        if (
            cell["generation_seed"] != registry["source"]
            or cell["motion"]["sha256"] != neutral["sha256"]
            or Path(cell["motion"]["path"]).resolve() != Path(neutral["path"]).resolve()
        ):
            raise ValueError("acquisition source or neutral motion differs from qualified registry")


def recording_alignment(row):
    document = json.loads(checked(Path(row["sensor"]["path"]), row["sensor"]["sha256"]).read_text())
    payload = load_reset_capture(
        checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
    )
    with np.load(
        checked(Path(row["bank"]["path"]), row["bank"]["sha256"]), allow_pickle=False
    ) as bank:
        reference_frames = bank["joint_pos"].shape[1]
    return audit_sensor_alignment(
        payload, document["observations"], reference_frames=reference_frames
    )


def base_decision_key(example):
    return (
        example["group"],
        example["phase_s"],
        example["state_history_sha256"],
        tuple(example["features"]),
    )


def recorded_history_id(key, phase, payload, packet):
    """Bind recorded tracker inputs/history; this is not a hidden-state snapshot."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [
                key,
                phase,
                {
                    k: packet[k]
                    for k in ("state", "root_pos_w", "root_quat_w", "features", "legal_mask")
                },
            ],
            sort_keys=True,
        ).encode()
    )
    n = round(phase * 50)
    for field in (
        "dof_pos",
        "dof_vel",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "applied_joint_action",
        "action_motion_token",
        "reference_g1_qpos",
        "motion_time_s",
    ):
        value = np.ascontiguousarray(np.asarray(payload[field])[:n])
        digest.update(json.dumps([field, str(value.dtype), value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def student_visit(row, phase, anchor_payload, names, packet, target):
    """Attest a student-visited neutral state and its actually executed continuation."""
    other_names, other = decision_packet(row, phase)
    if other is None or other["active_before"] != 0 or not row["measurement_admitted"]:
        return {"matched": False, "verified_gap": None}
    payload = load_reset_capture(
        checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
    )
    prefix = paired_prefix(anchor_payload, payload, phase)
    matched = (
        prefix["exact_match"]
        and other_names == names
        and all(
            other[k] == packet[k]
            for k in ("state", "root_pos_w", "root_quat_w", "features", "legal_mask")
        )
    )
    gap = None
    if matched and target["teacher_action"] is not None:
        best = target["passage_time_s"][target["teacher_action"]]
        if not full_episode_pass(row):
            gap = 1.0
        else:
            cost = row["costs"]["passage_time_s"]
            if cost is None or not np.isfinite(cost) or cost < 0:
                raise ValueError("successful student requires measured passage time")
            gap = float(max(0, (cost - best) / max(cost, best, 1e-12)))
    return {
        "matched": matched,
        "verified_gap": gap,
        "student_branch": row["trajectory"],
        "student_pass": full_episode_pass(row),
        "student_passage_endpoint_pass": row["pass"],
        "student_passage_time_s": row["costs"]["passage_time_s"],
        "prefix": prefix,
    }


def train(results, out, student_results=(), base_training=None):
    groups = defaultdict(list)
    option_ids = None
    seen = set()
    registry_hash = None
    expected_schedules = None
    for path in results:
        result = json.loads(path.read_text())
        manifest = json.loads(
            checked(Path(result["manifest"]["path"]), result["manifest"]["sha256"]).read_text()
        )
        if option_ids is None:
            option_ids, registry_hash = manifest["option_ids"], manifest["registry"]["sha256"]
            registry = json.loads(
                checked(Path(manifest["registry"]["path"]), registry_hash).read_text()
            )
            if option_ids != [reference["name"] for reference in registry["references"]]:
                raise ValueError("manifest options differ from qualified registry contents")
            expected_schedules = [(0, None)] + [
                (index, phase)
                for index, reference in enumerate(registry["references"])
                if index
                for phase in reference["qualified_entry_times_s"]
            ]
        elif (
            option_ids != manifest["option_ids"] or registry_hash != manifest["registry"]["sha256"]
        ):
            raise ValueError(
                "all teacher schedules must share the qualified online option registry"
            )
        validate_manifest_registry(manifest, registry)
        for row in result["rows"]:
            validate_result_row(row, manifest)
            if row["mode"] != "forced":
                continue
            identity = (row["trajectory"]["path"], row["trajectory"]["sha256"])
            if identity in seen:
                raise ValueError("duplicate recorded teacher branch")
            seen.add(identity)
            # Geometry groups branch acquisition; it is never appended to student features.
            key = group_key(row)
            groups[key].append(
                {
                    **row,
                    "input_result": artifact(path),
                    "_sensor_alignment": recording_alignment(row),
                }
            )
    students = {}
    for path in student_results:
        result = json.loads(path.read_text())
        manifest = json.loads(
            checked(Path(result["manifest"]["path"]), result["manifest"]["sha256"]).read_text()
        )
        if option_ids != manifest["option_ids"] or registry_hash != manifest["registry"]["sha256"]:
            raise ValueError("student and teacher must share qualified options")
        validate_manifest_registry(manifest, registry)
        for row in result["rows"]:
            validate_result_row(row, manifest)
            if row["mode"] != "learned":
                continue
            key = group_key(row)
            if key in students:
                raise ValueError("one preregistered student continuation per encounter required")
            students[key] = {**row, "_sensor_alignment": recording_alignment(row)}
    if student_results and not students:
        raise ValueError("no actual learned student executions")
    base_decisions = None
    if base_training is not None:
        if not students:
            raise ValueError("dataset aggregation requires actual student visits")
        base_decisions = {
            base_decision_key(example) for example in json.loads(base_training.read_text())
        }
    examples, feature_names = [], None
    for key, rows in groups.items():
        walks = [row for row in rows if row["option_index"] == 0 and row["measurement_admitted"]]
        if not walks:
            raise ValueError(
                "a measured neutral continuation is required for matched approach states"
            )
        anchor = walks[0]
        anchor_payload = load_reset_capture(
            checked(Path(anchor["trajectory"]["path"]), anchor["trajectory"]["sha256"])
        )
        payloads = {
            row["trajectory"]["sha256"]: load_reset_capture(
                checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
            )
            for row in rows
        }
        for phase in (0.2, 0.3, 0.4):
            names, packet = decision_packet(anchor, phase)
            if packet is None:
                continue
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("teacher feature names changed")
            tick = round(phase * 50)
            prefix_hash = recorded_history_id(key, phase, anchor_payload, packet)
            schedules, checks = [], []
            for row in rows:
                entry = None if row["option_index"] == 0 else row["decision_time_s"]
                if entry is not None and entry < phase - 1e-8:
                    continue
                other_names, other = decision_packet(row, phase)
                if other is None or other["active_before"] != 0:
                    continue
                prefix = paired_prefix(anchor_payload, payloads[row["trajectory"]["sha256"]], phase)
                matched = (
                    prefix["exact_match"]
                    and other_names == names
                    and all(
                        other[k] == packet[k]
                        for k in ("state", "root_pos_w", "root_quat_w", "features", "legal_mask")
                    )
                )
                checks.append({"branch": row["trajectory"], "matched": matched, "prefix": prefix})
                schedules.append(
                    ScheduledOutcome(
                        row["trajectory"]["path"],
                        row["option_index"],
                        entry,
                        row["physics_seed"],
                        full_episode_pass(row),
                        row["costs"]["passage_time_s"],
                        bool(row["measurement_admitted"] and matched),
                        {tick: prefix_hash},
                        row["acquisition"]["physics_steps"],
                    )
                )
            target = schedule_teacher(
                schedules,
                phase,
                prefix_hash,
                anchor["physics_seed"],
                packet["legal_mask"],
                expected_schedules=expected_schedules,
            )
            visit = (
                student_visit(students[key], phase, anchor_payload, names, packet, target)
                if key in students
                else {"matched": False, "verified_gap": None}
            )
            identity = (key, phase, prefix_hash, tuple(packet["features"]))
            if (
                base_decisions is not None
                and identity not in base_decisions
                and not visit["matched"]
            ):
                # New aggregation targets must actually be student-visited states.
                continue
            examples.append(
                {
                    **target,
                    "group": key,
                    "features": packet["features"],
                    "state_history_sha256": prefix_hash,
                    "matching_audit": checks,
                    "student_visit": visit,
                }
            )
    replay_weights, replay = None, None
    if students:
        _, supervised = physical_regret(
            [e["pass_labels"] for e in examples],
            [e["passage_time_s"] for e in examples],
            [e["admitted"] for e in examples],
            [e["legal_mask"] for e in examples],
        )
        keys = list(
            dict.fromkeys(
                e["group"] for e, teach in zip(examples, supervised, strict=True) if teach
            )
        )
        if not keys:
            raise ValueError("no consequential complete teacher groups available for replay")
        gaps, coverage = [], []
        for key in keys:
            subset = [example for example in examples if example["group"] == key]
            known = [
                example["student_visit"]["verified_gap"]
                for example in subset
                if example["student_visit"]["verified_gap"] is not None
            ]
            gaps.append(max(known) if known else None)
            coverage.append(
                tuple(
                    sorted({e["teacher_action"] for e in subset if e["teacher_action"] is not None})
                )
            )
        probabilities = curriculum_probabilities(gaps, coverage)
        counts = {
            key: sum(
                e["group"] == key and teach for e, teach in zip(examples, supervised, strict=True)
            )
            for key in keys
        }
        weights_by_group = dict(zip(keys, probabilities, strict=True))
        replay_weights = [
            weights_by_group[e["group"]] / counts[e["group"]] if teach else 1.0
            for e, teach in zip(examples, supervised, strict=True)
        ]
        replay = {
            "groups": keys,
            "verified_gaps": gaps,
            "probabilities": probabilities.tolist(),
            "supervised_decisions_per_group": [int(counts[key]) for key in keys],
            "effective_group_masses": [
                sum(
                    weight
                    for e, weight, teach in zip(examples, replay_weights, supervised, strict=True)
                    if teach and e["group"] == key
                )
                for key in keys
            ],
            "excluded_groups_without_consequential_targets": sorted(
                {e["group"] for e in examples} - set(keys)
            ),
            "uniform_fraction": 0.2,
            "coverage_fraction": 0.2,
            "regret_fraction": 0.6,
            "coverage": "set of verified cost-optimal immediate actions per encounter",
            "student_visited_decisions": sum(e["student_visit"]["matched"] for e in examples),
            "scope": "actual matched student continuations; no prediction-loss priorities",
        }
    out.mkdir(parents=True, exist_ok=False)
    code = []
    root = Path(__file__).resolve().parents[2]
    for path in sorted(closure([Path(__file__)])):
        snapshot = out / "source_snapshot" / path.relative_to(root)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        code.append({**artifact(path), "snapshot": artifact(snapshot)})
    write_new(out / "teachers.json", examples)
    write_new(
        out / "recording_alignment.json",
        [
            {"trajectory": row["trajectory"], "sensor": row["sensor"], **row["_sensor_alignment"]}
            for row in [*[r for rows in groups.values() for r in rows], *students.values()]
        ],
    )
    write_new(
        out / "registration.json",
        {
            "inputs": [artifact(path) for path in results],
            "student_inputs": [artifact(path) for path in student_results],
            "base_training": None if base_training is None else artifact(base_training),
            "teacher_table": artifact(out / "teachers.json"),
            "recording_alignment": artifact(out / "recording_alignment.json"),
            "registry_sha256": registry_hash,
            "option_ids": option_ids,
            "phases_s": [0.2, 0.3, 0.4],
            "expected_schedules": expected_schedules,
            "steps": 1500,
            "learning_rate": 0.05,
            "l2": 0.001,
            "role": "development finite-schedule teacher; not arbitrary course search or DAgger",
            "implementation": code,
            "replay": replay,
        },
    )
    started = time.monotonic()
    model, report = fit_multi_option_policy(
        [x["features"] for x in examples],
        feature_names,
        option_ids,
        [x["pass_labels"] for x in examples],
        [x["passage_time_s"] for x in examples],
        [x["admitted"] for x in examples],
        [x["legal_mask"] for x in examples],
        sample_weights=replay_weights,
    )
    np.savez_compressed(out / "policy.npz", **model)
    write_new(
        out / "result.json",
        {
            **report,
            "fit_seconds": time.monotonic() - started,
            "policy": artifact(out / "policy.npz"),
            "registration": artifact(out / "registration.json"),
            "unique_physical_branches": len(seen),
        },
    )
    print(json.dumps({**report, "policy": artifact(out / "policy.npz")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--student-results", type=Path, nargs="+", default=[])
    parser.add_argument(
        "--base-training", type=Path, help="previous teachers.json for visited-state aggregation"
    )
    args = parser.parse_args()
    train(args.results, args.out, args.student_results, args.base_training)
