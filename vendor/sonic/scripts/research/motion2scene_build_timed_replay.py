#!/usr/bin/env python3
"""Re-audit114D teachers/students and write immutable physical-gap replay evidence."""

import argparse
import copy
import json
import math
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_collect_timed_schedules import (  # noqa: E402
    analyze_cell,
    analyze_incomplete,
    validate_collection_context,
    verify_manifest,
)
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
from motion2scene_train_timed_schedules import (  # noqa: E402
    ARTIFACT_KEYS,
    COLLECTION_SCHEMA,
    audit_collection,
    history_id,
    validate_invocation,
)

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    definition_digest,
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_curriculum import (  # noqa: E402
    DEFAULT_RULE,
    SCHEMA,
    best_complete_teacher,
    coverage_key,
    deadline_eligibility,
    encounter_replay_weights,
    unique_capture_accounting,
    validate_rule,
    verified_decision_gap,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    HISTORY_FRAMES,
    HISTORY_SECONDS,
    expected_feature_names,
)


def read_bound(ref):
    return json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())


def group_identity(registry_ref, scene_ref, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("explicit nonnegative registered physics seed required")
    return [registry_ref["sha256"], scene_ref["sha256"], seed]


def dependency_identities(manifest):
    refs = manifest["dependencies"]
    result = {str(Path(r["path"]).resolve()): r["sha256"] for r in refs}
    if len(result) != len(refs):
        raise ValueError("duplicate frozen runtime dependency identity")
    return result


def raw_payload(ref):
    path = checked(Path(ref["path"]), ref["sha256"])
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local, hash-checked recorder artifact.
    if path.name == "aborted_raw_recording.pkl":
        if not isinstance(payload, dict) or set(payload) != {0}:
            raise ValueError("one explicit aborted physical environment required")
        payload = dict(payload[0], fps=50)
    return payload


def capture_record(ref, steps, role):
    checked(Path(ref["path"]), ref["sha256"])
    return {
        "path": str(Path(ref["path"]).resolve()),
        "sha256": ref["sha256"],
        "physics_steps": steps,
        "role": role,
    }


def audit_teachers(teachers_path):
    training = json.loads((teachers_path.parent / "result.json").read_text())
    if training["teachers"] != artifact(teachers_path):
        raise ValueError("teacher file differs from its training-result binding")
    registration = read_bound(training["registration"])
    registry_ref = registration["registry"]
    bank = load_verified_registry(Path(registry_ref["path"]), registry_ref["sha256"])
    if len(bank.option_ids) != 7 or len(expected_feature_names(7)) != 114:
        raise ValueError("this release requires the registered seven-option114D interface")
    for source in registration["implementation"]:
        checked(Path(source["snapshot"]["path"]), source["sha256"])
        if source["snapshot"]["sha256"] != source["sha256"]:
            raise ValueError("teacher source snapshot identity differs")
    stored = json.loads(teachers_path.read_text())
    refs = registration["collections"]
    if [g["collection"] for g in stored] != refs:
        raise ValueError("teacher groups differ from registered acquired collections")
    groups, captures = [], []
    for old in stored:
        path = checked(Path(old["collection"]["path"]), old["collection"]["sha256"])
        verified = audit_collection(path, bank, registry_ref)
        if verified != old:
            raise ValueError("stored teacher targets differ from independent physical re-audit")
        result = json.loads(path.read_text())
        manifest = read_bound(result["manifest"])
        scene = read_bound(manifest["scene_definition"])
        neutral = next(r for r in result["rows"] if r["forced_option_id"] == "neutral")
        identity = group_identity(
            registry_ref, manifest["scene_definition"], verified["physics_seed"]
        )
        groups.append(
            {
                **verified,
                "encounter_id": definition_digest(identity),
                "history_group": identity,
                "scene": scene,
                "manifest": manifest,
                "neutral_visibility": neutral.get("observation_timing", []),
                "visibility_source": neutral.get("sensor")
                or neutral.get("raw_artifacts", {}).get("sensor"),
            }
        )
        groups[-1]["unknown_teacher_attempts"] = []
        for row in result["rows"]:
            refs = row.get("raw_artifacts", {})
            original = (
                row.get("trajectory")
                or refs.get("trajectory")
                or refs.get("aborted_raw_recording")
                or refs.get("environment_pairs")
                or refs.get("all_body_contacts")
            )
            if original is not None and row["physics_steps"] is not None:
                captures.append(capture_record(original, row["physics_steps"], "teacher"))
            else:
                groups[-1]["unknown_teacher_attempts"].append(
                    {
                        "cell_id": row["cell_id"],
                        "attempt": row["attempt"],
                        "reason": "teacher_recorded_physics_accounting_unavailable",
                        "role": "teacher",
                    }
                )

    if len({g["encounter_id"] for g in groups}) != len(groups):
        raise ValueError("duplicate teacher scene/seed encounter")
    if sum(g["physics_steps"] for g in groups) != training["source_physics_steps"]:
        raise ValueError("teacher physical accounting differs from training receipt")
    return bank, registry_ref, groups, captures


def partial_source_bound(cell, manifest, bank, scene, interface):
    """Conservatively require actual capture identities even for a known early failure."""
    path = Path(cell["output"]) / "success_manifest.json"
    loaded_path = path.parent / "trajectories/loaded_reference_bank.json"
    if not path.exists() or not loaded_path.exists():
        return False
    runtime = json.loads(path.read_text())
    route_errors = json.loads(loaded_path.read_text()).get("source_route_max_error_m", [])
    return bool(
        route_errors
        and all(
            type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1e-4 for v in route_errors
        )
        and runtime["capture_context"]["use_encoder"] == "g1"
        and runtime["checkpoint_hash"] == bank.request["controller"]["sha256"]
        and runtime["capture_context"]["motion"]["hash"]
        == bank.request["references"][0]["motion"]["sha256"]
        and runtime["capture_context"]["scene"]["hash"] == scene["scene"]["sha256"]
        and interface.get("timed_schedule_mode") == cell["timed_schedule_mode"]
        and interface.get("timed_policy_sha256") == manifest["policy"]["sha256"]
        and interface.get("timed_registry_sha256") == manifest["registry"]["sha256"]
        and interface.get("timed_request_digest") == definition_digest(bank.request)
        and interface.get("feature_names") == list(expected_feature_names(len(bank.option_ids)))
        and interface.get("loaded_reference_ids")
        == [r["reference_id"] for r in bank.request["references"]]
        and interface.get("loaded_reference_frames")
        == [bank.frame_count] * len(bank.request["references"])
    )


def audit_students(path, bank, registry_ref):
    result = json.loads(path.read_text())
    manifest = read_bound(result["manifest"])
    archived, archived_bank, scene = verify_manifest(
        Path(result["manifest"]["path"]).parent, execution=False
    )
    if (
        result["schema"] != COLLECTION_SCHEMA
        or manifest != archived
        or manifest["split"] != "development"
        or scene["split"] != "development"
        or manifest["registry"] != registry_ref
        or definition_digest(archived_bank.request) != definition_digest(bank.request)
        or manifest["sensor"]
        != dict(
            rays_per_tick=65,
            history_seconds=HISTORY_SECONDS,
            history_frames=HISTORY_FRAMES,
            delay_s=0,
        )
        or not manifest.get("policy")
    ):
        raise ValueError(
            "actual learned development student must bind the common registry and sensor"
        )
    cells = {c["cell_id"]: c for c in manifest["cells"]}
    rows = {r["cell_id"]: r for r in result["rows"]}
    if (
        len(cells) != len(manifest["cells"])
        or len(rows) != len(result["rows"])
        or not set(rows).issubset(cells)
        or len({str(Path(c["output"]).resolve()) for c in cells.values()}) != len(cells)
    ):
        raise ValueError("unique registered student cells and actual result identities required")
    students, captures, unknown_attempts = [], [], []
    for cell_id, cell in cells.items():
        identity = group_identity(registry_ref, manifest["scene_definition"], cell["runtime_seed"])
        if cell["timed_schedule_mode"] != "learned":
            raise ValueError("replay input must be actual learned student episodes")
        base = dict(
            encounter_id=definition_digest(identity),
            history_group=identity,
            collection=artifact(path),
            manifest=manifest,
            cell=cell,
            phase_records={},
            source_admitted=False,
            task_outcome_admitted=False,
            passed=None,
            passage_time_s=None,
        )
        if cell_id not in rows:
            students.append({**base, "reason": "registered_student_result_missing"})
            unknown_attempts.append(
                {"cell_id": cell_id, "reason": "missing_result_and_unknown_steps"}
            )
            continue
        row = rows[cell_id]
        if row["mode"] != "learned" or row["forced_option_id"] != cell["forced_option_id"]:
            raise ValueError("stored student mode/preference differs from registered invocation")
        validate_collection_context(cell, manifest, bank, scene, actual=True)
        attempt_path = Path(cell["output"]) / "attempt.json"
        if row["attempt"] != artifact(attempt_path):
            raise ValueError("student actual attempt hash differs")
        attempt = json.loads(attempt_path.read_text())
        partial = "raw_artifacts" in row
        if partial:
            verified = analyze_incomplete(
                cell, manifest, bank, scene, attempt, row["analysis_errors"][0]
            )
            refs = verified["raw_artifacts"]
            for key in (
                "raw_artifacts",
                "physics_steps",
                "outcome",
                "task_outcome_admitted",
                "costs",
            ):
                if row[key] != verified[key]:
                    raise ValueError(f"student incomplete {key} differs from physical re-audit")
            trajectory = refs.get("trajectory") or refs.get("aborted_raw_recording")
            payload = None if trajectory is None else raw_payload(trajectory)
            interface = read_bound(refs["sensor"]) if refs.get("sensor") else {"observations": []}
            source_admitted = partial_source_bound(cell, manifest, bank, scene, interface)
            capture = trajectory or refs.get("environment_pairs") or refs.get("all_body_contacts")
        else:
            validate_invocation(cell, row, scene)
            verified, payload, interface = analyze_cell(cell, manifest, bank, scene)
            for key in ARTIFACT_KEYS + (
                "measurement_admitted",
                "pass",
                "costs",
                "schedule_audit",
                "observation_timing",
                "physics_steps",
            ):
                if row[key] != verified[key]:
                    raise ValueError(
                        f"student stored {key} differs from independent physical re-audit"
                    )
            for key in ("outcome", "task_outcome_admitted"):
                if key in row and row[key] != verified[key]:
                    raise ValueError(f"student stored {key} differs from outcome re-audit")
            source_admitted = partial_source_bound(cell, manifest, bank, scene, interface)
            capture = verified["trajectory"]
        steps = verified["physics_steps"]
        if capture is not None and steps is not None:
            captures.append(capture_record(capture, steps, "student"))
        else:
            unknown_attempts.append(
                {
                    "cell_id": cell_id,
                    "reason": "physical_accounting_unavailable",
                    "attempt": row["attempt"],
                }
            )
        outcome = verified["outcome"]
        phase_records = {}
        for phase in outcome["phase_availability"]:
            if not phase["sensor_preaction_available"] or payload is None:
                continue
            tick = phase["tick"]
            packet = interface["observations"][phase["packet_index"]]
            try:
                digest = history_id(identity, payload, interface, tick)
            except (ValueError, KeyError, TypeError):
                continue  # Retain unavailable prefix; never fabricate recorded hidden state.
            phase_records[tick] = {
                "recorded_history_sha256": digest,
                "features": packet["features"],
                "legal_mask": packet["legal_mask"],
                "neutral_phase_available": True,
                "actual_selected_option_id": packet["selected_option_id"],
            }
        students.append(
            {
                **base,
                "source_admitted": source_admitted,
                "task_outcome_admitted": outcome["task_outcome"] != "unknown",
                "passed": outcome["task_outcome"] == "pass",
                "passage_time_s": verified["costs"]["passage_time_s"],
                "phase_records": phase_records,
                "outcome": outcome,
                "actual_trajectory": capture,
                "attempt": row["attempt"],
            }
        )
    return students, captures, unknown_attempts


def validate_online_acquisition(receipt_path, groups, students, registry_ref):
    """Permit historical gaps only through an adopted, ordered acquisition prefix."""
    from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (
        validate_completed_replay_order,
    )

    receipt = json.loads(receipt_path.read_text())
    plan = read_bound(receipt["plan"])
    validate_completed_replay_order(plan, receipt)
    entries = receipt["entries"]
    if (
        not entries
        or [e["round_index"] for e in entries] != list(range(len(entries)))
        or [g["collection"] for g in groups] != [e["teacher_collection"] for e in entries]
    ):
        raise ValueError("replay teachers must be the exact completed acquisition prefix")
    by_collection = {s["collection"]["path"]: s for s in students}
    if len(by_collection) != len(students) or set(by_collection) != {
        e["student_collection"]["path"] for e in entries[1:]
    }:
        raise ValueError("actual students must match every completed pre-update acquisition round")
    records = {}
    for entry in entries[1:]:
        index = entry["round_index"]
        student = by_collection[entry["student_collection"]["path"]]
        training = read_bound(entry["model_training_result"])
        registration = read_bound(training["registration"])
        allowed = [e["teacher_collection"] for e in entries[:index]]
        if (
            student["encounter_id"] != groups[index]["encounter_id"]
            or student["collection"] != entry["student_collection"]
            or student["manifest"]["policy"] != entry["student_model"]
            or training.get("policy") != entry["student_model"]
            or registration["registry"] != registry_ref
            or registration["collections"] != allowed
            or entry["teacher_collection"] in registration["collections"]
        ):
            raise ValueError(
                "pre-update model must exclude current/future labels and bind actual student"
            )
        checked(Path(entry["student_model"]["path"]), entry["student_model"]["sha256"])
        trained_groups = read_bound(training["teachers"])
        if [g["collection"] for g in trained_groups] != allowed or training.get("audit_errors"):
            raise ValueError(
                "pre-update model teaching evidence differs from allowed earlier rounds"
            )
        for prior, actual in zip(trained_groups, groups[:index], strict=True):
            for key in ("collection", "targets", "trajectory_identities", "physics_steps"):
                if prior[key] != actual[key]:
                    raise ValueError(
                        "pre-update teacher evidence differs from re-audited earlier labels"
                    )
        records[student["encounter_id"]] = {
            "round_index": index,
            "generating_model_round_index": index - 1,
            "latest_completed_round": len(entries) - 1,
            "gap_age_in_completed_rounds": len(entries) - 1 - index,
            "student_model": entry["student_model"],
            "model_training_result": entry["model_training_result"],
            "model_training_registration": training["registration"],
            "training_l2": registration["l2"],
            "training_collections": allowed,
            "current_group_labels_excluded": True,
            "scope": "historical pre-update student gap; not an error of the current or final model",
        }
    return {
        "receipt": artifact(receipt_path),
        "plan": receipt["plan"],
        "run_id": receipt["run_id"],
        "rounds": records,
        "scope": "ordered historical gaps, each bound to its own pre-update model",
    }


def compute_evidence(teachers_path, student_paths, rule, acquisition_receipt=None):
    """Deterministically re-audit bound inputs without writing snapshots or fitting."""
    validate_rule(rule)
    if not student_paths or len({p.resolve() for p in student_paths}) != len(student_paths):
        raise ValueError("distinct actual student result files required")
    bank, registry_ref, groups, captures = audit_teachers(teachers_path)
    students = []
    unknown = [a for g in groups for a in g["unknown_teacher_attempts"]]
    for path in student_paths:
        actual, recorded, unavailable = audit_students(path, bank, registry_ref)
        students.extend(actual)
        captures.extend(recorded)
        unknown.extend(unavailable)
    indexed = {s["encounter_id"]: s for s in students}
    if len(indexed) != len(students):
        raise ValueError("one actual frozen student episode per scene/seed encounter required")
    model_hashes = {s["manifest"]["policy"]["sha256"] for s in students}
    online = None
    if acquisition_receipt is not None:
        online = validate_online_acquisition(acquisition_receipt, groups, students, registry_ref)
    elif len(model_hashes) != 1:
        raise ValueError("replay comparison requires one frozen student model")
    records = []
    for group in groups:
        student = indexed.get(group["encounter_id"])
        for target in group["targets"]:
            teacher = best_complete_teacher(target)
            continuation = None if teacher is None else teacher["continuation_option_index"]
            entry = (
                None
                if not continuation
                else bank.request["options"][continuation - 1]["entry_tick"]
            )
            deadline = deadline_eligibility(
                group["neutral_visibility"],
                group["scene"]["beam_collision_enabled"],
                target["phase_tick"],
                entry,
                rule,
            )
            matched = None
            runtime_same = None
            if student is not None:
                phase = student["phase_records"].get(target["phase_tick"], {})
                runtime_same = dependency_identities(group["manifest"]) == dependency_identities(
                    student["manifest"]
                )
                matched = {
                    **student,
                    **phase,
                    "source_admitted": student["source_admitted"] and runtime_same,
                    "matched_history": phase.get("recorded_history_sha256")
                    == target.get("recorded_history_sha256"),
                }
            evidence = verified_decision_gap(target, matched, deadline)
            records.append(
                {
                    "encounter_id": group["encounter_id"],
                    "phase_tick": target["phase_tick"],
                    "supervision_available": teacher is not None,
                    "coverage_key": coverage_key(bank, group["scene"], target, rule),
                    **evidence,
                    "deadline": deadline,
                    "visibility_source": group["visibility_source"],
                    "recorded_history_sha256": target.get("recorded_history_sha256"),
                    "teacher_collection": group["collection"],
                    "student_collection": None if student is None else student["collection"],
                    "student_trajectory": (
                        None if student is None else student.get("actual_trajectory")
                    ),
                    "student_actual_selected_option_id": (
                        None if matched is None else matched.get("actual_selected_option_id")
                    ),
                    "student_actual_passage_time_s": (
                        None if student is None else student["passage_time_s"]
                    ),
                    "student_actual_passed": None if student is None else student["passed"],
                    "online_acquisition": (
                        None if online is None else online["rounds"].get(group["encounter_id"])
                    ),
                    "matching": {
                        "registered_history_group": group["history_group"],
                        "runtime_dependency_identities_equal": runtime_same,
                        "student_source_admitted": (
                            None if student is None else student["source_admitted"]
                        ),
                        "student_recorded_history_sha256": (
                            None if matched is None else matched.get("recorded_history_sha256")
                        ),
                        "neutral_phase_available": matched is not None
                        and matched.get("neutral_phase_available") is True,
                        "feature_values_equal": matched is not None
                        and matched.get("features") == target.get("features"),
                        "legal_masks_equal": matched is not None
                        and matched.get("legal_mask") == target.get("legal_mask"),
                    },
                }
            )
    replay = encounter_replay_weights(records)
    evidence = dict(
        schema=SCHEMA,
        registry=registry_ref,
        teachers=artifact(teachers_path),
        student_model_sha256=next(iter(model_hashes)) if online is None else None,
        online_acquisition=online,
        rule=rule,
        records=records,
        replay=replay,
        accounting=unique_capture_accounting(captures),
        unknown_attempts=unknown,
        unmatched_students=[
            s["collection"]
            for s in students
            if s["encounter_id"] not in {g["encounter_id"] for g in groups}
        ],
        new_training_runs=0,
        scope=(
            "actual full student outcomes; neutral prefixes may already exist in the finite teacher bank"
        ),
    )
    return json.loads(json.dumps(evidence, sort_keys=True, allow_nan=False))


def build(teachers_path, student_paths, rule, out, acquisition_receipt=None):
    validate_rule(rule)
    if not student_paths or len({p.resolve() for p in student_paths}) != len(student_paths):
        raise ValueError("distinct actual student result files required")
    refs = [artifact(path) for path in student_paths]
    out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__).resolve()])):
        destination = out / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(destination)})
    write_new(
        out / "registration.json",
        dict(
            schema=SCHEMA,
            teachers=artifact(teachers_path),
            student_results=refs,
            acquisition_receipt=(
                None if acquisition_receipt is None else artifact(acquisition_receipt)
            ),
            rule=rule,
            implementation=sources,
            new_physics_steps=0,
            scope="CPU evidence/weights only; no training, physics or new-state claim",
        ),
    )
    evidence = compute_evidence(teachers_path, student_paths, rule, acquisition_receipt)
    evidence["registration"] = artifact(out / "registration.json")
    records, replay = evidence["records"], evidence["replay"]
    write_new(out / "evidence.json", evidence)
    write_new(
        out / "weights.json",
        {
            "schema": SCHEMA,
            "evidence": artifact(out / "evidence.json"),
            "rows": [
                dict(
                    encounter_id=row["encounter_id"],
                    phase_tick=row["phase_tick"],
                    teacher_collection=row["teacher_collection"],
                    recorded_history_sha256=row["recorded_history_sha256"],
                    weight=weight,
                )
                for row, weight in zip(records, replay["weights"], strict=True)
            ],
        },
    )
    return {
        "evidence": artifact(out / "evidence.json"),
        "weights": artifact(out / "weights.json"),
        "teacher_phase_rows": len(records),
        "known_gaps": sum(r["gap"] is not None for r in records),
        "accounting": evidence["accounting"],
    }


def verify_weights(weights_path, groups, registry_ref):
    """Recompute every receipt and align weights to all current group/phase targets.

    No snapshots, temporary training directories, fits or physics are created.
    Zero weights denote unavailable supervision and remain explicit in the return.
    """
    weights = json.loads(weights_path.read_text())
    if weights.get("schema") != SCHEMA:
        raise ValueError("explicit physical replay-weight schema required")
    evidence = read_bound(weights["evidence"])
    registration = read_bound(evidence["registration"])
    if registration["schema"] != SCHEMA or evidence["registry"] != registry_ref:
        raise ValueError("replay registry differs from current training registry")
    for source in registration["implementation"]:
        if source["snapshot"]["sha256"] != source["sha256"]:
            raise ValueError("replay implementation source snapshot differs")
        checked(Path(source["snapshot"]["path"]), source["sha256"])
    teachers_path = checked(
        Path(registration["teachers"]["path"]), registration["teachers"]["sha256"]
    )
    if json.loads(teachers_path.read_text()) != groups:
        raise ValueError("replay teacher targets differ from current independently audited groups")
    student_paths = [
        checked(Path(ref["path"]), ref["sha256"]) for ref in registration["student_results"]
    ]
    online = registration.get("acquisition_receipt")
    online_path = None if online is None else checked(Path(online["path"]), online["sha256"])
    rebuilt = compute_evidence(teachers_path, student_paths, registration["rule"], online_path)
    if rebuilt != {k: v for k, v in evidence.items() if k != "registration"}:
        raise ValueError("replay evidence differs from independent physical-gap recomputation")
    expected_rows = [
        dict(
            encounter_id=row["encounter_id"],
            phase_tick=row["phase_tick"],
            teacher_collection=row["teacher_collection"],
            recorded_history_sha256=row["recorded_history_sha256"],
            weight=weight,
        )
        for row, weight in zip(rebuilt["records"], rebuilt["replay"]["weights"], strict=True)
    ]
    if weights["rows"] != expected_rows:
        raise ValueError("stored weights differ from recomputed evidence or exact target alignment")
    expected_targets = [
        (g["collection"], t["phase_tick"], t["recorded_history_sha256"])
        for g in groups
        for t in g["targets"]
    ]
    if [
        (r["teacher_collection"], r["phase_tick"], r["recorded_history_sha256"])
        for r in expected_rows
    ] != expected_targets:
        raise ValueError("replay weight order differs from current complete target slots")
    return {
        "weights_in_group_target_order": [r["weight"] for r in expected_rows],
        "rows": expected_rows,
        "weights_artifact": artifact(weights_path),
        "evidence_artifact": weights["evidence"],
        "distribution": rebuilt["replay"]["distribution"],
        "accounting": rebuilt["accounting"],
        "online_acquisition": rebuilt["online_acquisition"],
        "new_physics_steps": 0,
        "new_training_runs": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--student-results", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--rule", type=Path, help="Explicit rule JSON, otherwise fixed documented defaults"
    )
    parser.add_argument(
        "--acquisition-receipt",
        type=Path,
        help="Adopted ordered online-prefix receipt; default uses one frozen model",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay_rule = json.loads(args.rule.read_text()) if args.rule else copy.deepcopy(DEFAULT_RULE)
    print(
        json.dumps(
            build(
                args.teachers, args.student_results, replay_rule, args.out, args.acquisition_receipt
            )
        )
    )
