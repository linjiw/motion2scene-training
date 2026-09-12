#!/usr/bin/env python3
"""Re-audit complete physical schedules and fit repeated neutral decision heads."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
from motion2scene_collect_timed_schedules import (  # noqa: E402
    COLLECTION_SCHEMA,
    analyze_cell,
    analyze_incomplete,
    validate_collection_context,
    verify_manifest,
)
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_integer_prefix import (  # noqa: E402
    PREFIX_KEYS,
    paired_prefix_on_ticks,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_sensor_alignment import (  # noqa: E402
    audit_sensor_alignment,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    definition_digest,
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    TimedScheduledOutcome,
    fit_timed_schedule_policy,
    schedule_layout,
    timed_schedule_teacher,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    HISTORY_FRAMES,
    HISTORY_SECONDS,
    expected_feature_names,
    validate_schedule_policy,
)

ARTIFACT_KEYS = (
    "trajectory",
    "sensor",
    "features",
    "all_body_contacts",
    "environment_pairs",
    "environment_mapping",
    "physics_beam_contacts",
)
PREACTION_KEYS = (
    "tick",
    "state",
    "root_pos_w",
    "root_quat_w",
    "features",
    "legal_mask",
    "active_before",
    "measurements",
    "normal_known_mask",
    "capture_elapsed_s",
    "delivered_capture_elapsed_s",
)


def preaction_prefix(interface, tick):
    """Exclude the current postdecision action, retain causal sensor and robot history."""
    rows = interface["observations"][:tick]
    if len(rows) != tick or [row["tick"] for row in rows] != list(range(1, tick + 1)):
        raise ValueError("complete consecutive preaction sensor history required")
    if rows[-1]["active_before"] != "neutral":
        raise ValueError("teacher requires an actual neutral preaction decision")
    return [{key: row[key] for key in PREACTION_KEYS} for row in rows]


def history_id(group, payload, interface, tick):
    """Bind recorded history; the simulator's unrecorded hidden state is not claimed."""
    paired_prefix_on_ticks(payload, payload, tick / 50)
    digest = hashlib.sha256()
    digest.update(json.dumps([group, preaction_prefix(interface, tick)], sort_keys=True).encode())
    for key in PREFIX_KEYS:
        value = np.ascontiguousarray(np.asarray(payload[key])[:tick])
        digest.update(json.dumps([key, str(value.dtype), value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def validate_invocation(cell, row, scene):
    checked(Path(scene["scene"]["path"]), scene["scene"]["sha256"])
    path = Path(cell["output"]) / "attempt.json"
    if row["attempt"] != artifact(path):
        raise ValueError("stored attempt differs from registered physical branch")
    attempt = json.loads(path.read_text())
    command = cell["command"]
    if attempt["exit_status"] != 0 or attempt["command"] != command:
        raise ValueError("successful actual invocation must match registered command")
    if (
        command.count("--extra") != 1
        or command.index("--extra") + 1 >= len(command)
        or command[command.index("--extra") + 1] != " ".join(cell["hydra_overrides"])
    ):
        raise ValueError("actual invocation and declared effective overrides disagree")
    seeds = [v for v in cell["hydra_overrides"] if v.lstrip("+").split("=", 1)[0] == "seed"]
    if len(seeds) != 1 or seeds[0].split("=", 1)[1] != str(cell["runtime_seed"]):
        raise ValueError("actual seed override differs from registered physics seed")
    runtime = json.loads((path.parent / "success_manifest.json").read_text())
    actual = runtime["capture_context"]["scene"]
    if (
        runtime["capture_context"]["scene_id"] != scene["scene_id"]
        or actual["hash"] != scene["scene"]["sha256"]
        or Path(actual["resolved"]).resolve() != Path(scene["scene"]["path"]).resolve()
    ):
        raise ValueError("actual runtime scene differs from registered native scene")


def audit_branch(cell, row, manifest, bank, scene):
    """Re-audit measured failure prefixes as well as complete physical episodes."""
    if row.get("status") not in ("failed_attempt", "invalid_or_partial_measurement"):
        for key in ARTIFACT_KEYS + ("attempt",):
            checked(Path(row[key]["path"]), row[key]["sha256"])
        validate_invocation(cell, row, scene)
        verified, payload, interface = analyze_cell(cell, manifest, bank, scene)
        keys = ARTIFACT_KEYS + (
            "measurement_admitted",
            "pass",
            "costs",
            "schedule_audit",
            "observation_timing",
            "physics_steps",
        )
        if "outcome" in row:
            keys += ("outcome", "task_outcome_admitted")
        for key in keys:
            if row[key] != verified[key]:
                raise ValueError(f"stored {key} differs from independent physical re-audit")
        return verified, payload, interface
    attempt_path = Path(cell["output"]) / "attempt.json"
    if row["attempt"] != artifact(attempt_path):
        raise ValueError("stored incomplete attempt differs from actual invocation")
    attempt = json.loads(checked(attempt_path, row["attempt"]["sha256"]).read_text())
    validate_collection_context(cell, manifest, bank, scene, actual=True)
    errors = row.get("analysis_errors")
    if not isinstance(errors, list) or not errors:
        raise ValueError("incomplete branch requires its explicit measurement diagnostic")
    verified = analyze_incomplete(cell, manifest, bank, scene, attempt, errors[0])
    for key in (
        "status",
        "measurement_admitted",
        "task_outcome_admitted",
        "pass",
        "physics_steps",
        "outcome",
        "raw_artifacts",
        "costs",
    ):
        if row.get(key) != verified.get(key):
            raise ValueError(f"stored incomplete {key} differs from independent raw re-audit")
    refs = verified["raw_artifacts"]
    payload, interface = None, {"observations": []}
    if refs.get("sensor"):
        interface = json.loads(
            checked(Path(refs["sensor"]["path"]), refs["sensor"]["sha256"]).read_text()
        )
    key = "trajectory" if refs.get("trajectory") else "aborted_raw_recording"
    if refs.get(key):
        ref = refs[key]
        with checked(Path(ref["path"]), ref["sha256"]).open("rb") as handle:
            raw = pickle.load(handle)  # noqa: S301 - independently SHA-bound local recorder data.
        payload = raw if key == "trajectory" else dict(raw[0], fps=50)
    return verified, payload, interface


def available_history(group, audited, tick, bank):
    """Return only a complete causal first-episode neutral prefix at this tick."""
    row, payload, interface = audited
    try:
        if payload is None:
            raise ValueError("no physical prefix capture")
        outcome = row.get("outcome")
        if outcome is not None:
            phase = [p for p in outcome["phase_availability"] if p["tick"] == tick]
            if len(phase) != 1 or phase[0]["sensor_preaction_available"] is not True:
                raise ValueError("no eligible neutral preaction phase receipt")
            prefix = dict(payload)
            for key in (
                "motion_time_s",
                "root_pos_w",
                "root_quat_w",
                "dof_pos",
                "dof_vel",
                "projected_gravity_b",
            ):
                if key in prefix:
                    prefix[key] = np.asarray(prefix[key])[:tick]
            alignment = audit_sensor_alignment(
                prefix, interface["observations"][:tick], reference_frames=bank.frame_count
            )
            if not all(alignment["packet_eligible"]):
                raise ValueError("an earlier sensor packet is ineligible in this causal prefix")
        digest = history_id(group, payload, interface, tick)
        packet = interface["observations"][tick - 1]
        names = expected_feature_names(len(bank.option_ids))
        features, legal = np.asarray(packet["features"]), np.asarray(packet["legal_mask"])
        if (
            features.shape != (len(names),)
            or features.dtype.kind not in "biuf"
            or not np.isfinite(features).all()
            or legal.dtype.kind != "b"
            or legal.shape != (len(bank.option_ids),)
            or not legal[0]
        ):
            raise ValueError("invalid actual preaction features or legality")
        return digest, None
    except (ValueError, KeyError, IndexError, TypeError) as error:
        return None, str(error)


def unavailable_target(tick, reason, count):
    return dict(
        phase_tick=tick,
        available=False,
        unavailable_reason=reason,
        teacher_action=None,
        features=None,
        feature_names=list(expected_feature_names(count)),
        recorded_history_sha256=None,
        pass_labels=None,
        passage_time_s=None,
        admitted=None,
        legal_mask=None,
        complete_legal_action_table=False,
        expected_continuation_counts=None,
        admitted_continuation_counts=None,
        continuation_branch_ids=None,
        continuation_option_indices=None,
        waiting_has_future_adaptation=None,
        scope="no eligible actual neutral history; no labels or continuation values invented",
    )


def audit_collection(path, bank, registry_ref):
    result = json.loads(path.read_text())
    ref = result["manifest"]
    manifest = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    archived, archived_bank, archived_scene = verify_manifest(
        Path(ref["path"]).parent, execution=False
    )
    if (
        archived != manifest
        or definition_digest(archived_bank.request) != definition_digest(bank.request)
        or manifest["sensor"]
        != dict(
            rays_per_tick=65,
            history_seconds=HISTORY_SECONDS,
            history_frames=HISTORY_FRAMES,
            delay_s=0,
        )
    ):
        raise ValueError("archival collection and actual sensor constants must match")
    if (
        result["schema"] != COLLECTION_SCHEMA
        or manifest["schema"] != COLLECTION_SCHEMA
        or manifest["split"] != "development"
        or manifest["registry"] != registry_ref
        or manifest["request_digest"] != definition_digest(bank.request)
    ):
        raise ValueError("matching development collection and verified registry required")
    ref = manifest["scene_definition"]
    scene = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    if scene != archived_scene or scene["split"] != "development":
        raise ValueError("reserved evaluation may not supply training targets")
    cells = {cell["forced_option_id"]: cell for cell in manifest["cells"]}
    rows = {row["forced_option_id"]: row for row in result["rows"]}
    count = len(bank.option_ids)
    if (
        len(manifest["cells"]) != count
        or len(result["rows"]) != count
        or set(cells) != set(bank.option_ids)
        or set(rows) != set(bank.option_ids)
        or len({cell["cell_id"] for cell in cells.values()}) != count
        or len({str(Path(cell["output"]).resolve()) for cell in cells.values()}) != count
        or len({cell["runtime_seed"] for cell in cells.values()}) != 1
        or any(
            type(cell["runtime_seed"]) is not int
            or cell["runtime_seed"] < 0
            or cell["timed_schedule_mode"] != "forced"
            for cell in cells.values()
        )
    ):
        raise ValueError("one unique forced branch per schedule and common physics seed required")
    actual = {}
    for option in bank.option_ids:
        row, cell = rows[option], cells[option]
        if row["cell_id"] != cell["cell_id"] or row["mode"] != "forced":
            raise ValueError("stored result identifies a different registered branch")
        actual[option] = audit_branch(cell, row, manifest, bank, scene)
    phases, _, entries = schedule_layout(bank)
    seed = next(iter(cells.values()))["runtime_seed"]
    group = [registry_ref["sha256"], manifest["scene_definition"]["sha256"], seed]
    anchor = actual["neutral"]
    hashes = {option: {} for option in bank.option_ids}
    matches = []
    anchor_hashes = {}
    for tick in phases.tolist():
        anchor_hash, anchor_error = available_history(group, anchor, tick, bank)
        anchor_hashes[tick] = (anchor_hash, anchor_error)
        for index, option in enumerate(bank.option_ids):
            if entries[index] is not None and entries[index] < tick:
                continue
            digest, error = available_history(group, actual[option], tick, bank)
            prefix = None
            if anchor_hash is not None and digest is not None:
                prefix = paired_prefix_on_ticks(anchor[1], actual[option][1], tick / 50)
            matched = bool(prefix is not None and prefix["exact_match"] and digest == anchor_hash)
            matches.append(
                dict(
                    phase_tick=tick,
                    option_id=option,
                    matched=matched,
                    prefix=prefix,
                    unavailable_reason=error or anchor_error,
                )
            )
            if digest is not None:
                hashes[option][tick] = digest
    branches, branch_assessments = [], []
    for index, option in enumerate(bank.option_ids):
        row = actual[option][0]
        outcome = row.get("outcome")
        admitted = (
            row["measurement_admitted"]
            if outcome is None
            else outcome["task_outcome"] in ("pass", "failure")
        )
        steps = row["physics_steps"]
        outcome_name = (
            outcome["task_outcome"]
            if outcome is not None
            else (("pass" if row["pass"] else "failure") if admitted else "unknown")
        )
        branch_assessments.append(
            dict(
                option_id=option,
                cell_id=cells[option]["cell_id"],
                status=row.get("status", "complete_capture"),
                task_outcome=outcome_name,
                task_outcome_admitted=admitted,
                measurement_admitted=row["measurement_admitted"],
                physics_steps=steps,
                available_prefix_ticks=sorted(hashes[option]),
                outcome=outcome,
                attempt=rows[option]["attempt"],
                raw_artifacts=row.get("raw_artifacts"),
            )
        )
        if hashes[option] and type(steps) is int and steps > 0:
            branches.append(
                TimedScheduledOutcome(
                    branch_id=cells[option]["cell_id"],
                    option_index=index,
                    physics_seed=seed,
                    passed=row["pass"],
                    passage_time_s=row["costs"]["passage_time_s"],
                    admitted=bool(admitted),
                    prefix_hash_by_tick=hashes[option],
                    physics_steps=steps,
                )
            )
    targets = []
    for tick in phases.tolist():
        anchor_hash, error = anchor_hashes[tick]
        if anchor_hash is None:
            targets.append(unavailable_target(tick, error, count))
            continue
        packet = anchor[2]["observations"][tick - 1]
        target = timed_schedule_teacher(
            bank, branches, tick, anchor_hash, seed, np.asarray(packet["legal_mask"], dtype=bool)
        )
        targets.append(
            dict(
                **target,
                available=True,
                features=packet["features"],
                feature_names=list(expected_feature_names(count)),
                recorded_history_sha256=anchor_hash,
            )
        )
    identities = []
    for option in bank.option_ids:
        row = actual[option][0]
        refs = row.get("raw_artifacts", {})
        identity = (
            row.get("trajectory") or refs.get("trajectory") or refs.get("aborted_raw_recording")
        )
        if identity is not None:
            identities.append(identity)
    return dict(
        scene_id=scene["scene_id"],
        physics_seed=seed,
        collection=artifact(path),
        trajectory_identities=identities,
        physics_steps=sum(row["physics_steps"] or 0 for row in branch_assessments),
        physics_steps_unknown_branches=sum(
            row["physics_steps"] is None for row in branch_assessments
        ),
        assigned_branches=count,
        branch_assessments=branch_assessments,
        option_ids=list(bank.option_ids),
        prefix_comparisons=matches,
        targets=targets,
    )


def weighted_targets(groups, weights=None):
    """Align all recorded target slots before excluding unavailable supervision."""
    slots = [target for group in groups for target in group["targets"]]
    if weights is None:
        return [row for row in slots if row.get("available", True)], None
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(slots),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("finite nonnegative weights for every recorded target slot required")
    rows, selected = [], []
    for row, weight in zip(slots, weights, strict=True):
        available = row.get("available", True)
        if not available:
            if weight != 0:
                raise ValueError("unavailable phase cannot receive replay supervision")
            continue
        consequential = (
            row["complete_legal_action_table"]
            and row["teacher_action"] is not None
            and sum(row["legal_mask"]) >= 2
        )
        if weight == 0:
            if consequential:
                raise ValueError(
                    "uniform replay component must retain complete consequential targets"
                )
            continue
        rows.append(row)
        selected.append(weight)
    return rows, np.asarray(selected, dtype=float)


def run(
    registry, collections, out, l2, replay_weights=None, *, allow_measured_tie_initialization=False
):
    if type(allow_measured_tie_initialization) is not bool:
        raise ValueError("explicit boolean measured-tie initialization setting required")
    registry_ref = artifact(registry)
    bank = load_verified_registry(registry, registry_ref["sha256"])
    if not collections or len(set(p.resolve() for p in collections)) != len(collections):
        raise ValueError("distinct nonempty collection paths required")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("finite nonnegative ridge penalty required")
    refs = [artifact(path) for path in collections]
    out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__)])):
        snapshot = out / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(snapshot)})
    write_new(
        out / "registration.json",
        dict(
            schema="motion2scene_timed_schedule_training_v1",
            registry=registry_ref,
            collections=refs,
            l2=l2,
            allow_measured_tie_initialization=allow_measured_tie_initialization,
            replay_weights=None if replay_weights is None else artifact(replay_weights),
            implementation=sources,
            weighting=(
                "uniform development scene-seed groups at each registered decision phase"
                if replay_weights is None
                else "independently recomputed causal physical-gap encounter replay"
            ),
            objective="complete finite-schedule physical regret; passage first then measured time",
            scope="recorded neutral prefixes; no hidden-state snapshot or repeated adaptation claim",
        ),
    )
    groups, audit_errors = [], []
    for path in collections:
        try:
            groups.append(audit_collection(path, bank, registry_ref))
        except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
            audit_errors.append(
                dict(
                    collection=artifact(path),
                    reason=str(error),
                    assigned_branches=len(bank.option_ids),
                    measurement_accounting="unverified",
                )
            )
    # The reviewable teaching receipts survive missing phase coverage or a failed
    # fit. Every assigned group remains represented, including hard audit errors.
    write_new(out / "teachers.json", groups)
    identities = [
        (str(Path(ref["path"]).resolve()), ref["sha256"])
        for group in groups
        for ref in group["trajectory_identities"]
    ]
    if len(set(identities)) != len(identities):
        audit_errors.append(
            dict(reason="same captured episode supplied through multiple collections")
        )
    replay_audit = None
    sample_weights = None
    rows, _ = weighted_targets(groups)
    if replay_weights is not None and not audit_errors:
        # Lazy import: the verifier independently reuses this module's teacher
        # audit, and importing it at module initialization would create a cycle.
        from motion2scene_build_timed_replay import verify_weights

        try:
            replay_audit = verify_weights(replay_weights, groups, registry_ref)
            rows, sample_weights = weighted_targets(
                groups, replay_audit["weights_in_group_target_order"]
            )
        except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
            audit_errors.append(dict(reason="replay verification failed: " + str(error)))
    phases, _, _ = schedule_layout(bank)
    coverage = []
    for tick in phases.tolist():
        selected = [row for row in rows if row["phase_tick"] == tick]
        coverage.append(
            dict(
                phase_tick=tick,
                available_decisions=len(selected),
                complete_consequential_decisions=sum(
                    row["complete_legal_action_table"] and row["teacher_action"] is not None
                    for row in selected
                ),
            )
        )
    base = dict(
        registration=artifact(out / "registration.json"),
        teachers=artifact(out / "teachers.json"),
        source_physics_steps=sum(group["physics_steps"] for group in groups),
        new_physics_steps=0,
        assigned_groups=len(collections),
        assigned_branches=len(collections) * len(bank.option_ids),
        audited_groups=len(groups),
        audit_errors=audit_errors,
        replay_audit=replay_audit,
        physics_steps_unknown_branches=sum(g["physics_steps_unknown_branches"] for g in groups),
        phase_coverage=coverage,
        unverified_assigned_branches=sum(
            error.get("assigned_branches", 0) for error in audit_errors
        ),
        source_physics_steps_complete=(
            not audit_errors
            and all(group["physics_steps_unknown_branches"] == 0 for group in groups)
        ),
        unavailable_phase_decisions=sum(
            not target.get("available", True) for group in groups for target in group["targets"]
        ),
    )
    if audit_errors or any(item["complete_consequential_decisions"] == 0 for item in coverage):
        status = "audit_failed" if audit_errors else "insufficient_phase_evidence"
        write_new(out / "result.json", dict(**base, status=status, policy=None, fit=None))
        print(
            json.dumps(
                dict(status=status, policy=None, phase_coverage=coverage, audit_errors=audit_errors)
            )
        )
        return
    try:
        model, report = fit_timed_schedule_policy(
            bank,
            np.asarray([row["features"] for row in rows]),
            expected_feature_names(len(bank.option_ids)),
            np.asarray([row["phase_tick"] for row in rows]),
            np.asarray([row["pass_labels"] for row in rows], dtype=bool),
            np.asarray([row["passage_time_s"] for row in rows], dtype=float),
            np.asarray([row["admitted"] for row in rows], dtype=bool),
            np.asarray([row["legal_mask"] for row in rows], dtype=bool),
            l2=l2,
            sample_weights=sample_weights,
            allow_measured_tie_initialization=allow_measured_tie_initialization,
        )
        validate_schedule_policy(model, bank)
    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
        write_new(
            out / "result.json",
            dict(
                **base, status="fit_validation_failed", policy=None, fit=None, fit_error=str(error)
            ),
        )
        print(json.dumps(dict(status="fit_validation_failed", policy=None, reason=str(error))))
        return
    np.savez_compressed(out / "policy.npz", **model)
    write_new(
        out / "result.json",
        dict(**base, status="complete", policy=artifact(out / "policy.npz"), fit=report),
    )
    print(json.dumps({"policy": artifact(out / "policy.npz"), "fit": report}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--replay-weights", type=Path)
    parser.add_argument("--allow-measured-tie-initialization", action="store_true")
    args = parser.parse_args()
    run(
        args.registry,
        args.collections,
        args.out,
        args.l2,
        args.replay_weights,
        allow_measured_tie_initialization=args.allow_measured_tie_initialization,
    )
