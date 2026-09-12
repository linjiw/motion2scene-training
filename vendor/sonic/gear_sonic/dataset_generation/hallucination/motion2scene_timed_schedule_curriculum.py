"""Offline replay weights from complete teachers and actual matched student outcomes.

No action lookup, predicted value or classification loss establishes a physical
student outcome. Files and history identities are independently audited by the CLI.
"""

from collections import Counter
from pathlib import Path

import numpy as np

SCHEMA = "motion2scene_timed_schedule_replay_v1"
DEFAULT_RULE = {
    "schema": "motion2scene_timed_schedule_replay_rule_v1",
    "cue": "surface",
    "sensing_margin_s": 0.0,
    "required_beams": "all_collision_enabled",
    "common_safe_wait_override": False,
    "length_edges_m": [0.25, 0.5, 1.0, 2.0],
    "mixture": {"uniform": 0.2, "coverage": 0.2, "verified_gap": 0.6},
    "encounter_gap": "maximum_current_eligible_matched_phase_gap",
    "no_positive_gap_fallback": "uniform_encounters_then_complete_phase_targets",
}


def validate_rule(rule):
    if set(rule) != set(DEFAULT_RULE):
        raise ValueError("explicit complete replay-rule schema required")
    for key in set(rule) - {"cue", "sensing_margin_s", "length_edges_m"}:
        if rule[key] != DEFAULT_RULE[key]:
            raise ValueError(f"unsupported replay rule: {key}")
    edges = np.asarray(rule["length_edges_m"], dtype=float)
    if (
        rule["cue"] not in ("surface", "underside")
        or type(rule["sensing_margin_s"]) not in (int, float)
        or not np.isfinite(rule["sensing_margin_s"])
        or rule["sensing_margin_s"] < 0
        or edges.ndim != 1
        or not len(edges)
        or not np.isfinite(edges).all()
        or edges[0] <= 0
        or (np.diff(edges) <= 0).any()
    ):
        raise ValueError(
            "declared cue, nonnegative timing margin and increasing length bins required"
        )


def best_complete_teacher(target):
    """Return the best admitted immediate action and its complete continuation."""
    if target.get("available") is False or target.get("complete_legal_action_table") is not True:
        return None
    legal = np.asarray(target["legal_mask"])
    admitted = np.asarray(target["admitted"])
    passed = np.asarray(target["pass_labels"])
    expected = np.asarray(target["expected_continuation_counts"])
    actual = np.asarray(target["admitted_continuation_counts"])
    size = len(legal)
    if (
        legal.ndim != 1
        or size < 2
        or legal.dtype.kind != "b"
        or admitted.dtype.kind != "b"
        or passed.dtype.kind != "b"
        or any(x.shape != (size,) for x in (admitted, passed, expected, actual))
        or expected.dtype.kind not in "iu"
        or actual.dtype.kind not in "iu"
        or (expected < 0).any()
        or (actual < 0).any()
        or len(target["passage_time_s"]) != size
        or len(target["continuation_option_indices"]) != size
    ):
        raise ValueError("complete aligned physical teacher arrays required")
    complete = bool(
        target["complete_legal_action_table"] is True
        and legal.any()
        and admitted[legal].all()
        and (expected[legal] > 0).all()
        and np.array_equal(expected[legal], actual[legal])
    )
    if not complete:
        return None
    candidates = np.flatnonzero(legal & passed)
    for index in candidates:
        time = target["passage_time_s"][index]
        continuation = target["continuation_option_indices"][index]
        if (
            type(time) not in (int, float)
            or not np.isfinite(time)
            or time < 0
            or type(continuation) is not int
            or not 0 <= continuation < size
        ):
            raise ValueError(
                "passing teacher requires actual finite cost and continuation identity"
            )
    if not len(candidates):
        return None
    choice = min(candidates.tolist(), key=lambda i: (target["passage_time_s"][i], i))
    if target["teacher_action"] != choice:
        raise ValueError("stored teacher differs from complete measured passage/time ranking")
    return {
        "immediate_action_index": choice,
        "continuation_option_index": target["continuation_option_indices"][choice],
        "passage_time_s": float(target["passage_time_s"][choice]),
        "immediate_wait": choice == 0,
    }


def deadline_eligibility(timing, enabled, tick, continuation_entry_tick, rule):
    """Current cues gate priority; later-entry cues are separately diagnostic.

    The receipts come from the matched actual neutral teacher trajectory. An
    observation after the current phase never becomes a current policy input.
    This is a visibility deadline, not semantic distinguishability or a measured
    transition-to-obstacle timing guarantee.
    """
    validate_rule(rule)
    if type(tick) is not int or tick <= 0 or any(type(v) is not bool for v in enabled):
        raise ValueError("explicit phase and collision-enabled flags required")
    if continuation_entry_tick is not None and (
        type(continuation_entry_tick) is not int or continuation_entry_tick < tick
    ):
        raise ValueError("continuation cannot enter before the neutral decision")

    def phase_receipt(phase):
        if phase is None:
            return None
        selected = [row for row in timing if row["entry_tick"] == phase]
        if len(selected) != 1 or selected[0].get("available") is False:
            return {"eligible": False, "reason": "phase_visibility_unavailable"}
        beams = selected[0].get("beams")
        if beams is None or len(beams) != len(enabled):
            return {"eligible": False, "reason": "complete_beam_visibility_unavailable"}
        deadline = (phase - 1) / 50 - rule["sensing_margin_s"]
        details = []
        for i, active in enumerate(enabled):
            if not active:
                continue
            cue = beams[i][rule["cue"]]
            seen = cue["first_delivery_elapsed_s"]
            valid_seen = type(seen) in (int, float) and np.isfinite(seen) and seen >= 0
            details.append(
                {
                    "beam_index": i,
                    "first_delivery_elapsed_s": seen,
                    "eligible": bool(
                        valid_seen and cue["delivered_by_entry"] is True and seen <= deadline
                    ),
                }
            )
        return {
            "phase_tick": phase,
            "physical_decision_elapsed_s": (phase - 1) / 50,
            "latest_delivery_elapsed_s": deadline,
            "beams": details,
            "eligible": all(row["eligible"] for row in details),
            "reason": "no_enabled_obstacle" if not details else "declared_causal_cue_deadline",
        }

    current = phase_receipt(tick)
    future = phase_receipt(continuation_entry_tick)
    return {
        "cue": rule["cue"],
        "current_phase": current,
        "teacher_continuation_entry": future,
        "current_eligible": current["eligible"],
        "common_safe_wait_override": False,
        "scope": "later entry visibility is diagnostic only; no future cue is credited now",
    }


def verified_decision_gap(target, student, deadline):
    """Compare a verified teacher to the actual full student outcome, never its label."""
    teacher = best_complete_teacher(target)
    reasons = []
    if teacher is None:
        reasons.append("no_complete_feasible_teacher")
    if student is None:
        reasons.append("student_missing")
    else:
        for flag, reason in (
            ("source_admitted", "student_source_unverified"),
            ("task_outcome_admitted", "student_task_outcome_unknown"),
            ("neutral_phase_available", "student_neutral_phase_unavailable"),
            ("matched_history", "student_recorded_history_unmatched"),
        ):
            if student.get(flag) is not True:
                reasons.append(reason)
        if student.get("recorded_history_sha256") != target.get("recorded_history_sha256"):
            reasons.append("student_history_identity_differs")
        if student.get("features") != target.get("features"):
            reasons.append("student_features_differ")
        if student.get("legal_mask") != target.get("legal_mask"):
            reasons.append("student_legality_differs")
    if not deadline["current_eligible"]:
        reasons.append("current_observed_deadline_ineligible")
    if reasons:
        return {"gap": None, "teacher": teacher, "reasons": reasons}
    if type(student["passed"]) is not bool:
        raise ValueError("actual physical student pass/fail is required")
    if not student["passed"]:
        return {"gap": 1.0, "teacher": teacher, "reasons": []}
    cost = student["passage_time_s"]
    if type(cost) not in (int, float) or not np.isfinite(cost) or cost < 0:
        raise ValueError("passing student requires measured full-episode passage time")
    gap = max(0.0, (cost - teacher["passage_time_s"]) / max(cost, teacher["passage_time_s"], 1e-12))
    return {"gap": gap, "teacher": teacher, "reasons": []}


def coverage_key(bank, scene, target, rule):
    """Privileged geometry organizes acquisition only; it is never a student feature."""
    validate_rule(rule)
    teacher = best_complete_teacher(target)
    if teacher is None:
        return None
    index = teacher["continuation_option_index"]
    option = None if index == 0 else bank.request["options"][index - 1]
    lengths = tuple(
        int(np.searchsorted(rule["length_edges_m"], beam["length_m"], side="right"))
        for beam, enabled in zip(scene["beams"], scene["beam_collision_enabled"], strict=True)
        if enabled
    )
    return (
        "neutral" if option is None else option["reference_id"],
        lengths,
        target["phase_tick"],
        None if option is None else option["entry_tick"],
        None if option is None else option["return_tick"],
    )


def encounter_replay_weights(rows):
    """Normalize encounters before phases; extra WAIT visits add no encounter mass."""
    keys = [(r["encounter_id"], r["phase_tick"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("one replay row per encounter and registered phase required")
    groups = {}
    for i, row in enumerate(rows):
        gap = row["gap"]
        if gap is not None and (not np.isfinite(gap) or not 0 <= gap <= 1):
            raise ValueError("verified gap must be None or finite in [0,1]")
        if row["supervision_available"]:
            if row["coverage_key"] is None:
                raise ValueError("supervised target requires declared coverage identity")
            groups.setdefault(row["encounter_id"], []).append(i)
    n = len(groups)
    components = {key: np.zeros(len(rows)) for key in ("uniform", "coverage", "gap", "fallback")}
    if n:
        counts = Counter(
            k for ids in groups.values() for k in {rows[i]["coverage_key"] for i in ids}
        )
        inverse = {
            g: np.array([1 / counts[rows[i]["coverage_key"]] for i in ids])
            for g, ids in groups.items()
        }
        raw_coverage = {g: float(v.mean()) for g, v in inverse.items()}
        group_gaps = {
            g: max((rows[i]["gap"] or 0 for i in ids), default=0) for g, ids in groups.items()
        }
        positive_total = sum(group_gaps.values())
        for group, ids in groups.items():
            components["uniform"][ids] = 0.2 / n / len(ids)
            components["coverage"][ids] = (
                0.2
                * raw_coverage[group]
                / sum(raw_coverage.values())
                * inverse[group]
                / inverse[group].sum()
            )
            if positive_total:
                local = np.array([rows[i]["gap"] or 0 for i in ids])
                if group_gaps[group]:
                    components["gap"][ids] = (
                        0.6 * group_gaps[group] / positive_total * local / local.sum()
                    )
            else:
                components["fallback"][ids] = 0.6 / n / len(ids)
    total = sum(components.values())
    return {
        "weights": total.tolist(),
        "components": {key: value.tolist() for key, value in components.items()},
        "encounter_masses": {g: float(total[ids].sum()) for g, ids in groups.items()},
        "supervised_encounters": n,
        "excluded_rows": [i for i, r in enumerate(rows) if not r["supervision_available"]],
        "no_positive_gap_fallback": bool(n and not np.any(components["gap"])),
        "distribution": {
            "minimum_positive_weight": float(total[total > 0].min()) if (total > 0).any() else None,
            "maximum_weight": float(total.max()) if len(total) else None,
            "maximum_to_minimum_positive_ratio": (
                float(total.max() / total[total > 0].min()) if (total > 0).any() else None
            ),
            "effective_sample_size": (
                float(1 / np.square(total).sum()) if (total > 0).any() else 0.0
            ),
            "component_masses": {key: float(value.sum()) for key, value in components.items()},
            "scope": "descriptive replay distribution; does not establish a learned-policy effect",
        },
        "scope": "maximum verified gap per encounter; no new student-state or DAgger claim",
    }


def unique_capture_accounting(captures):
    """An original resolved capture path counts once, even across teacher/student roles."""
    identities = {}
    for capture in captures:
        path = str(Path(capture["path"]).resolve())
        sha, steps = capture["sha256"], capture["physics_steps"]
        if type(steps) is not int or steps < 0:
            raise ValueError("recorded physics count must be explicitly verified")
        if path in identities and (
            identities[path]["sha256"] != sha or identities[path]["physics_steps"] != steps
        ):
            raise ValueError("one original capture has conflicting hash or physical accounting")
        if path not in identities:
            identities[path] = {**capture, "path": path, "roles": []}
        if capture["role"] not in identities[path]["roles"]:
            identities[path]["roles"].append(capture["role"])
    return {
        "unique_recorded_captures": len(identities),
        "unique_recorded_physics_steps": sum(r["physics_steps"] for r in identities.values()),
        "captures": list(identities.values()),
        "new_physics_steps": 0,
        "unrecorded_startup_physics": "unknown; not inferred from packet counts",
    }
