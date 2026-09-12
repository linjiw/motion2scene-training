"""Keep assigned task outcomes separate from incomplete observation supervision.

Contact and passage receipts must come from their separate native-geometry and
counterpart audits; this module does not turn a geometric proposal into evidence.
"""

import numpy as np

from .motion2scene_sensor_alignment import audit_sensor_alignment


def classify_timed_attempt(
    payload,
    observations,
    *,
    reference_frames,
    phase_ticks,
    exit_status,
    measurement_admitted=False,
    passage=None,
    contact_audit=None,
    schedule_audit=None,
    physics_steps=None,
):
    """Classify one assigned slot and retain only actual eligible neutral phases.

    ``physics_steps`` is the raw recorded physics counter vector, never a count
    inferred from sensor packets. Prefix contact receipts may establish failure
    without establishing full-episode measurement admission. The caller must
    preserve the raw artifacts and distinguish missing files from these values.
    """
    if (
        type(reference_frames) is not int
        or reference_frames < 3
        or not isinstance(phase_ticks, (list, tuple))
        or not phase_ticks
        or any(type(t) is not int or not 1 <= t < reference_frames for t in phase_ticks)
        or len(set(phase_ticks)) != len(phase_ticks)
        or type(measurement_admitted) is not bool
    ):
        raise ValueError("explicit finite reference/phase schema and measurement receipt required")
    observations = observations or []
    diagnostics, events = [], []
    n, complete_physics, valid_clock, fully_valid_physics = 0, False, False, False
    times = np.empty(0)
    root = gravity = None
    if payload is None:
        diagnostics.append("physical_capture_missing")
    else:
        try:
            times = np.asarray(payload["motion_time_s"], dtype=float)
            root = np.asarray(payload["root_pos_w"], dtype=float)
            gravity = np.asarray(payload["projected_gravity_b"], dtype=float)
            n = len(times)
            shapes = times.ndim == 1 and root.shape == gravity.shape == (n, 3) and n > 0
            if not shapes or float(payload["fps"]) != 50 or not np.isfinite(times).all():
                raise ValueError("physical array shape/clock invalid")
            ticks = np.rint(times * 50).astype(np.int64)
            valid_clock = bool(
                ticks[0] == 0
                and (ticks >= 0).all()
                and (ticks < reference_frames).all()
                and np.max(np.abs(times - ticks / 50)) <= 1e-8
                and np.all((np.diff(ticks) == 1) | (np.diff(ticks) < 0))
            )
            if not valid_clock:
                diagnostics.append("physical_reference_clock_invalid")
            else:
                resets = np.flatnonzero(np.diff(ticks) < 0) + 1
                if len(resets):
                    events.append(
                        dict(kind="recorded_physical_reference_reset", indices=resets.tolist())
                    )
                complete_physics = bool(
                    n == reference_frames - 1 and np.array_equal(ticks, np.arange(n))
                )
            finite_rows = np.isfinite(root).all(axis=1) & np.isfinite(gravity).all(axis=1)
            fully_valid_physics = bool(valid_clock and finite_rows.all())
            if not finite_rows.all():
                diagnostics.append("nonfinite_physical_rows")
            # A measured failure before a later corrupt row remains an observed
            # failure; the corrupt row cannot establish any new physical event.
            falls = np.flatnonzero(finite_rows & ((root[:, 2] < 0.5) | (-gravity[:, 2] < 0.5)))
            if valid_clock and len(falls):
                events.append(
                    dict(
                        kind="recorded_fall_or_upright_threshold_failure", first_index=int(falls[0])
                    )
                )
        except (KeyError, TypeError, ValueError, IndexError):
            diagnostics.append("physical_capture_invalid")
    measured_steps = None
    if physics_steps is not None:
        steps = np.asarray(physics_steps)
        if (
            steps.ndim == 1
            and steps.dtype.kind in "iu"
            and np.array_equal(steps, np.arange(1, len(steps) + 1))
        ):
            measured_steps = len(steps)
        else:
            diagnostics.append("physics_counter_stream_invalid")
    if contact_audit is not None:
        if contact_audit.get("complete_synchronized_streams") is True:
            maximum = contact_audit.get("maximum_undesired_environment_force_n")
            if maximum is None:
                maximum = contact_audit.get("maximum_undesired_contact_force_n")
            # The audit's strict outcome incorporates its explicit per-body
            # external counterpart scope; do not use unfiltered net self-force.
            if contact_audit.get("no_undesired_measured_contact") is False:
                events.append(dict(kind="verified_environment_contact", maximum_force_n=maximum))
        else:
            diagnostics.append("contact_measurement_incomplete_or_invalid")
    for index, row in enumerate(observations):
        if (
            not valid_clock
            or index >= n
            or type(row.get("tick")) is not int
            or row["tick"] != int(round(times[index] * 50)) + 1
            or row.get("physics_step") != 4 * (index + 1)
        ):
            continue
        transition = row.get("transition", {})
        if transition.get("allowed") is False:
            events.append(
                dict(
                    kind="recorded_transition_refusal",
                    observation_index=index,
                    tick=row.get("tick"),
                    reason=transition.get("reason"),
                )
            )
            break
        if any(
            transition.get(key) is False
            for key in ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
        ):
            events.append(
                dict(
                    kind="recorded_state_mutating_transition",
                    observation_index=index,
                    tick=row.get("tick"),
                )
            )
            break
    if (
        complete_physics
        and fully_valid_physics
        and passage is not None
        and passage.get("pass") is False
        and passage.get("passage_finish_frame_exclusive") is None
    ):
        events.append(
            dict(
                kind="finite_captured_reference_horizon_before_completion",
                final_physical_phase_s=float(times[-1]),
            )
        )
    if (
        complete_physics
        and fully_valid_physics
        and schedule_audit is not None
        and schedule_audit.get("complete_ticks") is True
        and any(
            schedule_audit.get(key) is False
            for key in (
                "switches_match_declared_schedule",
                "active_timeline_matches",
                "reference_guards_and_state_unchanged",
            )
        )
    ):
        events.append(dict(kind="verified_complete_schedule_contract_failure"))
    aligned, eligible_indices = None, set()
    available_count = min(n, len(observations))
    if payload is not None and available_count and valid_clock:
        # Missing later packets do not erase earlier synchronized measurements.
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
                prefix[key] = np.asarray(prefix[key])[:available_count]
        try:
            aligned = audit_sensor_alignment(
                prefix, observations[:available_count], reference_frames=reference_frames
            )
            eligible_indices = {i for i, ok in enumerate(aligned["packet_eligible"]) if ok}
        except (KeyError, TypeError, ValueError, IndexError):
            diagnostics.append("sensor_physical_alignment_invalid")
    phases = []
    for tick in phase_ticks:
        matches = [
            i
            for i, row in enumerate(observations)
            if row.get("tick") == tick and i in eligible_indices
        ]
        reason = None
        if len(matches) != 1:
            reason = "no_unique_eligible_first_episode_packet"
        else:
            packet = observations[matches[0]]
            features = np.asarray(packet.get("features", []))
            legal = np.asarray(packet.get("legal_mask", []))
            if packet.get("active_before") != "neutral":
                reason = "not_a_neutral_preaction_state"
            elif (
                features.ndim != 1
                or not len(features)
                or features.dtype.kind not in "biuf"
                or not np.isfinite(features).all()
                or legal.ndim != 1
                or legal.dtype.kind != "b"
                or not len(legal)
                or not legal[0]
            ):
                reason = "preaction_features_or_legality_unavailable"
        phases.append(
            dict(
                tick=tick,
                sensor_preaction_available=reason is None,
                packet_index=matches[0] if reason is None else None,
                reason=reason,
                teacher_target_available=False,
                teacher_scope="requires separately complete matched future schedule outcomes",
            )
        )
    complete_evidence = bool(
        measurement_admitted
        and complete_physics
        and not diagnostics
        and len(observations) == n
        and aligned is not None
        and all(aligned["packet_eligible"])
        and measured_steps == 4 * n
        and exit_status == 0
        and contact_audit is not None
        and contact_audit.get("complete_synchronized_streams") is True
        and schedule_audit is not None
    )
    # Admission is supplied only after source, invocation, geometry and complete
    # physical measurements are audited. A valid measurement can still violate
    # the task's initial approach criterion; this is not a contact or fall event.
    if (
        complete_evidence
        and fully_valid_physics
        and passage is not None
        and passage.get("initially_upstream") is False
    ):
        events.append(dict(kind="invalid_initial_approach"))
    passed = bool(
        complete_evidence
        and not events
        and passage is not None
        and passage.get("pass") is True
        and schedule_audit.get("valid") is True
        and contact_audit.get("no_undesired_measured_contact") is True
    )
    if passed:
        classification, task_outcome = "complete_pass", "pass"
    elif events:
        classification, task_outcome = "verified_task_failure", "failure"
    else:
        classification, task_outcome = "measurement_or_infrastructure_failure", "unknown"
    measurement_status = (
        "complete"
        if complete_evidence
        else (
            "invalid"
            if any(d != "physical_capture_missing" for d in diagnostics)
            else "partial" if n or measured_steps else "unavailable"
        )
    )
    return dict(
        schema="motion2scene_timed_attempt_outcome_v1",
        assigned_slots=1,
        classification=classification,
        task_outcome=task_outcome,
        complete_pass=passed,
        measurement_status=measurement_status,
        complete_measurement_admitted=complete_evidence,
        physical_events=events,
        measurement_diagnostics=diagnostics,
        physical_rows_recorded=n,
        physics_steps_recorded=measured_steps,
        sensor_packets_recorded=len(observations),
        exit_status=exit_status,
        phase_availability=phases,
        scope="one assigned finite traversal; unavailable later supervision is never invented",
    )
