"""Task failure remains distinct from unavailable future supervision."""

import copy
import json

import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_outcome import (
    classify_timed_attempt,
)


def fixture(n=79):
    times = np.arange(n) * 0.02
    payload = dict(
        fps=50,
        motion_time_s=times,
        root_pos_w=np.tile([0, 0, 0.8], (n, 1)),
        root_quat_w=np.tile([1, 0, 0, 0], (n, 1)),
        dof_pos=np.zeros((n, 29)),
        dof_vel=np.zeros((n, 29)),
        projected_gravity_b=np.tile([0.0, 0.0, -1.0], (n, 1)),
    )
    observations = [
        dict(
            tick=i + 1,
            time_s=(i + 1) / 50,
            capture_elapsed_s=i / 50,
            physics_step=4 * (i + 1),
            active_before="neutral",
            features=[0.0] * 114,
            legal_mask=[True] * 7,
            root_pos_w=payload["root_pos_w"][i].tolist(),
            root_quat_w=payload["root_quat_w"][i].tolist(),
            state={
                key: payload[key][i].tolist()
                for key in ("dof_pos", "dof_vel", "projected_gravity_b")
            },
            transition={"allowed": True},
        )
        for i in range(n)
    ]
    kwargs = dict(
        reference_frames=80,
        phase_ticks=[15, 50, 70],
        exit_status=0,
        physics_steps=np.arange(1, 4 * n + 1),
        measurement_admitted=n == 79,
        passage={"pass": True, "passage_finish_frame_exclusive": 60},
        schedule_audit={"valid": True},
        contact_audit={
            "complete_synchronized_streams": True,
            "no_undesired_measured_contact": True,
            "maximum_undesired_environment_force_n": 0.0,
        },
    )
    return payload, observations, kwargs


def test_complete_pass_and_available_phases_do_not_invent_teacher_targets():
    result = classify_timed_attempt(*fixture()[:2], **fixture()[2])
    assert result["classification"] == "complete_pass"
    assert result["measurement_status"] == "complete" and result["assigned_slots"] == 1
    assert all(p["sensor_preaction_available"] for p in result["phase_availability"])
    assert not any(p["teacher_target_available"] for p in result["phase_availability"])
    json.dumps(result)


def test_early_verified_fall_preserves_failure_and_only_actual_prior_phase():
    payload, observations, kwargs = fixture(30)
    payload["root_pos_w"][25:, 2] = 0.4
    for i in range(25, 30):
        observations[i]["root_pos_w"][2] = 0.4
    kwargs["exit_status"] = 1
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "failure" and result["measurement_status"] == "partial"
    assert [p["sensor_preaction_available"] for p in result["phase_availability"]] == [
        True,
        False,
        False,
    ]
    assert result["physical_events"][0]["kind"] == "recorded_fall_or_upright_threshold_failure"
    assert result["physics_steps_recorded"] == 120


def test_reset_is_a_failure_and_duplicate_later_episode_phases_are_not_reused():
    payload, observations, kwargs = fixture(79)
    payload["motion_time_s"][30:] = np.arange(49) * 0.02
    for i in range(30, 79):
        observations[i]["time_s"] = (i - 29) / 50
        observations[i]["tick"] = i - 29
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "failure"
    assert result["physical_events"][0]["kind"] == "recorded_physical_reference_reset"
    assert [p["sensor_preaction_available"] for p in result["phase_availability"]] == [
        True,
        False,
        False,
    ]


def test_complete_measured_contact_failure_is_not_missing_measurement():
    payload, observations, kwargs = fixture()
    kwargs["contact_audit"].update(
        no_undesired_measured_contact=False, maximum_undesired_environment_force_n=12.0
    )
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "failure" and result["measurement_status"] == "complete"
    assert result["physical_events"][0]["maximum_force_n"] == 12.0
    kwargs["contact_audit"]["complete_synchronized_streams"] = False
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "unknown" and result["measurement_status"] == "invalid"


def test_finite_horizon_and_observed_refusal_keep_assigned_failure_slots():
    payload, observations, kwargs = fixture()
    kwargs["passage"] = {"pass": False, "passage_finish_frame_exclusive": None}
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "failure" and result["assigned_slots"] == 1
    assert (
        result["physical_events"][0]["kind"]
        == "finite_captured_reference_horizon_before_completion"
    )
    payload, observations, kwargs = fixture(30)
    observations[-1]["transition"] = {"allowed": False, "reason": "reference_jump_guard"}
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "failure"
    assert result["physical_events"][0]["kind"] == "recorded_transition_refusal"
    # An unbound transition log cannot substitute for a physical capture.
    result = classify_timed_attempt(None, observations, **kwargs)
    assert result["task_outcome"] == "unknown"


def test_missing_corrupt_or_inactive_future_phase_is_never_a_physical_failure_label():
    _, _, kwargs = fixture()
    kwargs.update(exit_status=1, measurement_admitted=False, physics_steps=None)
    result = classify_timed_attempt(None, [], **kwargs)
    assert result["task_outcome"] == "unknown" and result["measurement_status"] == "unavailable"
    assert result["physics_steps_recorded"] is None and result["assigned_slots"] == 1
    payload, observations, kwargs = fixture()
    payload["root_pos_w"][0] = np.nan
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "unknown" and result["measurement_status"] == "invalid"
    payload, observations, kwargs = fixture()
    observations[49]["active_before"] = "short"
    observations[69]["features"] = ["corrupt"]
    result = classify_timed_attempt(payload, observations, **kwargs)
    phases = result["phase_availability"]
    assert phases[1]["reason"] == "not_a_neutral_preaction_state"
    assert phases[2]["reason"] == "preaction_features_or_legality_unavailable"
    changed = copy.deepcopy(kwargs)
    changed["physics_steps"][5] = 99
    assert (
        classify_timed_attempt(payload, observations, **changed)["measurement_status"] == "invalid"
    )


def test_absence_at_horizon_needs_valid_arrays_but_prior_observed_fall_survives_corruption():
    payload, observations, kwargs = fixture()
    payload["root_pos_w"][-1, 2] = np.nan
    kwargs["passage"] = {"pass": False, "passage_finish_frame_exclusive": None}
    kwargs["schedule_audit"] = {
        "valid": False,
        "complete_ticks": True,
        "active_timeline_matches": False,
    }
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "unknown"
    assert not result["physical_events"]
    payload["root_pos_w"][20, 2] = 0.4
    observations[20]["root_pos_w"][2] = 0.4
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "failure"
    assert [event["kind"] for event in result["physical_events"]] == [
        "recorded_fall_or_upright_threshold_failure"
    ]
