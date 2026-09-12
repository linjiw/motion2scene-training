"""Initial-approach failure requires complete admitted physical evidence."""

import copy

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_course import score_course
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_outcome import (
    classify_timed_attempt,
)


def fixture(*, course=False, frames=79, valid_approach=False):
    progress = np.linspace(0.0, 2.0, frames)
    root = np.c_[progress, np.zeros(frames), np.full(frames, 0.8)]
    body = np.repeat(root[:, None], 2, axis=1)
    body[:, :, 0] += [-0.04, 0.04]
    payload = dict(
        fps=50,
        motion_time_s=np.arange(frames) / 50,
        body_pos_w=body,
        root_pos_w=root,
        root_quat_w=np.tile([1, 0, 0, 0], (frames, 1)),
        dof_pos=np.zeros((frames, 29)),
        dof_vel=np.zeros((frames, 29)),
        projected_gravity_b=np.tile([0.0, 0.0, -1.0], (frames, 1)),
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
            root_pos_w=root[i].tolist(),
            root_quat_w=payload["root_quat_w"][i].tolist(),
            state={
                k: payload[k][i].tolist() for k in ("dof_pos", "dof_vel", "projected_gravity_b")
            },
            transition={"allowed": True},
        )
        for i in range(frames)
    ]
    stations = [0.3 if valid_approach else 0.0] + ([0.6] if course else [])
    beams = [
        dict(
            center_xy_m=[station, 0.0],
            yaw_rad=0.0,
            length_m=0.2,
            width_m=1.2,
            thickness_m=0.1,
            underside_m=1.25,
        )
        for station in stations
    ]
    if course:
        passage = score_course(
            payload,
            np.zeros((frames, 2, 2, 3)),
            beams,
            commands_valid=True,
            timeout_s=2,
        )
    else:
        passage = score_passage(payload, np.zeros((frames, 2, 3)), beams[0])
    kwargs = dict(
        reference_frames=80,
        phase_ticks=[15, 50, 70],
        exit_status=0,
        measurement_admitted=True,
        physics_steps=np.arange(1, frames * 4 + 1),
        passage=passage,
        schedule_audit={"valid": True},
        contact_audit=dict(
            complete_synchronized_streams=True,
            no_undesired_measured_contact=True,
            maximum_undesired_environment_force_n=0.0,
        ),
    )
    return payload, observations, kwargs


@pytest.mark.parametrize("course", [False, True])
def test_complete_admitted_invalid_start_is_a_named_task_failure(course):
    payload, observations, kwargs = fixture(course=course)
    original_passage = copy.deepcopy(kwargs["passage"])
    assert not original_passage["pass"] and not original_passage["initially_upstream"]
    assert original_passage["passage_finish_frame_exclusive"] is not None
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["classification"] == "verified_task_failure"
    assert result["task_outcome"] == "failure" and not result["complete_pass"]
    assert result["measurement_status"] == "complete"
    assert result["physical_events"] == [{"kind": "invalid_initial_approach"}]
    assert result["assigned_slots"] == 1 and result["physics_steps_recorded"] == 316
    assert result["physical_rows_recorded"] == result["sensor_packets_recorded"] == 79
    assert kwargs["passage"] == original_passage  # Retain actual crossing/hold diagnostics.


@pytest.mark.parametrize("course", [False, True])
def test_valid_initial_approach_still_passes(course):
    payload, observations, kwargs = fixture(course=course, valid_approach=True)
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["complete_pass"] and result["task_outcome"] == "pass"
    assert not result["physical_events"]


@pytest.mark.parametrize("flag", [None, 0, "false", np.bool_(False)])
def test_only_explicit_boolean_false_can_establish_initial_violation(flag):
    payload, observations, kwargs = fixture()
    kwargs["passage"]["initially_upstream"] = flag
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "unknown" and not result["physical_events"]


def test_missing_flag_does_not_infer_failure_from_other_diagnostics():
    payload, observations, kwargs = fixture()
    del kwargs["passage"]["initially_upstream"]
    assert kwargs["passage"]["invalid_initial_approach"]
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "unknown" and not result["physical_events"]


@pytest.mark.parametrize(
    "missing",
    [
        "source_admission",
        "invocation",
        "contact_measurements",
        "schedule",
        "physics_counter",
        "sensor_pose",
        "physical_pose",
    ],
)
def test_missing_or_invalid_evidence_cannot_establish_initial_failure(missing):
    payload, observations, kwargs = fixture()
    if missing == "source_admission":
        kwargs["measurement_admitted"] = False
    elif missing == "invocation":
        kwargs["exit_status"] = 1
    elif missing == "contact_measurements":
        kwargs["contact_audit"]["complete_synchronized_streams"] = False
    elif missing == "schedule":
        kwargs["schedule_audit"] = None
    elif missing == "physics_counter":
        kwargs["physics_steps"][0] = 100
    elif missing == "sensor_pose":
        observations[0]["root_pos_w"][0] += 0.01
    elif missing == "physical_pose":
        payload["root_pos_w"][0, 0] = np.nan
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "unknown" and not result["physical_events"]
    assert result["assigned_slots"] == 1


def test_partial_or_missing_capture_retains_unknown_and_actual_budget():
    payload, observations, kwargs = fixture(frames=30)
    assert kwargs["passage"]["passage_finish_frame_exclusive"] is not None
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "unknown" and not result["physical_events"]
    assert result["physics_steps_recorded"] == 120 and result["assigned_slots"] == 1
    result = classify_timed_attempt(None, [], **dict(kwargs, physics_steps=None))
    assert result["task_outcome"] == "unknown" and result["physics_steps_recorded"] is None
    assert result["assigned_slots"] == 1


@pytest.mark.parametrize("kind", ["contact", "fall"])
def test_existing_contact_and_fall_categories_remain_separate(kind):
    payload, observations, kwargs = fixture(valid_approach=True)
    if kind == "contact":
        kwargs["contact_audit"].update(
            no_undesired_measured_contact=False, maximum_undesired_environment_force_n=12.0
        )
        expected = "verified_environment_contact"
    else:
        payload["root_pos_w"][20, 2] = 0.4
        observations[20]["root_pos_w"][2] = 0.4
        expected = "recorded_fall_or_upright_threshold_failure"
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert result["task_outcome"] == "failure"
    assert [event["kind"] for event in result["physical_events"]] == [expected]
    kwargs["passage"]["initially_upstream"] = False
    result = classify_timed_attempt(payload, observations, **kwargs)
    assert [event["kind"] for event in result["physical_events"]] == [
        expected,
        "invalid_initial_approach",
    ]
