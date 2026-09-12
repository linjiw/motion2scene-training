from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_teacher import (
    ScheduledOutcome,
    schedule_teacher,
)


def test_waiting_inherits_successful_later_adaptation_not_failed_walk_commitment():
    branches = [
        ScheduledOutcome("walk", 0, None, 7, False, None, True, {10: "same", 15: "same"}, 796),
        ScheduledOutcome(
            "shallow_later", 1, 0.3, 7, True, 2.94, True, {10: "same", 15: "same"}, 796
        ),
        ScheduledOutcome("deep_now", 2, 0.2, 7, True, 2.96, True, {10: "same", 15: "adapted"}, 796),
    ]
    expected = [(0, None), (1, 0.3), (2, 0.2)]
    first = schedule_teacher(
        branches, 0.2, "same", 7, [True, False, True], expected_schedules=expected
    )
    assert first["teacher_action"] == 0
    assert first["waiting_has_future_adaptation"]
    assert first["continuation_branch_ids"][0] == "shallow_later"
    later = schedule_teacher(
        branches, 0.3, "same", 7, [True, True, False], expected_schedules=expected
    )
    assert later["teacher_action"] == 1
    assert not later["pass_labels"][0]


def test_missing_branches_and_wrong_history_do_not_create_failure_labels():
    branch = ScheduledOutcome("later", 1, 0.3, 7, True, 2.9, True, {10: "other"}, 796)
    result = schedule_teacher(
        [branch], 0.2, "same", 7, [True, True], expected_schedules=[(0, None), (1, 0.2), (1, 0.3)]
    )
    assert not any(result["admitted"])
    assert result["teacher_action"] is None
