"""Causal acquisition, immutable registration and conservative cost checks."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (
    ADOPTION_SCHEMA,
    ARMS,
    PLAN_SCHEMA,
    RECEIPT_SCHEMA,
    SEEDS,
    artifact,
    attempt_accounting,
    bind_preupdate_round,
    budget_contract,
    release_current_teachers,
    validate_completed_replay_order,
    validate_plan,
    validate_student_release,
    write_new,
)

OPTIONS = ["neutral", "short15", "long15", "prior15", "prior50", "short70", "long70"]


def proposed(tmp_path):
    """Small file-backed fixture; no real qualification or simulated result."""
    registry = write_new(tmp_path / "registry.json", {"test_fixture": True})
    source = write_new(tmp_path / "trainer.py", {"test_source": True})
    runs = []
    for seed in SEEDS:
        for arm in ARMS:
            run_id = f"seed{seed}_{arm}"
            rows = []
            for index in range(5):
                folder = tmp_path / "execution" / run_id / f"round{index}"
                model = tmp_path / "execution" / run_id / f"model{index - 1}"
                rows.append(
                    dict(
                        round_index=index,
                        physics_seed=seed,
                        teacher_branch_order=OPTIONS,
                        allowed_teacher_rounds=list(range(index)),
                        student_before_current_teachers=bool(index),
                        candidate_id=None if index == 0 else f"candidate{seed}_{index}",
                        scene_definition={"same_scene": index},
                        observation_availability=None,
                        teacher_directory=str(folder / "teachers"),
                        student_directory=None if index == 0 else str(folder / "student"),
                        expected_teacher_result_path=str(folder / "teachers" / "result.json"),
                        expected_student_result_path=(
                            None if index == 0 else str(folder / "student" / "result.json")
                        ),
                        expected_preupdate_policy_path=(
                            None if index == 0 else str(model / "policy.npz")
                        ),
                        expected_preupdate_training_result_path=(
                            None if index == 0 else str(model / "result.json")
                        ),
                        expected_preupdate_binding_path=(
                            None if index == 0 else str(folder / "preupdate.json")
                        ),
                        expected_teacher_release_path=(
                            None if index == 0 else str(folder / "release.json")
                        ),
                    )
                )
            runs.append(dict(run_id=run_id, physics_seed=seed, arm=arm, rounds=rows))
    return dict(
        schema=PLAN_SCHEMA,
        adoption_status="proposed_not_adopted",
        execution_authorized=False,
        budget=budget_contract(),
        option_ids=OPTIONS,
        registry=registry,
        test_training_implementation=[source],
        runs=runs,
    )


def adopted_fixture(tmp_path):
    plan = proposed(tmp_path)
    plan_ref = write_new(tmp_path / "plan.json", plan)
    learner = write_new(
        tmp_path / "learner.json",
        dict(
            status="frozen",
            feature_dimension=114,
            option_ids=OPTIONS,
            phase_ticks=[15, 50, 70],
            l2=0.1,
            training_implementation=plan["test_training_implementation"],
        ),
    )
    runtime = write_new(tmp_path / "runtime.json", dict(test_fixture=True))
    adoption = write_new(
        tmp_path / "adoption.json",
        dict(
            schema=ADOPTION_SCHEMA,
            status="adopted",
            plan=plan_ref,
            common_learner=learner,
            runtime_freeze=runtime,
        ),
    )
    return plan, plan_ref, adoption


def make_training(plan, row, earlier):
    policy = write_new(row["expected_preupdate_policy_path"], dict(test_model=True))
    registration = write_new(
        Path(row["expected_preupdate_training_result_path"]).with_name("registration.json"),
        dict(
            registry=plan["registry"],
            collections=earlier,
            l2=0.1,
            implementation=plan["test_training_implementation"],
        ),
    )
    training = write_new(
        row["expected_preupdate_training_result_path"],
        dict(status="complete", policy=policy, registration=registration),
    )
    return policy, training


def completed_prefix(tmp_path, end=1):
    plan, plan_ref, adoption = adopted_fixture(tmp_path)
    run = plan["runs"][0]
    initial = write_new(run["rounds"][0]["expected_teacher_result_path"], dict(status="complete"))
    entries = [dict(round_index=0, teacher_collection=initial)]
    earlier = [initial]
    for index in range(1, end + 1):
        row = run["rounds"][index]
        policy, training = make_training(plan, row, earlier)
        binding = bind_preupdate_round(plan_ref, adoption, run["run_id"], index, earlier)
        student = make_student(plan, row, policy)
        release = release_current_teachers(binding)
        teachers = write_new(row["expected_teacher_result_path"], dict(status="complete"))
        entries.append(
            dict(
                round_index=index,
                teacher_collection=teachers,
                student_collection=student,
                student_model=policy,
                model_training_result=training,
                preupdate_binding=binding,
                teacher_release=release,
            )
        )
        earlier.append(teachers)
    receipt = dict(
        schema=RECEIPT_SCHEMA,
        plan=plan_ref,
        adoption=adoption,
        run_id=run["run_id"],
        status="complete_prefix",
        completed_through_round=end,
        entries=entries,
    )
    return plan, receipt


def make_student(plan, row, policy):
    from gear_sonic.dataset_generation.hallucination.motion2scene_timed_outcome import (
        classify_timed_attempt,
    )

    n = 85
    root = np.tile([0.0, 0.0, 0.8], (n, 1))
    root[10:, 2] = 0.4
    outcome = classify_timed_attempt(
        dict(
            fps=50,
            motion_time_s=np.arange(n) * 0.02,
            root_pos_w=root,
            root_quat_w=np.tile([1, 0, 0, 0], (n, 1)),
            dof_pos=np.zeros((n, 29)),
            dof_vel=np.zeros((n, 29)),
            projected_gravity_b=np.tile([0.0, 0.0, -1.0], (n, 1)),
        ),
        [],
        reference_frames=299,
        phase_ticks=[15, 50, 70],
        exit_status=1,
        physics_steps=np.arange(1, 4 * n + 1),
    )
    assert outcome["task_outcome"] == "failure"
    folder = Path(row["expected_student_result_path"]).parent
    attempt = write_new(folder / "attempt.json", dict(exit_status=1))
    manifest = write_new(
        folder / "manifest.json",
        dict(policy=policy, registry=plan["registry"], scene_definition=row["scene_definition"]),
    )
    return write_new(
        row["expected_student_result_path"],
        dict(
            schema="motion2scene_timed_schedule_collection_v1",
            manifest=manifest,
            physics_steps=340,
            unmeasured_failed_attempts=0,
            rows=[
                dict(
                    mode="learned",
                    task_outcome_admitted=True,
                    physics_steps=340,
                    outcome=outcome,
                    attempt=attempt,
                )
            ],
        ),
    )


def test_budget_counts_fresh_bootstrap_students_and_all_teachers(tmp_path):
    plan = proposed(tmp_path)
    assert validate_plan(plan)["budget"]["maximum_planned_physics_steps"] == 46488
    assert plan["budget"]["unallocated_physics_steps"] == 3512
    assert plan["budget"]["total_planned_episodes"] == 39
    bad = copy.deepcopy(plan)
    bad["runs"][1]["rounds"][0]["expected_teacher_result_path"] = bad["runs"][0]["rounds"][0][
        "expected_teacher_result_path"
    ]
    with pytest.raises(ValueError, match="fresh physical"):
        validate_plan(bad)


def test_analytic_observation_requires_same_geometry_order_and_seed(tmp_path):
    plan = proposed(tmp_path)
    bad = copy.deepcopy(plan)
    bad["runs"][3]["rounds"][1]["candidate_id"] = "adaptively_replaced"
    with pytest.raises(ValueError, match="same candidate order"):
        validate_plan(bad)
    bad = copy.deepcopy(plan)
    bad["runs"][3]["rounds"][1]["physics_seed"] = 99999
    with pytest.raises(ValueError, match="matched seed"):
        validate_plan(bad)


def test_completed_prefix_supports_roundwise_replay_with_historical_gap_age(tmp_path):
    plan, receipt = completed_prefix(tmp_path, end=2)
    result = validate_completed_replay_order(plan, receipt)
    assert [g["gap_age_rounds"] for g in result["historical_gaps"]] == [1, 0]
    assert len(result["teacher_collections"]) == 3
    assert result["physical_admission_audited"] is False
    bad = copy.deepcopy(receipt)
    bad["status"] = "complete"
    with pytest.raises(ValueError, match="completed bootstrap/round prefix"):
        validate_completed_replay_order(plan, bad)


def test_current_labels_and_current_teacher_preparation_block_preupdate_binding(tmp_path):
    plan, plan_ref, adoption = adopted_fixture(tmp_path)
    run = plan["runs"][0]
    initial = write_new(run["rounds"][0]["expected_teacher_result_path"], dict(status="complete"))
    row = run["rounds"][1]
    leaked = write_new(tmp_path / "leaked_current_labels.json", dict(test_fixture=True))
    make_training(plan, row, [initial, leaked])
    with pytest.raises(ValueError, match="strictly earlier"):
        bind_preupdate_round(plan_ref, adoption, run["run_id"], 1, [initial])
    Path(row["teacher_directory"]).mkdir(parents=True)
    with pytest.raises(ValueError, match="must not exist"):
        bind_preupdate_round(plan_ref, adoption, run["run_id"], 1, [initial])


def test_proposal_without_separate_adoption_cannot_validate_completed_claim(tmp_path):
    plan, receipt = completed_prefix(tmp_path)
    receipt["adoption"] = write_new(
        tmp_path / "unadopted.json",
        dict(schema=ADOPTION_SCHEMA, status="proposed", plan=receipt["plan"]),
    )
    with pytest.raises(ValueError, match="separate adopted"):
        validate_completed_replay_order(plan, receipt)


def test_future_launch_and_hash_mutation_fail_completed_prefix(tmp_path):
    plan, receipt = completed_prefix(tmp_path)
    future = plan["runs"][0]["rounds"][2]
    Path(future["student_directory"]).mkdir(parents=True)
    with pytest.raises(ValueError, match="later rounds"):
        validate_completed_replay_order(plan, receipt)
    Path(future["student_directory"]).rmdir()
    ref = receipt["entries"][1]["student_model"]
    Path(ref["path"]).write_text("altered model")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_completed_replay_order(plan, receipt)


def test_unknown_attempts_are_reserved_without_inventing_actual_measurements():
    result = attempt_accounting(
        [
            dict(
                attempt_id="pass",
                launched=True,
                maximum_physics_steps=1192,
                actual_physics_steps=1192,
            ),
            dict(
                attempt_id="physical_failure",
                launched=True,
                maximum_physics_steps=1192,
                actual_physics_steps=500,
            ),
            dict(
                attempt_id="unmeasured_abort",
                launched=True,
                maximum_physics_steps=1192,
                actual_physics_steps=None,
            ),
            dict(
                attempt_id="resource_pause",
                launched=False,
                maximum_physics_steps=1192,
                actual_physics_steps=0,
            ),
        ]
    )
    assert result["actual_recorded_physics_steps"] == 1692
    assert result["unknown_attempt_reserved_steps"] == 1192
    assert result["conservative_charged_steps"] == 2884
    with pytest.raises(ValueError, match="duplicate attempt"):
        attempt_accounting(
            [
                dict(
                    attempt_id="duplicate",
                    launched=False,
                    maximum_physics_steps=1192,
                    actual_physics_steps=0,
                )
            ]
            * 2
        )


def test_exclusive_marker_preserves_old_bytes(tmp_path):
    ref = write_new(tmp_path / "immutable.json", {"first": True})
    with pytest.raises(FileExistsError):
        write_new(ref["path"], {"first": False})
    assert artifact(ref["path"]) == ref
    assert json.loads(Path(ref["path"]).read_text()) == {"first": True}


def test_collector_schema_without_status_retains_measured_physical_failures(tmp_path):
    plan, receipt = completed_prefix(tmp_path)
    entry = receipt["entries"][1]
    student = json.loads(Path(entry["student_collection"]["path"]).read_text())
    binding = json.loads(Path(entry["preupdate_binding"]["path"]).read_text())
    row = plan["runs"][0]["rounds"][1]
    assert "status" not in student
    validate_student_release(student, binding, plan, row)
    student["rows"][0]["outcome"]["task_outcome"] = "unknown"
    with pytest.raises(ValueError, match="measured pre-update"):
        validate_student_release(student, binding, plan, row)
