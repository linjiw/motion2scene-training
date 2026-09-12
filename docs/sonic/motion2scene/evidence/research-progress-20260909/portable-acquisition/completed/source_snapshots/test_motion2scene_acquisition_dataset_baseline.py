"""Portable acquisition reconstruction preserves chronological teachers and exact decisions."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_acquisition_dataset_baseline import (
    acquisition_curve,
    archived_weights,
    compare_models,
    measured_task_yield,
    prefix_targets,
    verify_student_models,
)


def target(value=1.0):
    return dict(
        features=[value],
        phase_tick=15,
        pass_labels=[True, False],
        passage_time_s=[1.0, None],
        admitted=[True, True],
        legal_mask=[True, True],
        continuation_option_indices=[0, 1],
    )


def test_reconstruction_uses_only_the_original_training_prefix():
    old = target()
    group = dict(collection=dict(path="original", sha256="bound"), targets=[old])
    future = target(999)
    rows = prefix_targets([copy.deepcopy(old), future], [group], [group["collection"]])
    assert rows == [old]
    with pytest.raises(ValueError, match="historical teacher prefix"):
        prefix_targets([old, future], [group], [dict(path="future", sha256="bound")])
    changed = target(2.0)
    with pytest.raises(ValueError, match="differ from the frozen fit"):
        prefix_targets([changed, future], [group], [group["collection"]])


def test_a_fitting_residual_cannot_replace_an_archived_physical_weight_binding():
    ref = dict(path="historical_weights", sha256="original")
    registration = dict(replay_weights=ref)
    result = dict(replay_audit=dict(weights_artifact=ref, weights_in_group_target_order=[0.2]))
    rows, weights = archived_weights(registration, result, [target()])
    assert rows == [target()] and weights.tolist() == [0.2]
    result["replay_audit"]["weights_artifact"] = dict(path="refreshed_residual", sha256="new")
    with pytest.raises(ValueError, match="originally audited historical"):
        archived_weights(registration, result, [target()])
    with pytest.raises(ValueError, match="uniform arm"):
        archived_weights(dict(replay_weights=None), result, [target()])


def model(bias):
    return dict(
        phase_ticks=np.array([15]),
        mean=np.zeros((1, 1)),
        std=np.ones((1, 1)),
        weights=np.zeros((1, 1, 2)),
        bias=np.array([bias]),
    )


def test_numerically_close_coefficients_do_not_excuse_an_actual_argmax_change():
    a = model([0.0, 0.0])
    b = model([0.0, 1e-14])
    with pytest.raises(ValueError, match="actual argmax differs"):
        compare_models(a, b, [target()])
    b = model([1e-14, 0.0])
    assert compare_models(a, b, [target()])["recorded_teacher_packet_choices_checked"] == 1


def test_reconstruction_does_not_silently_drop_a_zero_weighted_available_row():
    ref = dict(path="weights", sha256="bound")
    with pytest.raises(ValueError, match="positive weight"):
        archived_weights(
            dict(replay_weights=ref),
            dict(replay_audit=dict(weights_artifact=ref, weights_in_group_target_order=[0.0])),
            [target()],
        )


def test_later_fit_cannot_be_credited_with_an_earlier_student_outcome(tmp_path):
    models = [dict(policy=dict(path=f"M{i}", sha256=str(i))) for i in range(3)]
    (tmp_path / "manifest.json").write_text(json.dumps(dict(source_results=list(range(5)))))
    for round_index in (1, 2):
        folder = tmp_path / "provenance" / f"study_{round_index + 2:03d}"
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(
            json.dumps(
                dict(
                    policy=models[round_index - 1]["policy"],
                    cells=[dict(timed_schedule_mode="learned")],
                )
            )
        )
    assert [r["generating_checkpoint"] for r in verify_student_models(tmp_path, models)] == [0, 1]
    path = tmp_path / "provenance/study_003/manifest.json"
    stale = json.loads(path.read_text())
    stale["policy"] = models[2]["policy"]
    path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="different generating policy"):
        verify_student_models(tmp_path, models)


def test_task_yield_does_not_turn_unknown_neutral_into_adaptation_required():
    result = measured_task_yield(dict(neutral="unknown", adaptation="pass"))
    assert result["bank_solvable_lower"] == 1
    assert result["adaptation_required_lower"] == 0
    assert result["adaptation_required_upper"] == 1
    assert result["unknown_branch_outcomes"] == 1
    result = measured_task_yield(dict(neutral="failure", adaptation="unknown"))
    assert result["bank_solvable_lower"] == 0 and result["bank_solvable_upper"] == 1
    assert result["adaptation_required_lower"] == 0


def test_curve_excludes_bootstrap_tasks_but_counts_bootstrap_and_student_physics(tmp_path):
    episodes, teachers = [], []
    for group in range(3):
        ids = []
        for option in range(7):
            identifier = f"group{group}_option{option}"
            ids.append(identifier)
            episodes.append(
                dict(
                    episode_id=identifier,
                    configured_forced_option_id="neutral" if option == 0 else str(option),
                    collection_sha256=str(group),
                    physics_steps=10 * (group + 1),
                    assessment=dict(
                        outcome=dict(
                            task_outcome="failure" if group == 1 and option == 0 else "pass"
                        )
                    ),
                )
            )
        teachers.append(dict(episode_ids=ids))
    for index, steps in ((3, 17), (4, 19)):
        episodes.append(
            dict(episode_id=f"student{index}", collection_sha256=str(index), physics_steps=steps)
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(episodes=episodes, source_results=[dict(sha256=str(i)) for i in range(5)]))
    )
    (tmp_path / "teachers.json").write_text(json.dumps(teachers))
    a, b = acquisition_curve(tmp_path)
    assert (a["assigned_tasks"], b["assigned_tasks"]) == (1, 2)
    assert (a["adaptation_required_lower"], b["adaptation_required_lower"]) == (1, 1)
    assert (a["recorded_physics_steps"], b["recorded_physics_steps"]) == (227, 456)
