"""Independent admission and provenance checks for finite-schedule imitation."""

import copy
from dataclasses import replace
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_imitation import (
    physical_regret,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_teacher import (
    ScheduledOutcome,
    schedule_teacher,
)


def outcome(name, option, entry, passed, time=None, admitted=True):
    return ScheduledOutcome(name, option, entry, 7, passed, time, admitted, {10: "prefix"}, 796)


@pytest.mark.parametrize("missing_kind", ["omitted", "unadmitted_success", "unadmitted_failure"])
def test_missing_future_schedule_never_turns_failed_walk_into_verified_failed_wait(missing_kind):
    branches = [outcome("walk", 0, None, False), outcome("now", 1, 0.2, True, 3.0)]
    if missing_kind != "omitted":
        passed = missing_kind == "unadmitted_success"
        branches.append(outcome("later", 1, 0.3, passed, 2.8 if passed else None, False))
    target = schedule_teacher(
        branches,
        0.2,
        "prefix",
        7,
        [True, True],
        expected_schedules=[(0, None), (1, 0.2), (1, 0.3)],
    )
    assert not target["admitted"][0]
    assert not target["complete_legal_action_table"]
    assert target["teacher_action"] is None
    _, supervised = physical_regret(
        [target["pass_labels"]],
        [target["passage_time_s"]],
        [target["admitted"]],
        [target["legal_mask"]],
    )
    assert not supervised[0]


def test_complete_future_schedule_gives_wait_a_real_successful_continuation():
    target = schedule_teacher(
        [
            outcome("walk", 0, None, False),
            outcome("now", 1, 0.2, True, 3.0),
            outcome("later", 1, 0.3, True, 2.8),
        ],
        0.2,
        "prefix",
        7,
        [True, True],
        expected_schedules=[(0, None), (1, 0.2), (1, 0.3)],
    )
    assert target["teacher_action"] == 0
    assert target["waiting_has_future_adaptation"]
    assert target["continuation_branch_ids"][0] == "later"


def test_future_option_outside_registry_cannot_hide_behind_immediate_wait():
    with pytest.raises(ValueError, match="registry|option"):
        schedule_teacher(
            [outcome("outside", 2, 0.3, True, 2.8)],
            0.2,
            "prefix",
            7,
            [True, True],
            expected_schedules=[(0, None), (1, 0.2), (1, 0.3)],
        )


def test_duplicate_schedule_identity_is_not_an_independent_candidate():
    original = outcome("first", 1, 0.3, True, 2.8)
    with pytest.raises(ValueError, match="duplicate|unique"):
        schedule_teacher(
            [original, replace(original, branch_id="second")],
            0.2,
            "prefix",
            7,
            [True, True],
            expected_schedules=[(0, None), (1, 0.2), (1, 0.3)],
        )


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/research"))
    return importlib.import_module("motion2scene_train_schedule_policy")


@pytest.fixture
def recorded_visit(tmp_path, trainer, monkeypatch):
    payload = {
        key: np.zeros((50, 2))
        for key in (
            "dof_pos",
            "dof_vel",
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "applied_joint_action",
            "action_motion_token",
            "reference_g1_qpos",
        )
    }
    payload["motion_time_s"] = np.arange(50) / 50
    packet = {
        "time_s": 0.2,
        "capture_elapsed_s": 0.18,
        "active_before": 0,
        "state": {"dof_pos": [0.0, 0.0]},
        "root_pos_w": [0.0, 0.0, 1.0],
        "root_quat_w": [1.0, 0.0, 0.0, 0.0],
        "features": [0.0, 0.2],
        "legal_mask": [True, True],
    }
    sensor = tmp_path / "sensor.json"
    sensor.write_text(json.dumps({"feature_names": ["state", "phase_s"], "observations": [packet]}))
    trajectory = tmp_path / "trajectory.pkl"
    trajectory.write_bytes(b"trusted synthetic recording")
    row = {
        "mode": "learned",
        "measurement_admitted": True,
        "first_episode_frames": 50,
        "pass": True,
        "fall_observed": False,
        "reset_count": 0,
        "switches": [],
        "trajectory": trainer.artifact(trajectory),
        "sensor": trainer.artifact(sensor),
        "costs": {"passage_time_s": 4.0},
    }
    target = {"teacher_action": 1, "passage_time_s": [None, 3.0]}
    student_payload = {key: value.copy() for key, value in payload.items()}
    monkeypatch.setattr(trainer, "load_reset_capture", lambda _: student_payload)
    return row, payload, student_payload, packet, target


def test_student_gap_requires_exact_controller_history_not_only_identical_current_features(
    trainer, recorded_visit
):
    row, anchor, student, packet, target = recorded_visit
    result = trainer.student_visit(row, 0.2, anchor, ["state", "phase_s"], packet, target)
    assert result["matched"]
    assert result["verified_gap"] == 0.25
    student["action_motion_token"][3, 0] = 1
    result = trainer.student_visit(row, 0.2, anchor, ["state", "phase_s"], packet, target)
    assert not result["matched"]
    assert result["verified_gap"] is None


@pytest.mark.parametrize("failure", [{"fall_observed": True}, {"reset_count": 1}])
def test_student_passage_followed_by_fall_or_reset_is_not_successful_cost_regret(
    trainer, recorded_visit, failure
):
    row, anchor, _, packet, target = recorded_visit
    row.update(failure)
    result = trainer.student_visit(row, 0.2, anchor, ["state", "phase_s"], packet, target)
    assert result["matched"]
    assert not result["student_pass"]
    assert result["verified_gap"] == 1.0


def test_changed_student_sensor_artifact_is_rejected_before_replay(trainer, recorded_visit):
    row, anchor, _, packet, target = recorded_visit
    Path(row["sensor"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        trainer.student_visit(row, 0.2, anchor, ["state", "phase_s"], packet, target)


def test_post_reset_sensor_packet_cannot_supply_an_aligned_teacher_visit(trainer, recorded_visit):
    row, anchor, _, packet, target = recorded_visit
    row["_sensor_alignment"] = {"packet_eligible": [False]}
    result = trainer.student_visit(row, 0.2, anchor, ["state", "phase_s"], packet, target)
    assert not result["matched"]
    assert result["verified_gap"] is None


@pytest.fixture
def registered_branch(tmp_path, trainer):
    folder = tmp_path / "registered_branch"
    (folder / "trajectories").mkdir(parents=True)
    row = {
        "cell_id": "case",
        "source": 42,
        "physics_seed": 7,
        "condition": "present",
        "beam": {"underside_m": 1.2},
        "mode": "forced",
        "option_index": 1,
        "decision_time_s": 0.3,
        "costs": {"passage_time_s": 2.8},
        "trajectory": {"path": str(folder / "trajectories/recording.trajectory.pkl")},
        "sensor": {"path": str(folder / "trajectories/reactive_interface.json")},
    }
    cell = {
        "cell_id": "case",
        "generation_seed": 42,
        "runtime_seed": 7,
        "condition": "present",
        "beam": {"underside_m": 1.2},
        "multi_option_mode": "forced",
        "option_index": 1,
        "decision_time_s": 0.3,
        "output": str(folder),
    }
    (folder / "multi_option_row.json").write_text(json.dumps(row))
    manifest = {"cells": [cell]}
    trainer.validate_result_row(row, manifest)
    return row, manifest, folder


@pytest.mark.parametrize(
    "mutation",
    [
        {"source": 43},
        {"physics_seed": 8},
        {"beam": {"underside_m": 1.3}},
        {"option_index": 0},
        {"decision_time_s": 0.2},
        {"mode": "learned"},
        {"costs": {"passage_time_s": 1.0}},
    ],
)
def test_physical_rows_cannot_be_relabelled_after_acquisition(trainer, registered_branch, mutation):
    row, manifest, _ = registered_branch
    row.update(mutation)
    with pytest.raises(ValueError, match="registered cell|recorded physical branch"):
        trainer.validate_result_row(row, manifest)


def test_equal_copied_row_cannot_redirect_to_another_physical_branch(trainer, registered_branch):
    row, manifest, folder = registered_branch
    row["trajectory"]["path"] = str(folder.parent / "other_branch/recording.trajectory.pkl")
    (folder / "multi_option_row.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="outside registered branch"):
        trainer.validate_result_row(row, manifest)


def test_base_replay_identity_binds_phase_controller_history_and_actual_features(
    trainer, recorded_visit
):
    _, anchor, _, packet, _ = recorded_visit
    history = trainer.recorded_history_id("encounter", 0.2, anchor, packet)
    base = {
        "group": "encounter",
        "phase_s": 0.2,
        "state_history_sha256": history,
        "features": packet["features"],
    }
    key = trainer.base_decision_key(base)
    for mutation in (
        {"phase_s": 0.3},
        {"state_history_sha256": "different"},
        {"features": [1.0, 0.2]},
    ):
        assert trainer.base_decision_key({**base, **mutation}) != key
    changed = {name: value.copy() for name, value in anchor.items()}
    changed["action_motion_token"][25, 0] = 1
    assert trainer.recorded_history_id("encounter", 0.2, changed, packet) == history
    changed["action_motion_token"][3, 0] = 1
    assert trainer.recorded_history_id("encounter", 0.2, changed, packet) != history
    changed_packet = copy.deepcopy(packet)
    changed_packet["legal_mask"][1] = False
    assert trainer.recorded_history_id("encounter", 0.2, anchor, changed_packet) != history


@pytest.mark.parametrize("corruption", ["source", "neutral_path", "neutral_hash", "controller"])
def test_registry_binding_rejects_relabelled_motion_ancestry_or_tracker(
    trainer, tmp_path, corruption
):
    neutral = {"path": str(tmp_path / "neutral.pkl"), "sha256": "sha256:neutral"}
    controller = {"path": str(tmp_path / "tracker.pt"), "sha256": "sha256:tracker"}
    registry = {"source": 42, "controller": controller, "references": [{"motion": neutral}]}
    manifest = {
        "implementation": {"checkpoint": controller.copy()},
        "cells": [{"generation_seed": 42, "motion": neutral.copy()}],
    }
    trainer.validate_manifest_registry(manifest, registry)
    if corruption == "source":
        manifest["cells"][0]["generation_seed"] = 43
    elif corruption == "neutral_path":
        manifest["cells"][0]["motion"]["path"] = str(tmp_path / "other.pkl")
    elif corruption == "neutral_hash":
        manifest["cells"][0]["motion"]["sha256"] = "sha256:other"
    else:
        manifest["implementation"]["checkpoint"]["sha256"] = "sha256:other"
    with pytest.raises(ValueError, match="qualified registry"):
        trainer.validate_manifest_registry(manifest, registry)


@pytest.mark.parametrize("return_time", [None, 3.6, 3.4])
def test_full_continuation_requires_actual_qualified_neutral_return(trainer, return_time):
    switches = [{"from": 0, "to": 1, "time_s": 0.2}]
    if return_time is not None:
        switches.append({"from": 1, "to": 0, "time_s": return_time})
    row = {"pass": True, "fall_observed": False, "reset_count": 0, "switches": switches}
    assert trainer.full_episode_pass(row) == (return_time == 3.4)


def test_replay_preserves_encounter_mass_after_nonconsequential_decisions_are_filtered(
    trainer, tmp_path, recorded_visit, monkeypatch
):
    # Source-integrity helpers are tested separately above; here actual teaching
    # and fitting run against controlled recordings with unequal usable phases.
    monkeypatch.setattr(trainer, "validate_result_row", lambda *args: None)
    monkeypatch.setattr(trainer, "validate_manifest_registry", lambda *args: None)
    monkeypatch.setattr(trainer, "recording_alignment", lambda *args: {"packet_eligible": [True] * 3})
    monkeypatch.setattr(trainer, "closure", lambda *args: [])
    _, payload, _, base_packet, _ = recorded_visit
    monkeypatch.setattr(trainer, "load_reset_capture", lambda *args: payload)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "references": [
                    {"name": "neutral", "qualified_entry_times_s": [0.2, 0.3, 0.4]},
                    {"name": "adapt", "qualified_entry_times_s": [0.2, 0.3, 0.4]},
                ]
            }
        )
    )
    rows, students = [], []
    for group_index, group_name in enumerate(("one_phase", "three_phases", "no_target")):
        packets = []
        for phase in (0.2, 0.3, 0.4):
            packet = copy.deepcopy(base_packet)
            packet.update(time_s=phase, features=[float(group_index), phase])
            packet["legal_mask"] = [True, group_index != 0 or phase == 0.4]
            packets.append(packet)
        sensor = tmp_path / f"{group_name}_sensor.json"
        sensor.write_text(
            json.dumps({"feature_names": ["measured_state", "phase_s"], "observations": packets})
        )
        for option, entry, mode in [
            (0, 0.3, "forced"),
            (1, 0.2, "forced"),
            (1, 0.3, "forced"),
            (1, 0.4, "forced"),
            (0, 0.3, "learned"),
        ]:
            name = f"{group_name}_{mode}_{option}_{entry}"
            trajectory = tmp_path / f"{name}.pkl"
            trajectory.write_bytes(name.encode())
            elapsed = 3.2
            if option and group_index == 0:
                elapsed = 3.0
            elif option and group_index == 1:
                elapsed = {0.2: 2.6, 0.3: 2.8, 0.4: 3.0}[entry]
            row = {
                "cell_id": name,
                "source": 42,
                "physics_seed": 7,
                "condition": group_name,
                "beam": {"fixture": group_index},
                "mode": mode,
                "option_index": option,
                "decision_time_s": entry,
                "measurement_admitted": not (option and group_index == 0 and entry < 0.4),
                "pass": True,
                "fall_observed": False,
                "reset_count": 0,
                "first_episode_frames": 50,
                "costs": {"passage_time_s": elapsed},
                "sensor": trainer.artifact(sensor),
                "trajectory": trainer.artifact(trajectory),
                "acquisition": {"physics_steps": 796},
                "switches": (
                    [{"from": 0, "to": 1, "time_s": entry}, {"from": 1, "to": 0, "time_s": 3.3}]
                    if option
                    else []
                ),
            }
            (students if mode == "learned" else rows).append(row)
    paths = []
    for name, branches in (("teacher", rows), ("student", students)):
        manifest = tmp_path / f"{name}_manifest.json"
        manifest.write_text(
            json.dumps(
                {"option_ids": ["neutral", "adapt"], "registry": trainer.artifact(registry_path)}
            )
        )
        result = tmp_path / f"{name}_result.json"
        result.write_text(json.dumps({"manifest": trainer.artifact(manifest), "rows": branches}))
        paths.append(result)
    output = tmp_path / "fit"
    trainer.train([paths[0]], output, [paths[1]])
    replay = json.loads((output / "registration.json").read_text())["replay"]
    assert replay["supervised_decisions_per_group"] == [1, 3]
    np.testing.assert_allclose(replay["effective_group_masses"], replay["probabilities"])
    assert len(replay["excluded_groups_without_consequential_targets"]) == 1
    teachers = json.loads((output / "teachers.json").read_text())
    report = json.loads((output / "result.json").read_text())
    fitted_masses = dict.fromkeys(replay["groups"], 0.0)
    for decision, weight in zip(report["fitted_decisions"], report["replay_weights"], strict=True):
        fitted_masses[teachers[decision["recorded_decision_index"]]["group"]] += weight
    np.testing.assert_allclose(list(fitted_masses.values()), replay["probabilities"])
