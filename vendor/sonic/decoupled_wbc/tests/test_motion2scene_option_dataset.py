import json

import numpy as np
import pytest

from scripts.research import motion2scene_export_option_dataset as dataset


def test_sensor_whitelist_retains_normals_and_delayed_measurements_without_identity():
    legacy = {
        "time_s": 0.1,
        "capture_frame": 4,
        "rays": [{"hit": {"distance": 2.0, "path": "/World/Beam"}, "overhang": True}],
    }
    dense = {
        "time_s": 0.2,
        "measurements": [{"hit_normal_w": [0, 0, -1], "object_id": 7, "hit_distance_m": 1}],
        "delivered": legacy,
        "pass": True,
        "features": [1, 2],
        "state": {"dof_pos": [0.2], "scene": "secret"},
    }
    packets = dataset.sensor_packets({"observations": [dense]})
    assert packets[0]["measurements"] == [{"hit_normal_w": [0, 0, -1], "hit_distance_m": 1}]
    assert packets[0]["delivered"]["rays"] == [{"hit": {"distance": 2.0}}]
    assert packets[0]["delivered"]["capture_frame"] == 4
    assert packets[0]["state"] == {"dof_pos": [0.2]}
    assert not {"pass", "features"} & packets[0].keys()


def test_export_preserves_failure_reset_frames_and_checks_exact_causal_features(
    tmp_path, monkeypatch
):
    study = tmp_path / "synthetic_study"
    study.mkdir()

    def pinned(path):
        return {"path": str(path), "sha256": dataset.digest(path)}

    trajectory = study / "trajectory.pkl"
    trajectory.write_bytes(b"trusted-local-fixture")
    payload = {
        "motion_time_s": np.array([0.0, 0.02, 0.0]),
        "fps": 50,
        "dof_pos": np.ones((3, 2)),
        "partial_tracking_metrics": {"body_velocity_error": np.array([0.1, 0.3])},
    }
    monkeypatch.setattr(dataset, "load_reset_capture", lambda _: payload)
    features = np.arange(18, dtype=np.float32).reshape(3, 6)
    clocks = np.array([0.02, 0.04, 0.02])
    capture = np.arange(3, dtype=float) * 0.02
    feature_path = study / "features.npz"
    np.savez_compressed(
        feature_path,
        schema_version="fixture_named_6d_three_options",
        feature_names=np.array([f"feature_{i}" for i in range(6)]),
        features=features,
        phase_s=clocks,
        capture_elapsed_s=capture,
        active_before=np.zeros(3, dtype=int),
        legal_mask=np.ones((3, 3), dtype=bool),
        requested=np.ones(3, dtype=int),
        active_after=np.ones(3, dtype=int),
        teacher_action=np.ones(3, dtype=int),
    )
    sensor_path = study / "sensor.json"
    sensor = {
        "sensor": "ideal simulated query",
        "observations": [
            {
                "time_s": float(clocks[i]),
                "capture_elapsed_s": float(capture[i]),
                "features": features[i].tolist(),
                "measurements": [{"hit_distance_m": 1, "object_id": 42}],
                "active": 1,
            }
            for i in range(3)
        ],
    }
    dataset.write_json(sensor_path, sensor)
    for name in ("physics_beam_contacts.npz", "option_mechanical_work.npz"):
        np.savez_compressed(study / name, physics_clock_s=capture, recorded_value=[0, 1, 0])
    np.savez_compressed(study / "loaded_reference_bank.npz", root_xyz=np.zeros((3, 199, 3)))
    cell = {
        "cell_id": "case",
        "generation_seed": 42,
        "runtime_seed": 9,
        "beam": {"underside_m": 1.2},
    }
    manifest_path = study / "manifest.json"
    dataset.write_json(manifest_path, {"cells": [cell]})
    row = {
        "cell_id": "case",
        "trajectory": pinned(trajectory),
        "sensor": pinned(sensor_path),
        "features": pinned(feature_path),
        "pass": False,
        "first_episode_frames": 2,
        "reset_count": 1,
        "acquisition": {"physics_steps": 12},
    }
    result = {"manifest": pinned(manifest_path), "rows": [row]}
    result_path = study / "result.json"
    dataset.write_json(result_path, result)
    failed_startup = tmp_path / "failed_startup"
    failed_startup.mkdir()
    attempt = failed_startup / "attempt.json"
    dataset.write_json(attempt, {"exit_status": 1, "wall_seconds": 1.5, "command": ["fixture"]})
    output = tmp_path / "portable"
    report = dataset.export(output, [result_path], attempts=[attempt])
    assert report["episodes"] == 1
    infrastructure = json.loads((output / "infrastructure_attempts.json").read_text())
    assert infrastructure[0]["recorded_control_frames"] == 0
    assert infrastructure[0]["physics_steps"] is None
    assert infrastructure[0]["scored_physical_episode"] is False
    episodes = json.loads((output / "episodes.json").read_text())
    episode = episodes[0]
    assert episode["pass"] is False
    assert episode["reset_count"] == 1
    assert episode["first_episode_frames"] == 2
    assert episode["recorded_control_frames"] == 3
    assert episode["student_input"]["action_dimension"] == 3
    assert episode["sensor_alignment"]["eligible_packets"] == 2
    folder = output / episode["directory"]
    with np.load(folder / "sensor_alignment.npz", allow_pickle=False) as data:
        assert data["packet_eligible"].tolist() == [True, True, False]
    with np.load(folder / "student_history.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["features"], features)
        assert not {"requested", "active_after", "teacher_action"} & set(data.files)
    with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["motion_time_s"], payload["motion_time_s"])
        np.testing.assert_array_equal(
            data["partial_tracking_metrics/body_velocity_error"], [0.1, 0.3]
        )
    (folder / "sensor_history.json").write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        dataset.audit(output)
    sensor["observations"][0]["features"][0] = -100
    dataset.write_json(sensor_path, sensor)
    row["sensor"] = pinned(sensor_path)
    dataset.write_json(result_path, result)
    with pytest.raises(ValueError, match="policy inputs disagree"):
        dataset.export(tmp_path / "mismatched", [result_path])


def test_missing_or_duplicate_physical_rows_do_not_create_a_completed_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset.export(tmp_path / "output", [tmp_path / "unexecuted.json"])
    manifest = tmp_path / "manifest.json"
    dataset.write_json(manifest, {"cells": [{"cell_id": "a"}, {"cell_id": "b"}]})
    result = tmp_path / "result.json"
    dataset.write_json(
        result,
        {
            "manifest": {"path": str(manifest), "sha256": dataset.digest(manifest)},
            "rows": [{"cell_id": "a"}, {"cell_id": "a"}],
        },
    )
    with pytest.raises(ValueError, match="exactly once"):
        dataset.export(tmp_path / "output", [result])


@pytest.mark.parametrize("selected", [0, 2])
def test_multi_option_export_distinguishes_preferred_four_from_selected_two_or_neutral(
    tmp_path, monkeypatch, selected
):
    study = tmp_path / "multi_study"
    study.mkdir()

    def ref(path):
        return {"path": str(path), "sha256": dataset.digest(path)}

    options = ["neutral", "d040", "d055", "d070", "d085"]
    names = (
        [f"feature_{i}" for i in range(100)]
        + [f"active_option_{i}" for i in range(5)]
        + [f"option_{i}_legal" for i in range(5)]
    )
    features = np.zeros((199, 110), dtype=np.float32)
    phases = np.r_[np.arange(1, 199) / 50, 0.02]
    observations, switches = [], []
    active = 0
    for i, phase in enumerate(phases):
        before = active
        if phase == 0.2:
            active = selected
        if phase == 3.3:
            active = 0
        if before != active:
            switches.append({"from": before, "to": active, "time_s": float(phase)})
        observations.append(
            {
                "time_s": float(phase),
                "capture_elapsed_s": i / 50,
                "features": features[i].tolist(),
                "active_before": before,
                "active": active,
                "legal_mask": [True] * 5,
                "measurements": [{"origin_w": [0, 0, 1], "hit_distance_m": 1, "object_id": 9}],
                "transition": {"requested": active, "attempted": before != active, "allowed": True},
            }
        )
    registry_path = study / "registry.json"
    neutral = {"path": "/fixture/neutral.pkl", "sha256": "sha256:fixture-neutral"}
    controller = {"path": "/fixture/tracker.pt", "sha256": "sha256:fixture-tracker"}
    dataset.write_json(
        registry_path,
        {
            "source": 42,
            "controller": controller,
            "references": [{"name": name, "motion": neutral} for name in options],
        },
    )
    sensor_path = study / "sensor.json"
    dataset.write_json(
        sensor_path,
        {
            "observations": observations,
            "mode": "learned",
            "switches": switches,
            "feature_names": names,
            "feature_schema": dataset.MULTI_SCHEMA,
            "option_names": options,
            "option_registry_sha256": ref(registry_path)["sha256"],
        },
    )
    feature_path = study / "features.npz"
    np.savez_compressed(
        feature_path,
        schema_version=dataset.MULTI_SCHEMA,
        feature_names=names,
        features=features,
        phase_s=phases,
        capture_elapsed_s=np.arange(199) / 50,
        active_before=[o["active_before"] for o in observations],
        legal_mask=np.ones((199, 5), dtype=bool),
        option_ids=options,
        classes=np.arange(5),
        requested=[o["active"] for o in observations],
    )
    trajectory_path = study / "trajectory.pkl"
    trajectory_path.write_bytes(b"fixture")
    monkeypatch.setattr(
        dataset, "load_reset_capture", lambda _: {"fps": 50, "motion_time_s": np.arange(199) / 50}
    )
    for name in ("physics_beam_contacts.npz", "option_mechanical_work.npz"):
        np.savez_compressed(study / name, clocks=np.arange(199) / 50)
    np.savez_compressed(study / "loaded_reference_bank.npz", root_xyz=np.zeros((5, 199, 3)))
    cell = {
        "cell_id": "learner",
        "generation_seed": 42,
        "runtime_seed": 7,
        "motion": neutral,
        "option_index": 4,
        "multi_option_mode": "learned",
        "decision_time_s": 0.3,
    }
    manifest_path = study / "manifest.json"
    dataset.write_json(
        manifest_path,
        {
            "cells": [cell],
            "option_ids": options,
            "registry": ref(registry_path),
            "implementation": {"checkpoint": controller},
        },
    )
    row = {
        "cell_id": "learner",
        "mode": "learned",
        "option_index": 4,
        "decision_time_s": 0.3,
        "measurement_admitted": True,
        "switches": switches,
        "pass": True,
        "first_episode_frames": 199,
        "reset_count": 0,
        "trajectory": ref(trajectory_path),
        "sensor": ref(sensor_path),
        "features": ref(feature_path),
    }
    result_path = study / "result.json"
    dataset.write_json(result_path, {"manifest": ref(manifest_path), "rows": [row]})
    output = tmp_path / "portable_multi"
    dataset.export(output, [result_path])
    episode = json.loads((output / "episodes.json").read_text())[0]
    assert episode["configured_preference"]["option_index"] == 4
    assert episode["executed_option"]["option_index"] == selected
    assert episode["executed_option"]["stayed_neutral"] == (selected == 0)
    assert episode["executed_option"]["entry_time_s"] == (0.2 if selected else None)
    assert episode["executed_option"]["return_time_s"] == (3.3 if selected else None)
    assert episode["measurement_admitted"] is True
    assert episode["student_input"]["feature_dimension"] == 110
    folder = output / episode["directory"]
    with np.load(folder / "student_history.npz", allow_pickle=False) as arrays:
        assert arrays["option_ids"].tolist() == options
        np.testing.assert_array_equal(arrays["features"], features)
        assert "requested" not in arrays.files
    with np.load(folder / "sensor_alignment.npz", allow_pickle=False) as arrays:
        assert arrays["packet_eligible"].sum() == 198
        assert not arrays["packet_eligible"][-1]
    assert len(json.loads((folder / "sensor_history.json").read_text())) == 199


@pytest.mark.parametrize("eligible", [True, False])
def test_portable_wait_target_uses_recorded_later_adaptation_and_only_eligible_inputs(
    tmp_path, eligible
):
    output = tmp_path / "package"
    folder = output / "episodes/neutral"
    folder.mkdir(parents=True)
    np.savez_compressed(
        folder / "student_history.npz", features=[[1.0, 2.0, 3.0]], phase_s=[0.2], active_before=[0]
    )
    dataset.write_json(folder / "sensor_alignment.json", {"packet_eligible": [eligible]})
    walking = {"path": "/recorded/walk.pkl", "sha256": "sha256:walk"}
    adapting = {"path": "/recorded/adapt_later.pkl", "sha256": "sha256:adapt"}
    episodes = [
        {
            "episode_id": "walk",
            "directory": "episodes/neutral",
            "student_input": {"file": "student_history.npz"},
        },
        {"episode_id": "adapt", "directory": "episodes/adapt"},
    ]
    provenance = [
        {"episode_id": "walk", "artifacts": {"trajectory": walking}},
        {"episode_id": "adapt", "artifacts": {"trajectory": adapting}},
    ]
    source = tmp_path / "teachers.json"
    dataset.write_json(
        source,
        [
            {
                "phase_s": 0.2,
                "features": [1.0, 2.0, 3.0],
                "teacher_action": 0,
                "continuation_branch_ids": [adapting["path"], None],
                "matching_audit": [{"branch": walking, "matched": True}],
                "waiting_has_future_adaptation": True,
            }
        ],
    )
    if not eligible:
        with pytest.raises(ValueError, match="exact packaged causal student input"):
            dataset.export_schedule_targets(output, [source], episodes, provenance)
    else:
        assert dataset.export_schedule_targets(output, [source], episodes, provenance) == 1
        target = json.loads((output / "schedule_teachers.json").read_text())[0]
        assert target["teacher_action"] == 0
        assert target["continuation_episode_ids"] == ["adapt", None]
        assert target["student_input"]["episode_id"] == "walk"
        assert target["waiting_has_future_adaptation"]
        assert "features" not in target


@pytest.mark.parametrize("matching_student", [True, False])
def test_incremental_target_preserves_actual_student_in_related_package(
    tmp_path, monkeypatch, matching_student
):
    # The existing package is audited separately; this fixture isolates exact
    # student provenance and input admission across immutable package boundaries.
    monkeypatch.setattr(dataset, "audit", lambda path: {})
    output = tmp_path / "increment"
    teacher_folder = output / "episodes/teacher"
    teacher_folder.mkdir(parents=True)
    student_package = tmp_path / "old_policy_package"
    student_folder = student_package / "episodes/actual_student"
    student_folder.mkdir(parents=True)
    for folder, feature in (
        (teacher_folder, [1.0, 2.0, 3.0]),
        (student_folder, [1.0, 2.0, 3.0] if matching_student else [9.0, 2.0, 3.0]),
    ):
        np.savez_compressed(
            folder / "student_history.npz", features=[feature], phase_s=[0.3], active_before=[0]
        )
        dataset.write_json(folder / "sensor_alignment.json", {"packet_eligible": [True]})
    teacher_ref = {"path": "/recorded/teacher.pkl", "sha256": "sha256:teacher"}
    student_ref = {"path": "/recorded/actual_student.pkl", "sha256": "sha256:student"}
    teacher = {
        "episode_id": "teacher",
        "directory": "episodes/teacher",
        "student_input": {"file": "student_history.npz"},
    }
    student = {
        "episode_id": "actual_student",
        "directory": "episodes/actual_student",
        "student_input": {"file": "student_history.npz"},
    }
    dataset.write_json(student_package / "manifest.json", {"files": {}})
    dataset.write_json(student_package / "episodes.json", [student])
    dataset.write_json(
        student_package / "provenance.json",
        [{"episode_id": "actual_student", "artifacts": {"trajectory": student_ref}}],
    )
    source = tmp_path / "teachers.json"
    dataset.write_json(
        source,
        [
            {
                "phase_s": 0.3,
                "features": [1.0, 2.0, 3.0],
                "teacher_action": 0,
                "continuation_branch_ids": [teacher_ref["path"], None],
                "matching_audit": [{"branch": teacher_ref, "matched": True}],
                "student_visit": {"student_branch": student_ref, "matched": True},
            }
        ],
    )
    args = (
        output,
        [source],
        [teacher],
        [{"episode_id": "teacher", "artifacts": {"trajectory": teacher_ref}}],
        [student_package],
    )
    if not matching_student:
        with pytest.raises(ValueError, match="external student visit does not match"):
            dataset.export_schedule_targets(*args)
        return
    assert dataset.export_schedule_targets(*args) == 1
    target = json.loads((output / "schedule_teachers.json").read_text())[0]
    external = target["student_visit"]["external_student_episode"]
    assert external["episode_id"] == "actual_student"
    assert external["trajectory"] == student_ref
    assert external["dataset_manifest_sha256"] == dataset.digest(student_package / "manifest.json")
    assert target["student_input"]["episode_id"] == "teacher"
    assert not (output / "episodes/actual_student").exists()
