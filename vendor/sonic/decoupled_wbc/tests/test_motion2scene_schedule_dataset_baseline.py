"""Portable WAIT supervision without loading original simulation assets."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from decoupled_wbc.tests.test_motion2scene_timed_schedule_dataset import portable_teacher
from decoupled_wbc.tests.test_motion2scene_timed_schedule_training_audit import (
    collection as source_collection,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "schedule_dataset_baseline", ROOT / "scripts/research/motion2scene_schedule_dataset_baseline.py"
)
BASELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINE)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def refresh_inventory(package, manifest):
    manifest["files"] = [
        dict(
            path=str(p.relative_to(package)), sha256=BASELINE.sha256(p), size_bytes=p.stat().st_size
        )
        for p in sorted(package.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    ]
    write_json(package / "manifest.json", manifest)


@pytest.fixture
def package(tmp_path, monkeypatch):
    fixture = source_collection.__wrapped__(tmp_path / "private_source", monkeypatch)
    teacher, source = portable_teacher(fixture)
    package = tmp_path / "portable"
    identifier = fixture["registry"]["sha256"]
    teacher["registry_id"] = identifier
    request = dict(fixture["bank"].request, expected_loaded_frames=299)
    write_json(
        package / "registry.json",
        dict(schema="motion2scene_timed_option_registry_v1", request=request),
    )
    episodes = []
    for name, (episode, payload, interface) in source.items():
        folder = package / "episodes" / name
        observations = interface["observations"]
        features = np.asarray([r["features"] for r in observations], dtype=np.float64)
        names = BASELINE.expected_feature_names(7)
        folder.mkdir(parents=True)
        np.savez_compressed(
            folder / "student_inputs.npz",
            schema_version=BASELINE.INPUT_SCHEMA,
            feature_names=names,
            option_ids=fixture["bank"].option_ids,
            features=features,
            command_ticks=np.arange(1, 299),
            capture_elapsed_s=np.arange(298) / 50,
            legal_mask=[r["legal_mask"] for r in observations],
            active_before=[r["active_before"] for r in observations],
        )
        np.savez_compressed(folder / "trajectory.npz", **payload)
        write_json(
            folder / "trajectory_metadata.json", {key: {"npz_array": key} for key in payload}
        )
        write_json(folder / "sensor_alignment.json", dict(packet_eligible=[True] * 298))
        write_json(
            folder / "sensor_history.json",
            [
                {
                    key: r[key]
                    for key in BASELINE.PREACTION_KEYS
                    if key not in ("features", "active_before", "legal_mask")
                }
                for r in observations
            ],
        )
        write_json(
            folder / "commands.json",
            dict(
                records=[
                    {key: r[key] for key in ("tick", "active_before", "legal_mask")}
                    for r in observations
                ]
            ),
        )
        episodes.append(
            dict(
                **episode,
                episode_id=name,
                directory=f"episodes/{name}",
                mode="forced",
                split="development",
                registry_id=identifier,
                physical_control_frames=298,
                source_trajectory=dict(
                    path=f"/unavailable/private/{name}.pkl", sha256="same_bytes_allowed"
                ),
            )
        )
    write_json(package / "teachers.json", [teacher])
    manifest = dict(
        schema=BASELINE.DATA_SCHEMA,
        registries={identifier: dict(portable="registry.json")},
        episodes=episodes,
        failed_attempts=[],
        counts=dict(episodes=7, teacher_decisions=3, physics_steps=8344),
    )
    refresh_inventory(package, manifest)
    return package


def test_portable_wait_recomputed_then_pickle_free_fit_round_trip(package, tmp_path):
    bank, targets, report = BASELINE.inspect_package(package)
    assert report["teacher_decisions"] == 3
    assert targets[0]["waiting_has_future_adaptation"]
    assert targets[0]["continuation_option_indices"][0] == 6
    model, fit = BASELINE.fit_targets(bank, targets, l2=0.1)
    assert fit["status"] == "offline_fit_complete"
    path = tmp_path / "model.npz"
    np.savez_compressed(path, **model)
    BASELINE.validate_schedule_policy(BASELINE.arrays(path), bank)
    assert report["motion_assets_independently_qualified"] is False
    assert report["no_external_source_assets_opened"] is True


def test_missing_legal_branch_is_not_encoded_as_failure(package):
    bank, targets, _ = BASELINE.inspect_package(package)
    changed = copy.deepcopy(targets)
    changed[0]["admitted"][1] = False
    model, report = BASELINE.fit_targets(bank, changed, 0.1)
    assert model is None
    assert report["status"] == "insufficient_complete_consequential_phase_targets"
    assert report["supervised_phase_counts"]["15"] == 0
    assert report["excluded_decision_indices"] == [0]


def test_teacher_cannot_replace_wait_with_forced_neutral_or_drop_an_episode(package):
    teachers = BASELINE.read_json(package / "teachers.json")
    teachers[0]["targets"][0]["continuation_episode_ids"][0] = teachers[0]["episode_ids"][0]
    write_json(package / "teachers.json", teachers)
    manifest = BASELINE.read_json(package / "manifest.json")
    refresh_inventory(package, manifest)
    with pytest.raises(ValueError, match="continuation identity"):
        BASELINE.inspect_package(package)
    teachers[0]["episode_ids"].pop()
    write_json(package / "teachers.json", teachers)
    refresh_inventory(package, manifest)
    with pytest.raises(ValueError, match="seven distinct"):
        BASELINE.inspect_package(package)


def test_hash_inventory_external_digest_and_schema_changes_are_rejected(package):
    with pytest.raises(ValueError, match="external digest"):
        BASELINE.inspect_package(package, "0" * 64)
    original = (package / "teachers.json").read_bytes()
    (package / "teachers.json").write_text("[]")
    with pytest.raises(ValueError, match="hash/size"):
        BASELINE.inspect_package(package)
    (package / "teachers.json").write_bytes(original)
    manifest = BASELINE.read_json(package / "manifest.json")
    manifest["schema"] = "legacy_110D"
    write_json(package / "manifest.json", manifest)
    with pytest.raises(ValueError, match="dataset schema"):
        BASELINE.inspect_package(package)


def test_no_teachers_explicitly_produces_no_model_and_dataset_is_immutable(package, tmp_path):
    write_json(package / "teachers.json", [])
    manifest = BASELINE.read_json(package / "manifest.json")
    manifest["counts"]["teacher_decisions"] = 0
    refresh_inventory(package, manifest)
    before = BASELINE.sha256(package / "manifest.json")
    args = SimpleNamespace(
        dataset=package, out=tmp_path / "baseline", action="fit", l2=0.1, manifest_sha256=before
    )
    result = BASELINE.run(args)
    assert result["status"] == "no_complete_teacher_groups" and result["policy"] is None
    assert not (args.out / "policy.npz").exists()
    assert BASELINE.sha256(package / "manifest.json") == before
    args.out = package / "forbidden_output"
    with pytest.raises(ValueError, match="immutable dataset"):
        BASELINE.run(args)


def test_relative_paths_cannot_escape_package(package, tmp_path):
    with pytest.raises(ValueError, match="confined relative"):
        BASELINE.relative_path(package, "../outside")
    target = tmp_path / "outside.txt"
    target.write_text("private")
    (package / "escape").symlink_to(target)
    with pytest.raises(ValueError, match="escapes package"):
        BASELINE.relative_path(package, "escape")


def test_portable_opt_in_fits_observed_tie_and_records_setting(package, tmp_path, monkeypatch):
    bank, targets, inspection = BASELINE.inspect_package(package)
    targets = copy.deepcopy(targets)
    phase = next(row for row in targets if row["phase_tick"] == 50)
    for index, legal in enumerate(phase["legal_mask"]):
        if legal:
            phase["admitted"][index] = True
            phase["pass_labels"][index] = True
            phase["passage_time_s"][index] = 5.4
    legacy, report = BASELINE.fit_targets(bank, targets, 0.1)
    assert legacy is None
    assert report["status"] == "insufficient_complete_consequential_phase_targets"
    # Synthetic inspected targets isolate CLI-to-fitter propagation; these edits
    # are not represented as physically reverified archive contents.
    monkeypatch.setattr(BASELINE, "inspect_package", lambda *args: (bank, targets, inspection))
    args = SimpleNamespace(
        dataset=package,
        out=tmp_path / "opt_in",
        action="fit",
        l2=0.1,
        manifest_sha256=None,
        allow_measured_tie_initialization=True,
    )
    result = BASELINE.run(args)
    assert result["status"] == "offline_fit_complete"
    assert result["fit"]["initialized_phase_ticks"] == [50]
    assert (
        BASELINE.read_json(args.out / "registration.json")["allow_measured_tie_initialization"]
        is True
    )
    model = BASELINE.arrays(args.out / "policy.npz")
    assert np.count_nonzero(model["weights"][1]) == np.count_nonzero(model["bias"][1]) == 0


def test_portable_opt_in_retains_no_model_receipt_when_phase_is_missing(
    package, tmp_path, monkeypatch
):
    bank, targets, inspection = BASELINE.inspect_package(package)
    targets = copy.deepcopy(targets)
    targets[0]["admitted"][1] = False
    monkeypatch.setattr(BASELINE, "inspect_package", lambda *args: (bank, targets, inspection))
    args = SimpleNamespace(
        dataset=package,
        out=tmp_path / "rejected",
        action="fit",
        l2=0.1,
        manifest_sha256=None,
        allow_measured_tie_initialization=True,
    )
    result = BASELINE.run(args)
    assert result["status"] == "offline_fit_rejected" and result["policy"] is None
    assert BASELINE.read_json(args.out / "result.json") == result
    assert not (args.out / "policy.npz").exists()


@pytest.mark.parametrize("value", [0, 1, None])
def test_portable_tie_setting_rejects_non_boolean_before_creating_output(tmp_path, value):
    args = SimpleNamespace(out=tmp_path / "rejected", allow_measured_tie_initialization=value)
    with pytest.raises(ValueError, match="explicit boolean"):
        BASELINE.run(args)
    assert not args.out.exists()
