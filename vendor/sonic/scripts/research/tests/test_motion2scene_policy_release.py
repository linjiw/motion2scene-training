"""Adversarial checks for mixed scorer packages and partial physics accounting."""

import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from decoupled_wbc.tests.test_motion2scene_environment_contacts import fixture as contacts_fixture
from decoupled_wbc.tests.test_motion2scene_timed_outcome import fixture as outcome_fixture
from gear_sonic.dataset_generation.hallucination.motion2scene_environment_contacts import (
    BEAM_PATH,
    audit_environment_contacts,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_outcome import (
    classify_timed_attempt,
)

SCRIPTS = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PANEL = load("motion2scene_export_policy_panel")
WORKER = load("motion2scene_policy_release_worker")


def slot(index, complete=True, known=True):
    value = dict(
        collection_sha256="sha256:" + str(index),
        original_cell_id="cell",
        source_capture=dict(path="source/" + str(index)),
        capture_status="complete_episode" if complete else "partial_attempt",
        physical_rows=298 if complete else 189,
        sensor_packets=298 if complete else 188,
        physics_steps=1192 if complete else 756,
        task_outcome_admitted=known,
        costs=dict(passage_time_s=2.5 if complete else None),
    )
    value["pass"] = complete
    return value


def test_partial_physics_and_sensor_lengths_are_counted_independently():
    rows = [slot(i) for i in range(35)] + [slot(35, False)]
    counts = PANEL.count_slots(rows)
    assert counts["assigned_slots"] == 36
    assert counts["complete_episode_count"] == 35
    assert counts["partial_attempt_count"] == 1
    assert counts["physical_rows"] == 10619
    assert counts["sensor_packets"] == 10618
    assert counts["physics_steps"] == 42476
    assert counts["known_failure_count"] == 1 and counts["teacher_targets"] == 0
    rows[-1]["task_outcome_admitted"] = False
    counts = PANEL.count_slots(rows)
    assert counts["unknown_outcome_count"] == 1 and counts["known_failure_count"] == 0


def test_no_duplicate_capture_fabricated_cost_or_partial_pass():
    with pytest.raises(ValueError, match="once"):
        PANEL.count_slots([slot(0), slot(0)])
    rows = [slot(0), slot(1)]
    rows[1]["source_capture"] = rows[0]["source_capture"]
    with pytest.raises(ValueError, match="capture"):
        PANEL.count_slots(rows)
    bad = slot(0, False)
    bad["costs"]["passage_time_s"] = 5.96
    with pytest.raises(ValueError, match="fabricated"):
        PANEL.count_slots([bad])
    bad["pass"] = True
    with pytest.raises(ValueError, match="complete"):
        PANEL.count_slots([bad])


def test_relocation_only_changes_provenance_root_and_rejects_other_workspace():
    source = (
        "from pathlib import Path\nROOT = Path(__file__).resolve().parents[2]\n"
        'x = Path(dependency["path"]).relative_to(ROOT)\n'
    )
    mapped = PANEL.relocate_exporter(source, Path("/original/repo"))
    assert "ROOT = Path(__file__).resolve().parents[2]" in mapped
    assert PANEL.relocate_exporter(mapped, Path("/original/repo")) == mapped
    with pytest.raises(ValueError, match="original workspace"):
        PANEL.relocate_exporter(mapped, Path("/different/repo"))
    with pytest.raises(ValueError, match="mapping"):
        PANEL.relocate_exporter(
            source.replace("relative_to(ROOT)", "relative_to(OTHER)"), Path("/repo")
        )


def test_identity_overlay_preserves_hash_bytes_without_reading_changed_source(tmp_path):
    original, snapshot = tmp_path / "original.py", tmp_path / "frozen.py"
    original.write_text("declared bytes\n")
    snapshot.write_bytes(original.read_bytes())
    refs = [dict(original=PANEL.artifact(original), snapshot=PANEL.artifact(snapshot))]
    original.write_text("different current production source\n")
    original_open = io.open
    try:
        reads = WORKER.install_identity_overlay(refs)
        assert original.read_text() == "declared bytes\n"
        assert reads == [str(original)]
        with pytest.raises(PermissionError, match="read-only"):
            original.write_text("forbidden")
    finally:
        io.open = original_open
    assert original.read_text() == "different current production source\n"
    snapshot.write_text("corrupt frozen identity\n")
    with pytest.raises(ValueError, match="identity asset changed"):
        WORKER.install_identity_overlay(refs)


@pytest.fixture
def partial(tmp_path):
    folder = tmp_path / "failed"
    folder.mkdir()
    payload, observations, _ = outcome_fixture(30)
    observations = observations[:-1]  # Actual aborted capture has one extra physical row.
    pairs, net, mapping = contacts_fixture()
    pairs["force_w"] = np.repeat(pairs["force_w"][:1], 120, axis=0)
    pairs["force_w"][30, 1, -1, 0] = 385.0
    pairs["physics_steps"] = np.arange(1, 121)
    net["net_force_w"] = pairs["force_w"].sum(axis=2)[:, ::-1].copy()
    net["physics_steps"] = np.arange(1, 121)
    context = dict(
        reference_frames=80,
        phase_ticks=[15, 50, 70],
        attempt=dict(exit_status=1),
        scene_definition=dict(beam_paths=[BEAM_PATH]),
        declared_neutral_self_pairs=[["left_hip_roll_link", "left_wrist_yaw_link"]],
    )
    arrays = {"0/" + key: value for key, value in payload.items() if key != "fps"}
    metadata = {"0": {key: {"npz_array": "0/" + key} for key in payload if key != "fps"}}
    np.savez_compressed(folder / "aborted_raw_recording.npz", **arrays)
    np.savez_compressed(folder / "aborted_environment_pair_contacts.npz", **pairs)
    np.savez_compressed(folder / "aborted_all_body_contacts.npz", **net)
    WORKER.write_new(folder / "aborted_raw_recording_metadata.json", metadata)
    WORKER.write_new(folder / "aborted_environment_contact_mapping.json", mapping)
    WORKER.write_new(folder / "partial_interface.json", dict(observations=observations))

    def array_file(path):
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    exporter = SimpleNamespace(
        safe_path=PANEL.safe_path,
        array_file=array_file,
        scene_beam_paths=lambda scene: scene["beam_paths"],
        audit_environment_contacts=audit_environment_contacts,
        classify_timed_attempt=classify_timed_attempt,
    )
    contact = audit_environment_contacts(
        pairs, net, mapping, 120, neutral_self_pairs=context["declared_neutral_self_pairs"]
    )
    outcome = classify_timed_attempt(
        payload,
        observations,
        reference_frames=80,
        phase_ticks=context["phase_ticks"],
        exit_status=1,
        contact_audit=contact,
        physics_steps=pairs["physics_steps"],
    )
    assessment = dict(
        contact_audit=contact,
        outcome=outcome,
        physics_steps=120,
        measurement_admitted=False,
        task_outcome_admitted=True,
        costs=dict(passage_time_s=None, whole_episode_time_s=None, positive_mechanical_work_j=None),
    )
    assessment["pass"] = False
    return exporter, tmp_path, dict(directory="failed", assessment=assessment), context


def test_partial_native_contact_failure_retains_unavailable_later_phases(partial):
    result = WORKER.partial_audit(*partial)
    assert (result["physical_rows"], result["sensor_packets"], result["physics_steps"]) == (
        30,
        29,
        120,
    )
    assert result["outcome"]["task_outcome"] == "failure"
    assert not result["measurement_admitted"] and result["costs"]["passage_time_s"] is None
    assert [p["sensor_preaction_available"] for p in result["phase_availability"]] == [
        True,
        False,
        False,
    ]
    assert not any(p["teacher_target_available"] for p in result["phase_availability"])
    json.dumps(result, allow_nan=False)


def test_partial_contact_tamper_or_fabricated_outcome_is_detected(partial):
    exporter, root, failure, context = partial
    changed = copy.deepcopy(failure)
    changed["assessment"]["outcome"]["task_outcome"] = "success"
    with pytest.raises(ValueError, match="assessment differs"):
        WORKER.partial_audit(exporter, root, changed, context)
    path = root / "failed/aborted_environment_pair_contacts.npz"
    pairs = exporter.array_file(path)
    pairs["force_w"][30, 1, -1] = 0  # Leave independent net force unchanged.
    np.savez_compressed(path, **pairs)
    with pytest.raises(ValueError, match="contact streams"):
        WORKER.partial_audit(exporter, root, failure, context)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "counter", "cost"])
def test_partial_stream_ambiguity_missing_clock_and_imputed_cost_fail(partial, mutation):
    exporter, root, failure, context = partial
    source = root / "failed/aborted_environment_pair_contacts.npz"
    if mutation == "duplicate":
        (source.parent / "environment_pair_contacts.npz").write_bytes(source.read_bytes())
        expected = "unambiguous"
    elif mutation == "missing":
        source.unlink()
        expected = "unambiguous"
    elif mutation == "counter":
        pairs = exporter.array_file(source)
        pairs["physics_steps"] = pairs["physics_steps"][:-1]
        np.savez_compressed(source, **pairs)
        expected = "contact streams"
    else:
        failure["assessment"]["costs"]["whole_episode_time_s"] = 0.58
        expected = "assessment differs"
    with pytest.raises(ValueError, match=expected):
        WORKER.partial_audit(exporter, root, failure, context)
