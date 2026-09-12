"""Later portable curves preserve assigned prefixes and generating-policy history."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_acquisition_curve_dataset_baseline import (
    ARMS,
    CURVE_SCHEMA,
    SEEDS,
    artifact_identity,
    historical_weights,
    measured_curve,
    select_weighted_rows,
    student_bindings,
    validate_index,
    verify_history_snapshot,
    verify_model_provenance,
)
from motion2scene_export_acquisition_curve_dataset import history_sources, validate_audit
from motion2scene_timing_diagnostic import artifact
from motion2scene_train_timed_schedules import weighted_targets

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_curriculum import (
    encounter_replay_weights,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_export_size_metadata_does_not_change_the_original_source_identity():
    source = dict(path="recorded_collection.json", sha256="original")
    assert artifact_identity(dict(source, size_bytes=100)) == source
    assert artifact_identity(dict(source, sha256="different", size_bytes=100)) != source


def complete_inventory(budget):
    runs, corpora = [], []
    for seed in SEEDS:
        for arm in (*ARMS, "reference_contrast"):
            run_id = f"seed{seed}_{arm}"
            rounds = [
                dict(candidate_id=f"{seed}_{n}", scene=f"scene_{n}", branch_order=list(range(7)))
                for n in range(budget + 1)
            ]
            runs.append(dict(run_id=run_id, arm=arm, seed=seed, rounds=rounds))
            corpora.append(
                dict(
                    run_id=run_id,
                    arm=arm,
                    seed=seed,
                    models=[dict(checkpoint=n) for n in range(budget + 1)],
                    counts=dict(episodes=7 + 8 * budget, physics_steps=1000),
                    physical_cost=dict(assigned_episodes=7 + 8 * budget, total_recorded_steps=1000),
                    tasks=[dict(candidate_id=r["candidate_id"]) for r in rounds[1:]],
                )
            )
    plan = dict(schema="motion2scene_expanded_acquisition_v1", checkpoints=[8, 16, 32], runs=runs)
    index = dict(
        schema=CURVE_SCHEMA,
        budget=budget,
        reported_checkpoints=[n for n in (8, 16, 32) if n <= budget],
        corpora=corpora,
        total_episodes=15 * (7 + 8 * budget),
        total_physics_steps=15000,
    )
    audit = dict(schema="motion2scene_response_diversity_v1", budget=budget, corpora=corpora)
    return index, audit, plan


@pytest.mark.parametrize("budget", [8, 16, 32])
def test_every_historical_model_and_measured_episode_is_required(budget):
    index, audit, plan = complete_inventory(budget)
    assert validate_index(index, plan) == (budget, index["reported_checkpoints"])
    assert validate_audit(audit, plan) == budget
    index["corpora"][0]["models"].pop(2)
    with pytest.raises(ValueError, match="historical model slot"):
        validate_index(index, plan)


@pytest.mark.parametrize("change", ["duplicate", "cost", "seed", "undeclared"])
def test_inventory_rejects_identity_cost_and_checkpoint_substitution(change):
    index, _, plan = complete_inventory(8)
    if change == "duplicate":
        index["corpora"][0] = copy.deepcopy(index["corpora"][1])
    elif change == "cost":
        index["corpora"][0]["counts"]["physics_steps"] += 1
    elif change == "seed":
        plan["runs"][0]["seed"] = 93202
    else:
        index["reported_checkpoints"] = [4, 8]
    with pytest.raises(ValueError):
        validate_index(index, plan)


@pytest.mark.parametrize("field", ["candidate_id", "scene", "branch_order"])
def test_replay_comparison_keeps_candidate_identity_geometry_and_branch_order(field):
    _, audit, plan = complete_inventory(8)
    replay = next(r for r in plan["runs"] if r["arm"] == "observation_curriculum")
    replay["rounds"][0][field] = "changed"
    with pytest.raises(ValueError, match="contrast/replay"):
        validate_audit(audit, plan)


def test_student_encounter_eight_remains_attached_to_model_seven(tmp_path):
    originals = [dict(policy=dict(path=f"M{n}", sha256=str(n))) for n in range(9)]
    save(tmp_path / "manifest.json", dict(source_results=list(range(17))))
    for encounter in range(1, 9):
        path = tmp_path / "provenance" / f"study_{8 + encounter:03d}" / "manifest.json"
        save(
            path,
            dict(
                policy=originals[encounter - 1]["policy"],
                cells=[dict(timed_schedule_mode="learned")],
            ),
        )
    assert student_bindings(tmp_path, originals)[-1]["generating_checkpoint"] == 7
    wrong = json.loads(path.read_text())
    wrong["policy"] = originals[8]["policy"]
    save(path, wrong)
    with pytest.raises(ValueError, match="generating historical policy"):
        student_bindings(tmp_path, originals)


def test_prefix_curve_uses_variable_measured_steps_and_keeps_unknown_bounds(tmp_path):
    budget = 16
    episodes, teachers = [], []
    for group in range(budget + 1):
        ids = []
        for option in range(7):
            identifier = f"teacher{group}_{option}"
            ids.append(identifier)
            outcome = "unknown" if group == 1 and option == 0 else "pass"
            episodes.append(
                dict(
                    episode_id=identifier,
                    configured_forced_option_id="neutral" if option == 0 else str(option),
                    collection_sha256=str(group),
                    physics_steps=group + 1,
                    assessment=dict(outcome=dict(task_outcome=outcome)),
                )
            )
        teachers.append(dict(episode_ids=ids))
    for encounter in range(1, budget + 1):
        episodes.append(
            dict(
                episode_id=f"student{encounter}",
                collection_sha256=str(budget + encounter),
                physics_steps=100 + encounter,
            )
        )
    save(
        tmp_path / "manifest.json",
        dict(episodes=episodes, source_results=[dict(sha256=str(n)) for n in range(33)]),
    )
    save(tmp_path / "teachers.json", teachers)
    m8, m16 = measured_curve(tmp_path, budget, [8, 16])
    assert (m8["assigned_tasks"], m16["assigned_tasks"]) == (8, 16)
    assert m8["acquisition_episodes"] == 71
    assert m8["recorded_physics_steps"] == 7 * sum(range(1, 10)) + sum(range(101, 109))
    assert m16["recorded_physics_steps"] == 7 * sum(range(1, 18)) + sum(range(101, 117))
    assert m8["adaptation_required_lower"] == 0 and m8["adaptation_required_upper"] == 1
    assert m16["unknown_branch_outcomes"] == 1


def test_portable_filter_matches_native_for_unavailable_and_failed_supervision():
    useful = dict(
        available=True, complete_legal_action_table=True, teacher_action=1, legal_mask=[1, 1]
    )
    failed = dict(useful, teacher_action=None)
    rows = [dict(available=False), failed, useful]
    for weights in (None, [0.0, 0.0, 1.0], [0.0, 0.2, 0.8]):
        actual, values = select_weighted_rows(rows, weights)
        expected, native = weighted_targets([dict(targets=rows)], weights)
        assert actual == expected
        assert (values is None and native is None) or np.array_equal(values, native)
    for invalid in ([0, 1, 0], [1, 0, 1], [0, float("nan"), 1]):
        with pytest.raises(ValueError):
            select_weighted_rows(rows, invalid)
        with pytest.raises(ValueError):
            weighted_targets([dict(targets=rows)], invalid)


def expanded_replay(tmp_path):
    groups = [
        dict(collection=dict(path=f"T{n}", sha256=str(n)), targets=[dict(phase_tick=15)])
        for n in range(2)
    ]
    originals = [dict(policy=dict(path="M0", sha256="model0"))]
    students = [dict(path="S1", sha256="student1")]
    records = [
        dict(
            encounter_id=f"E{n}",
            phase_tick=15,
            gap=0.0 if n == 0 else 0.5,
            supervision_available=True,
            coverage_key=("coverage", (0, 2), 15, True),
            teacher_collection=groups[n]["collection"],
            generating_model=None if n == 0 else originals[0]["policy"],
            student_collection=None if n == 0 else students[0],
        )
        for n in range(2)
    ]
    replay = dict(
        records=records,
        weights=encounter_replay_weights(records),
        signal="historical measured generating-policy outcomes",
    )
    path = save(tmp_path / "replay.json", replay)
    proof = dict(
        original_sidecar=artifact(path),
        teacher_collections=[g["collection"] for g in groups],
        historical_outcomes_independently_reaudited=True,
        new_physics_steps=0,
    )
    proof_path = save(tmp_path / "replay_verification.json", proof)
    return (
        dict(schema="motion2scene_expanded_fit_v1", weighting="historical_observation_gap"),
        groups,
        originals,
        students,
        {"replay.json": path, "replay_verification.json": proof_path},
    )


@pytest.mark.parametrize(
    "change", [None, "generating_model", "student_collection", "weights", "proof"]
)
def test_expanded_weights_require_exact_historical_signals_and_export_proof(tmp_path, change):
    registration, groups, originals, students, files = expanded_replay(tmp_path)
    if change is None:
        assert historical_weights(
            registration, {}, files, groups, originals, students
        ) == pytest.approx([0.2, 0.8])
        return
    if change == "proof":
        proof = json.loads(files["replay_verification.json"].read_text())
        proof["historical_outcomes_independently_reaudited"] = False
        save(files["replay_verification.json"], proof)
    else:
        replay = json.loads(files["replay.json"].read_text())
        if change == "weights":
            replay["weights"]["weights"] = [0.5, 0.5]
        else:
            replay["records"][1][change] = dict(path="wrong", sha256="wrong")
        save(files["replay.json"], replay)
        proof = json.loads(files["replay_verification.json"].read_text())
        proof["original_sidecar"] = artifact(files["replay.json"])
        save(files["replay_verification.json"], proof)
    with pytest.raises(ValueError):
        historical_weights(registration, {}, files, groups, originals, students)


def test_model_dependency_hash_cannot_be_rebound_with_only_a_new_outer_manifest(tmp_path):
    source = save(
        tmp_path / "registration.json",
        dict(
            schema="motion2scene_timed_schedule_training_v1",
            collections=["bootstrap"],
            replay_weights=None,
        ),
    )
    teachers = save(tmp_path / "teachers.json", [])
    policy = save(tmp_path / "policy.npz", {})
    result = save(
        tmp_path / "result.json",
        dict(registration=artifact(source), teachers=artifact(teachers), policy=artifact(policy)),
    )
    files = {p.name: p for p in (source, teachers, policy, result)}
    verify_model_provenance(files, artifact(result), ["bootstrap"], 0, "uniform", None, None)
    save(teachers, ["later_data"])
    with pytest.raises(ValueError, match="dependency differs"):
        verify_model_provenance(files, artifact(result), ["bootstrap"], 0, "uniform", None, None)


def test_history_copy_inventory_excludes_extras_and_rejects_raw_assets(tmp_path):
    source = save(tmp_path / "receipt.json", dict(outcome="unknown"))
    save(tmp_path / "unrequested.json", dict(extra=True))
    history = dict(
        schema="motion2scene_portable_historical_startup_v1",
        sources=dict(receipt=dict(file="receipt.json", source=artifact(source))),
    )
    path = save(tmp_path / "history.json", history)
    _, sources = history_sources(path)
    assert set(sources) == {"history.json", "receipt.json"}
    history["sources"]["raw"] = dict(file="weights.pt", source=artifact(source))
    save(tmp_path / "weights.pt", {})
    save(path, history)
    with pytest.raises(ValueError, match="text receipts"):
        history_sources(path)


def test_reader_verifies_original_unknown_assessment_without_opening_its_private_path(tmp_path):
    folder = tmp_path / "historical_startup"
    receipt = save(folder / "startup.json", dict(task_outcome="unknown", measured_steps=0))
    ref = dict(artifact(receipt), path="/unavailable/original/startup.json")
    history = save(
        folder / "history.json",
        dict(
            schema="motion2scene_portable_historical_startup_v1",
            sources=dict(startup=dict(file="startup.json", source=ref)),
        ),
    )
    registration = dict(history=artifact(history))
    assert verify_history_snapshot(tmp_path, registration)["sources"]["startup"]["source"] == ref
    save(receipt, dict(task_outcome="failure", measured_steps=0))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_history_snapshot(tmp_path, registration)
