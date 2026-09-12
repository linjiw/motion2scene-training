"""No physics: frozen pairing, checkpoint provenance and failure denominators."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

import motion2scene_run_reserved_evaluation as runner  # noqa: E402
from test_motion2scene_primary_controller import FakeBackend, fixture  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    artifact,
    read_bound,
    write_new,
)


@pytest.fixture
def proposed(tmp_path):
    context = fixture(tmp_path)
    original = artifact(ROOT / "docs/motion2scene/TRAVERSAL_PROTOCOL_AMENDMENT_V3_PROPOSED.json")
    protocol = runner.propose_v4(original, context["plan_ref"])
    return context, protocol


def test_complete972_pairing_is_reproducible_and_does_not_mutate_v3(proposed, tmp_path):
    context, protocol = proposed
    before = artifact(ROOT / "docs/motion2scene/TRAVERSAL_PROTOCOL_AMENDMENT_V3_PROPOSED.json")
    table = runner.assignments(protocol, tmp_path / "future")
    assert (
        len(table)
        == len({(r["policy_id"], r["layout_id"], r["physics_seed"]) for r in table})
        == 972
    )
    assert {r["variant_id"] for r in table} == {"nominal"}
    assert {r["physics_seed"] for r in table} == {94301, 94302}
    assert len({r["layout_id"] for r in table}) == 18
    assert all(
        sum(r["policy_id"] == p for r in table) == 36 for p in {r["policy_id"] for r in table}
    )
    protocol["policies"].reverse()
    protocol["scenes"].reverse()
    assert runner.assignments(protocol, tmp_path / "future") == table
    assert artifact(Path(before["path"])) == before
    assert not (tmp_path / "future").exists()
    slots = runner.model_slots(context["plan"])
    assert len(slots) == 24
    assert {r["acquisition_assigned_maximum_steps"] for r in slots} == {27416, 46488}


@pytest.mark.parametrize("change", ["missing_policy", "extra_policy", "stress", "seed", "geometry"])
def test_changed_scope_never_silently_reduces_or_expands_pairing(proposed, tmp_path, change):
    _, protocol = proposed
    if change == "missing_policy":
        protocol["policies"].pop()
    elif change == "extra_policy":
        protocol["policies"].append(dict(protocol["policies"][0], policy_id="extra"))
    elif change == "stress":
        protocol["enabled_variant_ids"].append("yaw_rad_plus")
    elif change == "seed":
        protocol["physics_seeds"] = [94301]
    else:
        protocol["scenes"][0]["beams"][0]["underside_m"] += 0.001
    with pytest.raises(ValueError):
        runner.assignments(protocol, tmp_path / "future")


def test_proposed_protocol_cannot_prepare_any_child(proposed, tmp_path):
    _, protocol = proposed
    ref = write_new(tmp_path / "proposed.json", protocol)
    with pytest.raises(ValueError, match="proposed/unadopted"):
        runner.prepare(ref, tmp_path / "evaluation")
    assert not (tmp_path / "evaluation").exists()


def test_pending_or_partial_model_inventory_is_rejected(proposed, tmp_path):
    context, protocol = proposed
    ref = write_new(tmp_path / "proposed.json", protocol)
    inventory = runner.inventory_template(context["plan_ref"], ref)
    with pytest.raises(ValueError, match="all real primary"):
        runner.validate_inventory(inventory, protocol)
    inventory["status"] = "FROZEN_COMPLETE"
    inventory["models"].pop()
    with pytest.raises(ValueError, match="exactly24"):
        runner.validate_inventory(inventory, protocol)


class SeedBackend(FakeBackend):
    def prepare(self, context, row, kind, model):
        super().prepare(context, row, kind, model)
        path = Path(row[kind + "_directory"]) / "manifest.json"
        value = read_bound(artifact(path))
        for cell in value["cells"]:
            cell["runtime_seed"] = row["physics_seed"]
        path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture(scope="module")
def completed_inventory(tmp_path_factory):
    base = tmp_path_factory.mktemp("reserved-models")
    context = fixture(base)
    for run in context["plan"]["runs"]:
        current = dict(context, run=run, run_root=context["execution_root"] / run["run_id"])
        assert runner.primary.Controller(current, SeedBackend()).run()["status"] == "complete"
    previous = artifact(ROOT / "docs/motion2scene/TRAVERSAL_PROTOCOL_AMENDMENT_V3_PROPOSED.json")
    protocol = runner.propose_v4(previous, context["plan_ref"])
    models = []
    policies = {p["policy_id"]: p for p in protocol["policies"]}
    for slot in runner.model_slots(context["plan"]):
        item = dict(
            slot,
            completion=artifact(slot["expected_completion_path"]),
            checkpoint_receipt=artifact(slot["expected_checkpoint_receipt_path"]),
            model=artifact(slot["expected_model_path"]),
            training_result=artifact(slot["expected_training_result_path"]),
        )
        models.append(item)
        policies[slot["policy_id"]]["model"] = item["model"]
    validation = write_new(base / "baseline_validation.json", {"fixture_validation": True})
    inventory = dict(
        schema=runner.INVENTORY_SCHEMA,
        status="FROZEN_COMPLETE",
        primary_plan=context["plan_ref"],
        primary_adoption=context["adoption_ref"],
        models=models,
        baselines=[
            dict(policy=p, development_validation=validation)
            for p in protocol["policies"]
            if p["policy_id"] in runner.BASELINES
        ],
    )
    return context, protocol, inventory


def stub_native_models(monkeypatch):
    monkeypatch.setattr(runner.collector, "load_verified_registry", lambda *a: SimpleNamespace())
    monkeypatch.setattr(runner.collector, "load_schedule_policy", lambda *a: {})


def test_actual_M2_and_M4_causal_slots_accept_equal_model_bytes(completed_inventory, monkeypatch):
    _, protocol, inventory = completed_inventory
    stub_native_models(monkeypatch)
    result = runner.validate_inventory(copy.deepcopy(inventory), protocol)
    assert result["corpus_runs"] == 12 and result["checkpoint_policies"] == 24
    assert all(
        r["measured_steps_by_round"][2] == 27416 and r["measured_steps_by_round"][4] == 46488
        for r in result["acquisition_counts"].values()
    )
    # All synthetic M2 models have identical bytes but distinct recorded origins.
    assert len({r["model"]["sha256"] for r in inventory["models"] if r["checkpoint"] == "M2"}) == 1


@pytest.mark.parametrize(
    "change",
    [
        "M2_as_M4",
        "wrong_source_run",
        "missing_baseline",
        "duplicated_checkpoint",
        "missing_completion",
    ],
)
def test_checkpoint_substitution_cannot_fill_missing_inventory(
    completed_inventory, monkeypatch, change
):
    _, protocol, original = completed_inventory
    inventory = copy.deepcopy(original)
    stub_native_models(monkeypatch)
    if change == "M2_as_M4":
        inventory["models"][0]["model"] = inventory["models"][1]["model"]
    elif change == "wrong_source_run":
        inventory["models"][0]["completion"] = inventory["models"][2]["completion"]
    elif change == "missing_baseline":
        inventory["baselines"].pop()
    elif change == "duplicated_checkpoint":
        inventory["models"][0] = inventory["models"][1]
    else:
        inventory["models"][0]["completion"] = None
    with pytest.raises((ValueError, TypeError)):
        runner.validate_inventory(inventory, protocol)


def test_reporting_retains_unknown_failure_and_not_run_denominators():
    table = [dict(index=i, policy_id="one", layout_id=f"l{i}") for i in range(4)]

    def outcome(status, steps):
        return dict(
            row=dict(
                task_outcome_admitted=status != "unknown",
                measurement_admitted=False,
                outcome=dict(task_outcome=status),
                physics_steps=steps,
                costs=dict(passage_time_s=2.4),
            )
        )

    report = runner.summarize(
        table, {0: outcome("pass", 1192), 1: outcome("failure", 400), 2: outcome("unknown", 80)}
    )
    summary = report["policies"]["one"]
    assert (
        summary["assigned"] == 4 and summary["passed"] == 1 and summary["verified_task_failed"] == 1
    )
    assert summary["technical_missing"] == summary["not_run"] == 1
    assert summary["completion_lower"] == 0.25 and summary["completion_upper"] == 0.75
    assert summary["measured_only_pass_fraction"] == 0.5
    assert (
        report["actual_recorded_physics_steps"] == 1672 and report["unknown_reserved_steps"] == 1112
    )
    assert [row["successful_passage_time_s"] for row in report["rows"]] == [2.4, None, None, None]
    assert report["complete"] is False


def test_known_failure_with_unavailable_supervision_still_completes_assignment():
    row = dict(
        task_outcome_admitted=True,
        measurement_admitted=False,
        outcome=dict(task_outcome="failure"),
        physics_steps=400,
        costs=dict(passage_time_s=None),
    )
    result = runner.summarize([dict(index=0, policy_id="x")], {0: dict(row=row)})
    assert result["complete"] and result["policies"]["x"]["verified_task_failed"] == 1
    assert result["unknown_reserved_steps"] == 0


def test_named_initial_approach_failure_is_not_reported_as_collision_or_fall():
    row = dict(
        task_outcome_admitted=True,
        measurement_admitted=True,
        physics_steps=1192,
        outcome=dict(
            task_outcome="failure",
            classification="verified_task_failure",
            physical_events=[dict(kind="invalid_initial_approach")],
        ),
        costs=dict(passage_time_s=None),
    )
    result = runner.summarize([dict(index=0, policy_id="x")], {0: dict(row=row)})
    assert result["policies"]["x"]["verified_task_failed"] == 1
    assert result["policies"]["x"]["failure_categories"] == {"invalid_initial_approach": 1}
    assert result["rows"][0]["recorded_classification"] == "verified_task_failure"
    assert result["policies"]["x"]["measured_only_denominator"] == 1


def evaluation_fixture(proposed, tmp_path, monkeypatch):
    """Complete972-slot registration, synthetic files only, no protocol adoption."""
    context, protocol = proposed
    output = tmp_path / "eval_fixture"
    table = runner.assignments(protocol, output)
    protocol_ref = write_new(tmp_path / "fixture_protocol.json", protocol)
    source = context["learner"]["training_implementation"]
    execution = dict(
        shared_lock_path=str(context["execution_root"] / ".primary-acquisition.lock"),
        environment=context["runtime_ref"],
        runtime_assets_declaration=context["runtime_ref"],
        implementation=source,
        batch_implementation=source,
    )
    children = []
    for row in table:
        folder = Path(row["collection_directory"])
        cell = dict(
            cell_id="learned_neutral",
            output=str(folder / "rollouts" / "learned_neutral"),
            policy_id=row["policy_id"],
            runtime_seed=row["physics_seed"],
            timed_schedule_mode="learned",
            forced_option_id="neutral",
        )
        cell["command"] = ["fixture_only", cell["output"], "student"]
        manifest = dict(
            cells=[cell],
            fixture_scene=dict(scene_id=row["scene_id"]),
            limits=dict(minimum_free_gpu_mib=7500, timeout_s=375),
        )
        children.append(
            dict(index=row["index"], manifest=write_new(folder / "manifest.json", manifest))
        )
    spec = write_new(
        output / "fixture_batch_spec.json", dict(collections=[dict(index=i) for i in range(972)])
    )
    plan = write_new(
        output / "fixture_batch_plan.json",
        dict(
            specification=spec,
            expected_episodes=972,
            planned_maximum_physics_steps=1158624,
            budget_physics_steps=1158624,
            implementation=source,
        ),
    )
    registration = write_new(
        output / "batch" / "registration.json", dict(plan=plan, children=children)
    )
    assignment_ref = write_new(output / "assignments.json", table)
    write_new(
        output / "registration.json",
        dict(protocol=protocol_ref, assignments=assignment_ref, batch_registration=registration),
    )

    def verify(folder):
        manifest = read_bound(artifact(folder / "manifest.json"))
        return manifest, None, manifest["fixture_scene"]

    monkeypatch.setattr(runner.collector, "verify_manifest", verify)
    return dict(
        actual_context={},
        protocol=protocol,
        execution=execution,
        learner=context["learner"],
        output=output,
        protocol_ref=protocol_ref,
        table=table,
        acquisition_counts={
            run["run_id"]: dict(measured_steps_by_round={2: 27100, 4: 46300})
            for run in context["plan"]["runs"]
        },
    )


def test_serial_evaluation_pause_and_resume_keeps972_denominator(proposed, tmp_path, monkeypatch):
    evaluation = evaluation_fixture(proposed, tmp_path, monkeypatch)
    backend = FakeBackend()
    first = runner.EvaluationController(evaluation, backend).run(max_new_episodes=1)
    assert first["status"] == "paused" and len(backend.launches) == 1
    assert first["report"]["assigned_episodes"] == 972
    assert sum(row["status"] == "not_run" for row in first["report"]["rows"]) == 971
    again = runner.EvaluationController(evaluation, backend).run(max_new_episodes=1)
    assert again["status"] == "paused" and len(backend.launches) == 2
    assert again["ledger"]["actual_recorded_physics_steps"] == 2384
    assert again["ledger"]["reserved_future_assigned_steps"] == 970 * 1192
    learned = next(row for row in again["report"]["rows"] if row["policy_mode"] == "learned")
    assert (
        learned["acquisition_actual_physics_steps"]
        == {"M2": 27100, "M4": 46300}[learned["checkpoint"]]
    )
    baseline = next(row for row in again["report"]["rows"] if row["policy_mode"] == "always_walk")
    assert baseline["checkpoint"] is None and baseline["acquisition_seed"] is None
    assert baseline["acquisition_actual_physics_steps"] is None


def test_unresolved_evaluation_attempt_is_not_repeated(proposed, tmp_path, monkeypatch):
    evaluation = evaluation_fixture(proposed, tmp_path, monkeypatch)
    backend = FakeBackend()
    backend.crash_on_launch = True
    first = runner.EvaluationController(evaluation, backend).run()
    assert first["status"] == "paused" and len(backend.launches) == 1
    assert first["report"]["rows"][0]["status"] == "technical_missing"
    assert first["ledger"]["unknown_attempt_reserved_steps"] == 1192
    backend.crash_on_launch = False
    again = runner.EvaluationController(evaluation, backend).run()
    assert again["status"] == "paused" and len(backend.launches) == 1


def test_late_child_tampering_blocks_first_evaluation_launch(proposed, tmp_path, monkeypatch):
    evaluation = evaluation_fixture(proposed, tmp_path, monkeypatch)
    path = Path(evaluation["table"][-1]["collection_directory"]) / "manifest.json"
    path.write_text("tampered_last_child")
    backend = FakeBackend()
    with pytest.raises(ValueError):
        runner.EvaluationController(evaluation, backend).run()
    assert not backend.launches
