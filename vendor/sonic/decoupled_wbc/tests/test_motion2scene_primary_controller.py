"""Actual causal plan APIs with a file-backed CPU-only collection/fit backend."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

from motion2scene_register_primary_acquisition import run_slots  # noqa: E402
import motion2scene_run_primary_acquisition as runner  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    ADOPTION_SCHEMA,
    ARMS,
    PLAN_SCHEMA,
    SEEDS,
    artifact,
    budget_contract,
    read_bound,
    validate_completed_replay_order,
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_curriculum import (  # noqa: E402
    DEFAULT_RULE,
)

OPTIONS = ["neutral", "short15", "long15", "prior15", "prior50", "short70", "long70"]


def fixture(tmp_path, arm="uniform"):
    registry = write_new(tmp_path / "registry.json", {"fixture": True})
    source = write_new(tmp_path / "trainer.py", {"fixture_source": True})
    scene = write_new(tmp_path / "scene.json", {"fixture_scene": True})
    runs = []
    for seed in SEEDS:
        for name in ARMS:
            selections = [
                dict(
                    candidate_id=f"candidate_{seed}_{index}",
                    definition=scene,
                    stratum=f"stratum{index}",
                    future_branch_order=OPTIONS,
                )
                for index in range(4)
            ]
            run_id = f"seed{seed}_{name}"
            runs.append(
                dict(
                    run_id=run_id,
                    arm=name,
                    physics_seed=seed,
                    rounds=run_slots(
                        tmp_path / "execution", run_id, seed, selections, OPTIONS, scene
                    ),
                )
            )
    plan = dict(
        schema=PLAN_SCHEMA,
        adoption_status="proposed_not_adopted",
        execution_authorized=False,
        budget=budget_contract(),
        option_ids=OPTIONS,
        registry=registry,
        intended_execution_root=str(tmp_path / "execution"),
        runs=runs,
    )
    plan_ref = write_new(tmp_path / "plan.json", plan)
    learner = dict(
        status="frozen",
        feature_dimension=114,
        option_ids=OPTIONS,
        phase_ticks=[15, 50, 70],
        l2=0.1,
        training_implementation=[source],
    )
    learner_ref = write_new(tmp_path / "learner.json", learner)
    rule_ref = write_new(tmp_path / "rule.json", copy.deepcopy(DEFAULT_RULE))
    runtime = dict(replay_rule=rule_ref)
    runtime_ref = write_new(tmp_path / "runtime.json", runtime)
    adoption_ref = write_new(
        tmp_path / "adoption.json",
        dict(
            schema=ADOPTION_SCHEMA,
            status="adopted",
            plan=plan_ref,
            common_learner=learner_ref,
            runtime_freeze=runtime_ref,
        ),
    )
    selected = next(row for row in runs if row["run_id"] == f"seed{SEEDS[0]}_{arm}")
    return dict(
        plan=plan,
        plan_ref=plan_ref,
        adoption_ref=adoption_ref,
        learner=learner,
        runtime=runtime,
        runtime_ref=runtime_ref,
        run=selected,
        execution_root=tmp_path / "execution",
        run_root=tmp_path / "execution" / selected["run_id"],
    )


class FakeBackend:
    """Produces distinctly identified synthetic files, never simulated evidence."""

    def __init__(self):
        self.launches, self.fits, self.preflights, self.replays = [], [], [], []
        self.free = 9000
        self.crash_on_launch = False
        self.interrupt_audit = False
        self.student_outcome = "pass"

    def prepare(self, context, row, kind, model):
        out = Path(row[f"{kind}_directory"])
        assert not out.exists()
        if row["round_index"]:
            assert Path(row["expected_preupdate_binding_path"]).exists()
            if kind == "teacher":
                assert Path(row["expected_teacher_release_path"]).exists()
                assert Path(row["expected_student_result_path"]).exists()
            else:
                assert not Path(row["teacher_directory"]).exists()
        ids = row["teacher_branch_order"] if kind == "teacher" else ["neutral"]
        cells = []
        for option in ids:
            cell_id = ("forced_" if kind == "teacher" else "learned_") + option
            folder = out / "rollouts" / cell_id
            cells.append(
                dict(
                    cell_id=cell_id,
                    output=str(folder),
                    command=["fixture_only", str(folder), kind],
                    forced_option_id=option,
                    timed_schedule_mode="forced" if kind == "teacher" else "learned",
                )
            )
        write_new(
            out / "manifest.json",
            dict(
                schema=runner.collector.COLLECTION_SCHEMA,
                registry=context["plan"]["registry"],
                scene_definition=row["scene_definition"],
                policy=model,
                cells=cells,
                limits=dict(minimum_free_gpu_mib=7500, timeout_s=375),
            ),
        )

    def collection(self, context, row, kind, model):
        return runner.read(Path(row[f"{kind}_directory"]) / "manifest.json"), None, None

    def preflight(self, context, manifest):
        self.preflights.append(manifest)
        return dict(status="fixture_preflight", new_physics_steps=0)

    def free_gpu_mib(self):
        return self.free

    def launch(self, command, timeout):
        assert command not in self.launches, "a physical attempt was duplicated"
        self.launches.append(command)
        if self.crash_on_launch:
            raise KeyboardInterrupt("fixture interrupted dispatch")
        write_new(
            Path(command[1]) / "capture.json", dict(fixture_only=True, unique_path=command[1])
        )
        return 7 if command[2] == "student" and self.student_outcome == "failure" else 0

    def audit(self, cell, manifest, bank, scene):
        if self.interrupt_audit:
            self.interrupt_audit = False
            raise KeyboardInterrupt("fixture interruption after persisted attempt")
        attempt = artifact(Path(cell["output"]) / "attempt.json")
        student = cell["timed_schedule_mode"] == "learned"
        outcome = self.student_outcome if student else "pass"
        return dict(
            cell_id=cell["cell_id"],
            mode=cell["timed_schedule_mode"],
            task_outcome_admitted=outcome != "unknown",
            outcome=dict(task_outcome=outcome, exit_status=read_bound(attempt)["exit_status"]),
            physics_steps=None if outcome == "unknown" else (400 if outcome == "failure" else 1192),
            attempt=attempt,
            trajectory=artifact(Path(cell["output"]) / "capture.json"),
        )

    def analyze(self, out):
        manifest = runner.read(out / "manifest.json")
        rows = [self.audit(cell, manifest, None, None) for cell in manifest["cells"]]
        write_new(
            out / "result.json",
            dict(
                schema=runner.collector.COLLECTION_SCHEMA,
                manifest=artifact(out / "manifest.json"),
                rows=rows,
                physics_steps=sum(row["physics_steps"] or 0 for row in rows),
                unmeasured_failed_attempts=sum(row["physics_steps"] is None for row in rows),
            ),
        )

    def fit(self, context, collections, out, weights):
        assert out not in [entry[0] for entry in self.fits]
        self.fits.append((out, list(collections), weights))
        registration = write_new(
            out / "registration.json",
            dict(
                registry=context["plan"]["registry"],
                collections=collections,
                l2=context["learner"]["l2"],
                replay_weights=weights,
                implementation=context["learner"]["training_implementation"],
            ),
        )
        teachers = write_new(
            out / "teachers.json", [dict(fixture_group=ref) for ref in collections]
        )
        policy = write_new(
            out / "policy.npz", dict(fixture_model=True, label_groups=len(collections))
        )
        write_new(
            out / "result.json",
            dict(status="complete", policy=policy, teachers=teachers, registration=registration),
        )

    def build_replay(self, context, teachers, students, out, receipt):
        verified = validate_completed_replay_order(context["plan"], read_bound(receipt))
        assert len(verified["teacher_collections"]) == len(students) + 1
        self.replays.append(verified)
        registration = write_new(
            out / "registration.json",
            dict(
                teachers=teachers,
                student_results=students,
                acquisition_receipt=receipt,
                rule=read_bound(context["runtime"]["replay_rule"]),
            ),
        )
        evidence = write_new(out / "evidence.json", dict(registration=registration))
        weights = write_new(out / "weights.json", dict(evidence=evidence, fixture_weights=True))
        return dict(evidence=evidence, weights=weights, verification={"fixture_only": True})


def test_full_serial_order_and_resume_do_not_repeat_attempts(tmp_path):
    context, backend = fixture(tmp_path), FakeBackend()
    first = runner.Controller(context, backend).run(stop_after_round=1)
    assert first["status"] == "paused_at_round_boundary"
    assert len(backend.launches) == 15
    assert len(backend.fits) == 2
    result = runner.Controller(context, backend).run()
    assert result["status"] == "complete"
    assert len(backend.launches) == len(backend.preflights) == 39
    assert len(backend.fits) == 5
    assert result["ledger"]["actual_recorded_physics_steps"] == 46488
    assert result["ledger"]["budget_remaining_after_reservations"] == 3512
    assert result["ledger"]["distinct_original_recorded_captures"] == 39
    receipt = read_bound(result["completion"])
    assert receipt["final_model"] == artifact(
        context["run"]["rounds"][4]["expected_postupdate_policy_path"]
    )
    assert validate_completed_replay_order(context["plan"], receipt)["completed_through_round"] == 4
    again = runner.Controller(context, backend).run()
    assert again["completion"] == result["completion"]
    assert len(backend.launches) == 39


def test_observation_rounds_audit_before_replay_and_weighted_fit(tmp_path):
    context, backend = fixture(tmp_path, "observation_curriculum"), FakeBackend()
    result = runner.Controller(context, backend).run()
    assert result["status"] == "complete"
    assert len(backend.launches) == 39
    assert len(backend.fits) == 9  # five common fits plus four explicit CPU audit fits.
    assert result["ledger"]["cpu_fit_attempts_started"] == 9
    assert result["ledger"]["auxiliary_cpu_fit_attempts"] == 4
    assert result["ledger"]["completed_replay_receipts"] == 4
    assert len(backend.replays) == 4
    for index in range(1, 5):
        audit, weighted = backend.fits[2 * index - 1 : 2 * index + 1]
        assert audit[0].name == "unweighted_audit" and audit[2] is None
        assert weighted[2] is not None and len(weighted[1]) == index + 1
        assert [
            gap["gap_age_rounds"] for gap in backend.replays[index - 1]["historical_gaps"]
        ] == list(range(index - 1, -1, -1))
    runner.Controller(context, backend).run()
    assert len(backend.fits) == 9 and len(backend.replays) == 4


def test_resource_pause_is_zero_physics_and_can_resume(tmp_path):
    context, backend = fixture(tmp_path), FakeBackend()
    backend.free = 0
    result = runner.Controller(context, backend).run()
    assert result["status"] == "paused"
    assert not backend.launches
    assert result["ledger"]["actual_recorded_physics_steps"] == 0
    assert result["ledger"]["unknown_attempt_reserved_steps"] == 0
    assert result["ledger"]["resource_only_pauses"] == 1
    assert not list((context["run_root"] / "controller" / "attempts").glob("*/intent.json"))
    backend.free = 9000
    runner.Controller(context, backend).run(stop_after_round=0)
    assert len(backend.launches) == 7


def test_unresolved_launch_is_reserved_and_never_retried(tmp_path):
    context, backend = fixture(tmp_path), FakeBackend()
    backend.crash_on_launch = True
    result = runner.Controller(context, backend).run()
    assert result["status"] == "paused"
    assert result["ledger"]["unknown_attempt_reserved_steps"] == 1192
    assert result["ledger"]["actual_recorded_physics_steps"] == 0
    backend.crash_on_launch = False
    again = runner.Controller(context, backend).run()
    assert again["status"] == "paused"
    assert len(backend.launches) == 1


def test_interruption_after_actual_attempt_resumes_without_duplicate(tmp_path):
    context, backend = fixture(tmp_path), FakeBackend()
    backend.interrupt_audit = True
    with pytest.raises(KeyboardInterrupt):
        runner.Controller(context, backend).run(stop_after_round=0)
    assert len(backend.launches) == 1
    result = runner.Controller(context, backend).run(stop_after_round=0)
    assert result["status"] == "paused_at_round_boundary"
    assert len(backend.launches) == 7
    assert result["ledger"]["actual_recorded_physics_steps"] == 8344


def test_known_early_failure_releases_teachers_unknown_does_not(tmp_path):
    context, backend = fixture(tmp_path / "known"), FakeBackend()
    backend.student_outcome = "failure"
    result = runner.Controller(context, backend).run(stop_after_round=1)
    assert result["status"] == "paused_at_round_boundary"
    assert len(backend.launches) == 15
    assert result["ledger"]["actual_recorded_physics_steps"] == 14 * 1192 + 400
    context, backend = fixture(tmp_path / "unknown"), FakeBackend()
    backend.student_outcome = "unknown"
    result = runner.Controller(context, backend).run(stop_after_round=1)
    assert result["status"] == "paused"
    assert len(backend.launches) == 8
    assert not Path(context["run"]["rounds"][1]["teacher_directory"]).exists()
    assert result["ledger"]["unknown_attempt_reserved_steps"] == 1192


def test_no_resume_through_edited_model_or_attempt(tmp_path):
    context, backend = fixture(tmp_path), FakeBackend()
    runner.Controller(context, backend).run(stop_after_round=0)
    model = Path(context["run"]["rounds"][0]["expected_postupdate_policy_path"])
    model.write_text("edited fixture model")
    with pytest.raises(ValueError):
        runner.Controller(context, backend).run(stop_after_round=1)
    assert len(backend.launches) == 7


def test_interrupted_cpu_fit_is_preserved_and_pauses(tmp_path):
    context, backend = fixture(tmp_path), FakeBackend()
    out = Path(context["run"]["rounds"][0]["expected_postupdate_training_result_path"]).parent
    write_new(out / "partial.json", {"retained": True})
    result = runner.Controller(context, backend).run(stop_after_round=0)
    assert result["status"] == "paused"
    assert not backend.fits
    assert runner.read(out / "partial.json") == {"retained": True}
    runner.Controller(context, backend).run(stop_after_round=0)
    assert len(backend.launches) == 7


def test_common_lock_is_nonblocking_and_covers_all_runs(tmp_path):
    context, backend = fixture(tmp_path), FakeBackend()
    lock = context["execution_root"] / runner.EXECUTION_CONTRACT["lock_relative_path"]
    with runner.acquisition_lock(lock):
        with pytest.raises(runner.Paused, match="common lock"):
            runner.Controller(context, backend).run()
    assert not backend.launches


def test_proposed_adoption_is_rejected_before_execution_directories(tmp_path):
    context = fixture(tmp_path)
    adoption = read_bound(context["adoption_ref"])
    adoption["status"] = "proposed"
    proposed_ref = write_new(tmp_path / "proposed_adoption.json", adoption)
    with pytest.raises(ValueError, match="adopted protocol"):
        runner.load_context(
            Path(context["plan_ref"]["path"]), Path(proposed_ref["path"]), context["run"]["run_id"]
        )
    assert not context["execution_root"].exists()


def test_precreated_current_teacher_cannot_be_grandfathered(tmp_path):
    context, backend = fixture(tmp_path), FakeBackend()
    runner.Controller(context, backend).run(stop_after_round=0)
    row = context["run"]["rounds"][1]
    Path(row["teacher_directory"]).mkdir(parents=True)
    with pytest.raises(ValueError, match="must not exist"):
        runner.Controller(context, backend).run(stop_after_round=1)
    assert len(backend.launches) == 7


def test_unknown_outcome_retains_measured_prefix_and_reserves_remaining_horizon(tmp_path):
    context, backend = fixture(tmp_path), FakeBackend()
    original = backend.audit

    def partial(cell, manifest, bank, scene):
        value = original(cell, manifest, bank, scene)
        value.update(
            task_outcome_admitted=False,
            physics_steps=80,
            outcome=dict(task_outcome="unknown", exit_status=0),
        )
        return value

    backend.audit = partial
    result = runner.Controller(context, backend).run()
    assert result["status"] == "paused"
    assert result["ledger"]["actual_recorded_physics_steps"] == 80
    assert result["ledger"]["unknown_outcome_tail_reserved_steps"] == 1112
    assert result["ledger"]["conservative_charged_steps"] == 1192


def test_actual_invocation_edits_are_rejected_before_any_next_launch(tmp_path):
    context, backend = fixture(tmp_path), FakeBackend()
    runner.Controller(context, backend).run(stop_after_round=0)
    cell = runner.read(Path(context["run"]["rounds"][0]["teacher_directory"]) / "manifest.json")[
        "cells"
    ][0]
    path = Path(cell["output"]) / "attempt.json"
    actual = runner.read(path)
    actual["command"] = ["edited_invocation"]
    path.write_text(json.dumps(actual))
    with pytest.raises(ValueError, match="registered invocation"):
        runner.Controller(context, backend).run(stop_after_round=1)
    assert len(backend.launches) == 7


def test_runtime_declaration_binds_exact_executing_closures(tmp_path, monkeypatch):
    """Real closure and artifact checks; only physical-bank schema is stubbed."""
    context = fixture(tmp_path)
    controller = write_new(tmp_path / "controller.json", {"fixture": True})
    motion = write_new(tmp_path / "motion.json", {"fixture": True})
    request_value = dict(controller=controller, references=[dict(motion=motion)])
    request = write_new(tmp_path / "request.json", request_value)
    template = write_new(
        tmp_path / "template.json",
        dict(
            implementation=dict(checkpoint=controller, python="fixture_python"),
            cells=[dict(cell_id="neutral", motion=motion)],
        ),
    )
    environment = write_new(tmp_path / "environment.json", {"fixture_environment": True})
    assets = write_new(tmp_path / "assets.json", [environment])
    runtime = runner.runtime_description(
        context["plan_ref"],
        Path(request["path"]),
        Path(template["path"]),
        Path(assets["path"]),
        Path(environment["path"]),
        Path(context["runtime"]["replay_rule"]["path"]),
    )
    runtime_ref = write_new(tmp_path / "full_runtime.json", runtime)
    learner = dict(
        context["learner"],
        training_implementation=[
            artifact(path) for path in sorted(runner.closure([Path(runner.trainer.__file__)]))
        ],
    )
    learner_ref = write_new(tmp_path / "full_learner.json", learner)
    adoption = dict(
        read_bound(context["adoption_ref"]), common_learner=learner_ref, runtime_freeze=runtime_ref
    )
    adoption_ref = write_new(tmp_path / "full_adoption.json", adoption)
    monkeypatch.setattr(
        runner.collector,
        "load_verified_registry",
        lambda *args: SimpleNamespace(request=request_value),
    )
    monkeypatch.setattr(runner.collector, "validate_scene", lambda value: value)
    loaded = runner.load_context(
        Path(context["plan_ref"]["path"]), Path(adoption_ref["path"]), context["run"]["run_id"]
    )
    assert loaded["runtime_ref"] == runtime_ref
    assert not context["execution_root"].exists()
    runtime["collection_implementation"] = runtime["collection_implementation"][:-1]
    changed_ref = write_new(tmp_path / "incomplete_runtime.json", runtime)
    changed_adoption = write_new(
        tmp_path / "incomplete_adoption.json", dict(adoption, runtime_freeze=changed_ref)
    )
    with pytest.raises(ValueError, match="source closure differs"):
        runner.load_context(
            Path(context["plan_ref"]["path"]),
            Path(changed_adoption["path"]),
            context["run"]["run_id"],
        )
    assert not context["execution_root"].exists()


def test_preflight_failure_creates_neither_intent_nor_rollout(tmp_path):
    context, backend = fixture(tmp_path), FakeBackend()

    def refused(*args):
        raise ValueError("runtime inventory changed")

    backend.preflight = refused
    with pytest.raises(ValueError, match="runtime inventory"):
        runner.Controller(context, backend).run()
    assert not backend.launches
    assert not list((context["run_root"] / "controller" / "attempts").glob("*/intent.json"))


def test_interruption_between_preflight_and_intent_is_safe_to_resume(tmp_path):
    context, backend = fixture(tmp_path), FakeBackend()
    preflight = (
        context["run_root"]
        / "controller"
        / "attempts"
        / "round000_teacher_forced_neutral"
        / "runtime_preflight.json"
    )
    write_new(preflight, dict(status="fixture_preflight", new_physics_steps=0))
    controller = runner.Controller(context, backend)
    assert controller.run(stop_after_round=0)["status"] == "paused_at_round_boundary"
    assert controller.run(stop_after_round=1)["status"] == "paused_at_round_boundary"
    assert len(backend.launches) == 15
