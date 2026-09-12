"""Bootstrap reuse provenance and single-count accounting with synthetic receipts."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

import motion2scene_bootstrap_reuse as reuse  # noqa: E402
import motion2scene_run_primary_acquisition as controller_module  # noqa: E402


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return artifact(path)


@pytest.fixture
def bundle(tmp_path):
    run_id = "seed93203_analytic_contrast"
    root = tmp_path / "old" / run_id
    teacher = root / "bootstrap/teachers"
    options = ["neutral"] + [f"option{i}" for i in range(6)]
    registry = write(tmp_path / "registry.json", {})
    scene = write(tmp_path / "scene.json", {})
    rounds = [
        dict(
            round_index=0,
            physics_seed=93203,
            scene_definition=scene,
            teacher_branch_order=options,
            teacher_directory=str(teacher),
            expected_teacher_result_path=str(teacher / "result.json"),
            expected_postupdate_training_result_path=str(root / "models/model_000/result.json"),
        )
    ] + [
        dict(
            teacher_directory=str(root / f"round_{k}/teachers"),
            student_directory=str(root / f"round_{k}/student"),
        )
        for k in range(1, 5)
    ]
    run = dict(run_id=run_id, physics_seed=93203, arm="analytic_contrast", rounds=rounds)
    source_plan = dict(
        intended_execution_root=str(tmp_path / "old"),
        runs=[run],
        option_ids=options,
        registry=registry,
    )
    source_plan_ref = write(tmp_path / "source_plan.json", source_plan)
    source_adoption_ref = write(
        tmp_path / "source_adoption.json", dict(status="adopted", plan=source_plan_ref)
    )
    registration_ref = write(
        root / "controller/registration.json",
        dict(run_id=run_id, plan=source_plan_ref, adoption=source_adoption_ref),
    )
    fit_ref = write(
        root / "models/model_000/result.json",
        dict(status="fit_validation_failed", policy=None, audit_errors=[]),
    )
    cells = [
        dict(
            cell_id="forced_" + option,
            forced_option_id=option,
            command=["frozen", option],
            runtime_seed=93203,
            timed_schedule_mode="forced",
            output=str(teacher / "rollouts" / ("forced_" + option)),
        )
        for option in options
    ]
    manifest_ref = write(
        teacher / "manifest.json",
        dict(
            policy=None,
            script_parameters=None,
            registry=registry,
            scene_definition=scene,
            cells=cells,
        ),
    )
    rows, attempts = [], []
    for cell in cells:
        output = Path(cell["output"])
        attempt_ref = write(output / "attempt.json", dict(exit_status=0))
        capture_ref = write(output / "trajectory.json", dict(unique=cell["cell_id"]))
        row = dict(
            cell_id=cell["cell_id"],
            measurement_admitted=True,
            task_outcome_admitted=True,
            physics_steps=1192,
            outcome=dict(task_outcome="pass"),
            trajectory=capture_ref,
        )
        folder = root / "controller/attempts" / ("round000_teacher_" + cell["cell_id"])
        intent_ref = write(
            folder / "intent.json",
            dict(
                attempt_id=folder.name,
                manifest=manifest_ref,
                command=cell["command"],
                attempt_path=attempt_ref["path"],
            ),
        )
        assessment_ref = write(folder / "assessment.json", dict(row=row, attempt=attempt_ref))
        rows.append(row)
        attempts.append(dict(intent=intent_ref, assessment=assessment_ref))
    collection_ref = write(
        teacher / "result.json",
        dict(manifest=manifest_ref, rows=rows, physics_steps=8344),
    )
    declaration = dict(
        schema="motion2scene_same_corpus_bootstrap_reuse_v1",
        run_id=run_id,
        recorded_physics_steps=8344,
        source_plan=source_plan_ref,
        source_adoption=source_adoption_ref,
        source_controller_registration=registration_ref,
        failed_fit=fit_ref,
        collection=collection_ref,
        manifest=manifest_ref,
        attempts=attempts,
    )
    declaration_path = tmp_path / "reuse.json"
    declaration_ref = write(declaration_path, declaration)
    plan = copy.deepcopy(source_plan)
    plan["intended_execution_root"] = str(tmp_path / "new")
    plan["bootstrap_reuse"] = {run_id: declaration_ref}
    adoption = dict(bootstrap_reuse=copy.deepcopy(plan["bootstrap_reuse"]))
    return types.SimpleNamespace(
        plan=plan,
        adoption=adoption,
        run=plan["runs"][0],
        declaration=declaration,
        declaration_path=declaration_path,
        root=root,
        tmp=tmp_path,
    )


def refresh_declaration(b):
    ref = write(b.declaration_path, b.declaration)
    b.plan["bootstrap_reuse"] = {b.run["run_id"]: ref}
    b.adoption["bootstrap_reuse"] = copy.deepcopy(b.plan["bootstrap_reuse"])


def validate(b):
    return reuse.validate_bootstrap_reuse(b.plan, b.adoption, b.run)


def test_exact_complete_same_corpus_bootstrap_passes_without_output_writes(bundle):
    result = validate(bundle)
    assert len(result["intents"]) == 7 and result["recorded_physics_steps"] == 8344
    assert not Path(bundle.plan["intended_execution_root"]).exists()


@pytest.mark.parametrize("change", ["seed", "arm", "scene", "order", "path", "adoption"])
def test_changed_scientific_assignment_or_unadopted_reuse_is_rejected(bundle, change):
    if change == "seed":
        bundle.run["physics_seed"] += 1
    elif change == "arm":
        bundle.run["arm"] = "uniform"
    elif change == "scene":
        bundle.run["rounds"][0]["scene_definition"] = write(bundle.tmp / "other_scene.json", {})
    elif change == "order":
        bundle.run["rounds"][0]["teacher_branch_order"].reverse()
    elif change == "path":
        bundle.run["rounds"][0]["teacher_directory"] = str(bundle.tmp / "other")
    else:
        bundle.adoption["bootstrap_reuse"] = {}
    with pytest.raises(ValueError):
        validate(bundle)


@pytest.mark.parametrize(
    "change",
    ["later_teacher", "later_student", "old_policy", "extra_intent", "missing_branch"],
)
def test_reuse_does_not_hide_predecessor_work(bundle, change):
    if change in ("later_teacher", "later_student"):
        key = "teacher_directory" if change == "later_teacher" else "student_directory"
        Path(bundle.run["rounds"][1][key]).mkdir(parents=True)
    elif change == "old_policy":
        (bundle.root / "models/model_000/policy.npz").write_bytes(b"not inherited")
    elif change == "extra_intent":
        write(bundle.root / "controller/attempts/unaccounted/intent.json", {})
    else:
        bundle.declaration["attempts"].pop()
        refresh_declaration(bundle)
    with pytest.raises(ValueError):
        validate(bundle)


@pytest.mark.parametrize("change", ["partial", "unknown", "duplicate_capture", "wrong_command"])
def test_rehashed_but_invalid_physical_evidence_is_rejected(bundle, change):
    declaration = bundle.declaration
    result = reuse.read(declaration["collection"])
    item = declaration["attempts"][0]
    assessment = reuse.read(item["assessment"])
    if change == "wrong_command":
        intent = reuse.read(item["intent"])
        intent["command"] = ["different"]
        item["intent"] = write(Path(item["intent"]["path"]), intent)
    else:
        if change == "partial":
            assessment["row"]["measurement_admitted"] = False
        elif change == "unknown":
            assessment["row"]["outcome"]["task_outcome"] = "unknown"
        else:
            assessment["row"]["trajectory"] = result["rows"][1]["trajectory"]
        result["rows"][0] = assessment["row"]
        item["assessment"] = write(Path(item["assessment"]["path"]), assessment)
        declaration["collection"] = write(Path(declaration["collection"]["path"]), result)
    refresh_declaration(bundle)
    with pytest.raises(ValueError):
        validate(bundle)


@pytest.fixture(scope="module")
def primary_controller():
    return controller_module


def test_reused_collection_never_launches_and_counts_steps_once(bundle, primary_controller):
    reused = validate(bundle)
    context = dict(
        run_root=bundle.tmp / "new" / bundle.run["run_id"],
        run=bundle.run,
        plan=dict(budget=dict(budget_physics_steps=50000)),
        reused_bootstrap=reused,
    )

    class Backend:
        def reuse_bootstrap(self, supplied):
            assert supplied is context
            return reused["collection"]

        def launch(self, *args):
            raise AssertionError("reused physics may not be launched")

        def prepare(self, *args):
            raise AssertionError("reused collection may not be rewritten")

    controller = primary_controller.Controller(context, Backend())
    assert controller.collect(bundle.run["rounds"][0], "teacher") == reused["collection"]
    assert controller.collect(bundle.run["rounds"][0], "teacher") == reused["collection"]
    accounting = controller.accounting()
    assert accounting["reused_bootstrap_recorded_steps"] == 8344
    assert len(accounting["attempts"]) == 7
    assert sum(a["actual_physics_steps"] for a in accounting["attempts"]) == 8344
    assert accounting["unlaunched_assigned_episode_slots"] == 32
    assert all(a["origin"] == "reused_predecessor_bootstrap" for a in accounting["attempts"])
    assert not (context["run_root"] / "bootstrap").exists()
    with pytest.raises(ValueError):
        controller.collect(bundle.run["rounds"][0], "student")


def test_duplicate_reused_and_new_attempt_is_rejected(bundle, primary_controller):
    reused = validate(bundle)
    context = dict(
        run_root=bundle.tmp / "new" / bundle.run["run_id"],
        run=bundle.run,
        plan=dict(budget=dict(budget_physics_steps=50000)),
        reused_bootstrap=reused,
    )
    controller = primary_controller.Controller(context)
    value = json.loads(reused["intents"][0].read_text())
    write(controller.folder / "attempts/duplicate/intent.json", value)
    with pytest.raises(ValueError, match="duplicate"):
        controller.accounting()


@pytest.mark.parametrize("field", ["intent", "assessment", "collection", "declaration"])
def test_post_context_edit_cannot_change_inherited_ledger(bundle, primary_controller, field):
    reused = validate(bundle)
    context = dict(
        run_root=bundle.tmp / "new" / bundle.run["run_id"],
        run=bundle.run,
        plan=dict(budget=dict(budget_physics_steps=50000)),
        reused_bootstrap=reused,
    )
    if field in ("intent", "assessment"):
        ref = reused["attempt_artifacts"][0][field]
    else:
        ref = reused[field]
    Path(ref["path"]).write_text("{}")
    with pytest.raises(ValueError, match="frozen identity"):
        primary_controller.Controller(context).accounting()
