"""Prespecified common-set curves bind real checkpoints and shared references."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_development_learning_curve as curve
from motion2scene_timing_diagnostic import artifact


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return artifact(path)


def declaration_fixture(tmp_path, monkeypatch):
    options = ["neutral"] + [f"option{i}" for i in range(6)]
    scenes = [
        dict(
            scene_id=name,
            scene_definition=save(
                tmp_path / f"{name}.json", dict(scene_id=name, split="development")
            ),
        )
        for name in sorted(curve.anchor.CONTEXTS)
    ]
    runs = []
    for seed in (93201, 93202, 93203):
        for arm in sorted(curve.anchor.ARMS):
            run_id = f"seed{seed}_{arm}"
            rounds = [
                dict(
                    index=i,
                    scene=scenes[i % 6]["scene_definition"],
                    inherited_model=None,
                    inherited_teacher=None,
                    inherited_student=None,
                    model_directory=str(tmp_path / run_id / f"model_{i:03d}"),
                    teacher_directory=str(tmp_path / run_id / f"teacher_{i:03d}"),
                    student_directory=(
                        None if i == 0 else str(tmp_path / run_id / f"student_{i:03d}")
                    ),
                )
                for i in range(33)
            ]
            runs.append(dict(seed=seed, arm=arm, run_id=run_id, rounds=rounds))
    bank_ref = save(tmp_path / "registry.json", {})
    plan = dict(
        schema="motion2scene_expanded_acquisition_v1",
        registry=bank_ref,
        runs=runs,
        execution_root=str(tmp_path / "acquisition"),
    )
    plan_ref = save(tmp_path / "plan.json", plan)
    monkeypatch.setattr(
        curve, "load_verified_registry", lambda *args: SimpleNamespace(option_ids=options)
    )
    anchor_dir = tmp_path / "M8"
    save(
        anchor_dir / "study.json",
        dict(
            expanded_plan=plan_ref,
            registry=bank_ref,
            assignments=curve.anchor.assignments(runs, scenes, options),
            checkpoint=8,
            assigned_episodes=138,
            implementation=artifact(Path(curve.anchor.__file__)),
        ),
    )
    out = tmp_path / "curve"
    curve.declare(anchor_dir, out)
    return out, plan, options


def test_full_design_has_318_unique_episodes_and_exact_later_model_paths(tmp_path, monkeypatch):
    root, plan, _ = declaration_fixture(tmp_path, monkeypatch)
    _, study, _, _, _ = curve.read_study(root)
    assert study["unique_assigned_episodes"] == 318
    assert study["new_assigned_episodes"] == 180
    assert study["maximum_total_evaluation_physics_steps"] == 379056
    assert study["maximum_new_evaluation_physics_steps"] == 214560
    assert len(study["assignments"]) == 138
    for panel in study["panels"]:
        assert len(panel["models"]) == 15
        assert all(
            f'model_{panel["checkpoint"]:03d}' in r["expected_training_result"]
            for r in panel["models"]
        )
    assert len(curve.acquisition_ready(study, plan)) == 15
    with pytest.raises(FileNotFoundError, match="final M32"):
        curve.prepare(root, 16)
    assert not (root / "M16").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("evaluation_gate_acquisition_checkpoint", 16),
        ("new_assigned_episodes", 90),
        ("checkpoints", [8, 16]),
        ("maximum_new_evaluation_physics_steps", 1),
    ],
)
def test_cannot_reduce_the_gate_or_change_the_assigned_budget(tmp_path, monkeypatch, field, value):
    root, _, _ = declaration_fixture(tmp_path, monkeypatch)
    path = root / "study.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        curve.read_study(root)


def test_missing_model_or_extra_future_teacher_cannot_bind(tmp_path, monkeypatch):
    root, plan, _ = declaration_fixture(tmp_path, monkeypatch)
    models = curve.model_slots(plan, 16)
    with pytest.raises(ValueError, match="exact model slot"):
        curve.bind_models(plan, None, models[:-1], 16)
    with pytest.raises(FileNotFoundError):
        curve.bind_models(plan, None, models, 16)
    model = models[0]
    slots = curve.checkpoint_slots(plan, model["run_id"], 16)
    teachers = [save(s["teacher"], {}) for s in slots]
    registration = save(
        tmp_path / "registration.json",
        dict(collections=teachers + [teachers[-1]], registry=plan["registry"]),
    )
    save(
        Path(model["expected_training_result"]),
        dict(status="complete", audit_errors=[], registration=registration),
    )
    with pytest.raises(ValueError, match="earlier encounter prefix"):
        curve.bind_models(plan, None, models, 16)


def test_final_acquisition_gate_rejects_wrong_checkpoint_receipt(tmp_path, monkeypatch):
    root, plan, _ = declaration_fixture(tmp_path, monkeypatch)
    _, study, _, _, _ = curve.read_study(root)
    run = plan["runs"][0]
    save(
        Path(plan["execution_root"]) / run["run_id"] / "controller/complete_032.json",
        dict(expanded_plan={"path": "wrong", "sha256": "wrong"}),
    )
    with pytest.raises(ValueError, match="different trajectory"):
        curve.acquisition_ready(study, plan)


def outcome(path, status="pass", steps=1192):
    return dict(
        collection=dict(path=path, sha256=path),
        status=status,
        time_s=3.0 if status == "pass" else None,
        recorded_physics_steps=steps,
    )


def test_shared_baselines_are_not_new_episodes_and_partial_cost_is_retained():
    baseline = outcome("script")
    panels = [
        dict(rows=[baseline, outcome("M8")]),
        dict(rows=[baseline, outcome("M16", "failure", 756)]),
        dict(
            rows=[
                baseline,
                outcome("M32", "technical_missing", 80),
                outcome("pending", "not_run", None),
            ]
        ),
    ]
    result = curve.evaluation_accounting(panels)
    assert result["unique_assigned_episodes"] == 5
    assert result["analysis_rows"] == 7
    assert result["measured_physics_steps"] == 1192 * 2 + 756 + 80
    assert result["unknown_outcomes"] == result["unexecuted"] == 1
    assert result["unknown_physics_step_counts"] == 1


def test_inconsistent_reused_capture_is_rejected():
    with pytest.raises(ValueError, match="inconsistent"):
        curve.evaluation_accounting(
            [dict(rows=[outcome("same")]), dict(rows=[outcome("same", "failure", 756)])]
        )


def test_unknown_outcome_is_retained_and_cannot_advance_runner(tmp_path):
    manifest = save(tmp_path / "manifest.json", {})
    ref = save(
        tmp_path / "result.json",
        dict(
            manifest=manifest,
            rows=[
                dict(
                    **{"pass": False},
                    measurement_admitted=False,
                    costs=dict(passage_time_s=None),
                    outcome=dict(task_outcome="unknown", physics_steps_recorded=80),
                )
            ],
        ),
    )
    with pytest.raises(RuntimeError, match="unknown outcome retained"):
        curve.known_result(ref, manifest)
    assert (
        json.loads(Path(ref["path"]).read_text())["rows"][0]["outcome"]["physics_steps_recorded"]
        == 80
    )


@pytest.mark.parametrize(
    "row",
    [
        outcome("x", "failure") | {"time_s": 3.0},
        outcome("x") | {"time_s": float("nan")},
        outcome("x") | {"recorded_physics_steps": -1},
    ],
)
def test_measurement_definitions_are_not_changed(row):
    with pytest.raises(ValueError, match="measurement definitions"):
        curve.validate_measurement(row)


def test_comparison_contract_ignores_snapshot_locations_but_not_sensor_changes():
    keys = (
        "registry",
        "request",
        "template",
        "feature_names",
        "feature_schema",
        "sensor",
        "option_ids",
        "phase_ticks",
        "implementation",
        "runtime_artifacts",
        "scoring_artifacts",
        "runtime_assets_declaration",
        "scoring",
        "limits",
        "expected_physics_steps",
        "expected_recorded_control_steps",
        "qualified_environment_audits",
    )
    a = dict.fromkeys(keys, "same")
    a["dependencies"] = [dict(path="source", sha256="hash", snapshot=dict(path="capture_a"))]
    b = copy.deepcopy(a)
    b["dependencies"][0]["snapshot"]["path"] = "capture_b"
    assert curve.comparison_contract(a) == curve.comparison_contract(b)
    b["sensor"] = "changed"
    assert curve.comparison_contract(a) != curve.comparison_contract(b)


def fitted_prefix_fixture(tmp_path, monkeypatch):
    root, plan, _ = declaration_fixture(tmp_path, monkeypatch)
    _, study, _, _, bank = curve.read_study(root)
    models = curve.model_slots(plan, 16)
    for model in models:
        slots = curve.checkpoint_slots(plan, model["run_id"], 16)
        teachers = [save(s["teacher"], {}) for s in slots]
        directory = Path(model["expected_training_result"]).parent
        registration = save(
            directory / "registration.json",
            dict(
                schema="motion2scene_expanded_fit_v1",
                collections=teachers,
                registry=plan["registry"],
                l2=10.0,
                allow_measured_tie_initialization=True,
                weighting=(
                    "historical_observation_gap"
                    if model["arm"] == "observation_curriculum"
                    else "uniform_per_phase"
                ),
                plan=study["expanded_plan"],
            ),
        )
        save(
            Path(model["expected_training_result"]),
            dict(
                status="complete",
                audit_errors=[],
                registration=registration,
                teachers=save(directory / "teachers.json", [dict(collection=t) for t in teachers]),
                policy=save(directory / "policy.npz", {"fixture": "not a physical policy"}),
            ),
        )
    # Policy parsing is tested in the existing policy suite. These fixtures test
    # the controller's binding to all 15 exact model slots and teacher prefixes.
    monkeypatch.setattr(curve, "load_schedule_policy", lambda *args: None)
    return plan, bank, models


def test_all_exact_acquired_models_bind_and_common_learner_cannot_change(tmp_path, monkeypatch):
    plan, bank, models = fitted_prefix_fixture(tmp_path, monkeypatch)
    bindings = curve.bind_models(plan, bank, models, 16)
    assert set(bindings) == {m["run_id"] for m in models}
    path = Path(models[0]["expected_training_result"])
    result = json.loads(path.read_text())
    registration_path = Path(result["registration"]["path"])
    registration = json.loads(registration_path.read_text())
    registration["l2"] = 1.0
    result["registration"] = save(registration_path, registration)
    save(path, result)
    with pytest.raises(ValueError, match="common acquired ridge"):
        curve.bind_models(plan, bank, models, 16)


def runner_fixture(tmp_path, monkeypatch):
    root = tmp_path / "curve"
    root.mkdir()
    models_ref = save(root / "M16/models.json", dict(models={}))
    rows = []
    for name, reused in (("baseline", True), ("student0", False), ("student1", False)):
        folder = root / name
        manifest = save(folder / "manifest.json", dict(cells=[dict(output=str(folder / "native"))]))
        rows.append(dict(assignment_id=name, collection=manifest, reused_reference=reused))

    def physical_result(folder, status="pass", steps=1192):
        return save(
            folder / "result.json",
            dict(
                manifest=artifact(folder / "manifest.json"),
                rows=[
                    {
                        "pass": status == "pass",
                        "measurement_admitted": status != "unknown",
                        "costs": dict(passage_time_s=3.0 if status == "pass" else None),
                        "outcome": dict(task_outcome=status, physics_steps_recorded=steps),
                    }
                ],
            ),
        )

    physical_result(root / "baseline")
    prepared = save(root / "M16/prepared.json", dict(models=models_ref, assignments=rows))
    predecessor = save(
        root / "predecessor.json", dict(intended_execution_root=str(root / "acquisition"))
    )
    monkeypatch.setattr(
        curve, "read_study", lambda _: ({}, {}, {}, dict(predecessor=predecessor), None)
    )
    monkeypatch.setattr(curve, "acquisition_ready", lambda *args: [])
    monkeypatch.setattr(curve, "prepare", lambda *args: prepared)
    monkeypatch.setattr(
        curve.anchor, "check_assignment", lambda row, common, *args: common["cells"][0]
    )
    monkeypatch.setattr(
        curve.collection,
        "verify_manifest",
        lambda folder, **kw: (json.loads((folder / "manifest.json").read_text()), None, None),
    )
    monkeypatch.setattr(curve.anchor, "resource_ready", lambda _: True)
    monkeypatch.setattr(curve.anchor, "deduplicate_inventories", lambda _: {})
    launches = []

    def launch(cell, common, bank, scene, path):
        launches.append(path.parent.name)
        return dict(exit_status=0)

    monkeypatch.setattr(curve.anchor, "execute_cell", launch)
    monkeypatch.setattr(curve.collection, "analyze", physical_result)
    return root, launches, physical_result


def test_resource_pause_resumes_without_reexecuting_shared_or_completed_rows(tmp_path, monkeypatch):
    root, launches, _ = runner_fixture(tmp_path, monkeypatch)
    capacity = iter((True, False))
    monkeypatch.setattr(curve.anchor, "resource_ready", lambda _: next(capacity))
    paused = curve.run(root, 16)
    assert paused["current_assignment_not_launched"] == "student1"
    assert paused["processed_assignments"] == 2
    assert launches == ["student0"]
    monkeypatch.setattr(curve.anchor, "resource_ready", lambda _: True)
    result = curve.run(root, 16)
    assert launches == ["student0", "student1"]
    assert len(json.loads(Path(result["path"]).read_text())["assignments"]) == 3
    assert curve.run(root, 16) == result
    assert launches == ["student0", "student1"]


def test_unknown_attempt_halts_without_retry_and_retains_recorded_cost(tmp_path, monkeypatch):
    root, launches, physical_result = runner_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        curve.collection, "analyze", lambda folder: physical_result(folder, "unknown", 80)
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="unknown outcome retained"):
            curve.run(root, 16)
    assert launches == ["student0"]
    assert not (root / "student1/result.json").exists()
    assert not (root / "M16/result.json").exists()
    assert (
        json.loads((root / "student0/result.json").read_text())["rows"][0]["outcome"][
            "physics_steps_recorded"
        ]
        == 80
    )
