import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

import motion2scene_expanded_acquisition as expansion
import motion2scene_resume_expanded_acquisition as repair
import motion2scene_timing_diagnostic as artifacts


def write(path, value):
    path.write_text(json.dumps(value))
    return artifacts.artifact(path)


def inherited_fixture(tmp_path, monkeypatch):
    registry = dict(path="synthetic_registry", sha256="fixture")
    groups, rounds = [], []
    for index in range(2):
        groups.append(dict(collection=dict(id=index)))
        registration = write(
            tmp_path / f"registration{index}.json",
            dict(collections=[g["collection"] for g in groups], registry=registry),
        )
        teachers = write(tmp_path / f"teachers{index}.json", groups)
        model = write(
            tmp_path / f"model{index}.json",
            dict(registration=registration, teachers=teachers, policy=dict(id=index)),
        )
        rounds.append(
            dict(index=index, inherited_model=model["path"], inherited_student="student.json")
        )
    controller = expansion.ExpandedController(
        dict(
            plan=dict(registry=registry),
            run_root=tmp_path,
            expanded_plan_ref=dict(id="synthetic_plan"),
        ),
        dict(rounds=rounds, run_id="synthetic_corpus"),
        None,
    )
    monkeypatch.setattr(expansion, "load_verified_registry", lambda *args: None)
    monkeypatch.setattr(expansion.replay, "audit_students", lambda *args: ([{}], {}, []))
    monkeypatch.setattr(controller, "accounting", lambda: dict(actual_recorded_physics_steps=0))

    def forbidden(*args, **kwargs):
        pytest.fail("inherited import must not execute physics or fit another model")

    monkeypatch.setattr(controller, "collect", forbidden)
    monkeypatch.setattr(controller, "fit_expanded", forbidden)
    return controller, rounds


def test_original_error_reproduces_and_repair_reuses_exact_inherited_model(tmp_path, monkeypatch):
    controller, rounds = inherited_fixture(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in tmp_path.glob("*.json")}
    with pytest.raises(AttributeError, match="resolve"):
        controller.acquire_expanded(1)
    with repair.normalized_driver_artifacts():
        completion = expansion.bound(controller.acquire_expanded(1))
    assert completion["training_result"] == artifacts.artifact(Path(rounds[1]["inherited_model"]))
    assert completion["model"] == dict(id=1)
    assert completion["accounting"]["actual_recorded_physics_steps"] == 0
    assert all(p.read_bytes() == value for p, value in before.items())
    assert expansion.artifact is artifacts.artifact


def test_repair_preserves_prefix_order_rejection_and_restores_reader(tmp_path, monkeypatch):
    controller, rounds = inherited_fixture(tmp_path, monkeypatch)
    path = Path(rounds[1]["inherited_model"])
    model = json.loads(path.read_text())
    model["teachers"] = write(
        tmp_path / "wrong_teachers.json", [dict(collection=dict(id=1)), dict(collection=dict(id=0))]
    )
    write(path, model)
    with pytest.raises(ValueError, match="one teaching prefix"):
        with repair.normalized_driver_artifacts():
            controller.acquire_expanded(1)
    assert expansion.artifact is artifacts.artifact


def test_repair_retains_bound_artifact_hash_checks(tmp_path):
    path = tmp_path / "record.json"
    ref = write(path, dict(before=True))
    write(path, dict(after=True))
    with repair.normalized_driver_artifacts():
        with pytest.raises(ValueError, match="hash mismatch"):
            expansion.bound(ref)


def test_repair_requires_matching_declaration_and_plan_bound_original(tmp_path):
    plan = tmp_path / "plan.json"
    adoption = tmp_path / "adoption.json"
    write(plan, dict(implementation=[artifacts.artifact(Path(expansion.__file__))]))
    write(adoption, dict(fixture=True))
    declared = dict(
        repair=repair.REPAIR,
        plan=artifacts.artifact(plan),
        adoption=artifacts.artifact(adoption),
        adapter=artifacts.artifact(Path(repair.__file__)),
        original_driver=artifacts.artifact(Path(expansion.__file__)),
        original_artifact_reader=artifacts.artifact(Path(artifacts.__file__)),
    )
    receipt = tmp_path / "repair.json"
    write(receipt, declared)
    assert repair.validate_repair(receipt, plan, adoption) == artifacts.artifact(receipt)
    write(receipt, {**declared, "adapter": dict(incorrect=True)})
    with pytest.raises(ValueError, match="adapter"):
        repair.validate_repair(receipt, plan, adoption)
    write(plan, dict(implementation=[]))
    write(receipt, {**declared, "plan": artifacts.artifact(plan)})
    with pytest.raises(ValueError, match="plan-bound driver"):
        repair.validate_repair(receipt, plan, adoption)


def test_nested_or_unexpected_override_rejected():
    with repair.normalized_driver_artifacts():
        with pytest.raises(ValueError, match="unexpected existing"):
            with repair.normalized_driver_artifacts():
                pytest.fail("nested override should fail")
