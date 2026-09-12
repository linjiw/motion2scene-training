"""Synthetic provenance tests for exact scientific-setting consistency."""

import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/research"))
sys.path.insert(0, str(ROOT / "decoupled_wbc/tests"))

import motion2scene_run_primary_acquisition as primary  # noqa: E402
import motion2scene_run_reserved_evaluation as runner  # noqa: E402
from test_motion2scene_acquisition_plan import adopted_fixture, make_training  # noqa: E402
from test_motion2scene_primary_controller import (  # noqa: E402
    FakeBackend,
    fixture as primary_fixture,
)
from test_motion2scene_reserved_runner import (  # noqa: E402
    completed_inventory as original_completed_inventory,
    stub_native_models,
)

from gear_sonic.dataset_generation.hallucination import (  # noqa: E402
    motion2scene_acquisition_plan as plan_module,
)

MISSING = object()
FLAG = "allow_measured_tie_initialization"


@pytest.fixture(scope="module")
def completed_inventory(tmp_path_factory):
    return original_completed_inventory.__wrapped__(tmp_path_factory)


def set_flag(value, flag):
    result = copy.deepcopy(value)
    result.pop(FLAG, None)
    if flag is not MISSING:
        result[FLAG] = flag
    return result


@pytest.mark.parametrize(
    "adopted,registered,accepted",
    [
        (MISSING, MISSING, True),
        (MISSING, False, True),
        (False, MISSING, True),
        (False, False, True),
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (True, MISSING, False),
        (MISSING, True, False),
        (1, True, False),
        (True, 1, False),
        (0, False, False),
        (False, 0, False),
        (None, False, False),
        (False, None, False),
        ("false", False, False),
        (True, "true", False),
    ],
)
def test_preupdate_flag_is_an_exact_boolean_parameter(
    tmp_path, monkeypatch, adopted, registered, accepted
):
    plan, _, adoption_ref = adopted_fixture(tmp_path)
    learner = plan_module.read_bound(plan_module.read_bound(adoption_ref)["common_learner"])
    learner = set_flag(learner, adopted)
    row = plan["runs"][0]["rounds"][1]
    earlier = [plan_module.write_new(tmp_path / "teacher.json", {"synthetic": True})]
    model, training = make_training(plan, row, earlier)
    entry = dict(student_model=model, model_training_result=training)
    original_read = plan_module.read_bound

    def read(ref):
        value = original_read(ref)
        if "collections" in value and "implementation" in value:
            return set_flag(value, registered)
        return value

    monkeypatch.setattr(plan_module, "read_bound", read)
    if accepted:
        plan_module._training_provenance(plan, row, entry, earlier, learner)
    else:
        with pytest.raises(ValueError, match="measured-tie initialization"):
            plan_module._training_provenance(plan, row, entry, earlier, learner)


@pytest.mark.parametrize("flag", [MISSING, False, True])
def test_complete_inventory_preserves_old_default_and_new_explicit_optin(
    completed_inventory, monkeypatch, flag
):
    _, protocol, inventory = completed_inventory
    stub_native_models(monkeypatch)
    original_read = runner.read_bound

    def read(ref):
        value = original_read(ref)
        if ("collections" in value and "implementation" in value) or (
            "feature_dimension" in value and "training_implementation" in value
        ):
            return set_flag(value, flag)
        return value

    monkeypatch.setattr(runner, "read_bound", read)
    monkeypatch.setattr(plan_module, "read_bound", read)
    monkeypatch.setattr(
        runner, "validate_completed_replay_order", plan_module.validate_completed_replay_order
    )
    result = runner.validate_inventory(copy.deepcopy(inventory), protocol)
    assert result["checkpoint_policies"] == 24
    assert result["corpus_runs"] == 12


@pytest.mark.parametrize("checkpoint", ["M2", "M4"])
@pytest.mark.parametrize("registered", [False, MISSING, 1, None])
def test_checkpoint_validator_rejects_one_substituted_scientific_setting(
    completed_inventory, monkeypatch, checkpoint, registered
):
    _, protocol, inventory = completed_inventory
    stub_native_models(monkeypatch)
    item = next(row for row in inventory["models"] if row["checkpoint"] == checkpoint)
    registration_ref = runner.read_bound(item["training_result"])["registration"]
    original_read = runner.read_bound

    def read(ref):
        value = original_read(ref)
        if "feature_dimension" in value and "training_implementation" in value:
            return set_flag(value, True)
        if "collections" in value and "implementation" in value:
            return set_flag(value, registered if ref == registration_ref else True)
        return value

    # Keep the original causal audit intact and isolate the added checkpoint
    # gate: a final M4 model is not consumed by any pre-update causal receipt.
    monkeypatch.setattr(runner, "read_bound", read)
    with pytest.raises(ValueError, match="measured-tie initialization"):
        runner.validate_inventory(copy.deepcopy(inventory), protocol)


@pytest.mark.parametrize("flag", [MISSING, False, True, 0, 1, None, "true", [], {}])
def test_adopted_common_learner_requires_boolean_or_legacy_absence(tmp_path, monkeypatch, flag):
    plan, plan_ref, adoption_ref = adopted_fixture(tmp_path)
    learner_ref = plan_module.read_bound(adoption_ref)["common_learner"]
    original_read = plan_module.read_bound

    def read(ref):
        value = original_read(ref)
        return set_flag(value, flag) if ref == learner_ref else value

    monkeypatch.setattr(plan_module, "read_bound", read)
    if flag is MISSING or type(flag) is bool:
        learner = plan_module._adopted(plan, plan_ref, adoption_ref)
        assert learner.get(FLAG, False) is (False if flag is MISSING else flag)
    else:
        with pytest.raises(ValueError, match="one frozen common"):
            plan_module._adopted(plan, plan_ref, adoption_ref)


@pytest.mark.parametrize(
    "adopted,registered,accepted",
    [
        (MISSING, MISSING, True),
        (MISSING, False, True),
        (False, MISSING, True),
        (False, False, True),
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (True, MISSING, False),
        (MISSING, True, False),
        (1, True, False),
        (True, 1, False),
        (0, False, False),
        (False, 0, False),
        (None, False, False),
        (False, None, False),
        ("false", False, False),
        (True, "true", False),
    ],
)
def test_resumed_completed_fit_checks_exact_flag_before_accepting_model(
    tmp_path, monkeypatch, adopted, registered, accepted
):
    context, backend = primary_fixture(tmp_path), FakeBackend()
    context["learner"] = set_flag(context["learner"], adopted)
    out = tmp_path / "already_fitted"
    # Existing result is audited on resume; no fit or physical branch is repeated.
    backend.fit(context, [], out, None)
    registration_ref = primary.read(out / "result.json")["registration"]
    original_read = primary.read_bound

    def read(ref):
        value = original_read(ref)
        return set_flag(value, registered) if ref == registration_ref else value

    monkeypatch.setattr(primary, "read_bound", read)
    controller = primary.Controller(context, backend)
    if accepted:
        assert controller.fit([], out)["status"] == "complete"
    else:
        with pytest.raises(ValueError, match="completed fit differs"):
            controller.fit([], out)
        assert not (controller.folder / "cpu_fits").exists()
    assert len(backend.fits) == 1
    assert backend.launches == []
