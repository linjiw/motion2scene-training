from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

import motion2scene_expanded_acquisition as expansion
from motion2scene_expanded_acquisition import ExpandedController, validate_training_prefix
from motion2scene_run_primary_acquisition import Paused
from test_motion2scene_primary_controller import FakeBackend, fixture
from test_motion2scene_timed_schedule_learning import data


def test_extended_attempt_accounting_counts_inherited_cost_once_and_resumes(tmp_path):
    context = fixture(tmp_path)
    context["plan"]["budget"]["budget_physics_steps"] = 320000
    inherited = dict(
        actual_recorded_physics_steps=46488,
        conservative_charged_steps=47680,
        unknown_attempt_reserved_steps=1192,
        unknown_outcome_tail_reserved_steps=0,
    )
    controller = ExpandedController(context, context["run"], inherited)
    backend = FakeBackend()
    controller.backend = backend
    row = context["run"]["rounds"][0]
    controller.collect(row, "teacher")
    report = controller.accounting()
    assert report["actual_recorded_physics_steps"] == 46488 + 8344
    assert report["conservative_charged_steps"] == 47680 + 8344
    assert report["new_recorded_physics_steps"] == 8344
    assert report["unlaunched_assigned_episode_slots"] == 263 - 39 - 7
    controller.collect(row, "teacher")
    assert len(backend.launches) == 7
    assert (
        controller.accounting()["conservative_charged_steps"]
        == report["conservative_charged_steps"]
    )


def test_current_or_reordered_labels_cannot_enter_preupdate_training():
    registry = {"registry": "fixture"}
    prior = [{"id": 0}, {"id": 1}]
    validate_training_prefix(dict(collections=prior, registry=registry), prior, registry)
    with pytest.raises(ValueError, match="earlier encounter prefix"):
        validate_training_prefix(
            dict(collections=prior + [{"id": 2}], registry=registry), prior, registry
        )
    with pytest.raises(ValueError, match="earlier encounter prefix"):
        validate_training_prefix(dict(collections=prior[::-1], registry=registry), prior, registry)


def test_resource_pause_keeps_new_attempt_unlaunched(tmp_path):
    context = fixture(tmp_path)
    controller = ExpandedController(context, context["run"], None)
    controller.backend = FakeBackend()
    controller.backend.free = 0
    with pytest.raises(Paused, match="resource preflight"):
        controller.collect(context["run"]["rounds"][0], "teacher")
    assert controller.backend.launches == []
    assert controller.accounting()["new_recorded_physics_steps"] == 0


def test_real_cpu_fit_saves_bound_loadable_artifacts_and_resumes(tmp_path, monkeypatch):
    bank, names, x, ticks, passed, times, legal = data()
    monkeypatch.setattr(expansion, "load_verified_registry", lambda *args: bank)
    registry = dict(path="synthetic_fixture", sha256="fixture")
    context = dict(
        plan=dict(registry=registry),
        run_root=tmp_path,
        expanded_plan_ref=dict(path="synthetic_plan", sha256="fixture"),
    )
    targets = [
        dict(
            feature_names=list(names),
            features=x[i].tolist(),
            phase_tick=int(ticks[i]),
            pass_labels=passed[i].tolist(),
            passage_time_s=[float(v) if np.isfinite(v) else None for v in times[i]],
            admitted=[True] * len(bank.option_ids),
            legal_mask=legal[i].tolist(),
        )
        for i in range(len(x))
    ]
    groups = [
        dict(
            collection=dict(path="synthetic_collection", sha256="fixture"),
            physics_steps=0,
            targets=targets,
        )
    ]
    controller = ExpandedController(context, dict(arm="analytic_contrast"), None)
    result = controller.fit_expanded(groups, [None], tmp_path / "model")
    assert expansion.bound(result["teachers"]) == groups
    assert expansion.bound(result["registration"])["registry"] == registry
    with np.load(result["policy"]["path"]) as model:
        assert model["weights"].shape == (3, len(names), 4)
    assert controller.fit_expanded(groups, [None], tmp_path / "model") == result


def test_watch_waits_only_for_unlaunched_resource_states(monkeypatch):
    states = iter(
        [
            dict(status="waiting_for_primary_M4", missing_run="fixture"),
            dict(
                status="paused",
                reason="resource preflight paused before intent; assigned cell remains unlaunched",
            ),
            dict(status="paused", reason="unknown physical outcome"),
        ]
    )
    sleeps = []
    monkeypatch.setattr(expansion, "run", lambda *args: next(states))
    monkeypatch.setattr(expansion.time, "sleep", sleeps.append)
    result = expansion.watch(None, None, 32)
    assert result == dict(status="paused", reason="unknown physical outcome")
    assert sleeps == [45, 45]
