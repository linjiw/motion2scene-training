"""The M8 physical panel retains every model, context and matched bank branch."""

import copy
from collections import Counter
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_development_checkpoint_panel import (
    ARMS,
    CONTEXTS,
    PHYSICS_SEED,
    assignments,
    check_assignment,
)


def population():
    runs = [
        dict(seed=s, arm=a, run_id=f"seed{s}_{a}")
        for s in (93201, 93202, 93203)
        for a in sorted(ARMS)
    ]
    scenes = [dict(scene_id=s, scene_definition=dict(path=s, sha256=s)) for s in sorted(CONTEXTS)]
    return runs, scenes, ["neutral"] + [f"option{i}" for i in range(1, 7)]


def test_complete_balanced_panel_and_reproducible_order():
    args = population()
    rows = assignments(*args)
    assert rows == assignments(*args)
    assert len(rows) == len({r["assignment_id"] for r in rows}) == 138
    assert set(Counter(r["policy_id"] for r in rows).values()) == {6}
    assert set(Counter(r["scene_id"] for r in rows).values()) == {23}
    assert {r["physics_seed"] for r in rows} == {PHYSICS_SEED}
    assert Counter(r["mode"] for r in rows) == dict(learned=90, scripted_multi=6, forced=42)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_no_missing_model_scene_or_fixed_schedule(index):
    args = list(population())
    args[index] = args[index][:-1]
    with pytest.raises(ValueError):
        assignments(*args)


def context(mode):
    row = dict(
        mode=mode,
        scene_definition={"path": "scene"},
        run_id="seed93201_uniform",
        option_id="neutral",
    )
    study = dict(registry={"path": "bank"}, script={"path": "script"})
    models = {row["run_id"]: dict(policy={"path": "acquired"})}
    common = dict(
        split="development",
        registry=study["registry"],
        scene_definition=row["scene_definition"],
        expected_physics_steps=1192,
        policy=models[row["run_id"]]["policy"] if mode == "learned" else None,
        script_parameters=study["script"] if mode == "scripted_multi" else None,
        cells=[
            dict(runtime_seed=PHYSICS_SEED, timed_schedule_mode=mode, forced_option_id="neutral")
        ],
    )
    return row, common, study, models


@pytest.mark.parametrize("mode", ["learned", "scripted_multi", "forced"])
def test_assignment_has_the_paired_condition_and_correct_controller(mode):
    row, common, study, models = context(mode)
    assert check_assignment(row, common, study, models) == common["cells"][0]
    changed = copy.deepcopy(common)
    changed["cells"][0]["runtime_seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        check_assignment(row, changed, study, models)


def test_no_model_substitution_or_reserved_scene():
    row, common, study, models = context("learned")
    common["policy"] = {"path": "different"}
    with pytest.raises(ValueError, match="policy"):
        check_assignment(row, common, study, models)
    common["policy"] = models[row["run_id"]]["policy"]
    common["split"] = "reserved"
    with pytest.raises(ValueError, match="assignment"):
        check_assignment(row, common, study, models)


def test_no_script_weakening_or_fixed_schedule_substitution():
    row, common, study, models = context("scripted_multi")
    common["script_parameters"] = {"path": "weaker"}
    with pytest.raises(ValueError, match="script"):
        check_assignment(row, common, study, models)
    row, common, study, models = context("forced")
    common["cells"][0]["forced_option_id"] = "different"
    with pytest.raises(ValueError, match="schedule"):
        check_assignment(row, common, study, models)
