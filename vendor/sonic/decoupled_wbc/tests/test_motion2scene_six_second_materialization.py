"""Frozen world coordinates are the authority for geometry materialization."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/research/motion2scene_materialize_six_second_evaluation.py"
SPEC = importlib.util.spec_from_file_location("v3_materialize", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def first_beams():
    lock = json.loads((ROOT / "docs/motion2scene/TRAVERSAL_EVALUATION_LOCK_V3.json").read_text())
    return copy.deepcopy(lock["evaluation"]["layouts"][0]["nominal_beams"])


def test_preserve_frozen_world_authority_and_all_variants():
    beams = first_beams()
    assert MODULE.locked_world_beams(beams) is beams
    locked = MODULE.load_lock(ROOT / "docs/motion2scene/TRAVERSAL_EVALUATION_LOCK_V3.json")
    assert sum(len(row["fixed_world_variants"]) for row in locked["evaluation"]["layouts"]) == 162


def test_reject_relocated_or_inconsistent_authority():
    beams = first_beams()
    beams[0]["center_xy_m"][0] += 0.01
    with pytest.raises(ValueError, match="disagrees"):
        MODULE.locked_world_beams(beams)
    beams = first_beams()
    beams[0]["full_dimensions_xyz_m"][0] += 0.01
    with pytest.raises(ValueError, match="disagrees"):
        MODULE.locked_world_beams(beams)
    beams = first_beams()
    beams[0]["collision_enabled"] = False
    with pytest.raises(ValueError, match="authority"):
        MODULE.locked_world_beams(beams)
