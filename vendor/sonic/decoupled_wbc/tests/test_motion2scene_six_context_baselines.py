"""Extension invariants; all fixtures are synthetic, not physical evidence."""

import copy
from pathlib import Path
import sys

import pytest

from decoupled_wbc.tests.test_motion2scene_schedule_baselines import panel
from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_baselines import (
    context_oracle,
    rank_baselines,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_script import DEFAULT_CONFIG

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_select_six_context_baselines as selection  # noqa: E402


def synthetic_extension(monkeypatch):
    bank, _, _ = panel()
    old = {key: f"unchanged_{key}" for key in selection.UNCHANGED_RULES}
    old.update(batch="old_batch", registry="registry", manifests=list(range(5)))
    scenes = [
        dict(
            scene_id=f"scene_{i}",
            scene=dict(path=f"/scene_{i}.usd", sha256=f"hash_{i}"),
            beams=[i],
            beam_collision_enabled=[True],
            split="development",
        )
        for i in range(6)
    ]
    manifests = [dict(path=f"/group_{i}/manifest.json") for i in range(6)]
    old["manifests"] = manifests[:5]
    ranking = dict(path="/ranking.py", sha256="frozen-ranking")
    objects = dict(
        old=old,
        complement=dict(children=[dict(manifest=manifests[5])]),
        original_selection=dict(implementation_registration="implementation"),
        implementation=dict(source_closure=[ranking]),
        extra=dict(
            split="development",
            registry="registry",
            cells=[0] * 7,
            scene_id="scene_5",
            scene_definition="extra_scene",
        ),
        extra_scene=scenes[5],
    )

    def fake_read(ref):
        if isinstance(ref, dict):
            if ref == manifests[5]:
                return objects["extra"]
            return objects[ref["path"]]
        return objects[ref]

    monkeypatch.setattr(selection, "read_ref", fake_read)
    monkeypatch.setattr(selection, "checked", lambda path, sha: path)
    monkeypatch.setattr(
        selection, "validate_original_rules", lambda old: (bank, copy.deepcopy(scenes[:5]))
    )
    registration = dict(
        **{key: old[key] for key in selection.UNCHANGED_RULES},
        schema=selection.SCHEMA,
        assigned_contexts=6,
        assigned_branches=42,
        selection_runs=1,
        new_physics_steps=0,
        original_rules="old",
        original_batch="old_batch",
        complementary_batch="complement",
        manifests=manifests,
        collections=[dict(path=f"/group_{i}/result.json") for i in range(6)],
        ranking_implementation=ranking,
    )
    for key in ("original_selection", "original_model", "original_comparison", "complement_audit"):
        registration[key] = dict(path=key, sha256="unchanged")
    return registration, objects


def test_extension_rejects_rule_or_ranking_changes(monkeypatch):
    registration, _ = synthetic_extension(monkeypatch)
    assert len(selection.validate_extension(registration)[1]) == 6
    changed = copy.deepcopy(registration)
    changed["tie_rule"] = "pick a preferred winner"
    with pytest.raises(ValueError, match="exact original rules"):
        selection.validate_extension(changed)
    changed = copy.deepcopy(registration)
    changed["ranking_implementation"]["sha256"] = "new-ranking"
    with pytest.raises(ValueError, match="byte-identical"):
        selection.validate_extension(changed)


def test_extension_rejects_reused_geometry_or_wrong_result_assignment(monkeypatch):
    registration, objects = synthetic_extension(monkeypatch)
    registration["collections"][5]["path"] = "/group_4/result.json"
    with pytest.raises(ValueError, match="assigned manifest"):
        selection.validate_extension(registration)
    registration["collections"][5]["path"] = "/group_5/result.json"
    objects["extra_scene"]["beams"] = [0]
    with pytest.raises(ValueError, match="six distinct"):
        selection.validate_extension(registration)


def test_sixth_context_changes_selection_without_changing_original_five():
    b, ordinary, ordinary_rows = panel()
    low = dict(DEFAULT_CONFIG, minimum_observed_free_height_m=1.28)
    groups = [copy.deepcopy(ordinary) for _ in range(5)]
    oracles = [context_oracle(b, group, ordinary_rows) for group in groups]
    old = rank_baselines(b, groups, oracles, [DEFAULT_CONFIG, low])
    assert old["selected_setting_index"] == 0  # Exact ties retain the default.
    _, extra, extra_rows = panel(hazards={15: (0,)}, outcomes={"prior_15": ("failure", None)})
    # Measured 1.30m permits the low threshold to wait; the default commits and fails.
    from decoupled_wbc.tests.test_motion2scene_schedule_script_review import observation

    for target in extra["targets"]:
        target["features"] = observation((0,), gap=1.30)[1].tolist()
    ranked = rank_baselines(
        b, groups + [extra], oracles + [context_oracle(b, extra, extra_rows)], [DEFAULT_CONFIG, low]
    )
    assert ranked["selected_setting_index"] == 1
    assert ranked["script_candidates"][0]["selected_failure_count"] == 1
    assert ranked["script_candidates"][1]["assigned_contexts"] == 6
    assert old["selected_setting_index"] == 0


def test_unknown_complementary_branch_cannot_drop_sixth_context():
    b, ordinary, rows = panel()
    groups = [copy.deepcopy(ordinary) for _ in range(5)]
    oracles = [context_oracle(b, group, rows) for group in groups]
    _, extra, extra_rows = panel(outcomes={"short_15": ("unknown", None)})
    ranked = rank_baselines(
        b, groups + [extra], oracles + [context_oracle(b, extra, extra_rows)], [DEFAULT_CONFIG]
    )
    assert ranked["selected_setting_index"] is None
    assert ranked["preferred_constant_option_id"] is None
    assert ranked["script_candidates"][0]["assigned_contexts"] == 6
    assert ranked["script_candidates"][0]["known_contexts"] == 5
    assert len(ranked["constant_candidates"]) == 7
