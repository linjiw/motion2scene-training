"""Synthetic contract fixtures only; these tests contain no physical evidence."""

import copy
import json

import numpy as np
import pytest

from decoupled_wbc.tests.test_motion2scene_timed_options import artifact, evidence, fixture
from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import (
    FloorCeilingHistory,
    HistoryGrid,
    SensorRay,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_history_policy import (
    SCHEMA,
    choose_timed_option,
    expected_feature_names,
    load_timed_policy,
    timed_history_features,
    validate_history_bank,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (
    definition_digest,
    load_verified_registry,
    make_verified_registry,
    validate_request,
)


def bank_fixture(tmp_path):
    request = fixture(tmp_path)
    refs = []
    for option_id in validate_request(request).option_ids:
        path = tmp_path / (option_id + ".json")
        path.write_text(json.dumps(evidence(request, option_id)))
        refs.append(artifact(path))
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(make_verified_registry(request, refs)))
    return load_verified_registry(path, artifact(path)["sha256"])


def feature_fixture(hit=False):
    memory = FloorCeilingHistory(HistoryGrid())
    if hit:
        memory.push([SensorRay((0, 0, 0.4), (0.8, 0, 0.6), 4, 0.5, (0, 0, -1))], 0)
    obs = memory.snapshot((0, 0, 0), (1, 0, 0, 0), 0)
    state = {
        "projected_gravity_b": [0, 0, -1],
        "root_lin_vel_w": [0, 0, 0],
        "root_ang_vel_w": [0, 0, 0],
        "dof_pos": [0] * 29,
        "dof_vel": [0] * 29,
    }
    return timed_history_features(obs, memory.grid, state, 15, 0, np.ones(3, bool), 0)


def test_exact106_features_preserve_unknown_and_observed_overhead(tmp_path):
    bank = bank_fixture(tmp_path)
    names, unknown = feature_fixture()
    _, observed = feature_fixture(hit=True)
    assert names == expected_feature_names() and len(names) == 106
    assert unknown[names.index("corridor_0_0.75_unknown_fraction")] == 1
    assert unknown[names.index("corridor_0_0.75_upper_hit")] == 0
    assert observed[names.index("corridor_0_0.75_ceiling_fraction")] > 0
    common = (np.ones(3, bool), 0, 15, None, bank)
    assert choose_timed_option("scripted_sustained", names, unknown, *common)[0] == "neutral"
    assert (
        choose_timed_option("scripted_sustained", names, observed, *common)[0] == bank.option_ids[2]
    )


def test_experimental_request_cannot_be_used_by_any_online_mode(tmp_path):
    bank = validate_request(fixture(tmp_path))
    for mode in (
        "always_walk",
        "constant_short",
        "constant_sustained",
        "scripted_sustained",
        "learned",
    ):
        with pytest.raises(ValueError, match="physically verified"):
            validate_history_bank(bank, mode)
    validate_history_bank(bank, "forced")


def test_declared_mandatory_return_is_requested_even_when_guard_refuses(tmp_path):
    bank = bank_fixture(tmp_path)
    names, x = feature_fixture()
    # Current option hold remains legal, neutral's jump is illegal. The parent
    # must receive and record the refused return, not a fabricated valid hold.
    action, values = choose_timed_option(
        "constant_sustained",
        names,
        x,
        np.array([False, False, True]),
        2,
        265,
        0,
        bank,
    )
    assert action == "neutral" and values is None
    # Legacy3.3s return is not used; at that phase hold the active schedule.
    assert (
        choose_timed_option(
            "constant_sustained",
            names,
            x,
            np.array([False, False, True]),
            2,
            165,
            None,
            bank,
        )[0]
        == bank.option_ids[2]
    )


def test_exact_phase_and_option_masking_are_enforced(tmp_path):
    bank = bank_fixture(tmp_path)
    names, x = feature_fixture()
    action, _ = choose_timed_option(
        "constant_sustained",
        names,
        x,
        np.array([True, True, False]),
        0,
        15,
        None,
        bank,
    )
    assert action == "neutral"
    with pytest.raises(ValueError, match="phase"):
        choose_timed_option("always_walk", names, x, np.ones(3, bool), 0, 20, None, bank)
    bad = copy.deepcopy(bank)
    bad.definition["request"]["options"][0]["entry_tick"] = 20
    with pytest.raises(ValueError, match="single tick15"):
        validate_history_bank(bad, "always_walk")


def test_new_model_binds_exact_named_features_schedule_and_canonical_hash(tmp_path):
    bank = bank_fixture(tmp_path)
    names, x = feature_fixture()
    model = dict(
        schema_version=np.array(SCHEMA),
        feature_names=np.array(names),
        option_ids=np.array(bank.option_ids),
        request_digest=np.array(definition_digest(bank.request)),
        entry_tick=np.array(15),
        classes=np.arange(3),
        mean=np.zeros(106),
        std=np.ones(106),
        weights=np.zeros((106, 3)),
        bias=np.array([0.0, 1.0, 2.0]),
    )
    path = tmp_path / "policy.npz"
    np.savez(path, **model)
    loaded = load_timed_policy(path, artifact(path)["sha256"], bank)
    assert (
        choose_timed_option("learned", names, x, np.ones(3, bool), 0, 15, None, bank, loaded)[0]
        == bank.option_ids[2]
    )
    model["request_digest"] = np.array("sha256:" + "f" * 64)
    np.savez(path, **model)
    with pytest.raises(ValueError, match="binding"):
        load_timed_policy(path, artifact(path)["sha256"], bank)


def test_schedule_audit_detects_early_return_or_hidden_active_change(tmp_path):
    from gear_sonic.dataset_generation.hallucination.motion2scene_timed_history_policy import (
        audit_executed_schedule,
    )

    bank = bank_fixture(tmp_path)
    chosen = bank.option_ids[2]
    measured = evidence(bank.request, chosen)
    rows = [
        {"tick": tick, "active": active, "transition": {"allowed": True}}
        for tick, active in zip(
            measured["command_ticks"], measured["active_option_ids"], strict=True
        )
    ]
    assert audit_executed_schedule(bank, rows, measured["switches"], "forced", chosen)["valid"]
    altered = copy.deepcopy(measured["switches"])
    altered[-1]["tick"] = 165
    assert not audit_executed_schedule(bank, rows, altered, "forced", chosen)["valid"]
    rows[150]["active"] = bank.option_ids[1]
    assert not audit_executed_schedule(bank, rows, measured["switches"], "forced", chosen)["valid"]


def test_fit_requires_real_complete_target_for_each_option(tmp_path):
    from gear_sonic.dataset_generation.hallucination.motion2scene_timed_history_policy import (
        fit_timed_value_policy,
    )

    bank = bank_fixture(tmp_path)
    names, x = feature_fixture()
    model, report = fit_timed_value_policy(
        bank, [x], [[True] * 3], [[1.0, 2.0, 3.0]], np.ones((1, 3), bool), np.ones((1, 3), bool)
    )
    assert report["complete_consequential_decisions"] == 1
    assert (
        choose_timed_option("learned", names, x, np.ones(3, bool), 0, 15, None, bank, model)[0]
        == "neutral"
    )
    with pytest.raises(ValueError, match="complete consequential"):
        fit_timed_value_policy(
            bank,
            [x],
            [[True] * 3],
            [[1.0, 2.0, 3.0]],
            np.array([[True, True, False]]),
            np.ones((1, 3), bool),
        )
    altered = x.copy()
    altered[names.index("phase_s")] = 0.4
    with pytest.raises(ValueError, match="tick15"):
        fit_timed_value_policy(
            bank,
            [altered],
            [[True] * 3],
            [[1.0, 2.0, 3.0]],
            np.ones((1, 3), bool),
            np.ones((1, 3), bool),
        )
