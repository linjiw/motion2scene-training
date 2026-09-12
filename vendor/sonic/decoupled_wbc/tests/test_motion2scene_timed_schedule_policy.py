"""Synthetic multi-phase contracts; no test fixture is physical evidence."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from decoupled_wbc.tests.test_motion2scene_timed_options import artifact
from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import (
    FloorCeilingHistory,
    HistoryGrid,
    SensorRay,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (
    fit_timed_schedule_policy,
    schedule_layout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (
    HISTORY_FRAMES,
    HISTORY_SECONDS,
    audit_executed_schedule,
    choose_schedule_option,
    expected_feature_names,
    load_schedule_policy,
    schedule_history_features,
    validate_schedule_policy,
)


def bank_fixture():
    rows = [
        ("short", 15, 265),
        ("sustained", 15, 255),
        ("prior", 15, 265),
        ("prior", 50, 265),
        ("short", 70, 265),
        ("sustained", 70, 255),
    ]
    options = [
        dict(option_id=f"{name}_{entry}", reference_id=name, entry_tick=entry, return_tick=ret)
        for name, entry, ret in rows
    ]
    return SimpleNamespace(
        online_verified=True,
        frame_count=299,
        option_ids=("neutral", *(o["option_id"] for o in options)),
        request=dict(
            max_entries_per_episode=1,
            options=options,
            references=[
                dict(reference_id=name) for name in ("neutral", "short", "sustained", "prior")
            ],
        ),
    )


def features(tick=15, active=0, legal=None, hit=False):
    b = bank_fixture()
    if legal is None:
        phases, masks, _ = schedule_layout(b)
        legal = masks[np.flatnonzero(phases == tick)[0]]
    memory = FloorCeilingHistory(HistoryGrid(max_age_s=HISTORY_SECONDS, max_frames=HISTORY_FRAMES))
    if hit:
        memory.push([SensorRay((0, 0, 0.4), (0.8, 0, 0.6), 4, 0.5, (0, 0, -1))], 0)
    obs = memory.snapshot((0, 0, 0), (1, 0, 0, 0), 0.5)
    state = dict(
        projected_gravity_b=[0, 0, -1],
        root_lin_vel_w=[0, 0, 0],
        root_ang_vel_w=[0, 0, 0],
        dof_pos=[0] * 29,
        dof_vel=[0] * 29,
    )
    names, x = schedule_history_features(obs, memory.grid, state, tick, active, legal, 0)
    return names, x, legal


def model_fixture():
    b = bank_fixture()
    xs, ticks, passed, times, legal = [], [], [], [], []
    for tick in (15, 50, 70):
        for observed in (False, True):
            names, x, mask = features(tick=tick, hit=observed)
            y = np.zeros(7, bool)
            y[np.flatnonzero(mask)[-1] if observed else 0] = True
            xs.append(x)
            ticks.append(tick)
            passed.append(y)
            times.append(np.where(y, 3.0, np.nan))
            legal.append(mask)
    model, report = fit_timed_schedule_policy(
        b,
        np.asarray(xs),
        names,
        np.asarray(ticks),
        np.asarray(passed),
        np.asarray(times),
        np.ones((6, 7), bool),
        np.asarray(legal),
    )
    return b, model, report


def test_exact114_sensor_features_and_late_observed_reference_choice():
    bank = bank_fixture()
    names, unknown, legal = features(tick=70)
    _, occupied, _ = features(tick=70, hit=True)
    assert names == expected_feature_names(7) and len(names) == 114
    assert unknown[names.index("corridor_0_0.75_unknown_fraction")] == 1
    assert unknown[names.index("corridor_0_0.75_upper_hit")] == 0
    assert occupied[names.index("corridor_0_0.75_ceiling_fraction")] > 0
    assert (
        choose_schedule_option("scripted_reference", names, unknown, legal, 0, 70, None, bank)[0]
        == "neutral"
    )
    assert (
        choose_schedule_option("scripted_reference", names, occupied, legal, 0, 70, None, bank)[0]
        == "sustained_70"
    )


def test_constant_late_schedule_waits_while_early_options_are_available():
    bank = bank_fixture()
    for tick, expected in ((15, "neutral"), (50, "neutral"), (70, "short_70")):
        names, x, mask = features(tick)
        assert (
            choose_schedule_option(
                "constant_option",
                names,
                x,
                mask,
                0,
                tick,
                None,
                bank,
                preferred_option_id="short_70",
            )[0]
            == expected
        )


def test_mandatory_return_cannot_silently_hold_when_neutral_guard_fails():
    bank = bank_fixture()
    mask = np.array([False, False, True, False, False, False, False])
    names, x, _ = features(255, active=2, legal=mask)
    assert choose_schedule_option("learned", names, x, mask, 2, 255, 0, bank) == ("neutral", None)
    # No model invocation when the only legal behavior is holding an active schedule.
    assert choose_schedule_option("learned", names, x, mask, 2, 200, None, bank) == (
        "sustained_15",
        None,
    )


def test_fitted_phase_heads_round_trip_and_unsupported_heads_fail(tmp_path):
    bank, model, report = model_fixture()
    assert report["supervised_decisions"] == 6
    path = tmp_path / "model.npz"
    np.savez_compressed(path, **model)
    loaded = load_schedule_policy(path, artifact(path)["sha256"], bank)
    for tick in (15, 50, 70):
        for hit in (False, True):
            names, x, mask = features(tick, hit=hit)
            choice, _ = choose_schedule_option(
                "learned", names, x, mask, 0, tick, None, bank, policy=loaded
            )
            assert choice == (bank.option_ids[np.flatnonzero(mask)[-1]] if hit else "neutral")
    missing = copy.deepcopy(loaded)
    missing["trained_mask"][0, 1] = False
    names, x, mask = features()
    with pytest.raises(ValueError, match="lacks a measured"):
        choose_schedule_option("learned", names, x, mask, 0, 15, None, bank, policy=missing)
    missing = copy.deepcopy(loaded)
    missing["phase_ticks"] = np.array([15, 50, 71])
    with pytest.raises(ValueError, match="binding mismatch"):
        validate_schedule_policy(missing, bank)
    with pytest.raises(ValueError, match="phase or schedule"):
        choose_schedule_option("always_walk", names, x, mask, 0, 16, None, bank)
    bank.online_verified = False
    with pytest.raises(ValueError, match="physical online qualification"):
        validate_schedule_policy(loaded, bank)


def test_whole_schedule_audit_checks_preaction_packet_and_rejects_reentry():
    b = bank_fixture()
    chosen, entry, ret = "sustained_70", 70, 255
    rows = []
    for tick in range(1, b.frame_count):
        before = chosen if entry < tick <= ret else "neutral"
        after = chosen if entry <= tick < ret else "neutral"
        active = b.option_ids.index(before)
        mask = np.eye(7, dtype=bool)[active]
        _, x, _ = features(tick, active=active, legal=mask)
        rows.append(
            dict(
                tick=tick,
                active_before=before,
                active=after,
                features=x.tolist(),
                transition=dict(allowed=True),
            )
        )
    switches = [
        dict(
            tick=tick,
            **{"from": left, "to": right},
            joint_jump_rad=0.0,
            root_jump_m=0.0,
            root_state_unchanged=True,
            joint_state_unchanged=True,
            clock_unchanged=True,
        )
        for tick, left, right in ((entry, "neutral", chosen), (ret, chosen, "neutral"))
    ]
    result = audit_executed_schedule(b, rows, switches, "forced", chosen)
    assert result["valid"]
    json.dumps(result)
    bad = copy.deepcopy(rows)
    bad[entry - 1]["active_before"] = chosen
    assert not audit_executed_schedule(b, bad, switches, "forced", chosen)["valid"]
    assert not audit_executed_schedule(b, rows, switches * 2, "forced", chosen)["valid"]


def test_scripted_multi_uses_observed_features_and_bypasses_hold_and_return(tmp_path):
    from gear_sonic.dataset_generation.hallucination.motion2scene_schedule_script import (
        DEFAULT_CONFIG,
    )
    from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (
        load_script_parameters,
    )

    bank = bank_fixture()
    for row in bank.request["options"]:
        if row["reference_id"] == "prior":
            row["reference_id"] = "prior_splice"
    settings_path = tmp_path / "script.json"
    settings_path.write_text(json.dumps(DEFAULT_CONFIG))
    ref = artifact(settings_path)
    settings = load_script_parameters(settings_path, ref["sha256"], bank)
    names, x, legal = features(tick=15)
    assert (
        choose_schedule_option(
            "scripted_multi", names, x, legal, 0, 15, None, bank, script_parameters=settings
        )[0]
        == "neutral"
    )
    x[names.index("corridor_0_0.75_upper_hit")] = 1
    assert (
        choose_schedule_option(
            "scripted_multi", names, x, legal, 0, 15, None, bank, script_parameters=settings
        )[0]
        == "prior_15"
    )
    names, x, legal = features(tick=70, hit=True)
    assert (
        choose_schedule_option(
            "scripted_multi", names, x, legal, 0, 70, None, bank, script_parameters=settings
        )[0]
        == "short_70"
    )
    legal = np.array([False, True, False, False, False, False, False])
    names, x, _ = features(tick=200, active=1, legal=legal)
    # No valid settings are supplied: neither path may invoke the script.
    assert choose_schedule_option("scripted_multi", names, x, legal, 1, 200, None, bank) == (
        "short_15",
        None,
    )
    assert choose_schedule_option("scripted_multi", names, x, legal, 1, 265, 0, bank) == (
        "neutral",
        None,
    )
    settings_path.write_text(json.dumps({**DEFAULT_CONFIG, "minimum_observed_free_height_m": 1.4}))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_script_parameters(settings_path, ref["sha256"], bank)
