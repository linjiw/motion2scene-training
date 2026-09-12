"""Recorded command traces must not fabricate causal inputs or later choices."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_script_perception_traces as traces


def fixture(monkeypatch):
    names = [
        f"corridor_{a:g}_{b:g}_{key}"
        for a, b in traces.script.BANDS
        for key in ("upper_hit", "unknown_fraction")
    ]
    observations = []
    for tick in range(1, 71):
        observations.append(
            dict(
                tick=tick,
                phase_s=tick / 50,
                active_before="neutral" if tick <= 50 else "early",
                capture_elapsed_s=(tick - 1) / 50,
                delivered_capture_elapsed_s=(tick - 1) / 50,
                selected_option_id="early" if tick >= 50 else "neutral",
                legal_mask=[True, True, False] if tick <= 50 else [False, True, False],
                features=[0.0, 1.0] * 4,
                root_pos_w=[tick / 100, 0.0, 0.8],
            )
        )
    value = dict(
        feature_names=names,
        observations=observations,
        phase_ticks=[15, 50, 70],
        timed_schedule_mode="scripted_multi",
        switches=[dict(tick=50, **{"from": "neutral"}, to="early")],
    )
    bank = SimpleNamespace(option_ids=["neutral", "early", "late"])
    monkeypatch.setattr(
        traces.script,
        "choose_sensor_schedule",
        lambda names, features, legal, tick, bank, settings: (
            "early" if tick == 50 else "neutral",
            dict(hazard_bands=[False] * 4),
        ),
    )
    return value, bank


def test_only_precommit_neutral_phases_are_wait_or_commit(monkeypatch):
    interface, bank = fixture(monkeypatch)
    value = traces.trace(interface, bank, {})
    assert [d["decision"] for d in value["decisions"]] == ["WAIT", "COMMIT"]
    assert value["net_planar_displacement_first_decision_to_commit_m"] == pytest.approx(0.35)
    assert value["precommit_series"][-1]["tick"] == 50
    assert all(s["active_before"] == "neutral" for s in value["precommit_series"])


def test_future_packet_cannot_explain_a_recorded_decision(monkeypatch):
    interface, bank = fixture(monkeypatch)
    interface["observations"][14]["delivered_capture_elapsed_s"] = 0.32
    with pytest.raises(ValueError, match="causal"):
        traces.trace(interface, bank, {})


def test_recomputed_action_must_match_actual_recording(monkeypatch):
    interface, bank = fixture(monkeypatch)
    interface["observations"][49]["selected_option_id"] = "neutral"
    with pytest.raises(ValueError, match="recorded command"):
        traces.trace(interface, bank, {})


def test_command_and_actual_switch_must_agree(monkeypatch):
    interface, bank = fixture(monkeypatch)
    interface["switches"][0]["tick"] = 49
    with pytest.raises(ValueError, match="executed switch"):
        traces.trace(interface, bank, {})


def test_missing_control_tick_is_not_silently_interpolated(monkeypatch):
    interface, bank = fixture(monkeypatch)
    del interface["observations"][10]
    with pytest.raises(ValueError, match="consecutive"):
        traces.trace(interface, bank, {})


def test_final_neutral_choice_does_not_claim_later_choices_are_retained(monkeypatch):
    interface, bank = fixture(monkeypatch)
    for observation in interface["observations"]:
        observation["active_before"] = "neutral"
        observation["selected_option_id"] = "neutral"
        observation["legal_mask"] = [True, True, False]
    interface["switches"] = []
    monkeypatch.setattr(
        traces.script,
        "choose_sensor_schedule",
        lambda names, features, legal, tick, bank, settings: ("neutral", {}),
    )
    value = traces.trace(interface, bank, {})
    assert [d["decision"] for d in value["decisions"]] == ["WAIT", "WAIT", "CONTINUE_NEUTRAL"]
    assert value["decisions"][-1]["later_registered_entry_ticks"] == []
