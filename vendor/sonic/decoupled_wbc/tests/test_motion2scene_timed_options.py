import copy
import hashlib
import json

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (
    EVIDENCE_SCHEMA,
    HEIGHT_MEASUREMENT,
    REQUEST_SCHEMA,
    REQUIRED_CHECKS,
    TimedOptionState,
    apply_timed_request,
    assert_loaded_bank,
    definition_digest,
    guard_before_reference_advance,
    legal_timed_actions,
    load_verified_registry,
    make_verified_registry,
    seconds_to_tick,
    validate_evidence,
    validate_request,
)


def artifact(path):
    return {
        "path": str(path.resolve()),
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def fixture(tmp_path):
    refs = []
    for name in ("neutral", "short", "sustained"):
        path = tmp_path / (name + ".pkl")
        path.write_text("test fixture only " + name)
        refs.append(
            {
                "reference_id": name,
                "motion": artifact(path),
                "expected_loaded_frames": 299,
                "construction": "generated" if name == "neutral" else "authored_local_crouch",
            }
        )
        if name != "neutral":
            refs[-1]["parent_motion"] = refs[0]["motion"]
    request = {
        "schema": REQUEST_SCHEMA,
        "split": "development",
        "request_id": "test_only_not_physics",
        "controller": refs[0]["motion"],
        "reference_fps": 50,
        "expected_loaded_frames": 299,
        "max_entries_per_episode": 1,
        "references": refs,
        "height_measurement": HEIGHT_MEASUREMENT,
        "support_plane_z_m": 0.0,
        "options": [
            {
                "option_id": name + "_e015_r265",
                "reference_id": name,
                "profile_label": name,
                "entry_tick": 15,
                "return_tick": 265,
                "recovery_end_tick": 290,
                "maintenance": {
                    "start_tick": 100 if name == "short" else 70,
                    "end_tick": 160 if name == "short" else 220,
                    "maximum_body_height_m": 1.25,
                },
            }
            for name in ("short", "sustained")
        ],
    }
    return request


def evidence(request, option_id):
    ticks = np.arange(1, 299)
    active = np.full(len(ticks), "neutral", dtype=object)
    switches = []
    if option_id != "neutral":
        active[(ticks >= 15) & (ticks < 265)] = option_id
        for tick, left, right in ((15, "neutral", option_id), (265, option_id, "neutral")):
            switches.append(
                {
                    "tick": tick,
                    "from": left,
                    "to": right,
                    "joint_jump_rad": 0.02,
                    "root_jump_m": 0.001,
                    "root_state_unchanged": True,
                    "joint_state_unchanged": True,
                    "clock_unchanged": True,
                }
            )
    return {
        "schema": EVIDENCE_SCHEMA,
        "request_digest": definition_digest(request),
        "controller": request["controller"],
        "reference_motions": [r["motion"] for r in request["references"]],
        "option_id": option_id,
        "checks": dict.fromkeys(REQUIRED_CHECKS, True),
        "loaded_reference_ids": [r["reference_id"] for r in request["references"]],
        "loaded_reference_frames": [299] * 3,
        "fps": 50,
        "command_ticks": ticks.tolist(),
        "active_option_ids": active.tolist(),
        "recorded_physics_steps": 1192,
        "matched_prefix_frames": 15,
        "executed_body_height_m": [1.24] * 298,
        "switches": switches,
        "height_measurement": HEIGHT_MEASUREMENT,
        "support_plane_z_m": 0.0,
        "artifacts": dict.fromkeys(
            (
                "trajectory",
                "interface",
                "physics_contacts",
                "clock_audit",
                "prefix_audit",
                "geometry_audit",
            ),
            request["controller"],
        ),
    }


def test_request_cannot_be_used_online_and_loaded_horizon_is_measured(tmp_path):
    bank = validate_request(fixture(tmp_path))
    with pytest.raises(ValueError, match="forced qualification"):
        legal_timed_actions(bank, TimedOptionState(), 15, [0] * 3, [0] * 3)
    assert_loaded_bank(bank, ["neutral", "short", "sustained"], [299] * 3, 50)
    with pytest.raises(ValueError, match="equal-duration"):
        assert_loaded_bank(bank, ["neutral", "short", "sustained"], [299, 199, 299], 50)
    assert seconds_to_tick(5.3) == 265
    with pytest.raises(ValueError, match="exact"):
        seconds_to_tick(5.31)


def test_entry_maintenance_exact_return_and_no_second_entry(tmp_path):
    request = fixture(tmp_path)
    # A second schedule sharing a reference still cannot be entered after recovery.
    request["options"][1].update(entry_tick=280, return_tick=293, recovery_end_tick=297)
    request["options"][1]["maintenance"].update(start_tick=281, end_tick=290)
    bank = validate_request(request)
    state = TimedOptionState()
    state, decision = apply_timed_request(
        bank, state, 15, bank.option_ids[1], [0] * 3, [0] * 3, qualification_only=True
    )
    assert decision["allowed"] and state.entries == 1
    state, decision = apply_timed_request(
        bank, state, 100, bank.option_ids[2], [0] * 3, [0] * 3, qualification_only=True
    )
    assert not decision["allowed"] and state.active == bank.option_ids[1]
    state, decision = apply_timed_request(
        bank, state, 265, "neutral", [0] * 3, [0] * 3, qualification_only=True
    )
    assert decision["allowed"] and decision["mandatory_return"]
    state, decision = apply_timed_request(
        bank, state, 280, bank.option_ids[2], [0] * 3, [0] * 3, qualification_only=True
    )
    assert not decision["allowed"] and state.active == "neutral"


def test_unqualified_tick_jump_refusal_and_reference_wrap_are_not_motion(tmp_path):
    bank = validate_request(fixture(tmp_path))
    state, decision = apply_timed_request(
        bank, TimedOptionState(), 14, bank.option_ids[1], [0] * 3, [0] * 3, qualification_only=True
    )
    assert not decision["allowed"]
    state, decision = apply_timed_request(
        bank, state, 15, bank.option_ids[1], [0, 0.051, 0], [0] * 3, qualification_only=True
    )
    assert not decision["allowed"]
    active = TimedOptionState(bank.option_ids[1], 1, 264)
    active, decision = apply_timed_request(
        bank, active, 265, "neutral", [0.051, 0, 0], [0] * 3, qualification_only=True
    )
    assert not decision["allowed"] and active.active == bank.option_ids[1]
    with pytest.raises(RuntimeError, match="return was missed"):
        legal_timed_actions(bank, active, 266, [0] * 3, [0] * 3, qualification_only=True)
    guard_before_reference_advance(297, 299)
    with pytest.raises(RuntimeError, match="not a protective stop"):
        guard_before_reference_advance(298, 299)


def test_whole_episode_evidence_rejects_missing_tail_and_wrong_schedule(tmp_path):
    request = fixture(tmp_path)
    record = evidence(request, request["options"][0]["option_id"])
    assert validate_evidence(request, record)["qualified"]
    for modification in ("tail", "return", "maintenance", "prefix", "contact"):
        bad = copy.deepcopy(record)
        if modification == "tail":
            bad["command_ticks"] = bad["command_ticks"][:-1]
        if modification == "return":
            bad["switches"][-1]["tick"] = 264
        if modification == "maintenance":
            bad["executed_body_height_m"][120] = 1.3
        if modification == "prefix":
            bad["matched_prefix_frames"] = 14
        if modification == "contact":
            bad["checks"]["all_contacts_audited"] = False
        with pytest.raises(ValueError):
            validate_evidence(request, bad)


def test_registry_promotion_requires_all_bound_evidence_and_stays_separate(tmp_path):
    request = fixture(tmp_path)
    refs = []
    for option_id in ("neutral", *(o["option_id"] for o in request["options"])):
        path = tmp_path / (option_id + ".json")
        path.write_text(json.dumps(evidence(request, option_id)))
        refs.append(artifact(path))
    with pytest.raises(ValueError, match="every online schedule"):
        make_verified_registry(request, refs[:2])
    registry = make_verified_registry(request, refs)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    bank = load_verified_registry(path, artifact(path)["sha256"])
    assert bank.online_verified
    mask, required = legal_timed_actions(bank, TimedOptionState(), 15, [0] * 3, [0] * 3)
    assert mask.all() and required is None
    # These are synthetic fixtures testing validation, not physical evidence.
    (tmp_path / "short.pkl").write_text("tampered fixture")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_registry(path, artifact(path)["sha256"])
