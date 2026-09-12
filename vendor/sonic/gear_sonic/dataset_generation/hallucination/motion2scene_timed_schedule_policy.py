"""Registry-bound sensor value readouts for repeated neutral commitments."""

import json

import numpy as np

from .motion2scene_closed_loop_policy import history_policy_features
from .motion2scene_schedule_script import choose_sensor_schedule
from .motion2scene_timed_history_policy import expected_feature_names as initial_feature_names
from .motion2scene_timed_options import checked_artifact, definition_digest
from .motion2scene_timed_schedule_learning import SCHEMA, schedule_layout

MODES = (
    "always_walk",
    "constant_option",
    "scripted_reference",
    "scripted_multi",
    "learned",
    "forced",
)
HISTORY_SECONDS = 2.0
HISTORY_FRAMES = 101


def expected_feature_names(option_count):
    if type(option_count) is not int or option_count < 2:
        raise ValueError("neutral and at least one complete schedule required")
    return (
        initial_feature_names()[:100]
        + tuple(f"active_option_{i}" for i in range(option_count))
        + tuple(f"option_{i}_legal" for i in range(option_count))
    )


def schedule_history_features(observation, grid, state, tick, active, legality, age_s):
    legality = np.asarray(legality)
    if legality.ndim != 1 or legality.dtype.kind != "b" or not 0 <= active < len(legality):
        raise ValueError("boolean schedule legality and explicit active index required")
    count = len(legality)
    names, values = history_policy_features(
        observation,
        grid,
        state,
        tick / 50,
        int(active != 0),
        [legality[0], bool(legality[1:].any())],
        age_s,
    )
    names += tuple(f"active_option_{i}" for i in range(count))
    names += tuple(f"option_{i}_legal" for i in range(count))
    values = np.concatenate((values, np.eye(count)[active], legality)).astype(np.float32)
    if names != expected_feature_names(count) or not np.isfinite(values).all():
        raise ValueError("exact29-joint100+2K sensor/state schema required")
    return names, values


def validate_schedule_policy(policy, bank):
    phases, qualified, _ = schedule_layout(bank)
    count, size = len(bank.option_ids), 100 + 2 * len(bank.option_ids)
    if not isinstance(policy, dict) or (
        str(policy.get("schema_version")) != SCHEMA
        or tuple(policy.get("feature_names", ())) != expected_feature_names(count)
        or tuple(policy.get("option_ids", ())) != bank.option_ids
        or str(policy.get("request_digest")) != definition_digest(bank.request)
        or not np.array_equal(policy.get("classes"), np.arange(count))
        or not np.array_equal(policy.get("phase_ticks"), phases)
        or not np.array_equal(policy.get("qualified_mask"), qualified)
    ):
        raise ValueError("schedule policy schema/phase/option/request binding mismatch")
    shapes = {
        "mean": (len(phases), size),
        "std": (len(phases), size),
        "weights": (len(phases), size, count),
        "bias": (len(phases), count),
    }
    for key, shape in shapes.items():
        values = np.asarray(policy.get(key))
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError("invalid schedule value arrays")
    trained = np.asarray(policy.get("trained_mask"))
    if (
        trained.shape != qualified.shape
        or trained.dtype.kind != "b"
        or np.any(trained & ~qualified)
        or np.any(policy["std"] <= 0)
        or not trained[:, 0].all()
        or not (trained.sum(axis=1) >= 2).all()
    ):
        raise ValueError("legal trained heads and positive normalization required")
    if np.asarray(policy["phase_ticks"]).dtype.kind not in "iu":
        raise ValueError("exact integer decision ticks required")
    return policy


def load_schedule_policy(path, sha256, bank):
    path = checked_artifact({"path": str(path), "sha256": sha256})
    with np.load(path, allow_pickle=False) as values:
        model = {key: values[key].copy() for key in values.files}
    return validate_schedule_policy(model, bank)


def load_script_parameters(path, sha256, bank):
    settings = json.loads(checked_artifact({"path": str(path), "sha256": sha256}).read_text())
    phases, qualified, _ = schedule_layout(bank)
    # Validate the same explicit settings/schema used at execution, without a
    # physical outcome or a privileged scene field.
    choose_sensor_schedule(
        expected_feature_names(len(bank.option_ids)),
        np.zeros(100 + 2 * len(bank.option_ids)),
        qualified[0],
        int(phases[0]),
        bank,
        settings,
    )
    return settings


def choose_schedule_option(
    mode,
    names,
    features,
    legality,
    active,
    tick,
    mandatory,
    bank,
    *,
    policy=None,
    script_parameters=None,
    preferred_option_id="neutral",
    preferred_reference_id="sustained",
):
    """Select a complete available schedule; a return obligation bypasses learning."""
    phases, qualified, _ = schedule_layout(bank)
    names, x, legal = tuple(names), np.asarray(features), np.asarray(legality)
    count = len(bank.option_ids)
    if mode not in MODES or mode == "forced":
        raise ValueError("online modes required; forced schedules use the parent selector")
    if (
        names != expected_feature_names(count)
        or x.shape != (len(names),)
        or not np.isfinite(x).all()
        or legal.shape != (count,)
        or legal.dtype.kind != "b"
        or active not in range(count)
        or not legal[active]
    ):
        raise ValueError("invalid schedule feature or current legal state")
    if mandatory is not None:
        if mandatory != 0 or active == 0:
            raise ValueError("only a current schedule return can be mandatory")
        return "neutral", None
    if legal.sum() == 1:
        return bank.option_ids[active], None
    matches = np.flatnonzero(phases == tick)
    if active != 0 or len(matches) != 1 or np.any(legal & ~qualified[matches[0]]):
        raise ValueError("unsupported neutral decision phase or schedule")
    phase = int(matches[0])
    if mode == "learned":
        validate_schedule_policy(policy, bank)
        if np.any(legal & ~policy["trained_mask"][phase]):
            raise ValueError("a legal option lacks a measured trained phase head")
        values = ((x - policy["mean"][phase]) / policy["std"][phase]) @ policy["weights"][
            phase
        ] + policy["bias"][phase]
    else:
        preferred = 0
        if mode == "scripted_multi":
            if script_parameters is None:
                raise ValueError("scripted_multi requires explicit SHA-bound settings")
            selected, _ = choose_sensor_schedule(names, x, legal, tick, bank, script_parameters)
            preferred = bank.option_ids.index(selected)
        elif mode == "constant_option":
            if preferred_option_id not in bank.option_ids:
                raise ValueError("constant baseline requires an exact qualified schedule ID")
            index = bank.option_ids.index(preferred_option_id)
            if legal[index]:
                preferred = index
        elif mode == "scripted_reference":
            if preferred_reference_id not in [
                ref["reference_id"] for ref in bank.request["references"][1:]
            ]:
                raise ValueError("scripted baseline requires a qualified adapting reference")
            upper = any(
                value > 0
                for name, value in zip(names, x, strict=True)
                if name.endswith("upper_hit")
            )
            available = [
                i
                for i, option in enumerate(bank.request["options"], 1)
                if legal[i] and option["reference_id"] == preferred_reference_id
            ]
            if upper and available:
                preferred = available[0]
        values = np.asarray([float(i == preferred) for i in range(count)])
    if not np.isfinite(values).all():
        raise ValueError("nonfinite schedule value readout")
    action = int(np.argmax(np.where(legal, values, -np.inf)))
    return bank.option_ids[action], values.tolist()


def audit_executed_schedule(bank, observations, switches, mode, forced_option_id="neutral"):
    """Audit one complete entry/return without treating repeated decisions as entries."""
    schedule_layout(bank)
    if mode not in MODES or forced_option_id not in bank.option_ids:
        raise ValueError("registered mode and complete forced schedule ID required")
    ticks = np.asarray([row["tick"] for row in observations])
    complete = np.array_equal(ticks, np.arange(1, bank.frame_count))
    chosen = switches[0]["to"] if switches else "neutral"
    option = next((item for item in bank.request["options"] if item["option_id"] == chosen), None)
    expected = (
        []
        if chosen == "neutral"
        else (
            [(option["entry_tick"], "neutral", chosen), (option["return_tick"], chosen, "neutral")]
            if option is not None
            else None
        )
    )
    actual = [(item["tick"], item["from"], item["to"]) for item in switches]
    switch_schedule = expected is not None and actual == expected
    active = np.full(len(ticks), "neutral", dtype=object)
    before = active.copy()
    if option is not None:
        active[(ticks >= option["entry_tick"]) & (ticks < option["return_tick"])] = chosen
        before[(ticks > option["entry_tick"]) & (ticks <= option["return_tick"])] = chosen
    timeline = np.array_equal(active, [row["active"] for row in observations])
    preaction = np.array_equal(before, [row["active_before"] for row in observations])
    names = expected_feature_names(len(bank.option_ids))
    feature_active = np.asarray([row["features"] for row in observations])
    if feature_active.shape == (len(ticks), len(names)):
        expected_active = np.asarray(
            [np.eye(len(bank.option_ids))[bank.option_ids.index(value)] for value in before]
        )
        preaction = preaction and np.array_equal(
            feature_active[:, 100 : 100 + len(bank.option_ids)], expected_active
        )
    else:
        preaction = False
    flags = ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
    guards = all(
        all(item.get(key) is True for key in flags)
        and np.isfinite([item["joint_jump_rad"], item["root_jump_m"]]).all()
        and 0 <= item["joint_jump_rad"] <= 0.05
        and 0 <= item["root_jump_m"] <= 0.01
        for item in switches
    )
    no_refusals = all(row["transition"]["allowed"] is True for row in observations)
    forced_match = mode != "forced" or chosen == forced_option_id
    return {
        "complete_ticks": bool(complete),
        "switches_match_declared_schedule": bool(switch_schedule),
        "active_timeline_matches": bool(timeline),
        "preaction_timeline_and_features_match": bool(preaction),
        "reference_guards_and_state_unchanged": bool(guards),
        "no_refusals": no_refusals,
        "forced_choice_matches": forced_match,
        "chosen_option_id": chosen,
        "valid": bool(
            complete
            and switch_schedule
            and timeline
            and preaction
            and guards
            and no_refusals
            and forced_match
        ),
    }
