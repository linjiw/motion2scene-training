"""Sensor-only duration selection for one verified, finite three-option bank."""

import numpy as np

from .motion2scene_closed_loop_policy import history_policy_features
from .motion2scene_timed_options import checked_artifact, definition_digest

SCHEMA = "motion2scene_timed_history_value_v1"
MODES = (
    "always_walk",
    "constant_short",
    "constant_sustained",
    "scripted_sustained",
    "learned",
    "forced",
)
ENTRY_TICK = 15


def validate_history_bank(bank, mode):
    """The initial learner supports exactly one measured entry and two profiles."""
    if mode not in MODES:
        raise ValueError("unknown timed history mode")
    if mode != "forced" and not bank.online_verified:
        raise ValueError("online sensor policies require a physically verified timed registry")
    options = bank.request["options"]
    if (
        len(bank.option_ids) != 3
        or [value["profile_label"] for value in options] != ["short", "sustained"]
        or any(value["entry_tick"] != ENTRY_TICK for value in options)
        or bank.request["max_entries_per_episode"] != 1
    ):
        raise ValueError(
            "initial106D interface requires short/sustained at the single tick15 entry"
        )
    return bank


def expected_feature_names():
    names = []
    for band in ("0_0.75", "0.75_1.5", "1.5_2.5", "2.5_4"):
        names.extend(
            f"corridor_{band}_{name}"
            for name in (
                "floor_fraction",
                "ceiling_fraction",
                "minimum_ceiling_m",
                "maximum_floor_m",
                "upper_hit",
                "lower_free_fraction",
                "unknown_fraction",
            )
        )
    for name, size in (
        ("projected_gravity_b", 3),
        ("root_lin_vel_w", 3),
        ("root_ang_vel_w", 3),
        ("dof_pos", 29),
        ("dof_vel", 29),
    ):
        names.extend(f"{name}_{i}" for i in range(size))
    names.extend(("phase_s", "active_skill", "observation_age_s", "walk_legal", "adapt_legal"))
    names.extend(f"active_option_{i}" for i in range(3))
    names.extend(f"option_{i}_legal" for i in range(3))
    return tuple(names)


def timed_history_features(observation, grid, state, tick, active, legality, age_s):
    legality = np.asarray(legality)
    if legality.shape != (3,) or legality.dtype.kind != "b" or active not in (0, 1, 2):
        raise ValueError("three boolean legal options and an explicit active index required")
    names, features = history_policy_features(
        observation,
        grid,
        state,
        tick / 50,
        int(active != 0),
        [legality[0], bool(legality[1:].any())],
        age_s,
    )
    names += tuple(f"active_option_{i}" for i in range(3))
    names += tuple(f"option_{i}_legal" for i in range(3))
    features = np.concatenate((features, np.eye(3)[active], legality)).astype(np.float32)
    if names != expected_feature_names() or features.shape != (106,):
        raise ValueError("timed history requires exact106 named29-joint sensor/state features")
    return names, features


def validate_timed_policy(policy, bank):
    validate_history_bank(bank, "learned")
    if not isinstance(policy, dict):
        raise ValueError("timed value policy is required")
    if (
        str(policy.get("schema_version")) != SCHEMA
        or tuple(policy.get("feature_names", ())) != expected_feature_names()
        or tuple(policy.get("option_ids", ())) != bank.option_ids
        or str(policy.get("request_digest")) != definition_digest(bank.request)
        or np.asarray(policy.get("entry_tick")).shape != ()
        or np.asarray(policy.get("entry_tick")).dtype.kind not in "iu"
        or int(policy["entry_tick"]) != ENTRY_TICK
        or not np.array_equal(policy.get("classes"), np.arange(3))
    ):
        raise ValueError("timed policy schema/option/schedule binding mismatch")
    for key, shape in {"mean": (106,), "std": (106,), "weights": (106, 3), "bias": (3,)}.items():
        values = np.asarray(policy.get(key))
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError("invalid timed value parameter arrays")
    if np.any(policy["std"] <= 0):
        raise ValueError("positive timed feature normalization required")
    return policy


def load_timed_policy(path, sha256, bank):
    checked = checked_artifact({"path": str(path), "sha256": sha256})
    with np.load(checked, allow_pickle=False) as data:
        policy = {key: data[key].copy() for key in data.files}
    return validate_timed_policy(policy, bank)


def choose_timed_option(mode, names, features, mask, active, tick, mandatory, bank, policy=None):
    """Readout only; the physical parent applies current-jump and clock guards.

    Exact return is a schedule obligation. A refused return must be recorded by
    that parent, never converted to an apparently successful hold.
    """
    validate_history_bank(bank, mode)
    features, mask = np.asarray(features), np.asarray(mask)
    if (
        tuple(names) != expected_feature_names()
        or features.shape != (106,)
        or not np.isfinite(features).all()
        or mask.shape != (3,)
        or mask.dtype.kind != "b"
        or active not in (0, 1, 2)
        or not mask[active]
    ):
        raise ValueError("invalid named106D observation or current legal state")
    if mode == "forced":
        raise ValueError("forced schedules use the explicit qualification parent selector")
    if mandatory is not None:
        if mandatory != 0 or active == 0:
            raise ValueError("only a guarded return to neutral can be mandatory")
        return "neutral", None
    if np.count_nonzero(mask) == 1:
        return bank.option_ids[active], None
    if tick != ENTRY_TICK or active != 0:
        raise ValueError("unqualified learned decision phase or active option")
    if mode == "learned":
        validate_timed_policy(policy, bank)
        values = ((features - policy["mean"]) / policy["std"]) @ policy["weights"] + policy["bias"]
    else:
        preferred = {"always_walk": 0, "constant_short": 1, "constant_sustained": 2}.get(mode)
        if mode == "scripted_sustained":
            # Only actual occupied endpoint evidence; unknown cells are not hits.
            preferred = (
                2
                if any(
                    value > 0
                    for name, value in zip(names, features, strict=True)
                    if name.endswith("upper_hit")
                )
                else 0
            )
        values = np.asarray([float(index == preferred) for index in range(3)])
    if not np.isfinite(values).all():
        raise ValueError("nonfinite duration values")
    return bank.option_ids[int(np.argmax(np.where(mask, values, -np.inf)))], values.tolist()


def audit_executed_schedule(bank, observations, switches, mode, forced_option_id="neutral"):
    """Recompute complete schedule chronology independently of runtime claims."""
    validate_history_bank(bank, mode)
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
    if option is not None:
        active[(ticks >= option["entry_tick"]) & (ticks < option["return_tick"])] = chosen
    timeline = np.array_equal(active, [row["active"] for row in observations])
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
        "switches_match_declared_schedule": switch_schedule,
        "active_timeline_matches": bool(timeline),
        "reference_guards_and_state_unchanged": bool(guards),
        "no_refusals": no_refusals,
        "forced_choice_matches": forced_match,
        "chosen_option_id": chosen,
        "valid": bool(
            complete and switch_schedule and timeline and guards and no_refusals and forced_match
        ),
    }


def fit_timed_value_policy(bank, features, passed, passage_time_s, admitted, legality, *, l2=1e-6):
    """One fixed-phase head, supervised only by complete measured schedule rows."""
    # Lazy import avoids changing the existing K5 runtime or its model schema.
    from .motion2scene_value_imitation import fit_value_policy

    validate_history_bank(bank, "learned")
    names = expected_feature_names()
    x, legal = np.asarray(features), np.asarray(legality)
    if (
        x.ndim != 2
        or x.shape[1] != 106
        or not len(x)
        or not np.isfinite(x).all()
        or legal.shape != (len(x), 3)
        or legal.dtype.kind != "b"
        or not np.allclose(x[:, names.index("phase_s")], ENTRY_TICK / 50, atol=1e-7, rtol=0)
        or not np.array_equal(x[:, -6:-3], np.tile([1, 0, 0], (len(x), 1)))
        or not np.array_equal(x[:, -3:], legal)
    ):
        raise ValueError(
            "complete106D neutral tick15 decisions with matching measured legality required"
        )
    model, report = fit_value_policy(
        x,
        names,
        bank.option_ids,
        passed,
        passage_time_s,
        admitted,
        legal,
        l2=l2,
    )
    model.update(
        schema_version=np.array(SCHEMA),
        entry_tick=np.array(ENTRY_TICK),
        request_digest=np.array(definition_digest(bank.request)),
    )
    validate_timed_policy(model, bank)
    report.update(
        scope="finite measured development table; no physical generalization evidence",
        entry_tick=ENTRY_TICK,
        request_digest=definition_digest(bank.request),
    )
    return model, report
