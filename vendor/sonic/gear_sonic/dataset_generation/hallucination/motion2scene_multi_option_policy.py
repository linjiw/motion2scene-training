"""Pure guards and sensor-policy readout for a phase-aligned motion library."""

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "motion2scene_history_multioption_v1"


def load_option_registry(path, expected_sha256):
    data = Path(path).read_bytes()
    if "sha256:" + hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("option registry hash mismatch")
    registry = json.loads(data)
    entries = registry["references"]
    if (
        registry["schema"] != "motion2scene_qualified_option_registry_v1"
        or len(entries) < 2
        or entries[0]["name"] != "neutral"
        or len({entry["name"] for entry in entries}) != len(entries)
    ):
        raise ValueError("registry requires neutral followed by distinct qualified alternatives")
    for entry in entries:
        reference = entry["motion"]
        actual = "sha256:" + hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest()
        if actual != reference["sha256"]:
            raise ValueError("registered motion hash mismatch")
        ticks = entry["qualified_entry_times_s"]
        if not ticks or any(
            not np.isfinite(t) or not 0.2 <= t <= 0.4 or abs(t * 50 - round(t * 50)) > 1e-8
            for t in ticks
        ):
            raise ValueError("registry entry ticks must have finite 50 Hz qualification")
        evidence = entry["qualification"]
        raw = Path(evidence["path"]).read_bytes()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != evidence["sha256"]:
            raise ValueError("qualification result hash mismatch")
        rows = json.loads(raw)["rows"]
        for tick in ticks:
            matching = [
                row
                for row in rows
                if row["source"] == registry["source"]
                and row["option"]["reference"] == entry["name"]
                and abs(row["option"]["entry_time_s"] - tick) < 1e-8
                and row["qualified"]
            ]
            if not matching:
                raise ValueError("registered entry lacks successful matched physical qualification")
    return registry


def legal_option_mask(active, phase_s, joint_jumps, root_jumps, entry_times):
    """Allow only qualified 0->k entries and k->0 returns; no direct k->j changes."""
    joint, root = np.asarray(joint_jumps), np.asarray(root_jumps)
    count = len(entry_times)
    if (
        not isinstance(active, (int, np.integer))
        or not 0 <= active < count
        or joint.shape != (count,)
        or root.shape != (count,)
        or not np.isfinite(joint).all()
        or not np.isfinite(root).all()
        or (joint < 0).any()
        or (root < 0).any()
        or not np.isfinite(phase_s)
        or phase_s < 0
    ):
        raise ValueError("invalid option guard input")
    mask = np.zeros(count, dtype=bool)
    mask[active] = True
    for requested in range(count):
        if requested == active or (requested != 0 and active != 0):
            continue
        if requested:
            phase_allowed = 0.2 <= phase_s <= 0.4 and any(
                abs(phase_s - tick) < 1e-8 for tick in entry_times[requested]
            )
        else:
            phase_allowed = 3.3 <= phase_s <= 3.5
        mask[requested] = phase_allowed and joint[requested] <= 0.05 and root[requested] <= 0.01
    return mask


def extend_option_features(names, features, active, legality):
    count = len(legality)
    if not 0 <= active < count:
        raise ValueError("invalid active option")
    extra_names = [f"active_option_{i}" for i in range(count)]
    extra_names += [f"option_{i}_legal" for i in range(count)]
    extra_values = [int(active == i) for i in range(count)] + list(legality)
    return tuple(names) + tuple(extra_names), np.r_[features, extra_values].astype(np.float32)


def load_multi_policy(path, expected_sha256, option_names):
    raw = Path(path).read_bytes()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("multi-option policy hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        policy = {key: data[key].copy() for key in data.files}
    # Lazy import keeps the existing value fitter's SCHEMA dependency acyclic.
    from .motion2scene_phase_value_imitation import SCHEMA as PHASE_SCHEMA, validate_phase_model

    if str(policy["schema_version"]) == PHASE_SCHEMA:
        return validate_phase_model(policy, option_names)
    count, size = len(option_names), len(policy["feature_names"])
    if (
        str(policy["schema_version"]) != SCHEMA
        or list(policy["option_ids"]) != option_names
        or not np.array_equal(policy["classes"], np.arange(count))
    ):
        raise ValueError("policy and qualified option registry differ")
    for name, shape in {
        "mean": (size,),
        "std": (size,),
        "weights": (size, count),
        "bias": (count,),
    }.items():
        if policy[name].shape != shape or not np.isfinite(policy[name]).all():
            raise ValueError("malformed multi-option policy")
    if (policy["std"] <= 0).any():
        raise ValueError("policy normalization must be positive")
    return policy


def choose_option(mode, names, features, legality, active, phase_s, preferred, policy=None):
    mask = np.asarray(legality, dtype=bool)
    if mask.ndim != 1 or not 0 <= active < len(mask) or not mask[active]:
        raise ValueError("active option must be legal")
    if not 0 <= preferred < len(mask):
        raise ValueError("preferred option outside registry")
    if mode == "learned":
        if policy is None or tuple(policy["feature_names"]) != tuple(names):
            raise ValueError("policy feature schema mismatch")
        from .motion2scene_phase_value_imitation import SCHEMA as PHASE_SCHEMA, choose_phase_option

        if str(policy.get("schema_version", "")) == PHASE_SCHEMA:
            # A forced hold and guarded recovery have no learned action value.
            # Consequential decisions require a supported qualified phase/head.
            if active and phase_s >= 3.3:
                return (0 if mask[0] else active), None
            if mask.sum() == 1:
                return active, None
            if active != 0:
                raise ValueError("phase-value learner supports only neutral-state entries")
            return choose_phase_option(policy, names, features, mask, phase_s)
        logits = ((features - policy["mean"]) / policy["std"]) @ policy["weights"] + policy["bias"]
    elif mode in ("scripted", "always_walk", "always_adapt"):
        upper = any(
            value for name, value in zip(names, features, strict=True) if name.endswith("upper_hit")
        )
        desired = preferred if mode == "always_adapt" or (mode == "scripted" and upper) else 0
        logits = np.array([float(i == desired) for i in range(len(mask))])
    else:
        raise ValueError("unknown multi-option mode")
    # The installed option termination is mandatory and scene independent.
    if active and phase_s >= 3.3:
        requested = 0 if mask[0] else active
    else:
        requested = int(np.argmax(np.where(mask, logits, -np.inf)))
    return requested, np.asarray(logits).tolist()
