"""Shared sensor/state features and guarded single-encounter policy readouts."""

import hashlib
from pathlib import Path

import numpy as np

from .motion2scene_action_contract import command_permission

POLICY_SCHEMA = "motion2scene_history_linear_v1"


def legal_skill_mask(active, phase_s, joint_jump, root_jump):
    return np.array(
        [command_permission(skill, active, phase_s, joint_jump, root_jump)[0] for skill in (0, 1)],
        dtype=bool,
    )


def history_policy_features(observation, grid, state, phase_s, active, legality, age_s):
    """Fixed corridor summaries followed by measured proprioception and legality.

    Heights are relative to current root, occupancy comes only from delivered
    range rays. Unknown cells are counted explicitly. Summaries are deliberately
    compact; they do not preserve every distinction in the full history grid.
    """
    shape = grid.shape
    centers = [
        grid.lower_m[axis] + (np.arange(size) + 0.5) * grid.resolution_m
        for axis, size in enumerate(shape)
    ]
    x, y, z = centers
    names, values = [], []
    for start, end in ((0, 0.75), (0.75, 1.5), (1.5, 2.5), (2.5, 4)):
        horizontal = (x[:, None] >= start) & (x[:, None] < end) & (abs(y[None, :]) <= 0.5)
        if not horizontal.any():
            raise ValueError("history grid does not cover a policy corridor band")
        floor_mask = horizontal & observation["floor_observed"]
        ceiling_mask = horizontal & observation["ceiling_observed"]
        upper = horizontal[:, :, None] & (z[None, None, :] >= 0.25) & (z[None, None, :] < 0.9)
        lower = horizontal[:, :, None] & (z[None, None, :] >= -0.5) & (z[None, None, :] < 0.25)
        band = (
            ("floor_fraction", float(floor_mask.sum() / horizontal.sum())),
            ("ceiling_fraction", float(ceiling_mask.sum() / horizontal.sum())),
            (
                "minimum_ceiling_m",
                (
                    float(observation["ceiling_height_m"][ceiling_mask].min())
                    if ceiling_mask.any()
                    else 0
                ),
            ),
            (
                "maximum_floor_m",
                float(observation["floor_height_m"][floor_mask].max()) if floor_mask.any() else 0,
            ),
            ("upper_hit", float(observation["occupied"][upper].any())),
            ("lower_free_fraction", float(observation["free_sampled"][lower].mean())),
            ("unknown_fraction", float(observation["unknown"][horizontal].mean())),
        )
        for name, value in band:
            names.append(f"corridor_{start:g}_{end:g}_{name}")
            values.append(value)
    for name in ("projected_gravity_b", "root_lin_vel_w", "root_ang_vel_w", "dof_pos", "dof_vel"):
        array = np.asarray(state[name], dtype=float)
        if array.ndim != 1 or not array.size:
            raise ValueError("invalid proprioceptive vector")
        names.extend(f"{name}_{i}" for i in range(len(array)))
        values.extend(array)
    if np.asarray(legality).shape != (2,) or active not in (0, 1) or age_s < 0:
        raise ValueError("invalid action legality or observation age")
    names.extend(["phase_s", "active_skill", "observation_age_s", "walk_legal", "adapt_legal"])
    values.extend([phase_s, active, age_s, *legality])
    features = np.asarray(values, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite policy features")
    return tuple(names), features


def load_history_policy(path, expected_sha256):
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = str(expected_sha256).removeprefix("sha256:")
    if actual != expected:
        raise ValueError("history policy SHA256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        policy = {key: archive[key].copy() for key in archive.files}
    required = {"schema_version", "feature_names", "mean", "std", "weights", "bias", "classes"}
    if not required.issubset(policy) or str(policy["schema_version"]) != POLICY_SCHEMA:
        raise ValueError("unsupported history policy schema")
    size = len(policy["feature_names"])
    shapes = {"mean": (size,), "std": (size,), "weights": (size, 2), "bias": (2,)}
    if any(
        policy[key].shape != shape or not np.isfinite(policy[key]).all()
        for key, shape in shapes.items()
    ):
        raise ValueError("invalid history policy parameter arrays")
    if np.any(policy["std"] <= 0) or not np.array_equal(policy["classes"], [0, 1]):
        raise ValueError("invalid normalization or skill classes")
    if policy["feature_names"].ndim != 1 or len(set(policy["feature_names"].tolist())) != size:
        raise ValueError("invalid feature names")
    return policy


def choose_history_skill(mode, names, features, legality, active, phase_s, policy=None):
    """Return one supported skill, masking illegal alternatives before choice."""
    legality = np.asarray(legality, dtype=bool)
    features = np.asarray(features, dtype=float)
    if legality.shape != (2,) or active not in (0, 1) or not legality[active]:
        raise ValueError("current skill must remain legal")
    if features.shape != (len(names),) or not np.isfinite(features).all():
        raise ValueError("invalid policy feature vector")
    if mode == "learned":
        if policy is None or tuple(policy["feature_names"]) != tuple(names):
            raise ValueError("history policy feature schema mismatch")
        logits = ((features - policy["mean"]) / policy["std"]) @ policy["weights"] + policy["bias"]
    elif mode in ("scripted", "always_walk", "always_adapt"):
        if mode == "always_walk":
            preferred = 0
        elif mode == "always_adapt":
            preferred = 1 if phase_s < 3.3 else 0
        else:
            observed = dict(zip(names, features, strict=True))
            # A development baseline, using observed upper occupied cells only.
            # It is not a guaranteed collision-free response to an arbitrary wall.
            upper = any(value for name, value in observed.items() if name.endswith("upper_hit"))
            preferred = int(upper) if phase_s < 3.3 else 0
        logits = np.array([float(preferred == skill) for skill in (0, 1)])
    else:
        raise ValueError("unknown closed-loop policy mode")
    if not np.isfinite(logits).all():
        raise ValueError("nonfinite policy logits")
    masked = np.where(legality, logits, -np.inf)
    requested = int(np.argmax(masked))
    return requested, np.asarray(logits).tolist()
