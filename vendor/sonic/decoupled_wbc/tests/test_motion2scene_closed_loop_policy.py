import hashlib

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_closed_loop_policy import (
    POLICY_SCHEMA,
    choose_history_skill,
    history_policy_features,
    legal_skill_mask,
    load_history_policy,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import (
    FloorCeilingHistory,
    SensorRay,
)


def inputs(occupied=False):
    mapper = FloorCeilingHistory()
    if occupied:
        direction = np.array([1, 0, 0.5]) / np.linalg.norm([1, 0, 0.5])
        mapper.push([SensorRay((0, 0, 0), direction, 3, np.linalg.norm([1, 0, 0.5]))], 0)
    observed = mapper.snapshot((0, 0, 0), (1, 0, 0, 0), 0)
    state = {
        "projected_gravity_b": [0, 0, -1],
        "root_lin_vel_w": [0, 0, 0],
        "root_ang_vel_w": [0, 0, 0],
        "dof_pos": [0] * 29,
        "dof_vel": [0] * 29,
    }
    return history_policy_features(observed, mapper.grid, state, 0.3, 0, [True, True], 0)


def write_policy(path, names, **overrides):
    values = {
        "schema_version": POLICY_SCHEMA,
        "feature_names": np.array(names),
        "mean": np.zeros(len(names)),
        "std": np.ones(len(names)),
        "weights": np.zeros((len(names), 2)),
        "bias": np.array([0, 10]),
        "classes": np.array([0, 1]),
    }
    values.update(overrides)
    np.savez_compressed(path, **values)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_unknown_input_does_not_invent_geometry_for_scripted_decisions():
    names, features = inputs()
    assert len(names) == len(set(names)) == len(features) == 100
    values = dict(zip(names, features, strict=True))
    assert values["corridor_0_0.75_floor_fraction"] == 0
    assert values["corridor_0_0.75_ceiling_fraction"] == 0
    assert values["corridor_0_0.75_unknown_fraction"] == 1
    assert choose_history_skill("scripted", names, features, [True, True], 0, 0.3)[0] == 0
    names, features = inputs(occupied=True)
    assert choose_history_skill("scripted", names, features, [True, True], 0, 0.3)[0] == 1


def test_shared_mask_preserves_active_motion_outside_qualified_transition():
    names, features = inputs(occupied=True)
    assert legal_skill_mask(0, 0.1, 0, 0).tolist() == [True, False]
    assert legal_skill_mask(0, 0.3, 0.1, 0).tolist() == [True, False]
    assert legal_skill_mask(1, 0.4, 0, 0).tolist() == [False, True]
    assert legal_skill_mask(1, 3.3, 0, 0).tolist() == [True, True]
    assert choose_history_skill("scripted", names, features, [True, False], 0, 0.3)[0] == 0
    assert choose_history_skill("always_walk", names, features, [False, True], 1, 0.4)[0] == 1


def test_learned_readout_is_sha_pinned_schema_checked_and_legality_masked(tmp_path):
    names, features = inputs()
    path = tmp_path / "policy.npz"
    digest = write_policy(path, names)
    policy = load_history_policy(path, digest)
    canonical = load_history_policy(path, f"sha256:{digest}")
    assert np.array_equal(canonical["weights"], policy["weights"])
    assert choose_history_skill("learned", names, features, [True, True], 0, 0.3, policy)[0] == 1
    assert choose_history_skill("learned", names, features, [True, False], 0, 0.3, policy)[0] == 0
    with pytest.raises(ValueError, match="SHA256"):
        load_history_policy(path, "wrong")
    with pytest.raises(ValueError, match="schema mismatch"):
        choose_history_skill("learned", names[::-1], features, [True, True], 0, 0.3, policy)
    digest = write_policy(path, names, std=np.zeros(len(names)))
    with pytest.raises(ValueError, match="normalization"):
        load_history_policy(path, digest)


def test_policy_never_treats_no_legal_active_action_as_stop():
    names, features = inputs()
    with pytest.raises(ValueError, match="current skill"):
        choose_history_skill("scripted", names, features, [False, False], 0, 0.3)
