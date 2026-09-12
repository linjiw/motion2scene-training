import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_decision_study import OutcomeTrees
from motion2scene_outcome_policy import (
    choose_action,
    choose_schedule,
    export_policy,
    export_tree,
    predict_tree,
    validate_policy,
    validate_tree,
)
from test_motion2scene_checkpoint_controls import fixture


def fitted():
    bank, groups, _ = fixture()
    rows = [row for group in groups for row in group["targets"]]
    learner = OutcomeTrees(rows)
    return bank, rows, learner, export_policy(learner, bank)


def test_exported_choices_match_existing_feasibility_time_learner():
    bank, rows, learner, policy = fitted()
    for row in rows:
        action, estimates = choose_action(
            policy, row["features"], np.asarray(row["legal_mask"]), row["phase_tick"]
        )
        assert action == learner.choose(row)
        assert all(0 <= v["estimated_passage_probability"] <= 1 for v in estimates)
    assert validate_policy(policy, bank) is policy


def test_portable_tree_matches_sklearn_on_random_and_boundary_inputs():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(50, 3))
    models = [
        (DecisionTreeClassifier(max_depth=2, random_state=0).fit(x, x[:, 0] > 0), True),
        (DecisionTreeRegressor(max_depth=2, random_state=0).fit(x, abs(x[:, 0]) + 2), False),
    ]
    for model, feasibility in models:
        tree = export_tree(model, feasibility=feasibility)
        inputs = list(rng.normal(size=(100, 3)))
        for node, feature in enumerate(tree["feature"]):
            if feature >= 0:
                for threshold in (
                    tree["threshold"][node],
                    np.nextafter(tree["threshold"][node], np.inf),
                    np.nextafter(tree["threshold"][node], -np.inf),
                ):
                    v = np.zeros(3)
                    v[feature] = threshold
                    inputs.append(v)
        expected = model.predict_proba(inputs)[:, 1] if feasibility else model.predict(inputs)
        np.testing.assert_array_equal([predict_tree(tree, row) for row in inputs], expected)


def test_no_successful_time_remains_missing_and_does_not_invent_feasibility():
    model = DecisionTreeClassifier(max_depth=2).fit([[0], [1]], [False, False])
    tree = export_tree(model, feasibility=True)
    assert predict_tree(tree, [0.5]) == 0
    assert export_tree(None, feasibility=False) is None


@pytest.mark.parametrize("change", ["cycle", "unreachable", "probability", "feature"])
def test_invalid_tree_structure_or_estimates_are_rejected(change):
    tree = dict(
        left=[1, -1, -1],
        right=[2, -1, -1],
        feature=[0, -2, -2],
        threshold=[0, -2, -2],
        value=[0.5, 0, 1],
    )
    if change == "cycle":
        tree["left"][0] = 0
    elif change == "unreachable":
        for key, value in dict(left=-1, right=-1, feature=-2, threshold=-2, value=0).items():
            tree[key].append(value)
    elif change == "probability":
        tree["value"][2] = 1.1
    else:
        tree["feature"][0] = 1
    with pytest.raises(ValueError):
        validate_tree(tree, 1, probability=True)


def test_return_obligation_bypasses_outcome_estimates_and_hold_has_no_decision():
    bank, rows, _, policy = fitted()
    legal = np.zeros(len(bank.option_ids), dtype=bool)
    legal[0:2] = True
    assert choose_schedule(
        policy, policy["feature_names"], rows[0]["features"], legal, 1, 265, 0, bank
    ) == ("neutral", None)
    legal[0] = False
    assert choose_schedule(
        policy, policy["feature_names"], rows[0]["features"], legal, 1, 20, None, bank
    ) == (bank.option_ids[1], None)


def test_missing_phase_head_and_changed_bank_are_rejected():
    bank, _, _, policy = fitted()
    bad = copy.deepcopy(policy)
    bad["heads"].pop()
    with pytest.raises(ValueError, match="every legal"):
        validate_policy(bad, bank)
    bad = copy.deepcopy(policy)
    bad["request_digest"] = "changed"
    with pytest.raises(ValueError, match="common interface"):
        validate_policy(bad, bank)
