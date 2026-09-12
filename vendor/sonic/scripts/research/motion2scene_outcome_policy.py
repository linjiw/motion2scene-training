"""Portable shallow feasibility/time trees for finite-schedule traversal.

Deployment uses NumPy only. The fixed depth-two fitting experiment remains in
motion2scene_decision_study; this representation preserves its decision rule.
"""

import json
from pathlib import Path

import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (
    checked_artifact,
    definition_digest,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (
    schedule_layout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (
    expected_feature_names,
)

SCHEMA = "motion2scene_outcome_tree_policy_v1"


def export_tree(model, *, feasibility):
    if model is None:
        return None
    tree = model.tree_
    values = tree.value[:, 0]
    if feasibility:
        index = np.flatnonzero(model.classes_ == True)  # noqa: E712
        values = values[:, index[0]] / values.sum(axis=1) if len(index) else np.zeros(len(values))
    else:
        values = values[:, 0]
    return dict(
        left=tree.children_left.tolist(),
        right=tree.children_right.tolist(),
        feature=tree.feature.tolist(),
        threshold=tree.threshold.tolist(),
        value=values.tolist(),
    )


def export_policy(learner, bank):
    phases, qualified, _ = schedule_layout(bank)
    policy = dict(
        schema=SCHEMA,
        request_digest=definition_digest(bank.request),
        option_ids=list(bank.option_ids),
        phase_ticks=phases.tolist(),
        qualified_mask=qualified.tolist(),
        feature_names=list(expected_feature_names(len(bank.option_ids))),
        feasibility_threshold=0.5,
        feature_cast="float32_before_tree_comparison",
        max_depth=2,
        heads=[
            dict(
                tick=tick,
                action=action,
                feasibility=export_tree(pair[0], feasibility=True),
                successful_time=export_tree(pair[1], feasibility=False),
            )
            for (tick, action), pair in sorted(learner.heads.items())
        ],
    )
    validate_policy(policy, bank)
    return policy


def validate_tree(tree, size, *, probability):
    if not isinstance(tree, dict) or set(tree) != {
        "left",
        "right",
        "feature",
        "threshold",
        "value",
    }:
        raise ValueError("complete finite tree required")
    n = len(tree["value"])
    if not 1 <= n <= 7 or any(len(tree[k]) != n for k in tree):
        raise ValueError("a depth-two binary tree has at most seven nodes")
    if not np.isfinite(tree["threshold"] + tree["value"]).all():
        raise ValueError("finite tree thresholds and predictions required")
    visited = set()

    def visit(node, depth):
        if node in visited or not 0 <= node < n or depth > 2:
            raise ValueError("invalid, cyclic or deeper-than-declared tree")
        visited.add(node)
        left, right, feature = (tree[k][node] for k in ("left", "right", "feature"))
        if any(type(v) is not int for v in (left, right, feature)):
            raise ValueError("integer tree indices required")
        if left == right == -1:
            if feature >= 0:
                raise ValueError("leaf has a split feature")
        else:
            if not 0 <= feature < size or left <= node or right <= node or left == right:
                raise ValueError("forward, distinct child indices and a valid feature required")
            visit(left, depth + 1)
            visit(right, depth + 1)
        value = tree["value"][node]
        if value < 0 or (probability and value > 1):
            raise ValueError("invalid probability or successful-time estimate")

    visit(0, 0)
    if visited != set(range(n)):
        raise ValueError("unreachable tree node")


def validate_policy(policy, bank):
    phases, qualified, _ = schedule_layout(bank)
    names = expected_feature_names(len(bank.option_ids))
    if (
        policy.get("schema") != SCHEMA
        or policy.get("request_digest") != definition_digest(bank.request)
        or policy.get("option_ids") != list(bank.option_ids)
        or policy.get("feature_names") != list(names)
        or policy.get("phase_ticks") != phases.tolist()
        or policy.get("qualified_mask") != qualified.tolist()
        or policy.get("feasibility_threshold") != 0.5
        or policy.get("max_depth") != 2
        or policy.get("feature_cast") != "float32_before_tree_comparison"
    ):
        raise ValueError(
            "outcome policy does not bind the common interface and fixed decision rule"
        )
    expected = {(int(phases[p]), int(a)) for p, a in zip(*np.nonzero(qualified), strict=True)}
    found = set()
    for head in policy["heads"]:
        key = head["tick"], head["action"]
        if any(type(v) is not int for v in key) or key not in expected or key in found:
            raise ValueError("invalid or duplicate phase/action outcome head")
        found.add(key)
        validate_tree(head["feasibility"], len(names), probability=True)
        if head["successful_time"] is not None:
            validate_tree(head["successful_time"], len(names), probability=False)
    if found != expected:
        raise ValueError("every legal phase/action requires a fitted feasibility head")
    return policy


def load_policy(path, sha256, bank):
    ref = dict(path=str(Path(path)), sha256=sha256)
    return validate_policy(json.loads(checked_artifact(ref).read_text()), bank)


def predict_tree(tree, features):
    # sklearn's fitting/prediction convention is float32 input with float64
    # thresholds. Keeping float64 input here can change a boundary decision.
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("finite float32 tree inputs required")
    node = 0
    for _ in range(3):
        if tree["left"][node] == -1:
            return float(tree["value"][node])
        key = "left" if x[tree["feature"][node]] <= tree["threshold"][node] else "right"
        node = tree[key][node]
    raise ValueError("invalid depth-two tree")


def choose_action(policy, features, legal, tick):
    x, legal = np.asarray(features), np.asarray(legal)
    if (
        x.shape != (len(policy["feature_names"]),)
        or not np.isfinite(x).all()
        or legal.shape != (len(policy["option_ids"]),)
        or legal.dtype.kind != "b"
        or not legal.any()
        or tick not in policy["phase_ticks"]
    ):
        raise ValueError("finite named features, legal actions and a registered phase required")
    phase = policy["phase_ticks"].index(tick)
    if np.any(legal & ~np.asarray(policy["qualified_mask"][phase], dtype=bool)):
        raise ValueError("action is not supported at this phase")
    heads = {h["action"]: h for h in policy["heads"] if h["tick"] == tick}
    predictions = []
    for action in np.flatnonzero(legal):
        head = heads[int(action)]
        probability = predict_tree(head["feasibility"], x)
        time = (
            predict_tree(head["successful_time"], x)
            if head["successful_time"] is not None
            else np.inf
        )
        predictions.append((int(action), probability, time))
    feasible = [v for v in predictions if v[1] >= 0.5 and np.isfinite(v[2])]
    chosen = (
        min(feasible, key=lambda v: (v[2], -v[1], v[0]))
        if feasible
        else min(predictions, key=lambda v: (-v[1], v[2], v[0]))
    )
    return chosen[0], [
        dict(
            action=a,
            estimated_passage_probability=p,
            estimated_successful_time_s=t if np.isfinite(t) else None,
        )
        for a, p, t in predictions
    ]


def choose_schedule(policy, names, features, legal, active, tick, mandatory, bank):
    legal = np.asarray(legal)
    x = np.asarray(features)
    if (
        tuple(names) != tuple(policy["feature_names"])
        or legal.dtype.kind != "b"
        or legal.shape != (len(bank.option_ids),)
        or x.shape != (len(names),)
        or not np.isfinite(x).all()
    ):
        raise ValueError("common finite-schedule student interface required")
    if active not in range(len(bank.option_ids)) or not legal[active]:
        raise ValueError("current schedule is not legal")
    if mandatory is not None:
        if mandatory != 0 or active == 0:
            raise ValueError("only a non-neutral return can be mandatory")
        return "neutral", None
    if legal.sum() == 1:
        return bank.option_ids[active], None
    if active != 0:
        raise ValueError("new commitments require neutral")
    action, predictions = choose_action(policy, features, legal, tick)
    return bank.option_ids[action], predictions
