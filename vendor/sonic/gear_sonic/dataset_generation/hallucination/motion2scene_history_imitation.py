"""Cost-sensitive imitation from matched physical option branches.

Binary integration baseline for the repeated-decision runtime. Targets use
passage first and measured traversal time second. No feasible target is invented
for both-fail pairs. Multi-reference selection requires a larger action contract.
"""

import numpy as np

from .motion2scene_closed_loop_policy import POLICY_SCHEMA


def physical_targets(passed, passage_time_s, admitted):
    """Teacher classes and consequential weights for complete paired branches.

    With two successful branches, relative measured time regret supplies the
    weight. When only one passes its weight is one. Exact cost ties supply no
    preference and no fitting gradient. Missing labels are excluded, not failed.
    """
    passed = np.asarray(passed)
    admitted = np.asarray(admitted)
    times = np.asarray(passage_time_s, dtype=float)
    if (
        passed.ndim != 2
        or passed.shape[1] != 2
        or passed.dtype != bool
        or admitted.dtype != bool
        or admitted.shape != passed.shape
        or times.shape != passed.shape
    ):
        raise ValueError("aligned binary outcomes, admission masks, and time pairs required")
    if not np.isfinite(times[passed & admitted]).all() or (times[passed & admitted] < 0).any():
        raise ValueError("successful admitted branches require measured traversal times")
    labels = np.zeros(len(passed), dtype=np.int64)
    weights = np.zeros(len(passed))
    for i in range(len(passed)):
        if not admitted[i].all() or not passed[i].any():
            continue
        if passed[i].sum() == 1:
            labels[i] = int(passed[i, 1])
            weights[i] = 1
        else:
            labels[i] = int(np.argmin(times[i]))
            weights[i] = abs(times[i, 0] - times[i, 1]) / max(times[i].max(), 1e-12)
    return labels, weights


def fit_history_policy(
    features,
    feature_names,
    passed,
    passage_time_s,
    admitted,
    *,
    steps=1000,
    learning_rate=0.03,
    l2=0.001,
):
    """Fit a deterministic linear softmax with physical-regret weighted imitation.

    One row per verified matched decision. Callers must bind each row to actual
    prefixes, sensor schema, branch artifacts, and a permitted training split.
    Dataset aggregation calls this with old plus newly verified student states.
    """
    features = np.asarray(features, dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape[1] != len(feature_names)
        or not len(features)
        or not np.isfinite(features).all()
        or len(set(feature_names)) != len(feature_names)
        or steps < 1
        or not np.isfinite([learning_rate, l2]).all()
        or learning_rate <= 0
        or l2 < 0
    ):
        raise ValueError("finite aligned features and valid fitting hyperparameters required")
    labels, weights = physical_targets(passed, passage_time_s, admitted)
    if len(labels) != len(features) or not weights.any():
        raise ValueError("at least one consequential, physically verified matched target required")
    selected = weights > 0
    # Only supervised training rows determine normalization.
    mean = features[selected].mean(axis=0)
    scale = np.maximum(features[selected].std(axis=0), 0.05)
    x = (features - mean) / scale
    w = np.zeros((features.shape[1], 2))
    bias = np.zeros(2)
    onehot = np.eye(2)[labels]
    weights /= weights.sum()
    history = []
    for step in range(steps):
        logits = x @ w + bias
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        error = (probabilities - onehot) * weights[:, None]
        w -= learning_rate * (x.T @ error + l2 * w)
        bias -= learning_rate * error.sum(axis=0)
        if step % 100 == 0 or step == steps - 1:
            history.append(
                {
                    "step": step,
                    "weighted_cross_entropy": float(
                        -np.sum(
                            weights
                            * np.log(np.maximum(probabilities[np.arange(len(x)), labels], 1e-300))
                        )
                    ),
                }
            )
    model = {
        "schema_version": np.array(POLICY_SCHEMA),
        "feature_names": np.asarray(feature_names),
        "mean": mean,
        "std": scale,
        "weights": w,
        "bias": bias,
        "classes": np.array([0, 1]),
    }
    return model, {
        "rows": len(features),
        "consequential_supervised_rows": int(selected.sum()),
        "excluded_or_tied_rows": int((~selected).sum()),
        "history": history,
        "scope": "weighted imitation fit; no physical closed-loop performance measured here",
    }
