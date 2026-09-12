"""Measured-regret imitation for a qualified, matched multi-option teacher table."""

import numpy as np

from .motion2scene_multi_option_policy import SCHEMA


def physical_regret(passed, passage_time_s, admitted, legality):
    """Return per-option physical regret and which complete rows teach a decision.

    Failure has regret one; successful options have relative excess traversal
    time below one. An incomplete legal branch set or no verified solution
    supplies no target. Illegal options are masked, not assigned failed outcomes.
    """
    passed, admitted, legal = map(np.asarray, (passed, admitted, legality))
    times = np.asarray(passage_time_s, dtype=float)
    if (
        passed.ndim != 2
        or passed.shape[1] < 2
        or any(a.shape != passed.shape for a in (admitted, legal, times))
        or any(a.dtype != bool for a in (passed, admitted, legal))
        or not legal.any(axis=1).all()
    ):
        raise ValueError("aligned boolean option tables and legal alternatives required")
    observed_success = passed & admitted & legal
    if not np.isfinite(times[observed_success]).all() or (times[observed_success] < 0).any():
        raise ValueError("verified successful options require measured passage times")
    supervised = ((admitted | ~legal).all(axis=1)) & observed_success.any(axis=1)
    regret = np.zeros_like(times)
    for i in np.flatnonzero(supervised):
        success = observed_success[i]
        best, worst = times[i, success].min(), times[i, success].max()
        regret[i, legal[i] & ~passed[i]] = 1
        regret[i, success] = (times[i, success] - best) / max(worst, 1e-12)
        if np.ptp(regret[i, legal[i]]) == 0:
            supervised[i] = False
    return regret, supervised


def fit_multi_option_policy(
    features,
    feature_names,
    option_ids,
    passed,
    passage_time_s,
    admitted,
    legality,
    *,
    steps=1500,
    learning_rate=0.05,
    l2=0.001,
    sample_weights=None,
):
    """Minimize teacher regret under the student's masked action distribution.

    Gradients use the complete measured regret table rather than treating all
    nonselected, physically successful options as failures. This is an imitation
    objective on recorded branches; it does not perform new robot rollouts.
    """
    x = np.asarray(features, dtype=float)
    legal = np.asarray(legality)
    if (
        x.ndim != 2
        or not len(x)
        or x.shape[1] != len(feature_names)
        or not np.isfinite(x).all()
        or len(set(feature_names)) != len(feature_names)
        or len(option_ids) < 2
        or len(set(option_ids)) != len(option_ids)
        or legal.shape != (len(x), len(option_ids))
        or steps < 1
        or not np.isfinite([learning_rate, l2]).all()
        or learning_rate <= 0
        or l2 < 0
    ):
        raise ValueError("finite features and valid multi-option fitting configuration required")
    regret, supervised = physical_regret(passed, passage_time_s, admitted, legal)
    if not supervised.any():
        raise ValueError("no complete consequential physical teacher decisions")
    weights = np.ones(len(x)) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if weights.shape != (len(x),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("finite positive replay weights required for every recorded decision")
    weights = weights[supervised]
    weights /= weights.sum()
    mean = x[supervised].mean(axis=0)
    std = np.maximum(x[supervised].std(axis=0), 0.05)
    scaled = (x[supervised] - mean) / std
    regret, legal = regret[supervised], legal[supervised]
    w, bias = np.zeros((x.shape[1], len(option_ids))), np.zeros(len(option_ids))
    history = []
    for step in range(steps):
        logits = np.where(legal, scaled @ w + bias, -np.inf)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        expected = (probabilities * regret).sum(axis=1, keepdims=True)
        error = probabilities * (regret - expected) * weights[:, None]
        w -= learning_rate * (scaled.T @ error + l2 * w)
        bias -= learning_rate * error.sum(axis=0)
        if step % 100 == 0 or step == steps - 1:
            history.append(
                {
                    "step": step,
                    "mean_expected_physical_regret": float(expected.mean()),
                    "weighted_expected_physical_regret": float(weights @ expected[:, 0]),
                }
            )
    model = {
        "schema_version": np.array(SCHEMA),
        "feature_names": np.array(feature_names),
        "option_ids": np.array(option_ids),
        "classes": np.arange(len(option_ids)),
        "mean": mean,
        "std": std,
        "weights": w,
        "bias": bias,
    }
    fitted_actions = np.where(legal, scaled @ w + bias, -np.inf).argmax(axis=1)
    return model, {
        "recorded_decisions": len(x),
        "complete_consequential_decisions": int(supervised.sum()),
        "replay_weights": weights.tolist(),
        "fitted_decisions": [
            {
                "recorded_decision_index": int(index),
                "action": int(action),
                "measured_regret": float(regret[row, action]),
            }
            for row, (index, action) in enumerate(zip(np.flatnonzero(supervised), fitted_actions))
        ],
        "history": history,
        "scope": "offline physical-regret imitation; closed-loop success requires actual execution",
    }
