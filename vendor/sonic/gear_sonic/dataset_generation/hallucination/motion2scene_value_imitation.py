"""Linear value imitation of measured finite-schedule physical regret.

This is an alternative student objective on an unchanged teacher table. It does
not acquire scenes, invent outcome labels, or establish closed-loop performance.
"""

import numpy as np

from .motion2scene_multi_option_imitation import physical_regret
from .motion2scene_multi_option_policy import SCHEMA


def fit_value_policy(
    features,
    feature_names,
    option_ids,
    passed,
    passage_time_s,
    admitted,
    legality,
    *,
    l2=0.001,
    sample_weights=None,
):
    """Regress each option's measured regret using only its legal targets.

    Each option minimizes weighted mean squared error plus ``l2 * ||w||^2``.
    Its intercept is unpenalized. Complete consequential decision rows alone
    determine feature normalization, using the common student's 0.05 std floor.
    The existing runtime receives logits equal to negative predicted regret and
    still masks illegal options. Values are unconstrained estimates, not success
    probabilities. An option with no target is rejected rather than assigned a
    synthetic failure or optimistic value.
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
        or not np.isfinite(l2)
        or l2 < 0
    ):
        raise ValueError("finite features and valid value-imitation configuration required")
    regret, supervised = physical_regret(passed, passage_time_s, admitted, legal)
    if not supervised.any():
        raise ValueError("no complete consequential physical teacher decisions")
    replay = np.ones(len(x)) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if replay.shape != (len(x),) or not np.isfinite(replay).all() or (replay <= 0).any():
        raise ValueError("finite positive replay weights required for every recorded decision")
    mean = x[supervised].mean(axis=0)
    std = np.maximum(x[supervised].std(axis=0), 0.05)
    scaled = (x - mean) / std
    coefficients = np.zeros((x.shape[1], len(option_ids)))
    intercepts = np.zeros(len(option_ids))
    option_reports = []
    for option in range(len(option_ids)):
        mask = supervised & legal[:, option]
        if not mask.any():
            raise ValueError(f"option {option_ids[option]} has no complete legal value target")
        weights = replay[mask] / replay[mask].sum()
        values, inputs = regret[mask, option], scaled[mask]
        input_mean = weights @ inputs
        value_mean = float(weights @ values)
        design = (inputs - input_mean) * np.sqrt(weights[:, None])
        target = (values - value_mean) * np.sqrt(weights)
        if l2:
            design = np.vstack([design, np.sqrt(l2) * np.eye(x.shape[1])])
            target = np.r_[target, np.zeros(x.shape[1])]
        coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
        intercept = value_mean - input_mean @ coefficient
        coefficients[:, option], intercepts[option] = coefficient, intercept
        errors = inputs @ coefficient + intercept - values
        option_reports.append(
            {
                "option_index": option,
                "target_count": int(mask.sum()),
                "recorded_decision_indices": np.flatnonzero(mask).tolist(),
                "weighted_mean_squared_error": float(weights @ errors**2),
                "regularized_objective": float(
                    weights @ errors**2 + l2 * coefficient @ coefficient
                ),
            }
        )
    model = {
        "schema_version": np.array(SCHEMA),
        "feature_names": np.array(feature_names),
        "option_ids": np.array(option_ids),
        "classes": np.arange(len(option_ids)),
        "mean": mean,
        "std": std,
        "weights": -coefficients,
        "bias": -intercepts,
    }
    predicted_regret = scaled @ coefficients + intercepts
    actions = np.where(legal, predicted_regret, np.inf).argmin(axis=1)
    fit_rows = []
    for index in np.flatnonzero(supervised):
        action = int(actions[index])
        fit_rows.append(
            {
                "recorded_decision_index": int(index),
                "action": action,
                "measured_regret": float(regret[index, action]),
                "predicted_regret": predicted_regret[index].tolist(),
                "legal_mask": legal[index].tolist(),
            }
        )
    return model, {
        "method": "per-option ridge regression of measured physical regret",
        "l2": l2,
        "recorded_decisions": len(x),
        "complete_consequential_decisions": int(supervised.sum()),
        "excluded_decision_indices": np.flatnonzero(~supervised).tolist(),
        "option_fits": option_reports,
        "fitted_decisions": fit_rows,
        "scope": "offline learner development on finite recorded teacher values; no new physics",
    }
