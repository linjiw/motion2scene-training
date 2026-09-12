"""Controlled replay alternatives for development studies on fixed teacher data.

Prediction residuals measure learning on recorded inputs. Historical physical
gaps must come from executed generating policies; this module never estimates
new physical outcomes or changes the active acquisition algorithm.
"""

import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_imitation import (
    physical_regret,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_curriculum import (
    encounter_replay_weights,
)


def current_supervised_error(predicted_regret, passed, times, admitted, legality):
    """Mean squared regret residual over measured legal actions at each phase.

    All-failed, indifference and incomplete tables remain excluded according to
    the common consequential-regret learner. Return NaN for an excluded row.
    """
    target, eligible = physical_regret(passed, times, admitted, legality)
    prediction = np.asarray(predicted_regret, dtype=float)
    legal = np.asarray(legality)
    if prediction.shape != target.shape or not np.isfinite(prediction[legal]).all():
        raise ValueError("finite predictions on all legal actions required")
    result = np.full(len(target), np.nan)
    for index in np.flatnonzero(eligible):
        result[index] = np.mean(
            (prediction[index, legal[index]] - target[index, legal[index]]) ** 2
        )
    return result


def replay_control_weights(rows, mode, *, supervised_error=None, ungated_gaps=None):
    """Return encounter-balanced weights with the same coverage recipe in all arms.

    rows use the original replay schema, including *measured*, observation-gated
    historical ``gap``. Ungated gaps require separately verified matched outcomes.
    Error and hybrid controls use max phase score for encounter mass and normalized
    phase scores within the encounter, matching the historical priority rule.
    No positive priority falls back to 0.8 uniform plus 0.2 coverage.
    """
    modes = {
        "uniform",
        "coverage_only",
        "uniform_coverage",
        "historical_gated",
        "historical_ungated",
        "current_error",
        "historical_current_error",
    }
    if mode not in modes:
        raise ValueError("unknown replay control")
    records = [dict(row) for row in rows]
    if mode == "historical_ungated":
        if ungated_gaps is None or len(ungated_gaps) != len(rows):
            raise ValueError("separately measured ungated gaps required")
        for record, gap in zip(records, ungated_gaps, strict=True):
            record["gap"] = gap
    if mode in {"current_error", "historical_current_error"}:
        error = np.asarray(supervised_error, dtype=float)
        mask = np.asarray([r["supervision_available"] for r in rows], dtype=bool)
        if (
            error.shape != (len(rows),)
            or not np.isfinite(error[mask]).all()
            or (error[mask] < 0).any()
        ):
            raise ValueError("current nonnegative finite error required on supervised rows")
        # One common scale leaves the normalized priority distribution unchanged.
        scale = max(float(error[mask].max()) if mask.any() else 0.0, 1e-12)
        for i, record in enumerate(records):
            priority = float(error[i] / scale) if mask[i] else None
            if mode == "historical_current_error" and priority is not None:
                priority *= record["gap"] or 0.0
            record["gap"] = priority
    base = encounter_replay_weights(records)
    components = {k: np.asarray(v) for k, v in base["components"].items()}
    if mode == "uniform":
        weights = components["uniform"] / 0.2
    elif mode == "coverage_only":
        weights = components["coverage"] / 0.2
    elif mode == "uniform_coverage":
        weights = 4 * components["uniform"] + components["coverage"]
    else:
        weights = np.asarray(base["weights"])
    return dict(
        mode=mode,
        weights=weights.tolist(),
        no_positive_priority=base["no_positive_gap_fallback"],
        signal=(
            "recorded supervised error" if "error" in mode else "historical physical or coverage"
        ),
    )
