"""Observation-consistent regret targets with the existing phase-wise ridge fit.

Group targets first prefer maximal attainable passage count. Time distinguishes
actions attaining that count. Singleton targets equal the existing physical
regret for positive measured passage times. No outcomes are synthesized to call
the old fitter; the same ridge equations fit these explicit teaching targets.
"""

import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import definition_digest
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (
    SCHEMA,
    schedule_layout,
)


def group_regret(row):
    if not row["complete"] or not row["has_passing_supervision"]:
        return None
    maximum = row["maximum_causal_passages"]
    values = row["action_values"]
    passing = [v["successful_time_sum_s"] for v in values.values() if v["passage_count"] == maximum]
    best, worst = min(passing), max(passing)
    denominator = maximum * worst
    # The real passage task has strictly positive crossing/hold duration. This
    # explicit zero-time edge keeps even abstract test cases passage-first.
    if best == 0:
        denominator *= 2
    return {
        a: (
            (maximum - v["passage_count"]) / maximum
            if v["passage_count"] < maximum
            else (v["successful_time_sum_s"] - best) / denominator if denominator else 0.0
        )
        for a, v in values.items()
    }


def expand_group_targets(rows, features, phases, option_count):
    """One shared target for exactly equal observations, repeated per encounter."""
    features = np.asarray(features, float)
    if features.ndim != 3 or features.shape[1] != len(phases) or not np.isfinite(features).all():
        raise ValueError("finite encounter-by-phase feature tensor required")
    targets = np.full((*features.shape[:2], option_count), np.nan)
    legality = np.zeros(targets.shape, bool)
    seen = set()
    for row in rows:
        k = list(phases).index(row["phase_tick"])
        ids, actions = row["encounter_indices"], row["legal_actions"]
        if any((i, k) in seen for i in ids) or not np.array_equal(
            features[ids, k], np.broadcast_to(features[ids[0], k], features[ids, k].shape)
        ):
            raise ValueError(
                "each information group requires identical student inputs and disjoint membership"
            )
        seen.update((i, k) for i in ids)
        regret = group_regret(row)
        for i in ids:
            legality[i, k, actions] = True
            if regret is not None:
                targets[i, k, actions] = [regret[a] for a in actions]
    if len(seen) != features.shape[0] * features.shape[1]:
        raise ValueError("each encounter and phase must belong to an information group")
    return targets, legality


def fit_explicit_regret(bank, features, names, ticks, regret, legal, *, l2=10.0):
    """Same normalization, unpenalized intercepts and ridge equations as baseline."""
    phases, qualified, _ = schedule_layout(bank)
    x, ticks, regret, legal = (
        np.asarray(features, float),
        np.asarray(ticks),
        np.asarray(regret, float),
        np.asarray(legal),
    )
    if (
        x.ndim != 2
        or len(names) != x.shape[1]
        or len(set(names)) != len(names)
        or x.shape[1] != 100 + 2 * len(bank.option_ids)
        or not np.isfinite(x).all()
        or ticks.shape != (len(x),)
        or ticks.dtype.kind not in "iu"
        or legal.shape != regret.shape
        or regret.shape != (len(x), len(bank.option_ids))
        or legal.dtype.kind != "b"
        or not legal[:, 0].all()
        or np.isinf(regret).any()
        or np.any((regret[np.isfinite(regret)] < 0) | (regret[np.isfinite(regret)] > 1))
        or not np.isfinite(l2)
        or l2 < 0
    ):
        raise ValueError(
            "finite named observations, legal masks and bounded explicit regret required"
        )
    indices = np.array([list(phases).index(tick) for tick in ticks])
    if np.any(legal & ~qualified[indices]):
        raise ValueError("target action is unavailable at its phase")
    checks = {"phase_s": ticks / 50, "active_skill": np.zeros(len(x))}
    checks.update(
        {f"active_option_{i}": np.full(len(x), float(i == 0)) for i in range(len(bank.option_ids))}
    )
    checks.update({f"option_{i}_legal": legal[:, i] for i in range(len(bank.option_ids))})
    for key, expected in checks.items():
        if key not in names or not np.allclose(
            x[:, list(names).index(key)], expected, rtol=0, atol=1e-7
        ):
            raise ValueError("feature disagrees with the actual neutral phase/interface: " + key)
    model = dict(
        schema_version=np.array(SCHEMA),
        feature_names=np.asarray(names),
        option_ids=np.asarray(bank.option_ids),
        request_digest=np.array(definition_digest(bank.request)),
        classes=np.arange(len(bank.option_ids)),
        phase_ticks=phases,
        qualified_mask=qualified,
        trained_mask=np.zeros_like(qualified),
        mean=np.zeros((len(phases), x.shape[1])),
        std=np.ones((len(phases), x.shape[1])),
        weights=np.zeros((len(phases), x.shape[1], len(bank.option_ids))),
        bias=np.zeros((len(phases), len(bank.option_ids))),
        l2=np.array(l2),
    )
    supervised = np.array(
        [np.isfinite(r[m]).all() and np.ptp(r[m]) > 0 for r, m in zip(regret, legal, strict=True)]
    )
    for k in range(len(phases)):
        take = (indices == k) & supervised
        if not take.any():
            raise ValueError("each phase requires complete consequential explicit targets")
        mean = x[take].mean(0)
        std = np.maximum(x[take].std(0), 0.05)
        scaled = (x - mean) / std
        model["mean"][k], model["std"][k] = mean, std
        for a in np.flatnonzero(legal[take].any(0)):
            mask = take & legal[:, a]
            w = np.ones(mask.sum()) / mask.sum()
            values, inputs = regret[mask, a], scaled[mask]
            input_mean, value_mean = w @ inputs, float(w @ values)
            design = (inputs - input_mean) * np.sqrt(w[:, None])
            target = (values - value_mean) * np.sqrt(w)
            if l2:
                design = np.vstack([design, np.sqrt(l2) * np.eye(x.shape[1])])
                target = np.r_[target, np.zeros(x.shape[1])]
            coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
            model["weights"][k, :, a] = -coefficients
            model["bias"][k, a] = -(value_mean - input_mean @ coefficients)
            model["trained_mask"][k, a] = True
    return model
