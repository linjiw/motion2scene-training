"""Development control: RBF regression of feasibility and successful time.

Scores are uncalibrated feasibility estimates, not safety certificates. Kernel
scale and normalization use training inputs only. Failed/unknown passage costs
never train the time head. This module does not change the frozen ridge policy.
"""

import numpy as np


def _kernel(x, centers, bandwidth):
    return np.exp(-np.mean((x[:, None] - centers[None]) ** 2, axis=2) / bandwidth)


def fit_kernel_outcomes(features, ticks, passed, times, admitted, legal, *, l2=0.001):
    x = np.asarray(features, float)
    ticks = np.asarray(ticks)
    passed, admitted, legal = map(np.asarray, (passed, admitted, legal))
    times = np.asarray(times, float)
    if (
        x.ndim != 2
        or not len(x)
        or not np.isfinite(x).all()
        or ticks.shape != (len(x),)
        or ticks.dtype.kind not in "iu"
        or passed.ndim != 2
        or passed.shape[0] != len(x)
        or any(a.shape != passed.shape or a.dtype.kind != "b" for a in (passed, admitted, legal))
        or times.shape != passed.shape
        or not np.isfinite(l2)
        or l2 <= 0
        or np.any(~np.isfinite(times[passed & admitted & legal]))
        or np.any(times[passed & admitted & legal] < 0)
    ):
        raise ValueError("aligned finite training inputs and measured labels required")
    model = dict(l2=float(l2), phases={})
    for tick in np.unique(ticks):
        ids = np.flatnonzero(ticks == tick)
        mean, std = x[ids].mean(0), np.maximum(x[ids].std(0), 0.05)
        z = (x[ids] - mean) / std
        distances = np.mean((z[:, None] - z[None]) ** 2, axis=2)
        positive = distances[distances > 0]
        bandwidth = float(np.median(positive)) if len(positive) else 1.0
        heads = []
        for action in range(passed.shape[1]):
            observed = admitted[ids, action] & legal[ids, action]
            successful = observed & passed[ids, action]
            head = {}
            for name, mask, y in (
                ("feasibility", observed, passed[ids, action].astype(float)),
                ("time", successful, times[ids, action]),
            ):
                if mask.any():
                    centers, values = z[mask], y[mask]
                    offset = float(values.mean())
                    matrix = _kernel(centers, centers, bandwidth)
                    coefficients = np.linalg.solve(
                        matrix + l2 * np.eye(len(matrix)), values - offset
                    )
                    head[name] = dict(centers=centers, offset=offset, coefficients=coefficients)
                else:
                    head[name] = None
            heads.append(head)
        model["phases"][int(tick)] = dict(mean=mean, std=std, bandwidth=bandwidth, heads=heads)
    return model


def predict_kernel_outcomes(model, features, tick, legal):
    phase = model["phases"][int(tick)]
    x, legal = np.asarray(features, float), np.asarray(legal)
    if (
        x.shape != phase["mean"].shape
        or not np.isfinite(x).all()
        or legal.shape != (len(phase["heads"]),)
        or legal.dtype.kind != "b"
        or not legal.any()
    ):
        raise ValueError("finite query and legal action mask required")
    z = ((x - phase["mean"]) / phase["std"])[None]
    feasibility = np.full(len(legal), np.nan)
    times = np.full(len(legal), np.inf)
    for action in np.flatnonzero(legal):
        head = phase["heads"][action]
        if head["feasibility"] is None:
            raise ValueError("legal action has no observed feasibility supervision")
        for name, values in (("feasibility", feasibility), ("time", times)):
            h = head[name]
            if h is not None:
                values[action] = float(
                    (_kernel(z, h["centers"], phase["bandwidth"]) @ h["coefficients"])[0]
                    + h["offset"]
                )
    feasibility[legal] = np.clip(feasibility[legal], 0, 1)
    times[legal] = np.maximum(times[legal], 0)
    candidates = np.flatnonzero(legal & (feasibility >= 0.5) & np.isfinite(times))
    if len(candidates):
        choice = min(candidates.tolist(), key=lambda i: (times[i], -feasibility[i], i))
    else:
        choice = min(np.flatnonzero(legal).tolist(), key=lambda i: (-feasibility[i], times[i], i))
    return dict(action=choice, feasibility=feasibility, passage_time_s=times)
