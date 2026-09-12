"""Arc-length-relative proposal recipes, independent of geometric labels."""

import numpy as np


def route_candidates(root_xy, fallback_yaw=0.0):
    """Five arc-length stations, five lateral offsets, three heights/shapes.

    Zero-length routes use root heading. Duplicate recipes on stationary routes
    remain distinct latent cells, not independent spatial locations.
    """
    root = np.asarray(root_xy, dtype=float)
    if root.ndim != 2 or root.shape[1] != 2 or not len(root) or not np.isfinite(root).all():
        raise ValueError("invalid root path")
    delta = np.diff(root, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    valid = lengths > 1e-9
    starts, vectors, lengths = root[:-1][valid], delta[valid], lengths[valid]
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    specs, recipes, anchors = [], [], []
    for station in np.linspace(0, 1, 5):
        if len(lengths):
            distance = station * cumulative[-1]
            j = min(np.searchsorted(cumulative, distance, side="right") - 1, len(lengths) - 1)
            anchor = starts[j] + vectors[j] * (distance - cumulative[j]) / lengths[j]
            yaw = np.arctan2(vectors[j, 1], vectors[j, 0])
        else:
            anchor, yaw = root[0], fallback_yaw
        anchors.append([*anchor, float(yaw)])
        for lateral in np.linspace(-0.8, 0.8, 5):
            xy = anchor + lateral * np.array([-np.sin(yaw), np.cos(yaw)])
            for height in [0.4, 1.0, 1.6]:
                for kind in range(3):
                    specs.append([*xy, height, kind, yaw])
                    recipes.append([station, lateral, height, kind])
    return np.asarray(specs), np.asarray(recipes), np.asarray(anchors)
