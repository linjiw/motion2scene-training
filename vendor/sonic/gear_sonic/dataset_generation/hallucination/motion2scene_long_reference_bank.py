"""Length-aware route audit for separately registered finite reference horizons."""

import numpy as np


def audit_finite_root_route(root, raw, source_fps, expected_frames):
    root, raw = np.asarray(root), np.asarray(raw)
    if (
        expected_frames < 2
        or root.shape != (expected_frames, 3)
        or raw.ndim != 2
        or raw.shape[1] != 3
        or len(raw) < 2
        or not np.isfinite(source_fps)
        or source_fps <= 0
        or not np.isfinite(root).all()
        or not np.isfinite(raw).all()
    ):
        raise ValueError("invalid registered finite loaded/source route")
    target_times = np.arange(expected_frames) / 50
    source_times = np.arange(len(raw)) / source_fps
    if target_times[-1] >= source_times[-1]:
        raise ValueError("reference clock must exclude the source endpoint without padding")
    expected = np.stack(
        [np.interp(target_times, source_times, raw[:, j]) for j in range(2)], axis=1
    )
    error = float(np.max(np.abs(root[:, :2] - expected)))
    if error > 1e-4:
        raise ValueError(f"loaded finite reference differs from source route: {error} m")
    return error
