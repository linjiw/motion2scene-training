"""Offline body kinematics and explicitly heuristic event proposals."""

import numpy as np
from scipy.signal import savgol_filter

POINTS = ["com", "hip", "head", "left_hand", "right_hand", "left_foot", "right_foot"]


def smooth(x):
    n = len(x)
    if n < 5:
        return np.asarray(x).copy()
    width = min(7, n if n % 2 else n - 1)
    return savgol_filter(x, width, min(3, width - 1), axis=0, mode="interp")


def velocity_acceleration(positions, fps):
    if fps <= 0 or len(positions) < 3 or not np.isfinite(positions).all():
        raise ValueError("finite positions, positive fps and at least three frames required")
    p = smooth(np.asarray(positions))
    v = np.gradient(p, 1 / fps, axis=0, edge_order=2)
    a = np.gradient(v, 1 / fps, axis=0, edge_order=2)
    return p, v, a


def chord_root(root, center, half_window):
    """Kinematic shortcut in a local window; not a dynamically valid alternative."""
    start = max(0, center - half_window)
    end = min(len(root) - 1, center + half_window)
    result = root.copy()
    f = np.linspace(0, 1, end - start + 1)
    result[start : end + 1, :2] = (1 - f[:, None]) * root[start, :2] + f[:, None] * root[end, :2]
    return result, start, end


def observations(points, rotations, fps):
    """64 channels; rotations: hip/head/hands/feet 3x3, COM has no orientation."""
    p, v, a = velocity_acceleration(points, fps)
    relative = p - p[:, 1:2]
    increment = rotations[1:] @ rotations[:-1].swapaxes(-1, -2)
    angles = np.arccos(np.clip((np.trace(increment, axis1=-2, axis2=-1) - 1) / 2, -1, 1)) * fps
    angular = smooth(np.r_[angles[:1], angles])
    root = p[:, 1]
    speed = np.linalg.norm(v[:, 1, :2], axis=-1)
    yaw = np.unwrap(np.arctan2(rotations[:, 0, 1, 0], rotations[:, 0, 0, 0]))
    yawrate = np.gradient(smooth(yaw), 1 / fps)
    # atan2 is undefined at zero speed; use a velocity/acceleration identity.
    turn = (v[:, 1, 0] * a[:, 1, 1] - v[:, 1, 1] * a[:, 1, 0]) / np.maximum(speed**2, 0.05**2)
    turn = np.where(speed >= 0.05, turn, 0)
    deviation = []
    for t in range(len(root)):
        chord, _, _ = chord_root(root, t, max(1, round(0.5 * fps)))
        deviation.append(np.linalg.norm(chord[t, :2] - root[t, :2]))
    deviation = np.asarray(deviation)
    x = np.concatenate(
        [
            p[:, :, 2],
            np.linalg.norm(v, axis=-1),
            np.linalg.norm(a, axis=-1),
            relative.reshape(len(p), -1),
            angular,
            v[:, 0],
            a[:, 0],
            rotations[:, 0, :, :2].reshape(len(p), 6),
            yawrate[:, None],
            turn[:, None],
            deviation[:, None],
            speed[:, None],
        ],
        axis=-1,
    )
    assert x.shape == (len(p), 64) and np.isfinite(x).all()
    # Three human-specified event families. Scaling is fitted on training only.
    scores = np.stack(
        [
            np.abs(turn) + np.abs(yawrate),
            np.abs(v[:, 2, 2] - v[:, 1, 2]),
            np.linalg.norm(a[:, 0], axis=-1),
        ],
        axis=-1,
    )
    traces = dict(
        points=p,
        velocity=v,
        acceleration=a,
        angular_speed=angular,
        root_yaw=yaw,
        turn_rate=turn,
        chord_deviation=deviation,
        root_speed=speed,
    )
    return x, scores, traces


def event_anchors(score):
    """Endpoints + three peaks separated in time; geometry never selects peaks."""
    n = len(score)
    if n < 5 or not np.isfinite(score).all():
        raise ValueError("invalid event sequence")
    selected = [0, n - 1]
    gap = max(1, int(0.12 * (n - 1)))
    for i in np.argsort(-np.asarray(score), kind="stable"):
        if all(abs(int(i) - j) >= gap for j in selected):
            selected.append(int(i))
        if len(selected) == 5:
            break
    for i in np.linspace(0, n - 1, 5).astype(int):
        if len(selected) == 5:
            break
        if int(i) not in selected:
            selected.append(int(i))
    return np.array(sorted(selected))


def event_candidates(root, yaw, indices):
    specs = []
    recipes = []
    anchors = []
    for station, t in enumerate(indices):
        theta = yaw[t]
        anchor = root[t, :2]
        anchors.append([*anchor, theta])
        for lateral in np.linspace(-0.8, 0.8, 5):
            xy = anchor + lateral * np.array([-np.sin(theta), np.cos(theta)])
            for z in [0.4, 1.0, 1.6]:
                for kind in range(3):
                    specs.append([*xy, z, kind, theta])
                    recipes.append([t / (len(root) - 1), lateral, z, kind])
    return np.array(specs), np.array(recipes), np.array(anchors)
