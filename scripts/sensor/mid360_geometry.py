#!/usr/bin/env python3
"""MID-360 sensor geometry on the Unitree G1 (roadmap Phase 0.9, CPU only).

For each candidate MID-360 mount, forward kinematics of the G1 URDF gives:

1. the sensor height and its gravity-aligned ("body-frame") elevation field of view, standing and
   under the torso pitch/roll of executed and reference gaits;
2. the blind floor radius (closest visible floor point) per azimuth;
3. the last horizontal distance at which an overhead beam is still visible;
4. the fraction of rays blocked by the robot's own links: capsule/box proxies, checked against a
   triangle-mesh ray cast through MuJoCo when it is importable.

The geometry core is numpy-only and unit-tested in ``tests/test_mid360_geometry.py``. The driver
reads gitignored data under ``workspace/`` and writes ``results.json``, ``summary.md`` and figures.

Usage (CPU only; never inside Isaac)::

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 .venv_native/bin/python \
        scripts/sensor/mid360_geometry.py --out workspace/phase0/sensor_geometry
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
import struct
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Livox MID-360 datasheet values. Elevations are in the sensor frame (z along the housing axis).
MID360 = {
    "h_fov_deg": 360.0,
    "elev_min_deg": -7.0,
    "elev_max_deg": 52.0,
    "blind_zone_m": 0.1,
    "range_m": 40.0,
    "frame_rate_hz": 10.0,
    "scan": "non-repetitive",
}

REPO = Path(__file__).resolve().parents[2]
URDF_REL = "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
MESH_REL = "gear_sonic/data/assets/robot_description/meshes/g1"
MJCF_REL = "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
G1_29DOF_REL = "vendor/sonic/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof.urdf"

# Isaac spawn pose of the SONIC G1 (gear_sonic/envs/manager_env/robots/g1.py, init_state).
G1_DEFAULT_JOINTS = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}
G1_SPAWN_HEIGHT_M = 0.76
FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")


# --------------------------------------------------------------------------------------------
# Rotations
# --------------------------------------------------------------------------------------------


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_matrix(rpy) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (float(v) for v in rpy)
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def axis_angle_matrix(axis, angle) -> np.ndarray:
    """Rodrigues rotation about a fixed unit axis, batched over ``angle`` -> (..., 3, 3)."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    a = np.asarray(angle, dtype=float)[..., None, None]
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(a) * kx + (1.0 - np.cos(a)) * (kx @ kx)


def quat_wxyz_to_matrix(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def yaw_of(R) -> np.ndarray:
    """Heading of the body x axis projected on the horizontal plane."""
    R = np.asarray(R)
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def pitch_roll_of(R):
    """Z-Y-X Euler pitch (positive = x axis tipped down, i.e. forward lean) and roll, radians."""
    R = np.asarray(R)
    pitch = np.arcsin(np.clip(-R[..., 2, 0], -1.0, 1.0))
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    return pitch, roll


def heading_matrix(yaw) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    z, o = np.zeros_like(c), np.ones_like(c)
    rows = (np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1))
    return np.stack(rows, -2)


# --------------------------------------------------------------------------------------------
# Field of view
# --------------------------------------------------------------------------------------------


def direction(az, el) -> np.ndarray:
    """Unit vector from azimuth (from +x towards +y) and elevation (from the x-y plane), radians."""
    az, el = np.broadcast_arrays(np.asarray(az, float), np.asarray(el, float))
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


def az_el(d):
    d = np.asarray(d, float)
    return np.arctan2(d[..., 1], d[..., 0]), np.arcsin(np.clip(d[..., 2], -1.0, 1.0))


def sample_fov_directions(n: int, el_lo_deg: float, el_hi_deg: float) -> np.ndarray:
    """Deterministic golden-angle spiral, uniform in solid angle over the elevation band."""
    i = np.arange(n) + 0.5
    s_lo, s_hi = math.sin(math.radians(el_lo_deg)), math.sin(math.radians(el_hi_deg))
    z = s_lo + (s_hi - s_lo) * i / n
    az = np.mod(i * math.pi * (3.0 - math.sqrt(5.0)), 2 * math.pi)
    rho = np.sqrt(1.0 - z * z)
    return np.stack([rho * np.cos(az), rho * np.sin(az), z], -1)


def in_fov(d_sensor, el_lo_deg: float, el_hi_deg: float) -> np.ndarray:
    z = np.asarray(d_sensor)[..., 2]
    return (z >= math.sin(math.radians(el_lo_deg)) - 1e-12) & (
        z <= math.sin(math.radians(el_hi_deg)) + 1e-12
    )


def elevation_window(n_up, az, el_lo_deg: float, el_hi_deg: float):
    """Gravity-frame elevations covered by the sensor band at heading-frame azimuth ``az``.

    ``n_up`` is the sensor z axis expressed in the heading frame (z up, x forward). A direction at
    elevation ``e`` has sensor elevation asin(A cos e + B sin e) with A the horizontal component
    of ``n_up`` along ``az`` and B its vertical component. Returns (lo, hi, gap_lo, gap_hi) in
    radians: the covered set is [lo, hi] minus the open gap (gap is NaN when the set is one
    interval, lo/hi are NaN when it is empty).
    """
    n_up = np.asarray(n_up, float)
    az = np.asarray(az, float)
    A = n_up[..., 0] * np.cos(az) + n_up[..., 1] * np.sin(az)
    B = n_up[..., 2] * np.ones_like(A)
    C = np.hypot(A, B)
    phi = np.arctan2(A, B)  # A cos e + B sin e = C sin(e + phi)
    s_lo, s_hi = math.sin(math.radians(el_lo_deg)), math.sin(math.radians(el_hi_deg))
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = s_lo / C, s_hi / C
    empty = (a > 1.0) | (b < -1.0)
    ua, ub = np.arcsin(np.clip(a, -1, 1)), np.arcsin(np.clip(b, -1, 1))
    pieces = []
    for lo_u, hi_u in ((ua, ub), (np.pi - ub, np.pi - ua)):
        for k in (-2, -1, 0, 1):
            lo = np.maximum(lo_u + 2 * np.pi * k - phi, -np.pi / 2)
            hi = np.minimum(hi_u + 2 * np.pi * k - phi, np.pi / 2)
            ok = (hi >= lo) & ~empty
            pieces.append((np.where(ok, lo, np.nan), np.where(ok, hi, np.nan)))
    los = np.stack([p[0] for p in pieces], -1)
    his = np.stack([p[1] for p in pieces], -1)
    lo = np.min(np.where(np.isnan(los), np.inf, los), -1)
    hi = np.max(np.where(np.isnan(his), -np.inf, his), -1)
    lo, hi = np.where(np.isfinite(lo), lo, np.nan), np.where(np.isfinite(hi), hi, np.nan)
    # Gap: the largest uncovered stretch between the sorted pieces.
    order = np.argsort(np.where(np.isnan(los), np.inf, los), axis=-1)
    los_s = np.take_along_axis(los, order, -1)
    reach = np.fmax.accumulate(np.take_along_axis(his, order, -1), axis=-1)
    gap_start, gap_end = reach[..., :-1], los_s[..., 1:]
    g_len = np.where(gap_end > gap_start + 1e-9, gap_end - gap_start, -1.0)
    g_len = np.where(np.isnan(g_len), -1.0, g_len)
    j = np.argmax(g_len, -1)[..., None]
    has_gap = np.take_along_axis(g_len, j, -1)[..., 0] > 0
    gap_lo = np.where(has_gap, np.take_along_axis(gap_start, j, -1)[..., 0], np.nan)
    gap_hi = np.where(has_gap, np.take_along_axis(gap_end, j, -1)[..., 0], np.nan)
    return lo, hi, gap_lo, gap_hi


def floor_blind_radius(height, el_lo):
    """Horizontal distance to the closest floor point on the lowest ray (inf if it is not down)."""
    height, el_lo = np.broadcast_arrays(np.asarray(height, float), np.asarray(el_lo, float))
    out = np.full(height.shape, np.inf)
    down = el_lo < 0
    out[down] = height[down] / np.tan(-el_lo[down])
    return out


def overhead_last_visible(dh, el_lo, el_hi):
    """Smallest horizontal distance at which an edge ``dh`` above the sensor is inside [lo, hi].

    For dh > 0 the edge leaves through the top of the window, for dh < 0 through the bottom.
    Returns inf when the edge is never inside the window (e.g. dh > 0 but el_hi <= 0).
    """
    dh, el_lo, el_hi = np.broadcast_arrays(
        np.asarray(dh, float), np.asarray(el_lo, float), np.asarray(el_hi, float)
    )
    out = np.full(dh.shape, np.inf)
    up = (dh > 0) & (el_hi > 0)
    out[up] = dh[up] / np.tan(el_hi[up])
    down = (dh < 0) & (el_lo < 0)
    out[down] = -dh[down] / np.tan(-el_lo[down])
    level = (dh == 0) & (el_lo <= 0) & (el_hi >= 0)
    out[level] = 0.0
    return out


# --------------------------------------------------------------------------------------------
# Ray / primitive intersection (one origin, N unit directions). Misses are +inf; an origin
# inside the primitive returns 0.
# --------------------------------------------------------------------------------------------


def ray_sphere(o, d, center, radius) -> np.ndarray:
    oc = np.asarray(o, float) - np.asarray(center, float)
    b = d @ oc
    c = oc @ oc - radius * radius
    if c <= 0:
        return np.zeros(len(d))
    disc = b * b - c
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc >= 0) & (t >= 0), t, np.inf)


def ray_capsule(o, d, a, b, radius) -> np.ndarray:
    """Ray against the capsule of radius ``radius`` around segment a-b."""
    o, a, b = (np.asarray(v, float) for v in (o, a, b))
    ba = b - a
    baba = ba @ ba
    oa = o - a
    if baba < 1e-18:
        return ray_sphere(o, d, a, radius)
    h0 = np.clip((oa @ ba) / baba, 0.0, 1.0)
    if np.sum((oa - h0 * ba) ** 2) <= radius * radius:
        return np.zeros(len(d))
    bard = d @ ba
    baoa = ba @ oa
    rdoa = d @ oa
    oaoa = oa @ oa
    ka = baba - bard * bard
    kb = baba * rdoa - baoa * bard
    kc = baba * oaoa - baoa * baoa - radius * radius * baba
    t = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = kb * kb - ka * kc
        tc = (-kb - np.sqrt(np.maximum(disc, 0.0))) / ka
        y = baoa + tc * bard
        body = (disc >= 0) & (ka > 1e-12) & (y > 0) & (y < baba) & (tc >= 0)
    t[body] = tc[body]
    rest = ~body
    if rest.any():
        ts = np.minimum(ray_sphere(o, d[rest], a, radius), ray_sphere(o, d[rest], b, radius))
        t[rest] = ts
    return t


def ray_box(o, d, center, R, half) -> np.ndarray:
    """Ray against an oriented box (rotation R box->world, half extents ``half``)."""
    R = np.asarray(R, float)
    half = np.asarray(half, float)
    ol = R.T @ (np.asarray(o, float) - np.asarray(center, float))
    if np.all(np.abs(ol) <= half):
        return np.zeros(len(d))
    dl = d @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dl
        t1 = (-half - ol) * inv
        t2 = (half - ol) * inv
    t1 = np.where(np.isnan(t1), -np.inf, t1)
    t2 = np.where(np.isnan(t2), np.inf, t2)
    tmin = np.max(np.minimum(t1, t2), -1)
    tmax = np.min(np.maximum(t1, t2), -1)
    hit = (tmax >= tmin) & (tmax >= 0)
    return np.where(hit, np.maximum(tmin, 0.0), np.inf)


# --------------------------------------------------------------------------------------------
# Meshes and proxy fitting
# --------------------------------------------------------------------------------------------


def read_stl(path) -> np.ndarray:
    """Triangles (F, 3, 3) from a binary or ASCII STL."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                      ("a", "<u2")]), count=n, offset=84)
            return rec["v"].astype(float)
    verts = [
        [float(x) for x in line.split()[1:4]]
        for line in data.decode("ascii", "ignore").splitlines()
        if line.strip().startswith("vertex")
    ]
    return np.asarray(verts, float).reshape(-1, 3, 3)


def fit_capsule(points, quantile: float = 1.0):
    """PCA capsule (a, b, r) around ``points``; ``quantile`` < 1 trims the radius."""
    p = np.asarray(points, float)
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    axis = vt[0]
    s = (p - c) @ axis
    radial = np.linalg.norm((p - c) - s[:, None] * axis, axis=1)
    r = float(np.quantile(radial, quantile))
    # Pull each end cap in only as far as every point stays inside its hemisphere.
    reach = np.sqrt(np.clip(r * r - radial**2, 0.0, None))
    lo, hi = float(np.min(s + reach)), float(np.max(s - reach))
    if hi < lo:
        mid = 0.5 * (lo + hi)
        lo = hi = mid
        if quantile >= 1.0:
            r = float(np.max(np.linalg.norm(p - (c + mid * axis), axis=1)))
    return c + lo * axis, c + hi * axis, r


def fit_spheres(points, k: int, quantile: float = 1.0, iters: int = 25, seed: int = 0):
    """k-means sphere set (centres, radii) covering ``points``; ``quantile`` trims each radius."""
    p = np.asarray(points, float)
    k = min(k, len(p))
    rng = np.random.default_rng(seed)
    c = p[rng.choice(len(p), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((p[:, None, :] - c[None]) ** 2).sum(-1), 1)
        c = np.stack([p[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
    lab = np.argmin(((p[:, None, :] - c[None]) ** 2).sum(-1), 1)
    r = np.array([np.quantile(np.linalg.norm(p[lab == j] - c[j], axis=1), quantile)
                  if np.any(lab == j) else 0.0 for j in range(k)])
    keep = r > 0
    return c[keep], r[keep]


def fit_box(points):
    """PCA oriented bounding box (center, R box->frame, half extents)."""
    p = np.asarray(points, float)
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    R = vt.T
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    loc = (p - c) @ R
    lo, hi = loc.min(0), loc.max(0)
    return c + R @ ((lo + hi) / 2), R, (hi - lo) / 2


# --------------------------------------------------------------------------------------------
# URDF model and forward kinematics
# --------------------------------------------------------------------------------------------


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


@dataclass
class Geom:
    link: str
    kind: str
    xyz: np.ndarray
    rpy: np.ndarray
    radius: float = 0.0
    length: float = 0.0
    size: np.ndarray | None = None
    mesh: str | None = None


@dataclass
class UrdfModel:
    path: Path
    joints: list[Joint]
    collisions: list[Geom]
    visuals: list[Geom]
    root: str = ""
    by_child: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "UrdfModel":
        tree = ET.parse(path).getroot()
        joints = []
        for j in tree.findall("joint"):
            o = j.find("origin")
            a = j.find("axis")
            joints.append(
                Joint(
                    j.get("name"),
                    j.get("type"),
                    j.find("parent").get("link"),
                    j.find("child").get("link"),
                    _vec(o, "xyz"),
                    _vec(o, "rpy"),
                    np.array([float(v) for v in a.get("xyz").split()]) if a is not None else
                    np.array([1.0, 0.0, 0.0]),
                )
            )
        cols, vis = [], []
        for link in tree.findall("link"):
            for tag, out in (("collision", cols), ("visual", vis)):
                for el in link.findall(tag):
                    g = el.find("geometry")[0]
                    o = el.find("origin")
                    geom = Geom(link.get("name"), g.tag, _vec(o, "xyz"), _vec(o, "rpy"))
                    if g.tag in ("sphere", "cylinder", "capsule"):
                        geom.radius = float(g.get("radius"))
                        geom.length = float(g.get("length", 0.0))
                    elif g.tag == "box":
                        geom.size = np.array([float(v) for v in g.get("size").split()])
                    elif g.tag == "mesh":
                        geom.mesh = g.get("filename")
                    out.append(geom)
        children = {j.child for j in joints}
        parents = [j.parent for j in joints if j.parent not in children]
        model = cls(Path(path), [], cols, vis, root=parents[0])
        # Topological order from the root.
        pending, known = list(joints), {model.root}
        while pending:
            progressed = False
            for j in list(pending):
                if j.parent in known:
                    model.joints.append(j)
                    known.add(j.child)
                    pending.remove(j)
                    progressed = True
            if not progressed:
                raise ValueError(f"disconnected joints in {path}")
        model.by_child = {j.child: j for j in model.joints}
        return model

    def joint(self, name: str) -> Joint:
        return next(j for j in self.joints if j.name == name)

    @property
    def revolute_names(self) -> list[str]:
        return [j.name for j in self.joints if j.kind in ("revolute", "continuous")]

    def fk(self, q: dict, root_pos, root_R) -> dict:
        """World (R, p) of every link, batched over frames. Missing joints are held at zero."""
        root_pos = np.atleast_2d(np.asarray(root_pos, float))
        root_R = np.asarray(root_R, float).reshape(-1, 3, 3)
        T = max(len(root_pos), len(root_R))
        poses = {self.root: (np.broadcast_to(root_R, (T, 3, 3)),
                             np.broadcast_to(root_pos, (T, 3)))}
        for j in self.joints:
            Rp, pp = poses[j.parent]
            Ro = rpy_to_matrix(j.rpy)
            R = Rp @ Ro
            p = pp + Rp @ j.xyz
            if j.kind in ("revolute", "continuous"):
                angle = np.broadcast_to(np.asarray(q.get(j.name, 0.0), float), (T,))
                R = R @ axis_angle_matrix(j.axis, angle)
            poses[j.child] = (R, p)
        return poses


def _vec(origin, key) -> np.ndarray:
    if origin is None or origin.get(key) is None:
        return np.zeros(3)
    return np.array([float(v) for v in origin.get(key).split()])


@dataclass
class Mount:
    name: str
    parent: str
    xyz: np.ndarray
    rpy: np.ndarray
    source: str

    @property
    def R(self) -> np.ndarray:
        return rpy_to_matrix(self.rpy)

    def pose(self, parent_R, parent_p):
        """Sensor (R, p) in world from batched parent poses."""
        return parent_R @ self.R, parent_p + parent_R @ self.xyz


def mount_from_urdf(path, name: str, joint: str = "mid360_joint") -> Mount:
    j = UrdfModel.load(path).joint(joint)
    return Mount(name, j.parent, j.xyz, j.rpy, str(path))


def mount_in_pelvis(path, joint: str = "mid360_joint"):
    """Sensor (R, p) in the pelvis frame of its own URDF with all joints at zero."""
    model = UrdfModel.load(path)
    mount = mount_from_urdf(path, "probe", joint)
    poses = model.fk({}, np.zeros((1, 3)), np.eye(3)[None])
    R, p = mount.pose(*poses[mount.parent])
    return R[0], p[0]


def transplant_mount(path, dst: UrdfModel, name: str, joint: str = "mid360_joint") -> Mount:
    """Re-express a mount from another URDF in ``dst``'s parent-link frame (zero joints).

    The decoupled_wbc g1_29dof.urdf places torso_link 0.010 m higher than main.urdf
    (waist offsets 0.035 + 0.019 vs 0.044 + 0), so its raw offset is not comparable.
    """
    src = mount_from_urdf(path, name, joint)
    R_s, p_s = mount_in_pelvis(path, joint)
    poses = dst.fk({}, np.zeros((1, 3)), np.eye(3)[None])
    R_d, p_d = poses[src.parent][0][0], poses[src.parent][1][0]
    R_src_parent = UrdfModel.load(path).fk({}, np.zeros((1, 3)), np.eye(3)[None])[src.parent][0][0]
    if not np.allclose(R_d, R_src_parent, atol=1e-9):
        raise ValueError("parent frames are not parallel; cannot keep the source rpy")
    return Mount(name, src.parent, R_d.T @ (p_s - p_d), src.rpy, str(path))


def sensor_up_in_heading(R_sensor, yaw) -> np.ndarray:
    """Sensor z axis in the gravity-aligned heading frame (z up, x along ``yaw``)."""
    return np.einsum("...ji,...j->...i", heading_matrix(yaw), np.asarray(R_sensor)[..., :, 2])


def foot_floor_z(poses, model: UrdfModel) -> np.ndarray:
    """Lowest point of the foot collision capsules (Isaac replaces cylinders with capsules)."""
    lows = []
    for g in model.collisions:
        if g.link not in FOOT_LINKS or g.kind != "cylinder":
            continue
        R, p = poses[g.link]
        Rg = rpy_to_matrix(g.rpy)
        half = 0.5 * g.length * Rg[:, 2]
        for s in (-1.0, 1.0):
            end = p + R @ (g.xyz + s * half)
            lows.append(end[..., 2] - g.radius)
    return np.min(np.stack(lows, 0), 0)


# --------------------------------------------------------------------------------------------
# Self-occlusion proxies
# --------------------------------------------------------------------------------------------


@dataclass
class Proxy:
    link: str
    kind: str  # "sphere" | "capsule" | "box"
    a: np.ndarray  # sphere/box centre or capsule end, link frame
    b: np.ndarray | None = None  # capsule end
    radius: float = 0.0
    R: np.ndarray | None = None  # box rotation in link frame
    half: np.ndarray | None = None
    tag: str = ""

    def world(self, Rl, pl) -> "Proxy":
        return Proxy(
            self.link,
            self.kind,
            pl + Rl @ self.a,
            None if self.b is None else pl + Rl @ self.b,
            self.radius,
            None if self.R is None else Rl @ self.R,
            self.half,
            self.tag,
        )

    def contains(self, x) -> bool:
        x = np.asarray(x, float)
        if self.kind == "sphere":
            return float(np.sum((x - self.a) ** 2)) <= self.radius**2
        if self.kind == "capsule":
            ba = self.b - self.a
            h = np.clip(((x - self.a) @ ba) / max(ba @ ba, 1e-18), 0, 1)
            return float(np.sum((x - self.a - h * ba) ** 2)) <= self.radius**2
        return bool(np.all(np.abs(self.R.T @ (x - self.a)) <= self.half))

    def cast(self, o, d) -> np.ndarray:
        if self.kind == "sphere":
            return ray_sphere(o, d, self.a, self.radius)
        if self.kind == "capsule":
            return ray_capsule(o, d, self.a, self.b, self.radius)
        return ray_box(o, d, self.a, self.R, self.half)


def cast_proxies(proxies_world, o, d):
    """First-hit distance and proxy index per ray (index -1 and inf for a miss)."""
    if not proxies_world:
        return np.full(len(d), np.inf), np.full(len(d), -1)
    t = np.stack([p.cast(o, d) for p in proxies_world], 0)
    idx = np.argmin(t, 0)
    tmin = t[idx, np.arange(len(d))]
    return tmin, np.where(np.isfinite(tmin), idx, -1)


def mesh_path(mesh_dir, filename: str) -> Path:
    return Path(mesh_dir) / Path(filename.replace("package://", "")).name


def collision_proxies(model: UrdfModel, mesh_dir) -> list[Proxy]:
    """The URDF collision set as Isaac spawns it (``replace_cylinders_with_capsules=True``).

    A cylinder of length L becomes a capsule whose straight segment is L long (assumed). Mesh
    collisions (wrists, Dex3 hand) are replaced by bounding capsules of their vertices.
    """
    out = []
    for g in model.collisions:
        Rg = rpy_to_matrix(g.rpy)
        if g.kind == "sphere":
            out.append(Proxy(g.link, "sphere", g.xyz, radius=g.radius, tag="urdf-sphere"))
        elif g.kind in ("cylinder", "capsule"):
            half = 0.5 * g.length * Rg[:, 2]
            out.append(Proxy(g.link, "capsule", g.xyz - half, g.xyz + half, g.radius,
                             tag=f"urdf-{g.kind}"))
        elif g.kind == "box":
            out.append(Proxy(g.link, "box", g.xyz, R=Rg, half=g.size / 2, tag="urdf-box"))
        elif g.kind == "mesh":
            v = read_stl(mesh_path(mesh_dir, g.mesh)).reshape(-1, 3) @ Rg.T + g.xyz
            a, b, r = fit_capsule(v)
            out.append(Proxy(g.link, "capsule", a, b, r, tag="urdf-mesh-capsule"))
    return out


RIGID_WITH_SENSOR = ("head_link", "torso_link", "logo_link")


def sphere_proxies(model: UrdfModel, mesh_dir, k: int = 16, quantile: float = 0.9,
                   skip=RIGID_WITH_SENSOR) -> list[Proxy]:
    """k-means spheres per moving visual mesh (radius = ``quantile`` of member distances).

    Links rigid with the sensor are skipped: their occlusion is a fixed sensor-frame mask,
    which the exact mesh cast shows to be empty for both mounts.
    """
    out = []
    for g in model.visuals:
        if g.kind != "mesh" or g.link in skip:
            continue
        v = read_stl(mesh_path(mesh_dir, g.mesh)).reshape(-1, 3)
        v = np.unique(np.round(v, 4), axis=0) @ rpy_to_matrix(g.rpy).T + g.xyz
        c, r = fit_spheres(v, k, quantile)
        out += [Proxy(g.link, "sphere", ci, radius=float(ri), tag=f"sphere-k{k}")
                for ci, ri in zip(c, r)]
    return out


def visual_proxies(model: UrdfModel, mesh_dir, slabs=None, skip=("head_link",)) -> list[Proxy]:
    """One capsule or box per visual mesh (whichever is smaller); ``slabs`` splits big links
    into boxes along their principal axis."""
    slabs = {"torso_link": 4, "pelvis_contour_link": 2} if slabs is None else slabs
    out = []
    for g in model.visuals:
        if g.kind != "mesh" or g.link in skip:
            continue
        v = read_stl(mesh_path(mesh_dir, g.mesh)).reshape(-1, 3)
        v = np.unique(np.round(v, 5), axis=0) @ rpy_to_matrix(g.rpy).T + g.xyz
        k = slabs.get(g.link, 1)
        if k > 1:
            c = v.mean(0)
            axis = np.linalg.svd(v - c, full_matrices=False)[2][0]
            s = (v - c) @ axis
            edges = np.quantile(s, np.linspace(0, 1, k + 1))
            for i in range(k):
                part = v[(s >= edges[i]) & (s <= edges[i + 1])]
                cb, Rb, hb = fit_box(part)
                out.append(Proxy(g.link, "box", cb, R=Rb, half=hb, tag=f"visual-box-slab{i}"))
            continue
        a, b, r = fit_capsule(v)
        cap_vol = math.pi * r * r * float(np.linalg.norm(b - a)) + 4 / 3 * math.pi * r**3
        cb, Rb, hb = fit_box(v)
        if 8 * float(np.prod(hb)) < cap_vol:
            out.append(Proxy(g.link, "box", cb, R=Rb, half=hb, tag="visual-box"))
        else:
            out.append(Proxy(g.link, "capsule", a, b, r, tag="visual-capsule"))
    return out


def place_proxies(proxies, poses, frame: int = 0):
    return [p.world(poses[p.link][0][frame], poses[p.link][1][frame]) for p in proxies]


class ProxyCaster:
    """Ray caster over proxies placed at one frame; drops proxies that contain the sensor."""

    def __init__(self, proxies, poses, frame, origin):
        placed = place_proxies(proxies, poses, frame)
        self.dropped = sorted({p.link + ":" + p.tag for p in placed if p.contains(origin)})
        self.proxies = [p for p in placed if not p.contains(origin)]
        self.links = [p.link for p in self.proxies]

    def __call__(self, o, d):
        t, idx = cast_proxies(self.proxies, o, d)
        return t, np.array([self.links[i] if i >= 0 else "" for i in idx], dtype=object)


class MeshCaster:
    """Exact triangle ray casting against the URDF visual meshes through MuJoCo (optional)."""

    def __init__(self, urdf_path, mesh_dir, exclude_links=("head_link",)):
        import mujoco

        self.mj = mujoco
        text = Path(urdf_path).read_text()
        compiler = (f'<mujoco><compiler meshdir="{mesh_dir}" discardvisual="false" '
                    'fusestatic="false" strippath="true"/></mujoco>')
        text = re.sub(r"(<robot[^>]*>)", lambda m: m.group(1) + compiler, text, count=1)
        self.m = mujoco.MjModel.from_xml_string(text)
        self.d = mujoco.MjData(self.m)
        body = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.m.nbody)]
        self.geom_link = np.array([body[b] for b in self.m.geom_bodyid], dtype=object)
        for i in range(self.m.ngeom):
            if self.m.geom_group[i] == 1 and self.geom_link[i] in exclude_links:
                self.m.geom_group[i] = 5
        self.group = np.array([0, 1, 0, 0, 0, 0], np.uint8)
        self.qadr = {
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j): int(self.m.jnt_qposadr[j])
            for j in range(self.m.njnt)
        }
        self.body_index = {n: i for i, n in enumerate(body)}
        self.root_R, self.root_p = np.eye(3), np.zeros(3)

    def set_pose(self, q: dict, root_R, root_p):
        """``q`` maps joint name to angle; the MuJoCo model is rooted at the pelvis."""
        self.d.qpos[:] = 0.0
        for name, adr in self.qadr.items():
            self.d.qpos[adr] = float(q.get(name, 0.0))
        self.mj.mj_kinematics(self.m, self.d)
        self.root_R, self.root_p = np.asarray(root_R, float), np.asarray(root_p, float)

    def link_position(self, link):
        return self.root_p + self.root_R @ self.d.xpos[self.body_index[link]]

    def __call__(self, o, d):
        o_l = self.root_R.T @ (np.asarray(o, float) - self.root_p)
        d_l = np.ascontiguousarray(np.asarray(d, float) @ self.root_R)
        n = len(d_l)
        gid = np.zeros(n, np.int32)
        dist = np.zeros(n)
        self.mj.mj_multiRay(self.m, self.d, o_l, d_l.reshape(-1), self.group, 1, -1, gid, dist,
                            None, n, 100.0)
        hit = dist >= 0
        t = np.where(hit, dist, np.inf)
        return t, np.where(hit, self.geom_link[np.maximum(gid, 0)], "")


def frame_q(q: dict, i: int) -> dict:
    return {k: float(np.asarray(v).reshape(-1)[i] if np.ndim(v) else v) for k, v in q.items()}


def floor_scan(caster, R_s, p_s, yaw, height, az_deg, el_step_deg=0.1, fov=None):
    """Closest visible floor distance per heading azimuth, with the given self-occlusion caster.

    Scans gravity-frame elevations below the horizon; a floor point is visible when its ray is
    inside the sensor band and hits no robot link. ``caster=None`` ignores self-occlusion.
    """
    fov = fov or (MID360["elev_min_deg"], MID360["elev_max_deg"])
    E = np.radians(np.arange(-89.95, 0.0, el_step_deg))
    AZ = np.radians(np.asarray(az_deg, float))
    d_h = direction(AZ[:, None], E[None, :])
    d_w = d_h @ heading_matrix(yaw).T
    inside = in_fov(d_w @ R_s, *fov)
    visible = inside.copy()
    blockers = np.zeros(inside.shape, dtype=object)
    if caster is not None and inside.any():
        t, link = caster(p_s, d_w[inside])
        floor_t = height / np.sin(-E)[None, :].repeat(len(AZ), 0)[inside]
        blocked = t < floor_t
        visible[inside] = ~blocked
        blockers[inside] = np.where(blocked, link, "")
    el_vis = np.where(visible, E[None, :], np.nan)
    lowest = np.full(len(AZ), np.nan)
    has = visible.any(1)
    lowest[has] = np.nanmin(el_vis[has], 1)
    radius = floor_blind_radius(np.full(len(AZ), height), np.where(has, lowest, 0.0))
    # Links that hide the floor between the FOV-only edge and the first visible point.
    hidden = {}
    for row in blockers:
        for name in row:
            if name:
                hidden[name] = hidden.get(name, 0) + 1
    return radius, np.degrees(lowest), hidden


def approach_last_seen(threshold, dt: float, speed: float = 0.6, scan_hz: float = 10.0,
                       d_max: float = 15.0, n_arrivals: int = 2000, seed: int = 0) -> np.ndarray:
    """Distance of the last 10 Hz scan that still contains an edge while walking in at ``speed``.

    ``threshold[f]`` is the smallest horizontal distance at which the edge is inside the band at
    pose ``f`` (inf: not visible at any distance). The pose sequence is replayed in time order
    (wrapping around), each approach gets a random arrival time and scan phase, and the edge is
    seen by a scan at distance d when d >= threshold of the pose at that instant. Returns one
    distance per approach; inf when no scan within ``d_max`` saw it.
    """
    threshold = np.asarray(threshold, float)
    T = len(threshold)
    rng = np.random.default_rng(seed)
    t_a = rng.uniform(0.0, T * dt, n_arrivals)
    phase = rng.uniform(0.0, 1.0 / scan_hz, n_arrivals)
    n = np.arange(int(d_max / speed * scan_hz) + 1)
    lag = phase[:, None] + n[None, :] / scan_hz
    d = speed * lag
    f = np.mod(np.floor((t_a[:, None] - lag) / dt).astype(int), T)
    seen = d >= threshold[f]
    return np.where(seen, d, np.inf).min(1)


def occlusion_stats(caster, R_s, p_s, dirs_sensor, blind_m=MID360["blind_zone_m"]):
    """Blocked fraction of FOV rays (uniform in solid angle) and first-hit link shares."""
    d_w = dirs_sensor @ R_s.T
    t, link = caster(p_s, d_w)
    blocked = np.isfinite(t)
    names, counts = np.unique(link[blocked].astype(str), return_counts=True)
    order = np.argsort(-counts)
    return {
        "blocked_fraction": float(blocked.mean()),
        "blocked_within_blind_zone_fraction": float((blocked & (t < blind_m)).mean()),
        "self_returns_fraction": float((blocked & (t >= blind_m)).mean()),
        "by_link": {str(names[i]): float(counts[i] / len(t)) for i in order},
        "mask": blocked,
    }


# --------------------------------------------------------------------------------------------
# Data: default pose, executed release rollouts, reference clips
# --------------------------------------------------------------------------------------------

GAIT_EXCLUDE = re.compile(
    r"carr|box|door|chair|cabinet|shelf|crouch|duck|kneel|crawl|sit|stair|ledge|platform|"
    r"backward|retrace|sidestep|sideways|squeez|arms held|hop|clutter|lower",
    re.I,
)
BEAM_HEIGHTS = tuple(round(1.0 + 0.05 * i, 2) for i in range(11))
NAMED_AZ = {"front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0}
AZ_GRID_DEG = np.arange(0.0, 360.0, 5.0)


def clip_kind(prompt: str) -> str:
    """gait = plain (unstyled) forward walking; crouch/duck and crawl are kept apart."""
    if re.search(r"crawl", prompt, re.I):
        return "crawl"
    if re.search(r"crouch|duck|lowers their body", prompt, re.I):
        return "crouch"
    if (prompt.startswith("A person ") and re.search(r"walk|steps forward", prompt)
            and not GAIT_EXCLUDE.search(prompt)):
        return "gait"
    return "other"


def find_workspace(start: Path = REPO) -> Path:
    for p in (start, *start.parents):
        if (p / "workspace" / "vendor" / "sonic").is_dir():
            return p / "workspace"
    raise FileNotFoundError("no workspace/ with vendor/sonic data above the repo; use --workspace")


def sonic_asset(rel: str, workspace: Path) -> Path:
    for base in (REPO / "vendor/sonic", workspace / "vendor/sonic"):
        if (base / rel).exists():
            return base / rel
    raise FileNotFoundError(rel)


def default_clip(model: UrdfModel) -> dict:
    q = {k: np.array([v]) for k, v in G1_DEFAULT_JOINTS.items()}
    poses = model.fk(q, np.zeros((1, 3)), np.eye(3)[None])
    z0 = float(foot_floor_z(poses, model)[0])
    return {"q": q, "root_pos": np.array([[0.0, 0.0, -z0]]), "root_R": np.eye(3)[None],
            "dt": None, "floor": 0.0, "id": "default", "prompt": "Isaac init_state pose"}


def load_executed(path: Path) -> dict:
    z = np.load(path)
    names = [str(n) for n in z["joint_names_isaac"]]
    return {
        "q": {n: z["robot_joint_pos"][:, i].astype(float) for i, n in enumerate(names)},
        "root_pos": z["robot_root_pos"].astype(float),
        "root_R": quat_wxyz_to_matrix(z["robot_root_quat"].astype(float)),
        "dt": float(z["dt"]),
        "floor": 0.0,  # Isaac ground plane; checked against the foot soles in results.json
        "terminated": bool(z["native_terminated"]),
        "progress": float(z["native_progress"]),
    }


def load_reference(path: Path) -> dict:
    z = np.load(path)
    names = [str(n) for n in z["joint_names"]]
    qpos = z["qpos"].astype(float)
    return {
        "q": {n: qpos[:, 7 + i] for i, n in enumerate(names)},
        "root_pos": qpos[:, :3],
        "root_R": quat_wxyz_to_matrix(qpos[:, 3:7]),
        "dt": 1.0 / float(z["fps"]),
        "floor": None,  # per-frame lowest foot sole: kinematic references are not floor-aligned
    }


def planar_speed(root_pos, dt, window_s=0.2) -> np.ndarray:
    if dt is None or len(root_pos) < 3:
        return np.zeros(len(root_pos))
    v = np.gradient(root_pos[:, :2], dt, axis=0)
    k = max(1, int(round(window_s / dt)))
    v = np.stack([np.convolve(v[:, i], np.ones(k) / k, mode="same") for i in range(2)], 1)
    return np.linalg.norm(v, axis=1)


def head_points_in_torso(model: UrdfModel, mesh_dir) -> np.ndarray:
    """Upper part of the head shell in the torso frame (for head-top clearance)."""
    j = model.by_child["head_link"]
    g = next(v for v in model.visuals if v.link == "head_link")
    v = np.unique(np.round(read_stl(mesh_path(mesh_dir, g.mesh)).reshape(-1, 3), 4), axis=0)
    v = v @ rpy_to_matrix(g.rpy).T + g.xyz
    v = v @ rpy_to_matrix(j.rpy).T + j.xyz
    top = v[v[:, 2] > v[:, 2].max() - 0.1]
    return top[:: max(1, len(top) // 1500)]


def sensor_frames(model: UrdfModel, mount: Mount, clip: dict, head_pts) -> dict:
    """Per-frame sensor pose, height above the floor and gravity-frame elevation windows."""
    poses = model.fk(clip["q"], clip["root_pos"], clip["root_R"])
    R_t, p_t = poses["torso_link"]
    R_s, p_s = mount.pose(*poses[mount.parent])
    T = len(p_s)
    floor = foot_floor_z(poses, model) if clip["floor"] is None else np.full(T, clip["floor"])
    yaw = yaw_of(R_t)
    n_up = sensor_up_in_heading(R_s, yaw)
    pitch, roll = pitch_roll_of(R_t)
    lo_e, hi_e = MID360["elev_min_deg"], MID360["elev_max_deg"]
    out = {
        "height": p_s[:, 2] - floor,
        "pitch": np.degrees(pitch),
        "roll": np.degrees(roll),
        "yaw": yaw,
        "R_s": R_s,
        "p_s": p_s,
        "n_up": n_up,
        "floor": floor,
        "head_top": np.max(p_t[:, None, 2] + np.einsum("tij,kj->tki", R_t, head_pts)[..., 2], 1)
        - floor,
        "foot_min": foot_floor_z(poses, model) - floor,
        "speed": planar_speed(clip["root_pos"], clip["dt"]),
        "dt": np.full(T, np.nan if clip["dt"] is None else clip["dt"]),
        "qmat": np.stack([np.broadcast_to(np.asarray(clip["q"].get(j, 0.0), float), (T,))
                          for j in model.revolute_names], 1),
    }
    for name, az in NAMED_AZ.items():
        lo, hi, glo, _ = elevation_window(n_up, math.radians(az), lo_e, hi_e)
        out[f"{name}_lo"], out[f"{name}_hi"] = np.degrees(lo), np.degrees(hi)
        out[f"{name}_gap"] = ~np.isnan(glo)
    lo, hi, _, _ = elevation_window(n_up[:, None, :], np.radians(AZ_GRID_DEG)[None, :], lo_e, hi_e)
    out["lowest"] = np.degrees(np.nanmin(lo, 1))
    out["highest"] = np.degrees(np.nanmax(hi, 1))
    for name in NAMED_AZ:
        out[f"blind_{name}"] = floor_blind_radius(out["height"], np.radians(out[f"{name}_lo"]))
    out["blind_min"] = floor_blind_radius(out["height"], np.radians(out["lowest"]))
    out["poses"] = poses
    return out


def select(frames: dict, mask) -> dict:
    mask = np.asarray(mask, bool)
    out = {}
    for k, v in frames.items():
        if k == "poses":
            out[k] = {n: (R[mask], p[mask]) for n, (R, p) in v.items()}
        elif isinstance(v, np.ndarray) and len(v) == len(mask):
            out[k] = v[mask]
        else:
            out[k] = v
    return out


def concat(frames_list: list) -> dict:
    out = {}
    for k in frames_list[0]:
        if k == "poses":
            out[k] = {n: (np.concatenate([f[k][n][0] for f in frames_list]),
                          np.concatenate([f[k][n][1] for f in frames_list]))
                      for n in frames_list[0][k]}
        else:
            out[k] = np.concatenate([np.atleast_1d(f[k]) for f in frames_list])
    return out


def corridor_last_visible(n_up, height, hb, half_width=0.3, d_max=8.0, step=0.01, fov=None):
    """Smallest distance at which any point of the beam's lower front edge with |y| <= half_width
    is inside the band (a beam spanning the robot's swept corridor)."""
    fov = fov or (MID360["elev_min_deg"], MID360["elev_max_deg"])
    d = np.arange(step, d_max + step / 2, step)
    y = np.linspace(-half_width, half_width, 25)
    D, Y = np.meshgrid(d, y, indexing="ij")
    az = np.arctan2(Y, D)
    el = np.arctan2(hb - height, np.hypot(D, Y))
    lo, hi, glo, ghi = elevation_window(np.broadcast_to(n_up, D.shape + (3,)), az, *fov)
    inside = (el >= lo - 1e-9) & (el <= hi + 1e-9)
    inside &= ~((el > glo) & (el < ghi))
    rng = np.sqrt(D**2 + Y**2 + (hb - height) ** 2)
    inside &= (rng >= MID360["blind_zone_m"]) & (rng <= MID360["range_m"])
    vis = inside.any(1)
    return float(d[vis].min()) if vis.any() else math.inf


def stats(x) -> dict:
    x = np.asarray(x, float)
    if x.size == 0:
        return {"n": 0}
    finite = x[np.isfinite(x)]
    out = {"n": int(x.size), "inf_fraction": float(np.mean(~np.isfinite(x)))}
    if finite.size:
        p = np.percentile(finite, [5, 50, 95])
        out.update(p5=float(p[0]), p50=float(p[1]), p95=float(p[2]),
                   min=float(finite.min()), max=float(finite.max()))
    return out


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items() if k != "mask"}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if math.isnan(x):
            return None
        return "inf" if math.isinf(x) else float(f"{x:.6g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------


def survey_mounts(workspace: Path) -> list:
    """Every G1 description on this host that declares a MID-360, with its orientation."""
    candidates = [
        workspace / "vendor/sonic" / URDF_REL,
        REPO / G1_29DOF_REL,
        REPO / G1_29DOF_REL.replace("g1_29dof.urdf", "g1_29dof_with_hand.urdf"),
        REPO / "vendor/sonic/decoupled_wbc/sim2mujoco/resources/robots/g1/g1.urdf",
        Path("/opt/ros/humble/share/unitree_description/urdf/g1/main.urdf"),
    ]
    out = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            m = mount_from_urdf(path, path.name)
        except StopIteration:
            continue
        R_p, p_p = mount_in_pelvis(path)
        z = m.R[:, 2]
        out.append({
            "file": str(path),
            "xyz_in_parent": m.xyz, "rpy": m.rpy,
            "sensor_position_in_pelvis_frame_zero_joints": p_p,
            "sensor_z_axis_in_torso": z,
            "orientation": "inverted" if z[2] < 0 else "upright",
            "tilt_from_torso_axis_deg": math.degrees(math.acos(abs(z[2]))),
        })
    return out


def posture_summary(fr: dict) -> dict:
    out = {k: stats(fr[k]) for k in ("height", "pitch", "roll", "head_top", "speed")}
    for name in NAMED_AZ:
        out[f"window_{name}_deg"] = {"lo": stats(fr[f"{name}_lo"]), "hi": stats(fr[f"{name}_hi"]),
                                     "gap_fraction": float(np.mean(fr[f"{name}_gap"]))}
    out["lowest_elevation_any_az_deg"] = stats(fr["lowest"])
    out["highest_elevation_any_az_deg"] = stats(fr["highest"])
    out["blind_floor_radius_fov_only_m"] = {n: stats(fr[f"blind_{n}"]) for n in (*NAMED_AZ, "min")}
    return out


def overhead_table(fr: dict, corridor_frames: int = 64, speed=0.6, approach=True) -> dict:
    """Last horizontal distance (from the sensor) at which the beam's lower front edge is in the
    band: per pose (``center_m``: on the heading line; ``corridor_0.3m_m``: anywhere within
    |y| <= 0.3 m), and for a replayed 0.6 m/s approach scanned at 10 Hz (``approach_m``)."""
    out = {}
    T = len(fr["height"])
    sub = np.unique(np.linspace(0, T - 1, min(T, corridor_frames)).astype(int))
    dt = float(fr["dt"][0])
    for hb in BEAM_HEIGHTS:
        center = overhead_last_visible(hb - fr["height"], np.radians(fr["front_lo"]),
                                       np.radians(fr["front_hi"]))
        center = np.where(fr["front_gap"], np.nan, center)
        corridor = np.array([corridor_last_visible(fr["n_up"][i], fr["height"][i], hb)
                             for i in sub])
        row = {
            "beam_above_sensor_fraction": float(np.mean(hb > fr["height"])),
            "beam_above_head_top_fraction": float(np.mean(hb > fr["head_top"])),
            "center_m": stats(center),
            "corridor_0.3m_m": stats(corridor),
        }
        if approach and T > 1 and np.isfinite(dt):
            last = approach_last_seen(np.where(np.isnan(center), np.inf, center), dt, speed)
            row["approach_m"] = stats(last)
            row["approach_time_to_arrival_s"] = stats(last / speed)
        out[f"{hb:.2f}"] = row
    return out


def occlusion_block(casters: dict, fr: dict, frames, dirs_sensor, az_deg, el_step) -> dict:
    """Blocked fractions and occlusion-aware blind floor radius over the given frames.

    ``casters`` maps a name to a per-frame factory; the first one is the reference for IoU.
    """
    res, ref_masks = {}, {}
    ref_name = next(iter(casters))
    for cname, make in casters.items():
        fracs, within, ious, links = [], [], [], {}
        radii = {n: [] for n in (*NAMED_AZ, "min")}
        dropped = set()
        for i in frames:
            caster = make(i)
            dropped.update(getattr(caster, "dropped", []))
            R_s, p_s = fr["R_s"][i], fr["p_s"][i]
            st = occlusion_stats(caster, R_s, p_s, dirs_sensor)
            fracs.append(st["blocked_fraction"])
            within.append(st["blocked_within_blind_zone_fraction"])
            if cname == ref_name:
                ref_masks[int(i)] = st["mask"]
            else:
                ref = ref_masks[int(i)]
                union = np.sum(ref | st["mask"])
                ious.append(float(np.sum(ref & st["mask"]) / union) if union else 1.0)
            for k, v in st["by_link"].items():
                links[k] = links.get(k, 0.0) + v / len(frames)
            r, _, _ = floor_scan(caster, R_s, p_s, fr["yaw"][i], fr["height"][i], az_deg, el_step)
            for n, az in NAMED_AZ.items():
                radii[n].append(r[int(np.argmin(np.abs(np.asarray(az_deg) - az)))])
            radii["min"].append(float(np.min(r)))
        res[cname] = {
            "blocked_fraction": stats(fracs),
            "blocked_within_blind_zone_fraction": stats(within),
            "mean_share_by_link": dict(sorted(links.items(), key=lambda kv: -kv[1])),
            "blind_floor_radius_m": {n: stats(v) for n, v in radii.items()},
            "proxies_dropped_for_containing_sensor": sorted(dropped),
        }
        if cname != ref_name:
            res[cname][f"iou_vs_{ref_name}"] = stats(ious)
    return res


def pitch_sweep(model: UrdfModel, mounts: dict, clip: dict, pitches_deg) -> dict:
    out = {}
    for name, mount in mounts.items():
        rows = []
        for p in pitches_deg:
            c = dict(clip)
            c["root_R"] = (rot_y(math.radians(p)) @ clip["root_R"][0])[None]
            fr = sensor_frames(model, mount, c, np.zeros((1, 3)))
            rows.append({
                "root_pitch_deg": p,
                "torso_pitch_deg": float(fr["pitch"][0]),
                "front": [float(fr["front_lo"][0]), float(fr["front_hi"][0])],
                "back": [float(fr["back_lo"][0]), float(fr["back_hi"][0])],
                "left": [float(fr["left_lo"][0]), float(fr["left_hi"][0])],
            })
        out[name] = rows
    return out


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_frames(model, mounts, clips, head_pts) -> dict:
    """Per mount: default standing, executed/reference gait, crouch and crawl frame sets."""
    stand = default_clip(model)
    frames = {name: {} for name in mounts}
    loaded = []
    for c in clips:
        if c["kind"] in ("gait", "crouch", "crawl"):
            ref = load_reference(c["reference"]) if c["kind"] != "crawl" else None
            loaded.append((c, load_executed(c["executed"]), ref))
    for name, mount in mounts.items():
        per = frames[name]
        per["default_standing"] = sensor_frames(model, mount, stand, head_pts)
        h_stand = float(per["default_standing"]["height"][0])
        ex = {"gait": [], "crouch": [], "crawl": []}
        rf = {"gait": [], "crouch": []}
        for c, e, r in loaded:
            ex[c["kind"]].append(sensor_frames(model, mount, e, head_pts))
            if r is not None:
                rf[c["kind"]].append(sensor_frames(model, mount, r, head_pts))
        walking = lambda f: (f["speed"] >= 0.3) & (f["speed"] <= 1.6)  # noqa: E731
        crouched = lambda f: (f["height"] <= h_stand - 0.15) & (f["pitch"] < 60)  # noqa: E731
        g, rg = concat(ex["gait"]), concat(rf["gait"])
        per["gait_executed"] = select(g, walking(g))
        per["gait_reference"] = select(rg, walking(rg))
        cr, rcr = concat(ex["crouch"]), concat(rf["crouch"])
        per["crouch_executed"] = select(cr, crouched(cr))
        per["crouch_reference"] = select(rcr, crouched(rcr))
        cw = concat(ex["crawl"])
        per["crawl_executed"] = select(cw, cw["pitch"] >= 45)
    return frames


def analyze(args) -> tuple[dict, dict]:
    ws = args.workspace or find_workspace()
    urdf = sonic_asset(URDF_REL, ws)
    mesh_dir = sonic_asset(MESH_REL, ws)
    upright_urdf = REPO / G1_29DOF_REL
    model = UrdfModel.load(urdf)
    mounts = {"inverted": mount_from_urdf(urdf, "inverted"),
              "upright": transplant_mount(upright_urdf, model, "upright")}
    raw_upright = mount_from_urdf(upright_urdf, "upright-raw")
    head_pts = head_points_in_torso(model, mesh_dir)
    lo_e, hi_e = MID360["elev_min_deg"], MID360["elev_max_deg"]

    hind = ws / "m2s-hindsight-dataset-v1-20260911" / "motions"
    release = ws / "teacher-8192-500-review" / "eval" / "release"
    clips = []
    for split in ("train", "development"):
        for f in sorted((release / split / "metrics").glob("hindsight_*.pose.npz")):
            mid = f.name.split(".")[0].removeprefix("hindsight_")
            prompt = json.loads((hind / mid / "metadata.json").read_text())["prompts"][0]
            clips.append({"id": mid, "split": split, "prompt": prompt, "kind": clip_kind(prompt),
                          "executed": f, "reference": hind / mid / "reference.npz"})

    results = {
        "item": "roadmap Phase 0.9: MID-360 sensor geometry on G1 (CPU)",
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "code_commit": git_sha(),
        "spec": MID360,
        "conventions": {
            "heading_frame": "origin at the sensor, z up (gravity), x = torso heading",
            "elevation": "degrees above the horizontal plane in the heading frame",
            "distances": "horizontal, from the sensor (not from the robot's leading edge)",
            "inf": "never inside the field of view",
            "blocked_fraction": "share of FOV directions (uniform in solid angle) hitting a link",
            "stats": "p5/p50/p95/min/max over frames; inf_fraction = share of inf values",
        },
        "inputs": {
            "workspace": str(ws),
            "kinematics_urdf": {"path": str(urdf), "sha256": sha256(urdf)},
            "upright_mount_urdf": {"path": str(upright_urdf), "sha256": sha256(upright_urdf)},
            "mesh_dir": str(mesh_dir),
            "executed_rollouts": str(release),
            "reference_clips": str(hind),
        },
        "mounts": {},
        "mount_survey": survey_mounts(ws),
        "checks": {
            "upright_raw_xyz_in_its_own_torso_frame": raw_upright.xyz,
            "upright_xyz_in_main_urdf_torso_frame": mounts["upright"].xyz,
            "sensor_position_gap_between_mounts_m": float(
                np.linalg.norm(mounts["upright"].xyz - mounts["inverted"].xyz)),
        },
    }
    for name, m in mounts.items():
        wins = {}
        for az_name, az in NAMED_AZ.items():
            lo, hi, _, _ = elevation_window(m.R[:, 2], math.radians(az), lo_e, hi_e)
            wins[az_name] = [math.degrees(lo), math.degrees(hi)]
        results["mounts"][name] = {"parent": m.parent, "xyz": m.xyz, "rpy": m.rpy,
                                   "source": m.source, "sensor_z_axis_in_torso": m.R[:, 2],
                                   "window_in_level_torso_frame_deg": wins}

    # FK validation against the MuJoCo-native reference bodies and the Isaac reference bodies.
    probe = next(c for c in clips if c["kind"] == "gait")
    z = np.load(probe["reference"])
    ref = load_reference(probe["reference"])
    poses = model.fk(ref["q"], ref["root_pos"], ref["root_R"])
    val = {"probe_clip": probe["id"]}
    val["reference_npz_max_body_error_m"] = max(
        float(np.abs(poses[str(b)][1] - z["body_position_m"][:, i]).max())
        for i, b in enumerate(z["body_names"]))
    pz = np.load(probe["executed"])
    q = {str(n): pz["ref_track_joint_pos"][:, i].astype(float)
         for i, n in enumerate(pz["joint_names_isaac"])}
    P = model.fk(q, pz["ref_track_root_pos"].astype(float),
                 quat_wxyz_to_matrix(pz["ref_track_root_quat"].astype(float)))
    val["isaac_ref_track_max_body_error_m"] = max(
        float(np.abs(P[str(b)][1] - pz["ref_track_body_pos"][:, i]).max())
        for i, b in enumerate(pz["body_names"]))
    results["checks"]["fk_validation"] = val

    frames = load_frames(model, mounts, clips, head_pts)
    results["clip_sets"] = {
        k: [c["id"] for c in clips if c["kind"] == k] for k in ("gait", "crouch", "crawl")}
    results["clip_sets"]["selection"] = {
        "gait": "unstyled 'A person ...' forward-walking prompts; frames with planar root "
                "speed 0.3-1.6 m/s (0.2 s smoothing)",
        "crouch": "crouch/duck/lower-body prompts; frames with the sensor >= 0.15 m below the "
                  "default standing height and torso pitch < 60 deg",
        "crawl": "crawl prompts; frames with torso pitch >= 45 deg",
        "executed": "release SONIC physics rollouts (Isaac, 50 Hz, floor z=0) of the repaired "
                    "hindsight clips (workspace/teacher-8192-500-review/eval/release)",
        "reference": "kinematic reference clips (30 Hz); floor = per-frame lowest foot sole",
    }
    inv = frames["inverted"]
    results["checks"]["executed_foot_sole_above_floor_m"] = stats(
        np.concatenate([inv["gait_executed"]["foot_min"], inv["crouch_executed"]["foot_min"]]))
    results["postures"] = {}
    for name, per in frames.items():
        results["postures"][name] = {}
        for pname, fr in per.items():
            results["postures"][name][pname] = posture_summary(fr)
            results["postures"][name][pname]["n_frames"] = int(len(fr["height"]))
    results["pitch_sweep"] = pitch_sweep(model, mounts, default_clip(model), list(range(-10, 31)))

    results["overhead"] = {
        name: {p: overhead_table(per[p]) for p in
               ("default_standing", "gait_executed", "gait_reference", "crouch_executed")}
        for name, per in frames.items()}

    # Self-occlusion and the occlusion-aware floor radius.
    dirs = sample_fov_directions(args.rays, lo_e, hi_e)
    proxy_sets = {
        "isaac_collision_capsules": collision_proxies(model, mesh_dir),
        "bounding_capsules_boxes": visual_proxies(model, mesh_dir),
        "sphere_proxies_k16": sphere_proxies(model, mesh_dir),
    }
    mesh = None
    if not args.no_mesh:
        try:
            mesh = MeshCaster(urdf, mesh_dir)
        except ImportError:
            results["checks"]["mesh_caster"] = "mujoco not importable; mesh check skipped"
    joint_names = model.revolute_names
    rng = np.random.default_rng(0)
    occ = {}
    plan = (("default_standing", 1, 2.0, 0.05),
            ("gait_executed", args.occlusion_frames, 10.0, 0.1),
            ("crouch_executed", max(1, args.occlusion_frames // 2), 10.0, 0.1))
    for name, per in frames.items():
        occ[name] = {}
        for pname, n_fr, az_step, el_step in plan:
            fr = per[pname]
            T = len(fr["height"])
            idx = np.arange(T) if T <= n_fr else np.sort(rng.choice(T, n_fr, replace=False))
            casters = {}
            if mesh is not None:
                def posed_mesh(i, fr=fr):
                    mesh.set_pose(dict(zip(joint_names, fr["qmat"][i])),
                                  fr["poses"]["pelvis"][0][i], fr["poses"]["pelvis"][1][i])
                    return mesh
                casters["visual_mesh_exact"] = posed_mesh
            for pxname, px in proxy_sets.items():
                casters[pxname] = (
                    lambda i, fr=fr, px=px: ProxyCaster(px, fr["poses"], i, fr["p_s"][i]))
            occ[name][pname] = occlusion_block(casters, fr, idx, dirs,
                                               np.arange(0.0, 360.0, az_step), el_step)
            occ[name][pname]["n_frames"] = int(len(idx))
    extent = body_extent(model, mesh_dir, frames["inverted"]["default_standing"])
    results["body_extent_default_standing_m"] = extent
    if mesh is not None:
        fr = frames["inverted"]["default_standing"]
        mesh.set_pose(dict(zip(joint_names, fr["qmat"][0])), fr["poses"]["pelvis"][0][0],
                      fr["poses"]["pelvis"][1][0])
        results["checks"]["mesh_vs_fk_torso_position_error_m"] = float(np.abs(
            mesh.link_position("torso_link") - fr["poses"]["torso_link"][1][0]).max())
    results["self_occlusion"] = occ
    results["self_occlusion_setup"] = {
        "rays_per_frame": args.rays,
        "isaac_collision_capsules": "main.urdf collision set; cylinders -> capsules (segment = "
                                    "cylinder length, assumed); mesh collisions -> bounding "
                                    "capsules",
        "bounding_capsules_boxes": "one PCA bounding capsule or box per visual mesh (smaller "
                                   "volume wins); torso 4 boxes, pelvis contour 2 boxes",
        "sphere_proxies_k16": "16 k-means spheres per moving visual mesh (radius = 90th "
                              "percentile of member-vertex distances); torso/logo/head are "
                              "rigid with the sensor and left to a static mask",
        "visual_mesh_exact": "MuJoCo mj_multiRay on the main.urdf visual meshes (reference for "
                             "IoU; matches brute-force triangle intersection to 1e-7 m)",
        "excluded": "head_link shell and the head collision disk: the sensor sits inside them; "
                    "the real head's aperture is not described by these files",
        "n_proxies": {k: len(v) for k, v in proxy_sets.items()},
    }
    return results, {"frames": frames, "mounts": mounts, "model": model, "mesh_dir": mesh_dir,
                     "dirs": dirs, "mesh": mesh, "proxy_sets": proxy_sets,
                     "joint_names": joint_names}


def body_extent(model: UrdfModel, mesh_dir, fr: dict) -> dict:
    """Front/back/side extent of the visual meshes relative to the sensor, frame 0 of ``fr``."""
    pts = []
    for g in model.visuals:
        if g.kind != "mesh":
            continue
        v = np.unique(np.round(read_stl(mesh_path(mesh_dir, g.mesh)).reshape(-1, 3), 3), axis=0)
        v = v @ rpy_to_matrix(g.rpy).T + g.xyz
        R, p = fr["poses"][g.link][0][0], fr["poses"][g.link][1][0]
        pts.append(v @ R.T + p)
    pts = np.concatenate(pts) - fr["p_s"][0]
    H = heading_matrix(fr["yaw"][0])
    loc = pts @ H
    return {"front_m": float(loc[:, 0].max()), "back_m": float(-loc[:, 0].min()),
            "half_width_m": float(np.abs(loc[:, 1]).max()),
            "top_above_sensor_m": float(loc[:, 2].max())}


# --------------------------------------------------------------------------------------------
# Figures (matplotlib, light surface; series colours are categorical slots 1-2)
# --------------------------------------------------------------------------------------------

STYLE = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
    "body": "#b9b8b2", "inverted": "#2a78d6", "upright": "#eb6834",
}


def _axes(ax):
    ax.set_facecolor(STYLE["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(STYLE["ink2"])
    ax.tick_params(colors=STYLE["ink2"], labelsize=8)
    ax.grid(True, color=STYLE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def make_figures(results: dict, ctx: dict, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "text.color": STYLE["ink"],
                         "axes.labelcolor": STYLE["ink2"], "axes.titlecolor": STYLE["ink"]})
    files = []
    frames, model, mesh_dir = ctx["frames"], ctx["model"], ctx["mesh_dir"]

    # 1. Side view with the FOV wedges, floor blind points and the beam band.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, facecolor=STYLE["surface"])
    stand = frames["inverted"]["default_standing"]
    pts = []
    for g in model.visuals:
        if g.kind == "mesh":
            v = read_stl(mesh_path(mesh_dir, g.mesh)).reshape(-1, 3)[::7]
            v = v @ rpy_to_matrix(g.rpy).T + g.xyz
            R, p = stand["poses"][g.link][0][0], stand["poses"][g.link][1][0]
            pts.append(v @ R.T + p)
    pts = np.concatenate(pts)
    for ax, name in zip(axes, ("inverted", "upright")):
        _axes(ax)
        fr = frames[name]["default_standing"]
        col = STYLE[name]
        h, xs = float(fr["height"][0]), float(fr["p_s"][0][0])
        ax.scatter(pts[:, 0], pts[:, 2], s=0.2, color=STYLE["body"], rasterized=True)
        ax.axhspan(1.0, 1.5, color=STYLE["grid"], alpha=0.6, lw=0)
        ax.text(7.9, 1.25, "beam undersides\n1.0-1.5 m", ha="right", va="center", fontsize=8,
                color=STYLE["ink2"])
        ax.axhline(0.0, color=STYLE["ink2"], lw=1)
        for side, sgn in (("front", 1.0), ("back", -1.0)):
            lo, hi = math.radians(fr[f"{side}_lo"][0]), math.radians(fr[f"{side}_hi"][0])
            L = 12.0
            poly = [(xs, h)]
            for e in np.linspace(lo, hi, 50):
                poly.append((xs + sgn * L * math.cos(e), h + L * math.sin(e)))
            ax.fill([q[0] for q in poly], [q[1] for q in poly], color=col, alpha=0.12, lw=0)
            for e in (lo, hi):
                ax.plot([xs, xs + sgn * L * math.cos(e)], [h, h + L * math.sin(e)], color=col,
                        lw=2)
            r = float(fr[f"blind_{side}"][0])
            if np.isfinite(r) and r < 9:
                ax.plot([xs + sgn * r], [0.0], "o", ms=6, color=col, mec=STYLE["surface"],
                        mew=1.5, zorder=5)
                ax.annotate(f"{r:.2f} m", (xs + sgn * r, 0.0), (0, 8), textcoords="offset points",
                            ha="center", fontsize=8, color=STYLE["ink"])
        ax.plot([xs], [h], "o", ms=5, color=STYLE["ink"])
        ax.set_xlim(-2.5, 8.0)
        ax.set_ylim(-0.05, 2.0)
        ax.set_aspect("equal")
        ax.set_xlabel("forward distance from the sensor (m)")
        w = results["mounts"][name]["window_in_level_torso_frame_deg"]
        ax.set_title(f"{name} mount: front {w['front'][0]:.1f} to {w['front'][1]:+.1f} deg, "
                     f"back {w['back'][0]:.1f} to {w['back'][1]:+.1f} deg", fontsize=9)
    axes[0].set_ylabel("height (m)")
    fig.tight_layout()
    f = out / "fig_side_view.png"
    fig.savefig(f, dpi=160, facecolor=STYLE["surface"], bbox_inches="tight")
    plt.close(fig)
    files.append(f.name)

    # 2. Front elevation window versus torso pitch, with executed gait/crouch pitch ranges.
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=STYLE["surface"])
    _axes(ax)
    for label, key in (("gait p5-p95", "gait_executed"), ("crouch p5-p95", "crouch_executed")):
        st = results["postures"]["inverted"][key]["pitch"]
        ax.axvspan(st["p5"], st["p95"], color=STYLE["grid"], alpha=0.7 if "gait" in label else 0.35,
                   lw=0)
        ax.text((st["p5"] + st["p95"]) / 2, 57, f"executed {label}", ha="center", fontsize=8,
                color=STYLE["ink2"])
    for name in ("inverted", "upright"):
        rows = results["pitch_sweep"][name]
        p = np.array([r["torso_pitch_deg"] for r in rows])
        lo = np.array([r["front"][0] for r in rows])
        hi = np.array([r["front"][1] for r in rows])
        ax.fill_between(p, lo, hi, color=STYLE[name], alpha=0.14, lw=0)
        ax.plot(p, lo, color=STYLE[name], lw=2)
        ax.plot(p, hi, color=STYLE[name], lw=2, label=f"{name} mount")
        ax.annotate(name, (p[-1], (lo[-1] + hi[-1]) / 2), (4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=STYLE["ink"])
    ax.axhline(0, color=STYLE["ink2"], lw=1)
    ax.set_xlim(-10, 36)
    ax.set_ylim(-90, 62)
    ax.set_xlabel("torso pitch (deg, forward lean positive)")
    ax.set_ylabel("elevation covered straight ahead (deg)")
    ax.set_title("Forward field of view relative to gravity", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    f = out / "fig_window_vs_pitch.png"
    fig.savefig(f, dpi=160, facecolor=STYLE["surface"], bbox_inches="tight")
    plt.close(fig)
    files.append(f.name)

    # 3. Last distance at which the beam's lower front edge is seen, 0.6 m/s approach at 10 Hz.
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=STYLE["surface"])
    _axes(ax)
    hbs = np.array(BEAM_HEIGHTS)
    for name in ("inverted", "upright"):
        tab = results["overhead"][name]["gait_executed"]
        p5, p50, p95 = (np.array([tab[f"{hb:.2f}"]["approach_m"].get(q, np.nan) for hb in hbs])
                        for q in ("p5", "p50", "p95"))
        ax.fill_between(hbs, p5, p95, color=STYLE[name], alpha=0.14, lw=0)
        ax.plot(hbs, p50, color=STYLE[name], lw=2, marker="o", ms=4,
                label=f"{name}: walking (median, p5-p95)")
        st = results["overhead"][name]["default_standing"]
        ax.plot(hbs, [st[f"{hb:.2f}"]["center_m"]["p50"] for hb in hbs], color=STYLE[name],
                lw=1.2, ls="--", label=f"{name}: standing still")
    s = results["postures"]["inverted"]["default_standing"]
    for x, lab in ((s["height"]["p50"], "sensor"), (s["head_top"]["p50"], "head top")):
        ax.axvline(x, color=STYLE["ink2"], lw=1, ls=":")
        ax.text(x, 5.6, lab, rotation=90, ha="right", va="top", fontsize=8,
                color=STYLE["ink2"])
    ax.set_ylim(0, 6)
    ax.set_xlabel("beam underside height (m)")
    ax.set_ylabel("last distance seen (m, from sensor)")
    ax.set_title("Overhead beam, approach at 0.6 m/s, 10 Hz scans (executed release gait)",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    f = out / "fig_overhead_last_seen.png"
    fig.savefig(f, dpi=160, facecolor=STYLE["surface"], bbox_inches="tight")
    plt.close(fig)
    files.append(f.name)

    # 4. Self-occlusion map (exact mesh) for the inverted mount, standing and crouched.
    mesh = ctx.get("mesh")
    if mesh is not None:
        dirs = sample_fov_directions(30000, MID360["elev_min_deg"], MID360["elev_max_deg"])
        cr = frames["inverted"]["crouch_executed"]
        k = int(np.argmin(np.abs(cr["height"] - np.median(cr["height"]))))
        panels = (("standing", frames["inverted"]["default_standing"], 0), ("crouched", cr, k))
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True, facecolor=STYLE["surface"])
        for ax, (label, fr, i) in zip(axes, panels):
            _axes(ax)
            mesh.set_pose(dict(zip(ctx["joint_names"], fr["qmat"][i])), fr["poses"]["pelvis"][0][i],
                          fr["poses"]["pelvis"][1][i])
            d_w = dirs @ fr["R_s"][i].T
            t, _ = mesh(fr["p_s"][i], d_w)
            az, el = az_el(d_w @ heading_matrix(fr["yaw"][i]))
            blocked = np.isfinite(t)
            ax.scatter(np.degrees(az[~blocked]), np.degrees(el[~blocked]), s=0.3,
                       color=STYLE["body"], rasterized=True)
            ax.scatter(np.degrees(az[blocked]), np.degrees(el[blocked]), s=0.6,
                       color=STYLE["inverted"], rasterized=True)
            ax.set_xlim(-180, 180)
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_xlabel("azimuth (deg, 0 = heading, +90 = left)")
            ax.set_title(f"inverted mount, {label}: sensor {fr['height'][i]:.2f} m, torso pitch "
                         f"{fr['pitch'][i]:.0f} deg, blocked {blocked.mean():.1%}", fontsize=9)
        axes[0].set_ylabel("elevation (deg, gravity frame)")
        fig.tight_layout()
        f = out / "fig_self_occlusion_map.png"
        fig.savefig(f, dpi=160, facecolor=STYLE["surface"], bbox_inches="tight")
        plt.close(fig)
        files.append(f.name)
    return files


# --------------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------------

BEAM_DEPTH_M = 0.20  # assumed beam depth along the direction of travel
SPEED_MPS = 0.6


def _f(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return "never"
    return f"{x:.{nd}f}"


def _band(st, nd=2, scale=1.0, unit=""):
    """'p50 (p5 to p95)' with 'never' for an all-inf set and the inf share when mixed."""
    if not st or st.get("n", 0) == 0:
        return "-"
    if "p50" not in st:
        return "never"
    v = lambda k: f"{st[k] * scale:.{nd}f}{unit}"  # noqa: E731
    if st["n"] == 1:
        return v("p50")
    extra = f"; never in {st['inf_fraction']:.0%}" if st.get("inf_fraction", 0) > 0 else ""
    return f"{v('p50')} ({v('p5')} to {v('p95')}{extra})"


def _pct(st):
    return _band(st, 1, 100.0, "%")


def _win(p, side):
    w = p[f"window_{side}_deg"]
    if w["lo"].get("n", 0) == 1:
        return f"{w['lo']['p50']:+.1f} to {w['hi']['p50']:+.1f}"
    return (f"{w['lo']['p50']:+.1f} to {w['hi']['p50']:+.1f} "
            f"(lo {w['lo']['p5']:+.0f}/{w['lo']['p95']:+.0f}, hi {w['hi']['p5']:+.0f}/"
            f"{w['hi']['p95']:+.0f})")


def _rel(path: str, roots) -> str:
    for root in roots:
        if path.startswith(root.rstrip("/") + "/"):
            return path[len(root.rstrip("/")) + 1:]
    return path


def render_summary(r: dict) -> str:
    P, OV, S = r["postures"], r["overhead"], r["self_occlusion"]
    inv, up = P["inverted"], P["upright"]
    ext = r.get("body_extent_default_standing_m", {})
    back = ext.get("back_m", 0.07)
    pnames = {
        "default_standing": "standing (Isaac init pose)",
        "gait_executed": "walking, executed (release SONIC, physics)",
        "gait_reference": "walking, kinematic reference",
        "crouch_executed": "crouched, executed",
        "crouch_reference": "crouched, kinematic reference",
        "crawl_executed": "crawling, executed",
    }
    mesh_key = "visual_mesh_exact" if "visual_mesh_exact" in S["inverted"]["default_standing"] \
        else "isaac_collision_capsules"
    occ_inv = S["inverted"]

    def floor_occ(mount, posture, side):
        st = S[mount][posture][mesh_key]["blind_floor_radius_m"][side]
        return _band(st)

    ap = lambda m, p, hb: OV[m][p][f"{hb:.2f}"].get("approach_m", {})  # noqa: E731
    walk_under = (1.30, 1.40, 1.50)
    horizon = {hb: ((ap("inverted", "gait_executed", hb)["p50"] + BEAM_DEPTH_M + back) / SPEED_MPS,
                    (ap("inverted", "gait_executed", hb)["p95"] + BEAM_DEPTH_M + back) / SPEED_MPS)
               for hb in walk_under}
    h_med = (min(v[0] for v in horizon.values()), max(v[0] for v in horizon.values()))
    h_p95 = max(v[1] for v in horizon.values())
    keep_s = math.ceil(h_p95)
    last_p95 = max(ap("inverted", "gait_executed", hb)["p95"] for hb in walk_under)
    floor_walk = inv["gait_executed"]["blind_floor_radius_fov_only_m"]["front"]
    rigid_share = max(
        (share for m in S for p in S[m] if p != "n_frames"
         for c, st in S[m][p].items() if c == mesh_key
         for link, share in st["mean_share_by_link"].items() if link in RIGID_WITH_SENSOR),
        default=0.0)
    roots = [str(REPO), str(Path(r["inputs"].get("workspace", "/nonexistent")).parent)]

    L = []
    add = L.append
    add("# MID-360 sensor geometry on the G1 (roadmap Phase 0.9)\n")
    add(f"Generated {r['generated']} by `scripts/sensor/mid360_geometry.py` at commit "
        f"`{r['code_commit'][:10]}`. CPU only. Numbers come from `results.json` in this "
        "directory. To reproduce: `CUDA_VISIBLE_DEVICES= .venv_native/bin/python "
        "scripts/sensor/mid360_geometry.py`, which takes about 1.5 min on 4 threads and needs "
        "3.3 GB RAM.\n")
    add("Tags:")
    add("- **[C]** computed: URDF forward kinematics plus ray geometry.")
    add("- **[M]** measured input: executed release-SONIC physics rollouts from "
        "`teacher-8192-500-review/eval/release`.")
    add("- **[A]** assumed.\n")
    add("Values are the median, with p5 to p95 in brackets, over frames or simulated approaches. "
        "\"never\" means the edge is never inside the field of view (FOV). Distances are "
        "horizontal and measured from the sensor.\n")

    # --- bottom line ---------------------------------------------------------------------
    s_inv = inv["default_standing"]
    g_inv, g_up = inv["gait_executed"], up["gait_executed"]
    add("## Bottom line\n")
    add(f"- **Both mounts put the sensor at the same point: {s_inv['height']['p50']:.3f} m above "
        f"the floor standing [C]; the roadmap had assumed about 1.2 m.** They differ only in "
        "orientation. `g1_29dof.urdf`'s +0.40618 m is relative to a `torso_link` frame that "
        "sits 0.010 m higher than `main.urdf`'s. Head top: "
        f"{s_inv['head_top']['p50']:.3f} m. Executed walking: sensor "
        f"{_band(g_inv['height'], 3)} m, torso pitch {_band(g_inv['pitch'], 1)} deg, roll "
        f"{_band(g_inv['roll'], 1)} deg [C from M].")
    add(f"- **Inverted mount (`main.urdf`, and Unitree's own description):** covers "
        f"{_win(s_inv, 'front')} deg straight ahead when level.")
    sb = s_inv["blind_floor_radius_fov_only_m"]
    add(f"  - It sees the floor from {sb['front']['p50']:.2f} m ahead, {sb['left']['p50']:.2f} m "
        f"to the side ({floor_occ('inverted', 'default_standing', 'left')} m once the shoulders "
        f"are counted) and {sb['back']['p50']:.2f} m behind.")
    add(f"  - Its upper edge is only {s_inv['window_front_deg']['hi']['p50']:+.1f} deg ahead. "
        "Overheads above the sensor therefore drop out early. While walking, a 1.40 m underside "
        f"is last seen {_band(ap('inverted', 'gait_executed', 1.40))} m out.")
    add("  - Crouched, the upper edge ahead is "
        f"{_band(inv['crouch_executed']['window_front_deg']['hi'], 1)} deg, so an overhead ahead "
        "is essentially invisible [C].")
    ub = up["default_standing"]["blind_floor_radius_fov_only_m"]
    add(f"- **Upright mount (`g1_29dof.urdf`):** covers {_win(up['default_standing'], 'front')} "
        "deg ahead. It sees overheads almost until it is underneath them, but it is floor-blind "
        f"out to {ub['front']['p50']:.1f} m ahead standing "
        f"({g_up['blind_floor_radius_fov_only_m']['front']['p50']:.1f} m median while walking) "
        f"and {ub['back']['p50']:.1f} m behind [C]. Ground obstacles near the feet would need "
        "another sensor, such as the D435i.")
    so = occ_inv["default_standing"]
    add(f"- **Self-occlusion (exact mesh, inverted):** "
        f"{_pct(so[mesh_key]['blocked_fraction'])} of FOV rays standing, "
        f"{_pct(occ_inv['gait_executed'][mesh_key]['blocked_fraction'])} walking and "
        f"{_pct(occ_inv['crouch_executed'][mesh_key]['blocked_fraction'])} crouched. The "
        "blocked rays are a thin band at the lower edge, from the shoulder pitch/roll links. "
        "The upright mount blocks 0% [C]. Isaac's collision capsules capture only "
        f"{_pct(so['isaac_collision_capsules']['blocked_fraction'])} standing.")
    add(f"- **Memory is mandatory for the inverted mount.** Overheads that are walked under "
        f"need {h_med[0]:.1f}–{h_med[1]:.1f} s of registered memory at 0.6 m/s (median; "
        f"{h_p95:.1f} s at p95). The floor under the feet needs "
        f"{floor_walk['p50'] / SPEED_MPS:.1f} s ({floor_walk['p95'] / SPEED_MPS:.1f} s at p95) "
        "[C+A]; see §6.\n")

    # --- mounts ---------------------------------------------------------------------------
    add("## 1. Mount survey [C]\n")
    add("| File | Orientation | Offset in parent (m) | Sensor in pelvis frame, zero joints (m) |")
    add("|---|---|---|---|")
    for s in r["mount_survey"]:
        f = _rel(s["file"], roots)
        p = s["sensor_position_in_pelvis_frame_zero_joints"]
        add(f"| `{f}` | {s['orientation']} (tilt {s['tilt_from_torso_axis_deg']:.1f} deg) | "
            f"{s['xyz_in_parent'][2]:+.5f} | ({p[0]:+.4f}, {p[1]:+.5f}, {p[2]:+.5f}) |")
    add("")
    add("- The inverted rpy (0, 3.101, 3.1415) is Rx(π)·Ry(−0.0406). The sensor z axis points "
        "down, tilted 2.3 deg backwards, so the band leans 2.3 deg down at the front.")
    add("- The upright rpy (0, 0.0401, 0) tilts the sensor z axis 2.3 deg forwards, so the band "
        "is 2.3 deg lower at the front.")
    add("- The only asset spawned in Isaac (`main.urdf`) and Unitree's ROS description "
        "(`/opt/ros/humble/.../g1/main.urdf`, same as its MJCF site) are inverted. The "
        "decoupled_wbc copies are upright.")
    fk = r["checks"]["fk_validation"]
    add(f"- FK check: max body error {fk['reference_npz_max_body_error_m']:.1e} m "
        "against the MuJoCo-native `reference.npz`, and "
        f"{fk['isaac_ref_track_max_body_error_m']:.1e} m against "
        "Isaac's `ref_track_body_pos`. In the executed rollouts, the foot soles sit on z=0 "
        f"(p50 {r['checks']['executed_foot_sole_above_floor_m']['p50']:.1e} m).\n")

    # --- FOV table ------------------------------------------------------------------------
    add("## 2. Sensor height and gravity-frame elevation window [C]\n")
    add("Windows are in degrees above the horizon, at heading azimuths front, left and back. "
        "For walking and crouching the window shows its median; lo and hi also give their "
        "p5/p95. A ±5 deg pitch sweep is in `results.json` → `pitch_sweep` and "
        "`fig_window_vs_pitch.png`.\n")
    add("| Posture | Frames | Sensor height (m) | Torso pitch (deg) | Mount | Front | Left | "
        "Back |")
    add("|---|---|---|---|---|---|---|---|")
    for p in ("default_standing", "gait_executed", "gait_reference", "crouch_executed",
              "crouch_reference", "crawl_executed"):
        for m in ("inverted", "upright"):
            q = P[m][p]
            add(f"| {pnames[p]} | {q['n_frames']} | {_band(q['height'], 3)} | "
                f"{_band(q['pitch'], 1)} | {m} | {_win(q, 'front')} | {_win(q, 'left')} | "
                f"{_win(q, 'back')} |")
    add("")
    add("The kinematic references lean further forward than the executed gait (median "
        f"{P['inverted']['gait_reference']['pitch']['p50']:.1f} vs "
        f"{P['inverted']['gait_executed']['pitch']['p50']:.1f} deg). The executed rollouts are "
        "what a robot running the release tracker would see.\n")

    # --- floor ----------------------------------------------------------------------------
    add("## 3. Blind floor radius (closest visible floor point, m) [C]\n")
    add("\"FOV only\" ignores the robot's own body. \"+ self-occlusion\" also requires a clear "
        "ray against the exact visual meshes, on a subsample of frames.\n")
    add("| Posture | Mount | Front, FOV only | Front + self-occl. | Left, FOV only | "
        "Left + self-occl. | Back, FOV only | Back + self-occl. |")
    add("|---|---|---|---|---|---|---|---|")
    for p in ("default_standing", "gait_executed", "crouch_executed"):
        for m in ("inverted", "upright"):
            b = P[m][p]["blind_floor_radius_fov_only_m"]
            add(f"| {pnames[p]} | {m} | {_band(b['front'])} | {floor_occ(m, p, 'front')} | "
                f"{_band(b['left'])} | {floor_occ(m, p, 'left')} | {_band(b['back'])} | "
                f"{floor_occ(m, p, 'back')} |")
    add("")

    # --- overhead -------------------------------------------------------------------------
    add("## 4. Overhead beam: last distance at which its lower front edge is seen [C]\n")
    add("- **Static pose:** the geometric threshold on the heading line.")
    add("- **Approach:** the executed pose sequence is replayed while the robot closes in at "
        "0.6 m/s. Scans run at 10 Hz with a random phase (2000 approaches). The value is the "
        "distance of the last scan that still contains the edge; the scan spacing is 0.06 m.")
    add("- For a beam below the sensor, the \"edge\" is the lower edge of its front face, seen "
        "from above. The underside itself is never seen from above.")
    add(f"- Standing, the head top is at {s_inv['head_top']['p50']:.2f} m. Beams below about "
        "1.30 m need a duck, so the standing and walking rows apply to the approach before "
        "the duck.\n")
    add("| Underside (m) | Inverted, standing | Inverted, walking approach | Inverted, crouched "
        "approach | Upright, standing | Upright, walking approach | Upright, crouched approach |")
    add("|---|---|---|---|---|---|---|")
    for hb in BEAM_HEIGHTS:
        k = f"{hb:.2f}"
        add(f"| {k} | {_band(OV['inverted']['default_standing'][k]['center_m'])} | "
            f"{_band(ap('inverted', 'gait_executed', hb))} | "
            f"{_band(ap('inverted', 'crouch_executed', hb))} | "
            f"{_band(OV['upright']['default_standing'][k]['center_m'])} | "
            f"{_band(ap('upright', 'gait_executed', hb))} | "
            f"{_band(ap('upright', 'crouch_executed', hb))} |")
    add("")
    add("The approach replays the executed crouch frames at 0.6 m/s. Real crouched transit "
        "speed is unknown (roadmap 0.7). The corridor variant, where any point of the edge "
        "within |y| ≤ 0.3 m counts, and the kinematic-reference gait are in `results.json`.\n")

    # --- self-occlusion -------------------------------------------------------------------
    add("## 5. Self-occlusion (share of FOV rays, uniform in solid angle) [C]\n")
    add("The reference is the exact MuJoCo ray cast on the `main.urdf` visual meshes. The head "
        "shell and the head collision disk are excluded, because the sensor sits inside them. "
        "IoU is per ray, against the exact mesh.\n")
    add("| Mount | Posture | Frames | Caster | Blocked | Within 0.1 m blind zone | "
        "IoU vs mesh | Main blockers (share of rays) |")
    add("|---|---|---|---|---|---|---|---|")
    for m in ("inverted", "upright"):
        for p in ("default_standing", "gait_executed", "crouch_executed"):
            block = S[m][p]
            for c, st in block.items():
                if c == "n_frames":
                    continue
                top = ", ".join(f"{k.replace('_link', '')} {v:.1%}" for k, v in
                                list(st["mean_share_by_link"].items())[:4]) or "-"
                iou = st.get(f"iou_vs_{mesh_key}")
                add(f"| {m} | {pnames[p]} | {block['n_frames']} | {c} | "
                    f"{_band(st['blocked_fraction'], 3)} | "
                    f"{_band(st['blocked_within_blind_zone_fraction'], 3)} | "
                    f"{_band(iou) if iou else 'ref'} | {top} |")
    add("")
    rigid_txt = ("block no ray" if rigid_share == 0 else f"block at most {rigid_share:.2%} of rays")
    add(f"Links that are rigid with the sensor (torso, logo, head) {rigid_txt} in any posture of "
        "the exact cast. Everything that is blocked comes from the moving arm links.")
    add("- A simulated MID-360 therefore needs no torso mask.")
    add("- The arm links need a good ray cast. Bounding primitives over-block, and the Isaac "
        "collision set, which has no shoulder pitch/roll geometry, under-blocks while "
        "standing.\n")

    # --- implications ---------------------------------------------------------------------
    add("## 6. Implications\n")
    gi = floor_walk
    ov = {hb: ap("inverted", "gait_executed", hb) for hb in walk_under}
    up_floor = up["gait_executed"]["blind_floor_radius_fov_only_m"]["front"]
    add(f"**Memory horizon (inverted mount, 0.6 m/s).** The horizon is: last-seen distance + "
        f"beam depth {BEAM_DEPTH_M:.2f} m [A] + body behind the sensor {back:.2f} m, divided "
        "by speed.")
    for hb, (h50, h95) in horizon.items():
        add(f"- Underside {hb:.2f} m (walk-under): {h50:.1f} s median, {h95:.1f} s at p95.")
    add(f"- Floor: the ground under the feet was last seen at ≥ {gi['p50']:.2f} m (median; "
        f"p95 {gi['p95']:.2f} m) in front, so a floor/obstacle layer must persist ≥ "
        f"{gi['p50'] / SPEED_MPS:.1f}–{gi['p95'] / SPEED_MPS:.1f} s.")
    add("- Duck-under beams (≤1.30 m): once crouched, the edge ahead is out of view in most "
        "frames (table §4). Memory must span the whole transit, from the start of the duck "
        "until the back clears the beam. That is at least (duck lead distance + beam depth + "
        f"{back:.2f} m) / crouched speed, where lead distance and speed come from Phase 0.7 "
        f"[A]. With a 1 m lead at 0.3 m/s it is about {(1.0 + BEAM_DEPTH_M + back) / 0.3:.1f} s.")
    add("- **Recommendation:** a registered map (FAST-LIO2-style odometry plus a rolling "
        f"voxel/elevation memory) that keeps at least {keep_s} s, or "
        f"{keep_s * SPEED_MPS:.1f} m of travel at 0.6 m/s. That covers the p95 horizon.")
    add("  - The teacher's visibility mask should use this horizon, not the instantaneous FOV.")
    add(f"  - Memory must reach at least {last_p95:.1f} m ahead: an overhead is last seen that "
        "far out at p95.")
    add(f"  - The policy's map crop must extend at least {BEAM_DEPTH_M + back:.2f} m behind the "
        "sensor, plus margin, so a beam stays in view until the body clears it.")
    add("  - Its forward reach only needs to cover the duck lead distance (Phase 0.7).\n")
    add("**Can the inverted mount see the floor ahead?** Yes. The forward floor becomes visible "
        f"at {s_inv['blind_floor_radius_fov_only_m']['front']['p50']:.2f} m standing, "
        f"{_band(gi)} m walking, and closer when crouched. The chest and shoulders never hide "
        "the forward floor band (§3); the shoulders only push the side radius out. This is "
        "the mount that suits stepping and foothold perception.\n")
    fh = g_inv["window_front_deg"]["hi"]
    add("**Can it see overheads?** Only far ahead. The upper edge is "
        f"{s_inv['window_front_deg']['hi']['p50']:+.1f} deg ahead when level. While walking it "
        f"is {_band(fh, 1)} deg, so forward lean often pushes it below the horizon. While "
        "crouching it looks down entirely. An underside 0.1–0.3 m above the sensor is last "
        f"seen at {_band(ov[1.30])} m (1.30 m) to {_band(ov[1.50])} m (1.50 m) while walking. "
        "After that the robot relies on memory.")
    up_ov = [ap("upright", "gait_executed", hb)["p95"] for hb in walk_under]
    add(f"- The upright mount sees the same overheads until {min(up_ov):.2f}–{max(up_ov):.2f} m "
        "(p95).")
    add(f"- But the upright mount is floor-blind within {up_floor['p50']:.1f} m ahead while "
        f"walking ({up_floor['p5']:.1f} to {up_floor['p95']:.1f} m).\n")
    add("**Consequences for the plan:**")
    add("- The Phase 4.1 sensor proxy must model the inverted band (−54/+5 deg ahead).")
    add("- Phase 4.2's pre-registered signature is expected to hold for the inverted mount: "
        "a student distilled from a full-geometry teacher fails on overheads that leave the "
        "FOV before arrival.")
    add("- The distance at which they leave is the approach column in §4.\n")

    # --- question -------------------------------------------------------------------------
    add("## 7. Question for the user\n")
    add("**Is the MID-360 on your G1 mounted inverted, as in `main.urdf` and Unitree's "
        "`unitree_description`, or upright, as in `decoupled_wbc/.../g1_29dof.urdf`?** A "
        "10-second check: with the robot standing, look at the raw Livox point cloud in the "
        "sensor frame. Floor returns sit at z ≈ **+1.2 m** if the sensor is inverted and at "
        "z ≈ −1.2 m if it is upright. Also: is the "
        "D435i fitted, and is it pitched about 48 deg down as in `g1_29dof.urdf`? With an "
        "upright MID-360, it would be the only near-floor sensor.\n")

    # --- caveats --------------------------------------------------------------------------
    add("## 8. Caveats\n")
    add("- [A] Ray density is uniform in solid angle over the −7..+52 deg band. The real "
        "non-repetitive MID-360 pattern (OmniPerception's `mid360.npy`, not on this host) is "
        "not uniform. A 10 Hz frame covers about 1 point per deg², so thin bars can be missed "
        "in a single frame.")
    add("- [A] The head shell is treated as transparent, because the sensor sits inside the "
        "`head_link` mesh and its window geometry is not in these files. The 0.1 m blind zone "
        "only removes returns; blocked rays still count as blocked.")
    add("- [A] Walking uses executed release-SONIC rollouts of Kimodo reference clips, at their "
        "own speeds, with the 0.6 m/s approach synthesised on top. The motor student and "
        "navigation adapter may walk differently. Crouch frames come from clips that were "
        "never qualified for crouch transit.")
    add("- [A] Beam: a horizontal bar across the path, depth "
        f"{BEAM_DEPTH_M:.2f} m for the memory horizon. Distances are from the sensor; the toes "
        f"lead it by {ext.get('front_m', float('nan')):.2f} m.")
    add("- The kinematic chain and mount are from `main.urdf`. The upright mount is transplanted "
        "through the pelvis frame, and both mounts give the same point to 1e-12 m.\n")

    if r.get("figures"):
        add("## Figures\n")
        for f in r["figures"]:
            add(f"![{f}]({f})\n")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workspace", type=Path, default=None,
                    help="data root (default: the first workspace/ above the repo)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: <workspace>/phase0/sensor_geometry)")
    ap.add_argument("--rays", type=int, default=20000, help="FOV rays per occlusion frame")
    ap.add_argument("--occlusion-frames", type=int, default=48)
    ap.add_argument("--no-mesh", action="store_true", help="skip the MuJoCo mesh check")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    ws = args.workspace or find_workspace()
    out = args.out or ws / "phase0" / "sensor_geometry"
    out.mkdir(parents=True, exist_ok=True)
    results, ctx = analyze(args)
    if not args.no_figures:
        results["figures"] = make_figures(results, ctx, out)
    (out / "results.json").write_text(json.dumps(jsonable(results), indent=1) + "\n")
    (out / "summary.md").write_text(render_summary(results))
    print(f"wrote {out / 'results.json'} and {out / 'summary.md'}")


if __name__ == "__main__":
    main()

