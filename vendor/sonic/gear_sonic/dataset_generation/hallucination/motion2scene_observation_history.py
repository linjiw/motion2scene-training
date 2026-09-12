"""Causal sensor-ray history for gravity-aligned floor/ceiling observations.

Inputs are ray measurements and robot poses, never scene parameters or object
identities. Surface normals are optional measured raycast/depth normals: absent
normals leave floor/ceiling unknown while retaining occupied endpoint evidence.
The map is a finite history of a static scene, not a free-space certificate.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from .motion2scene_ray_observer import normalize_hit

DEFAULT_ELEVATIONS_DEG = (-75, -45, -20, -10, -5, 0, 2.5, 5, 7.5, 10, 15, 30, 60)
DEFAULT_AZIMUTHS_DEG = (-30, -15, 0, 15, 30)


def _vector(values, name, length=3):
    result = np.asarray(values, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"invalid {name}")
    return result


def _rotation(quat_wxyz):
    w, x, y, z = _vector(quat_wxyz, "quaternion", 4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = np.array([w, x, y, z]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass(frozen=True)
class SensorRay:
    """One ideal range measurement in world coordinates.

    ``hit_distance_m=None`` means a valid no-return out to ``range_m``. Missing
    or invalid depth pixels must be omitted; they are not valid no-return rays.
    """

    origin_w: tuple[float, float, float]
    direction_w: tuple[float, float, float]
    range_m: float
    hit_distance_m: float | None = None
    hit_normal_w: tuple[float, float, float] | None = None

    def __post_init__(self):
        origin = _vector(self.origin_w, "ray origin")
        direction = _vector(self.direction_w, "ray direction")
        if not np.isclose(np.linalg.norm(direction), 1, atol=1e-6):
            raise ValueError("ray direction must be unit length")
        if not math.isfinite(self.range_m) or self.range_m <= 0:
            raise ValueError("invalid ray range")
        if self.hit_distance_m is not None and (
            not math.isfinite(self.hit_distance_m) or not 0 <= self.hit_distance_m <= self.range_m
        ):
            raise ValueError("invalid hit distance")
        if self.hit_normal_w is not None:
            normal = _vector(self.hit_normal_w, "hit normal")
            if self.hit_distance_m is None or not np.isclose(np.linalg.norm(normal), 1, atol=1e-5):
                raise ValueError("a unit hit normal requires a measured hit")
            object.__setattr__(self, "hit_normal_w", tuple(normal))
        object.__setattr__(self, "origin_w", tuple(origin))
        object.__setattr__(self, "direction_w", tuple(direction))


def rays_from_overhang_packet(packet, upper_range_m=3.0):
    """Adapt existing upper/lower sensor logs without inferring missing surfaces.

    Existing logs omit normals, so their endpoints populate occupancy only.
    Object paths and ``upper_candidate``/``overhang`` labels are ignored.
    """
    measurements = []
    for upper in packet["rays"]:
        for ray, origin, limit in [
            (upper, packet["origin"], upper_range_m),
            *[(lower, lower["origin"], lower["range_m"]) for lower in upper.get("lower_rays", [])],
        ]:
            hit = ray["hit"]
            distance = None if hit is None else float(hit["distance"])
            if hit is not None:
                endpoint = (
                    _vector(origin, "ray origin")
                    + _vector(ray["direction"], "ray direction") * distance
                )
                if not np.allclose(endpoint, _vector(hit["position"], "hit position"), atol=1e-4):
                    raise ValueError("hit position and ray distance disagree")
            measurements.append(
                SensorRay(
                    origin,
                    ray["direction"],
                    limit,
                    distance,
                    None if hit is None else hit.get("normal"),
                )
            )
    return tuple(measurements)


def capture_ray_fan(
    root_pos_w,
    root_quat_wxyz,
    query,
    *,
    sensor_offset_b=(0.2, 0.0, 0.4),
    elevations_deg=DEFAULT_ELEVATIONS_DEG,
    azimuths_deg=DEFAULT_AZIMUTHS_DEG,
    range_m=4.0,
    robot_path_prefix="/World/envs/env_0/Robot",
):
    """PhysX range fan at one body-mounted sensor pose; retain nearest returns.

    Unlike virtual lower rays at arbitrary heights, every ray originates at the
    supplied sensor pose. Optional normals are ideal PhysX collision-query
    normals, not reconstructed noisy depth-sensor normals.
    The robot is excluded; no environmental object is selected by its identity.
    """
    rotation = _rotation(root_quat_wxyz)
    origin = _vector(root_pos_w, "root position") + rotation @ _vector(
        sensor_offset_b, "sensor offset"
    )
    measurements = []
    if not robot_path_prefix:
        raise ValueError("robot exclusion prefix cannot be empty")
    for elevation in elevations_deg:
        for azimuth in azimuths_deg:
            if not np.isfinite([elevation, azimuth]).all():
                raise ValueError("nonfinite ray angle")
            a, e = math.radians(azimuth), math.radians(elevation)
            direction = rotation @ np.array(
                [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
            )
            # Validate configuration before invoking the simulator.
            SensorRay(origin, direction, range_m)
            hits, errors = [], []

            def collect(raw):
                try:
                    hit = normalize_hit(raw)
                    path = hit["path"]
                    if path == robot_path_prefix or path.startswith(robot_path_prefix + "/"):
                        return True
                    normal = (
                        raw.get("normal")
                        if isinstance(raw, Mapping)
                        else getattr(raw, "normal", None)
                    )
                    measured = SensorRay(origin, direction, range_m, hit["distance"], normal)
                    if not np.allclose(
                        np.asarray(origin) + direction * hit["distance"],
                        hit["position"],
                        atol=1e-4,
                    ):
                        raise ValueError("hit position and ray distance disagree")
                    hits.append(measured)
                    return True
                except Exception as error:
                    errors.append(error)
                    return False

            query.raycast_all(tuple(origin), tuple(direction), float(range_m), collect)
            if errors:
                raise RuntimeError("invalid range sensor callback") from errors[0]
            measurements.append(
                min(hits, key=lambda ray: ray.hit_distance_m)
                if hits
                else SensorRay(origin, direction, range_m)
            )
    return tuple(measurements)


@dataclass(frozen=True)
class HistoryGrid:
    """Half-open gravity-aligned voxel bounds relative to the current root."""

    lower_m: tuple[float, float, float] = (-1.0, -1.5, -1.5)
    upper_m: tuple[float, float, float] = (4.0, 1.5, 1.5)
    resolution_m: float = 0.2
    max_age_s: float = 2.0
    max_frames: int = 101
    surface_normal_z_min: float = 0.8

    def __post_init__(self):
        lower = _vector(self.lower_m, "grid lower bound")
        upper = _vector(self.upper_m, "grid upper bound")
        if (
            not np.isfinite([self.resolution_m, self.max_age_s, self.surface_normal_z_min]).all()
            or self.resolution_m <= 0
            or self.max_age_s < 0
            or type(self.max_frames) is not int
            or self.max_frames < 1
            or not 0 < self.surface_normal_z_min <= 1
            or np.any(upper <= lower)
        ):
            raise ValueError("invalid history grid")
        count = (upper - lower) / self.resolution_m
        if not np.allclose(count, np.round(count)):
            raise ValueError("grid extents must be integer multiples of resolution")
        object.__setattr__(self, "lower_m", tuple(lower))
        object.__setattr__(self, "upper_m", tuple(upper))

    @property
    def shape(self):
        return tuple(
            np.round((np.array(self.upper_m) - self.lower_m) / self.resolution_m).astype(int)
        )


class FloorCeilingHistory:
    """Register delivered rays in world coordinates, then reproject at decisions.

    Capture times must be monotonic and not exceed delivery time. Reset between
    episodes. Old evidence expires; occupied evidence dominates free samples
    while retained, making this unsuitable for dynamic obstacles without an
    additional motion/clearing model. ``free_sampled`` is sparse ray evidence,
    not a statement that an entire voxel or vertical interval is clear.
    """

    def __init__(self, grid=None):
        self.grid = HistoryGrid() if grid is None else grid
        self._frames = deque(maxlen=self.grid.max_frames)
        self._last_capture_s = -math.inf
        self._last_now_s = -math.inf

    def reset(self):
        self._frames.clear()
        self._last_capture_s = self._last_now_s = -math.inf

    def _advance(self, now_s):
        if not math.isfinite(now_s) or now_s < 0 or now_s < self._last_now_s:
            raise ValueError("history time must be finite, nonnegative and monotonic")
        self._last_now_s = now_s
        while self._frames and self._frames[0][0] < now_s - self.grid.max_age_s:
            self._frames.popleft()

    def push(self, rays, capture_time_s, *, delivered_time_s=None):
        if delivered_time_s is None:
            delivered_time_s = capture_time_s
        if (
            not math.isfinite(capture_time_s)
            or capture_time_s < 0
            or capture_time_s <= self._last_capture_s
            or capture_time_s > delivered_time_s
        ):
            raise ValueError("capture times must increase and precede delivery")
        rays = tuple(rays)
        if any(not isinstance(ray, SensorRay) for ray in rays):
            raise ValueError("history requires SensorRay measurements")
        self._advance(delivered_time_s)
        self._last_capture_s = capture_time_s
        if capture_time_s >= delivered_time_s - self.grid.max_age_s:
            self._frames.append((capture_time_s, rays))

    def snapshot(self, root_pos_w, root_quat_wxyz, now_s):
        root = _vector(root_pos_w, "root position")
        rotation = _rotation(root_quat_wxyz)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        world_to_ego = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        self._advance(now_s)
        lower = np.asarray(self.grid.lower_m)
        shape = self.grid.shape
        occupied = np.zeros(shape, dtype=bool)
        free = np.zeros(shape, dtype=bool)
        floor = np.full(shape[:2], -np.inf)
        ceiling = np.full(shape[:2], np.inf)
        floor_seen = np.full(shape[:2], np.nan)
        ceiling_seen = np.full(shape[:2], np.nan)

        def indices(points):
            index = np.floor((points - lower) / self.grid.resolution_m).astype(int)
            return index[np.all((index >= 0) & (index < shape), axis=1)]

        for captured, rays in self._frames:
            for ray in rays:
                origin = world_to_ego @ (np.asarray(ray.origin_w) - root)
                direction = world_to_ego @ np.asarray(ray.direction_w)
                distance = ray.range_m if ray.hit_distance_m is None else ray.hit_distance_m
                # Keep free samples at least one voxel diagonal from a hit.
                free_limit = distance - (
                    math.sqrt(3) * self.grid.resolution_m if ray.hit_distance_m is not None else 0
                )
                samples = np.arange(0, max(0, free_limit), self.grid.resolution_m / 2)
                sampled = indices(origin[None, :] + samples[:, None] * direction)
                if len(sampled):
                    free[tuple(sampled.T)] = True
                if ray.hit_distance_m is None:
                    continue
                endpoint = origin + direction * distance
                hit_index = indices(endpoint[None, :])
                if not len(hit_index):
                    continue
                i, j, k = hit_index[0]
                occupied[i, j, k] = True
                normal = ray.hit_normal_w
                if normal is None:
                    continue
                if normal[2] >= self.grid.surface_normal_z_min and endpoint[2] < origin[2]:
                    if endpoint[2] > floor[i, j]:
                        floor[i, j], floor_seen[i, j] = endpoint[2], captured
                elif normal[2] <= -self.grid.surface_normal_z_min and endpoint[2] > origin[2]:
                    if endpoint[2] < ceiling[i, j]:
                        ceiling[i, j], ceiling_seen[i, j] = endpoint[2], captured
        floor_mask, ceiling_mask = np.isfinite(floor), np.isfinite(ceiling)
        free &= ~occupied
        interval_observed = floor_mask & ceiling_mask & (ceiling > floor)
        return {
            "floor_height_m": np.where(floor_mask, floor, 0).astype(np.float32),
            "ceiling_height_m": np.where(ceiling_mask, ceiling, 0).astype(np.float32),
            "floor_observed": floor_mask,
            "ceiling_observed": ceiling_mask,
            "floor_unknown": ~floor_mask,
            "ceiling_unknown": ~ceiling_mask,
            "floor_capture_time_s": floor_seen,
            "ceiling_capture_time_s": ceiling_seen,
            "interval_observed": interval_observed,
            "vertical_gap_m": np.where(interval_observed, ceiling - floor, 0).astype(np.float32),
            "occupied": occupied,
            "free_sampled": free,
            "unknown": ~(occupied | free),
            "retained_frames": len(self._frames),
            "time_s": float(now_s),
        }


def transition_timing_eligibility(
    first_seen_s,
    obstacle_arrival_s,
    *,
    sensing_latency_s,
    transition_duration_s,
    margin_s,
    legal_entry_times_s,
    decision_time_s=None,
):
    """Screen observability against measured duration and legal command times.

    Arrival/transition forecasts are constructor/teacher inputs, not policy
    features. This timing screen does not certify passage. ``None`` visibility
    stays ineligible; it never implies a protective stop is available.
    """
    values = [obstacle_arrival_s, sensing_latency_s, transition_duration_s, margin_s]
    if first_seen_s is not None:
        values.append(first_seen_s)
    if decision_time_s is not None:
        values.append(decision_time_s)
    legal = np.asarray(legal_entry_times_s, dtype=float)
    if (
        not np.isfinite(values).all()
        or min(values) < 0
        or legal.ndim != 1
        or not np.isfinite(legal).all()
        or np.any(legal < 0)
        or np.any(np.diff(legal) <= 0)
    ):
        raise ValueError("invalid timing inputs or unordered legal entries")
    result = {"eligible": False, "entry_time_s": None, "ready_time_s": None, "slack_s": None}
    if first_seen_s is None:
        return {**result, "reason": "unobserved"}
    delivered_s = first_seen_s + sensing_latency_s
    available_s = max(delivered_s, decision_time_s or 0)
    candidates = legal[legal >= available_s - 1e-9]
    if not len(candidates):
        return {**result, "reason": "no_legal_entry_after_observation"}
    entry_s = float(candidates[0])
    ready_s = entry_s + transition_duration_s + margin_s
    slack_s = float(obstacle_arrival_s - ready_s)
    return {
        "eligible": slack_s >= -1e-9,
        "entry_time_s": entry_s,
        "ready_time_s": ready_s,
        "slack_s": slack_s,
        "reason": "timely" if slack_s >= -1e-9 else "late_observation_or_transition",
    }
