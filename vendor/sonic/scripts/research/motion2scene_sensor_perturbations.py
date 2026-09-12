"""Causal sensor-channel dropout, range noise and latency for development tests.

Missing measurements are omitted, never converted to valid no-hit rays. Normals
remain ideal when a noisy return is retained; this is not a depth-camera model.
"""

from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_observation_delay import (
    ObservationDelay,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_observation_history import (
    FloorCeilingHistory,
    HistoryGrid,
    SensorRay,
)


@dataclass(frozen=True)
class SensorPerturbation:
    dropout_probability: float = 0.0
    range_noise_std_m: float = 0.0
    latency_s: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if (
            not all(
                math.isfinite(v)
                for v in (self.dropout_probability, self.range_noise_std_m, self.latency_s)
            )
            or not 0 <= self.dropout_probability <= 1
            or self.range_noise_std_m < 0
            or self.latency_s < 0
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("finite sensor settings and an unsigned 32-bit seed required")


def corrupt_rays(rays, settings, frame):
    """Index random draws by episode seed, frame and fixed fan-channel position.

    Separate streams pair dropout uniforms and range standard normals across
    strengths and policies. Random draws do not depend on which channels hit.
    Gaussian ranges outside [0, maximum range] become missing measurements.
    """
    rays = tuple(rays)
    if type(frame) is not int or frame < 0 or any(not isinstance(r, SensorRay) for r in rays):
        raise ValueError("nonnegative frame and typed range measurements required")
    dropped = (
        np.random.default_rng(np.random.SeedSequence([settings.seed, frame, 0])).random(len(rays))
        < settings.dropout_probability
    )
    noise = (
        np.random.default_rng(np.random.SeedSequence([settings.seed, frame, 1])).standard_normal(
            len(rays)
        )
        * settings.range_noise_std_m
    )
    measured, source_indices, invalid = [], [], []
    for i, ray in enumerate(rays):
        if dropped[i]:
            continue
        if ray.hit_distance_m is not None and settings.range_noise_std_m:
            distance = float(ray.hit_distance_m + noise[i])
            if not 0 <= distance <= ray.range_m:
                invalid.append(i)
                continue
            ray = replace(ray, hit_distance_m=distance)
        measured.append(ray)
        source_indices.append(i)
    valid = np.zeros(len(rays), dtype=bool)
    valid[source_indices] = True
    return tuple(measured), dict(
        measurement_valid_mask=valid.tolist(),
        retained_channel_indices=source_indices,
        dropout_channel_indices=np.flatnonzero(dropped).tolist(),
        out_of_range_channel_indices=invalid,
        capture_frame=frame,
    )


class PerturbedObservationStream:
    """Corrupt at capture, delay the packet, then update world-registered history."""

    def __init__(self, settings, grid=None):
        self.settings = settings
        self.delay = ObservationDelay(settings.latency_s)
        self.history = FloorCeilingHistory(HistoryGrid() if grid is None else grid)
        self.frame = 0

    def push(self, rays, phase_s):
        rays = tuple(rays)
        measured, metadata = corrupt_rays(rays, self.settings, self.frame)
        elapsed = self.frame / 50
        packet = dict(measurements=[asdict(r) for r in measured], phase_s=phase_s)
        delivered = self.delay.push(packet)
        if delivered is not None:
            self.history.push(
                [SensorRay(**r) for r in delivered["measurements"]],
                delivered["capture_elapsed_s"],
                delivered_time_s=elapsed,
            )
        cache = dict(
            capture_elapsed_s=elapsed,
            delivered_capture_elapsed_s=(
                None if delivered is None else delivered["capture_elapsed_s"]
            ),
            observation_age_s=(
                max(0, elapsed - delivered["capture_elapsed_s"])
                if delivered is not None
                else self.settings.latency_s + elapsed
            ),
            measurements=packet["measurements"],
            normal_known_mask=[r.hit_normal_w is not None for r in measured],
            raw_measurements=[asdict(r) for r in rays],
            sensor_perturbation=metadata,
        )
        self.frame += 1
        return cache, delivered
