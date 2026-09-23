"""Scripted goal controllers over the deploy planner runtime, and kinematic goal metrics.

The "robot" is the reference itself (perfect tracking): each controller reads the frame the
tracker would observe and commands the planner at the planner-thread rate (10 Hz).

- P0 (direction / speed / stop): face and move toward the goal in SLOW_WALK / WALK, slow down
  in distance bands, and command IDLE once the calibrated stopping distance covers the rest.
  Mode 0 cannot drive the approach because it ignores velocity commands.
- P1 (waypoint, extension E1): a moving mode with ``has_specific_target`` = 1 and the goal in
  every waypoint slot; the planner itself decides where to stop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math

import numpy as np

from gear_sonic.research.planner.planner_runtime import (
    CONTROL_FPS,
    PLANNER_PERIOD_TICKS,
    SLOW_WALK,
    WALK,
    DeployPlannerRuntime,
    PlannerCommand,
    idle_command,
    locomotion_command,
    waypoint_command,
    yaw_from_quat,
)

VELOCITY_WINDOW_FRAMES = 10


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class KinematicState:
    time_s: float
    xy: np.ndarray
    yaw: float
    speed: float


def observe(history: Sequence[np.ndarray], current: np.ndarray, tick: int) -> KinematicState:
    """Root xy, yaw and horizontal speed over the last 0.2 s of observed frames."""
    frames = list(history[-VELOCITY_WINDOW_FRAMES:]) + [current]
    if len(frames) > 1:
        span = (len(frames) - 1) / CONTROL_FPS
        speed = float(np.linalg.norm(frames[-1][:2] - frames[0][:2]) / span)
    else:
        speed = 0.0
    return KinematicState(
        tick / CONTROL_FPS,
        np.asarray(current[:2], dtype=np.float64),
        float(yaw_from_quat(current[3:7])),
        speed,
    )


@dataclass(frozen=True)
class SpeedBand:
    """Use ``mode``/``speed`` while the goal is farther than ``min_distance``."""

    min_distance: float
    mode: int
    speed: float

    @property
    def key(self) -> str:
        return f"mode{self.mode}_speed{self.speed:g}"


DEFAULT_BANDS = (
    SpeedBand(2.5, WALK, -1.0),
    SpeedBand(1.0, SLOW_WALK, 0.4),
    SpeedBand(0.0, SLOW_WALK, 0.2),
)


@dataclass
class DirectionSpeedStopController:
    """P0: turn toward the goal, slow down in bands, IDLE to stop (latched)."""

    goal_xy: Sequence[float]
    stop_distance: Mapping[str, float]
    bands: Sequence[SpeedBand] = DEFAULT_BANDS
    # Chosen on 20 development goals (seed 70602) before the 200-goal study (seed 70601).
    heading_deadband_rad: float = math.radians(5.0)
    freeze_radius: float = 0.3
    name: str = "P0"
    stopped: bool = field(default=False, init=False)

    def __post_init__(self):
        self.goal_xy = np.asarray(self.goal_xy, dtype=np.float64)
        missing = [band.key for band in self.bands if band.key not in self.stop_distance]
        if missing:
            raise ValueError(f"no calibrated stopping distance for {missing}")
        if sorted(b.min_distance for b in self.bands)[0] != 0.0:
            raise ValueError("the last band must cover distance 0")
        self.reset()

    def reset(self):
        self.stopped = False
        self.command_yaw = None
        self.stop_time_s = None

    def band(self, distance: float) -> SpeedBand:
        for band in sorted(self.bands, key=lambda b: -b.min_distance):
            if distance > band.min_distance:
                return band
        return min(self.bands, key=lambda b: b.min_distance)

    def __call__(self, state: KinematicState) -> PlannerCommand:
        offset = self.goal_xy - state.xy
        distance = float(np.linalg.norm(offset))
        if self.command_yaw is None:
            self.command_yaw = math.atan2(offset[1], offset[0])
        if self.stopped:
            return idle_command(self.command_yaw)
        bearing = math.atan2(offset[1], offset[0])
        band = self.band(distance)
        behind = abs(wrap(bearing - self.command_yaw)) > math.pi / 2
        if distance <= self.stop_distance[band.key] or (distance < self.freeze_radius and behind):
            self.stopped = True
            self.stop_time_s = state.time_s
            return idle_command(self.command_yaw)
        if distance > self.freeze_radius and (
            abs(wrap(bearing - self.command_yaw)) > self.heading_deadband_rad
        ):
            self.command_yaw = bearing
        return locomotion_command(band.mode, self.command_yaw, self.command_yaw, band.speed)


@dataclass
class WaypointController:
    """P1: the goal in every waypoint slot, final heading = start bearing, constant command.

    With ``carrot_m`` set (P1c, exploratory) the waypoint is instead placed ``carrot_m`` ahead
    toward the goal (or on it once closer) and re-placed every ``update_period_s``, on the same
    clock as the deploy's replan timer, with the final heading along the current bearing.
    The goal-as-waypoint planner tries to reach the target within one ~1.4 s plan, so a
    distant goal makes it sprint; the carrot bounds the implied speed.
    """

    goal_xy: Sequence[float]
    mode: int = SLOW_WALK
    speed: float | None = None
    carrot_m: float | None = None
    update_period_s: float = 1.0
    name: str = "P1"

    def __post_init__(self):
        self.goal_xy = np.asarray(self.goal_xy, dtype=np.float64)
        if self.carrot_m is not None and not self.carrot_m > 0:
            raise ValueError("carrot length must be positive")
        self.reset()

    def reset(self):
        self.command = None
        self.next_update_s = 0.0

    def __call__(self, state: KinematicState) -> PlannerCommand:
        offset = self.goal_xy - state.xy
        distance = float(np.linalg.norm(offset))
        if self.command is None:
            self.heading = math.atan2(offset[1], offset[0])
        if self.carrot_m is None:
            if self.command is None:
                self.command = waypoint_command(
                    self.mode, self.goal_xy, self.heading, speed=self.speed
                )
            return self.command
        if self.command is None or state.time_s + 1e-9 >= self.next_update_s:
            if distance > 0.3:
                self.heading = math.atan2(offset[1], offset[0])
            target = (
                self.goal_xy
                if distance <= self.carrot_m
                else state.xy + offset / distance * self.carrot_m
            )
            self.command = waypoint_command(self.mode, target, self.heading, speed=self.speed)
            self.next_update_s = (math.floor(state.time_s / self.update_period_s + 1e-9) + 1) * (
                self.update_period_s
            )
        return self.command


def run_controller(runtime: DeployPlannerRuntime, controller, duration_s: float) -> np.ndarray:
    """Reset both, then run ``duration_s`` of 50 Hz ticks; the controller acts at 10 Hz."""
    runtime.reset()
    controller.reset()
    frames: list[np.ndarray] = []
    command = idle_command()
    for tick in range(int(round(duration_s * CONTROL_FPS))):
        if tick % PLANNER_PERIOD_TICKS == 0:
            command = controller(observe(frames, runtime.current_reference, tick))
        frames.append(runtime.step(command))
    return np.asarray(frames)


def goal_metrics(
    frames: np.ndarray,
    goal_xy: Sequence[float],
    *,
    arrive_radius: float = 0.10,
    coarse_radius: float = 0.25,
    stop_speed: float = 0.10,
    end_window_s: float = 0.5,
    smooth_window_s: float = 0.5,
) -> dict:
    """Root-XY goal metrics of one observed 50 Hz reference stream."""
    xy = np.asarray(frames, dtype=np.float64)[:, :2]
    goal = np.asarray(goal_xy, dtype=np.float64)
    error = np.linalg.norm(xy - goal, axis=1)
    straight = float(np.linalg.norm(goal - xy[0]))
    raw_length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    width = max(1, int(round(smooth_window_s * CONTROL_FPS)))
    kernel = np.ones(width) / width
    if len(xy) > width:
        smooth = np.stack([np.convolve(xy[:, i], kernel, mode="valid") for i in range(2)], 1)
        smooth = np.vstack([xy[:1], smooth, xy[-1:]])
    else:
        smooth = xy
    smooth_length = float(np.sum(np.linalg.norm(np.diff(smooth, axis=0), axis=1)))
    n_end = max(1, int(round(end_window_s * CONTROL_FPS)))
    window = min(n_end, len(xy) - 1)
    window_speed = np.linalg.norm(xy[window:] - xy[:-window], axis=1) / (window / CONTROL_FPS)
    end_speed = float(np.linalg.norm(xy[-1] - xy[-1 - n_end]) / (n_end / CONTROL_FPS))

    def first(radius):
        hits = np.flatnonzero(error <= radius)
        return float(hits[0] / CONTROL_FPS) if len(hits) else None

    outside = np.flatnonzero(error > arrive_radius)
    if len(outside) == 0:
        settle = 0.0
    elif outside[-1] == len(error) - 1:
        settle = None
    else:
        settle = float((outside[-1] + 1) / CONTROL_FPS)
    return {
        "straight_line_m": straight,
        "final_error_m": float(error[-1]),
        "min_error_m": float(error.min()),
        "success_010": bool(error[-1] <= arrive_radius),
        "success_025": bool(error[-1] <= coarse_radius),
        "time_to_arrive_010_s": first(arrive_radius),
        "time_to_arrive_025_s": first(coarse_radius),
        "time_to_settle_010_s": settle,
        "path_length_ratio_raw": raw_length / straight if straight > 0 else None,
        "path_length_ratio_smoothed": smooth_length / straight if straight > 0 else None,
        "end_speed_m_s": end_speed,
        "stops": bool(end_speed < stop_speed),
        "min_root_z_m": float(np.min(frames[:, 2])),
        "peak_speed_0p5s_m_s": float(window_speed.max()),
        "time_above_1p5_m_s_s": float(np.count_nonzero(window_speed > 1.5) / CONTROL_FPS),
    }


def stop_distance_trial(
    runtime: DeployPlannerRuntime,
    band: SpeedBand,
    stop_after_s: float,
    *,
    settle_s: float = 3.0,
) -> dict:
    """Walk straight along +x in ``band``, command IDLE at a planner tick, measure the run-out."""
    stop_tick = int(round(stop_after_s * CONTROL_FPS))
    if stop_tick % PLANNER_PERIOD_TICKS:
        raise ValueError("stop time must fall on a planner tick")
    runtime.reset()
    walk = locomotion_command(band.mode, 0.0, 0.0, band.speed)
    frames = []
    for tick in range(stop_tick + int(round(settle_s * CONTROL_FPS))):
        frames.append(runtime.step(walk if tick < stop_tick else idle_command(0.0)))
    frames = np.asarray(frames)
    at_stop = frames[stop_tick, :2]
    walk_speed = float(np.linalg.norm(frames[stop_tick, :2] - frames[stop_tick - 50, :2]) / 1.0)
    return {
        "band": band.key,
        "stop_after_s": stop_after_s,
        "walk_speed_m_s": walk_speed,
        "run_out_m": float(frames[-1, 0] - at_stop[0]),
        "lateral_m": float(frames[-1, 1] - at_stop[1]),
        "end_speed_m_s": float(np.linalg.norm(frames[-1, :2] - frames[-26, :2]) / 0.5),
    }
