"""Deploy-faithful closed-loop kinematic runner for the SONIC V2 planner (no ONNX import).

Every rule below is ported from ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref`` in
``/home/robotixx/GR00T-WholeBodyControl`` (line numbers as of the installed checkout):

- ``LocalMotionPlannerBase`` (``include/localmotion_kplanner.hpp``): ``Initialize`` and
  ``InitializeContext`` (standing context at the default height, identity root rotation, measured
  joints), ``UpdatePlanning`` (context taken ``motion_look_ahead_steps = 2`` control ticks after the
  cursor), ``UpdateContextFromMotion`` (four samples at 30 Hz spacing, clamped at the end, linear
  position and joint interpolation, quaternion slerp) and ``ResampleGeneratedSequence50Hz``
  (``floor(n / 30 * 50)`` samples of the valid 30 Hz frames).
- ``LocalMotionPlannerTensorRT`` (the active backend): token mask ``[0,0,0,1,1,1,0,0,0,0,0]``
  (9, 10 or 11 tokens), V2 mode range 27, seed 1234 kept for every call, waypoints disabled.
- ``G1Deploy::Planner`` (``src/g1_deploy_onnx_ref.cpp:3561-3760``): 10 Hz planner thread, replan
  triggers (mode, facing or height change always; speed, direction or the periodic timer only in
  non-static modes, the timer only while speed != 0) and intervals (0.1 s running, 0.2 s crawling
  (mode 8 only), 1.0 s boxing, 1.0 s otherwise; ``:228-231``), with the float32 counter.
- ``G1Deploy::CurrentFrameAdvancement`` (``:3150-3380``): rebase so the cursor becomes frame 0,
  linear 8-frame cross-fade starting at ``gen_frame - cursor``, cursor clamped at the last frame,
  and the IDLE re-adaptation state machine (``:237-243``).
- ``GamepadManager`` (``input_interface/gamepad_manager.hpp:425-560, 905-980``): per-mode speed and
  height defaults, zero movement and speed 0 in the static kneel/squat modes, and the staged crawl
  entry (IDLE -> IDEL_KNEEL_TWO_LEGS -> 2 s -> CRAWLING [-> 2 s -> ELBOW_CRAWLING]).

Declared extensions (not in the deploy binary):

- E1 ``has_specific_target`` is exposed. The deploy never sets it. A change of the waypoint
  fields triggers a replan like a facing change.
- E2 The mode whitelist is ``COMMAND_MODES`` = {0, 1, 2, 4, 8, 14, 18, 22}, plus
  ``TRANSITION_MODES`` = {5}, used only by the deploy's staged crawl entry.
- E3 The two deploy threads are serialized: the planner runs every fifth 50 Hz tick, and a plan
  is blended ``latency_ticks`` ticks after its call (0 = before the next cursor advance, the
  TensorRT case of about 1-2 ms per call).

Coordinates: world metres, Z up, qpos in MuJoCo order (xyz, wxyz, 29 named joints).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import time

import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_planner_adapter import (
    ACTIVE_TOKEN_MASK,
    INPUT_SPEC,
    OUTPUT_SPEC,
    PLANNER_FPS,
    REFERENCE_FPS,
)

CONTROL_FPS = REFERENCE_FPS
PLANNER_DT = 0.1
PLANNER_PERIOD_TICKS = 5
LOOKAHEAD_FRAMES = 2
BLEND_FRAMES = 8
MAX_PLANNER_FRAMES = 64
DEFAULT_HEIGHT = 0.788740
INITIAL_RANDOM_SEED = 1234
V2_MODE_COUNT = 27
DEPLOY_TOKEN_MASK = tuple(ACTIVE_TOKEN_MASK)
# The deprecated ONNX Runtime backend's mask (localmotion_kplanner_onnx.hpp); reference only.
ONNX_BACKEND_TOKEN_MASK = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0)

# fmt: off
MODE_NAMES = (
    "IDLE", "SLOW_WALK", "WALK", "RUN", "IDEL_SQUAT", "IDEL_KNEEL_TWO_LEGS", "IDEL_KNEEL",
    "IDEL_LYING_FACE_DOWN", "CRAWLING", "IDEL_BOXING", "WALK_BOXING", "LEFT_PUNCH", "RIGHT_PUNCH",
    "RANDOM_PUNCH", "ELBOW_CRAWLING", "LEFT_HOOK", "RIGHT_HOOK", "FORWARD_JUMP", "STEALTH_WALK",
    "INJURED_WALK", "LEDGE_WALKING", "OBJECT_CARRYING", "STEALTH_WALK_2", "HAPPY_DANCE_WALK",
    "ZOMBIE_WALK", "GUN_WALK", "SCARE_WALK",
)
# fmt: on
IDLE, SLOW_WALK, WALK, RUN, IDEL_SQUAT, IDEL_KNEEL_TWO_LEGS = 0, 1, 2, 3, 4, 5
CRAWLING, ELBOW_CRAWLING, STEALTH_WALK, STEALTH_WALK_2 = 8, 14, 18, 22
COMMAND_MODES = frozenset({0, 1, 2, 4, 8, 14, 18, 22})
TRANSITION_MODES = frozenset({IDEL_KNEEL_TWO_LEGS})
STATIC_MODES = frozenset({0, 4, 5, 6, 7, 9})
BOXING_REPLAN_MODES = frozenset({11, 12, 13, 15, 16})
REPLAN_INTERVAL_RUNNING = 0.1
REPLAN_INTERVAL_CRAWLING = 0.2
REPLAN_INTERVAL_BOXING = 1.0
REPLAN_INTERVAL_DEFAULT = 1.0
# GamepadManager::applySpeedAndHeight; unlisted modes send (-1, -1) = mode defaults.
GAMEPAD_SPEED_HEIGHT = {
    SLOW_WALK: (0.4, -1.0),
    RUN: (1.5, -1.0),
    CRAWLING: (0.7, 0.4),
    ELBOW_CRAWLING: (0.7, 0.3),
    IDEL_SQUAT: (-1.0, 0.4),
    IDEL_KNEEL_TWO_LEGS: (-1.0, 0.4),
    6: (-1.0, 0.4),
}
STAGED_CRAWL_DWELL_S = 2.0
# policy_parameters.hpp default_angles, MuJoCo joint order.
# fmt: off
DEFAULT_ANGLES_MUJOCO = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
)
# fmt: on
LOWER_BODY_MUJOCO = tuple(range(12))
IDLE_ADAPT_TRIGGER = 0.10
IDLE_ADAPT_STOP = 0.05
IDLE_RECOVER_TRIGGER = 0.045
QPOS_DIM = 36

Vector3 = tuple[float, float, float]


def _unit_or_zero(vector: Sequence[float], name: str, *, allow_zero: bool) -> Vector3:
    values = tuple(float(v) for v in vector)
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must be three finite values")
    norm = math.sqrt(sum(v * v for v in values))
    if allow_zero and norm == 0.0:
        return values
    if abs(norm - 1.0) > 1e-5 or abs(values[2]) > 1e-6:
        raise ValueError(f"{name} must be a horizontal unit vector" + (" or zero" * allow_zero))
    return values


def yaw_vector(yaw_rad: float) -> Vector3:
    return (math.cos(yaw_rad), math.sin(yaw_rad), 0.0)


@dataclass(frozen=True)
class PlannerCommand:
    """One MovementState (plus E1 waypoint fields) as the planner thread reads it."""

    mode: int = IDLE
    movement_direction: Vector3 = (0.0, 0.0, 0.0)
    facing_direction: Vector3 = (1.0, 0.0, 0.0)
    speed: float = -1.0
    height: float = -1.0
    has_specific_target: bool = False
    target_positions: tuple[Vector3, ...] = ((0.0, 0.0, 0.0),) * 4
    target_headings: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        if isinstance(self.mode, bool) or int(self.mode) != self.mode:
            raise ValueError("mode must be an integer")
        if self.mode not in COMMAND_MODES | TRANSITION_MODES:
            raise ValueError(f"mode {self.mode} is outside the audited whitelist")
        movement = _unit_or_zero(self.movement_direction, "movement_direction", allow_zero=True)
        facing = _unit_or_zero(self.facing_direction, "facing_direction", allow_zero=False)
        positions = tuple(tuple(float(v) for v in p) for p in self.target_positions)
        headings = tuple(float(h) for h in self.target_headings)
        if len(positions) != 4 or any(len(p) != 3 for p in positions) or len(headings) != 4:
            raise ValueError("waypoints need four xyz positions and four headings")
        if not all(math.isfinite(v) for p in positions for v in p) or not all(
            math.isfinite(h) for h in headings
        ):
            raise ValueError("waypoints must be finite")
        if not (math.isfinite(self.speed) and math.isfinite(self.height)):
            raise ValueError("speed and height must be finite")
        object.__setattr__(self, "mode", int(self.mode))
        object.__setattr__(self, "movement_direction", movement)
        object.__setattr__(self, "facing_direction", facing)
        object.__setattr__(self, "speed", float(self.speed))
        object.__setattr__(self, "height", float(self.height))
        object.__setattr__(self, "has_specific_target", bool(self.has_specific_target))
        object.__setattr__(self, "target_positions", positions)
        object.__setattr__(self, "target_headings", headings)

    @property
    def waypoint_key(self):
        return (self.has_specific_target, self.target_positions, self.target_headings)


# The deploy's last_movement_state_ is default-constructed (speed 0, height 0), so the first
# planner tick after Initialize always sees a height/speed change and replans.
DEPLOY_INITIAL_LAST_COMMAND = PlannerCommand(IDLE, (0, 0, 0), (1, 0, 0), speed=0.0, height=0.0)


def idle_command(facing_yaw: float = 0.0) -> PlannerCommand:
    """Standing-set dead-zone state: IDLE, zero movement, speed and height -1."""
    return PlannerCommand(IDLE, (0.0, 0.0, 0.0), yaw_vector(facing_yaw), -1.0, -1.0)


def locomotion_command(
    mode: int,
    movement_yaw: float,
    facing_yaw: float | None = None,
    speed: float | None = None,
    height: float | None = None,
) -> PlannerCommand:
    """A moving-mode command; ``None`` speed/height take the GamepadManager defaults."""
    if mode in STATIC_MODES:
        raise ValueError("static modes take static_posture_command")
    default_speed, default_height = GAMEPAD_SPEED_HEIGHT.get(mode, (-1.0, -1.0))
    return PlannerCommand(
        mode,
        yaw_vector(movement_yaw),
        yaw_vector(movement_yaw if facing_yaw is None else facing_yaw),
        default_speed if speed is None else speed,
        default_height if height is None else height,
    )


def static_posture_command(mode: int, height: float, facing_yaw: float = 0.0) -> PlannerCommand:
    """Static kneel/squat modes always send zero movement and speed 0 (gamepad_manager.hpp)."""
    if mode not in STATIC_MODES or mode == IDLE:
        raise ValueError("static_posture_command needs a static non-IDLE mode")
    return PlannerCommand(mode, (0.0, 0.0, 0.0), yaw_vector(facing_yaw), 0.0, float(height))


def encode_waypoints(
    targets_xy: Sequence[Sequence[float]],
    headings_rad: Sequence[float] | float,
    *,
    z: float = 0.0,
) -> tuple[tuple[Vector3, ...], tuple[float, ...]]:
    """World-frame waypoint tensors. 1-4 targets; missing slots repeat the last target.

    Measured on this graph: the last slot sets the end point, target z is ignored and the
    heading is the final world yaw (see the kinematic study report).
    """
    targets = [tuple(float(v) for v in t) for t in targets_xy]
    if not 1 <= len(targets) <= 4 or any(len(t) != 2 for t in targets):
        raise ValueError("need one to four xy targets")
    if np.isscalar(headings_rad):
        headings = [float(headings_rad)] * len(targets)
    else:
        headings = [float(h) for h in headings_rad]
    if len(headings) != len(targets):
        raise ValueError("one heading per target")
    wrapped = [math.atan2(math.sin(h), math.cos(h)) for h in headings]
    targets += [targets[-1]] * (4 - len(targets))
    wrapped += [wrapped[-1]] * (4 - len(wrapped))
    return tuple((x, y, float(z)) for x, y in targets), tuple(wrapped)


def waypoint_command(
    mode: int,
    target_xy: Sequence[float],
    heading_rad: float,
    *,
    movement_yaw: float | None = None,
    speed: float | None = None,
) -> PlannerCommand:
    """E1: a moving mode with ``has_specific_target`` = 1 and the goal in every slot."""
    positions, headings = encode_waypoints([target_xy], heading_rad)
    base = locomotion_command(
        mode, heading_rad if movement_yaw is None else movement_yaw, heading_rad, speed
    )
    return PlannerCommand(
        base.mode,
        base.movement_direction,
        base.facing_direction,
        base.speed,
        base.height,
        True,
        positions,
        headings,
    )


def replan_interval(mode: int, scale: float = 1.0) -> np.float32:
    if mode == RUN:
        base = REPLAN_INTERVAL_RUNNING
    elif mode == CRAWLING:  # the deploy checks CRAWLING only, not ELBOW_CRAWLING
        base = REPLAN_INTERVAL_CRAWLING
    elif mode in BOXING_REPLAN_MODES:
        base = REPLAN_INTERVAL_BOXING
    else:
        base = REPLAN_INTERVAL_DEFAULT
    return np.float32(base * scale)


def quat_slerp_deploy(q0: np.ndarray, q1: np.ndarray, t) -> np.ndarray:
    """``quat_slerp_d`` (math_utils.hpp): shortest arc, normalized lerp when dot > 0.9995."""
    q0 = np.atleast_2d(np.asarray(q0, dtype=np.float64))
    q1 = np.atleast_2d(np.asarray(q1, dtype=np.float64))
    t = np.broadcast_to(np.asarray(t, dtype=np.float64), (len(q0),))
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where(dot[:, None] < 0.0, -q1, q1)
    dot = np.abs(dot)
    lerp = q0 + t[:, None] * (q1 - q0)
    lerp /= np.linalg.norm(lerp, axis=1, keepdims=True)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.where(dot > 0.9995, 1.0, np.sin(theta))
    f0 = np.sin((1.0 - t) * theta) / sin_theta
    f1 = np.sin(t * theta) / sin_theta
    slerp = f0[:, None] * q0 + f1[:, None] * q1
    return np.where((dot > 0.9995)[:, None], lerp, slerp)


def standing_context(
    joint_positions: Sequence[float] = DEFAULT_ANGLES_MUJOCO,
    default_height: float = DEFAULT_HEIGHT,
) -> np.ndarray:
    """``InitializeContext``: xy 0, default height, identity rotation, the measured joints."""
    joints = np.asarray(joint_positions, dtype=np.float64)
    if joints.shape != (29,) or not np.all(np.isfinite(joints)):
        raise ValueError("need 29 finite MuJoCo-order joint positions")
    context = np.zeros((4, QPOS_DIM))
    context[:, 2] = default_height
    context[:, 3] = 1.0
    context[:, 7:] = joints
    return context


def sample_context(motion_50hz: np.ndarray, gen_frame: int) -> np.ndarray:
    """``UpdateContextFromMotion``: four samples from ``gen_frame / 50`` at 30 Hz spacing.

    Out-of-range samples clamp to the last frame, as in the deploy (no rejection).
    """
    motion = np.asarray(motion_50hz, dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != QPOS_DIM or len(motion) == 0:
        raise ValueError("motion not ready, cannot update context")
    last = len(motion) - 1
    gen_time = float(gen_frame) / CONTROL_FPS
    context = np.empty((4, QPOS_DIM))
    for n in range(4):
        t = gen_time + float(n) / PLANNER_FPS
        f_50 = t * CONTROL_FPS
        f0 = min(int(math.floor(f_50)), last)
        f1 = min(f0 + 1, last)
        w0 = 1.0 - (f_50 - f0)
        w1 = 1.0 - w0
        context[n, 3:7] = quat_slerp_deploy(motion[f0, 3:7], motion[f1, 3:7], f_50 - f0)[0]
        context[n, :3] = w0 * motion[f0, :3] + w1 * motion[f1, :3]
        context[n, 7:] = w0 * motion[f0, 7:] + w1 * motion[f1, 7:]
    return context


def resample_plan_50hz(qpos_30hz: np.ndarray, num_pred_frames: int) -> np.ndarray:
    """``ResampleGeneratedSequence50Hz`` over the valid frames only."""
    count = int(num_pred_frames)
    data = np.asarray(qpos_30hz, dtype=np.float64)
    if count < 2 or count > MAX_PLANNER_FRAMES or data.shape[0] < count:
        raise ValueError(f"invalid planner frame count {count}")
    valid = data[:count]
    if valid.shape[1] != QPOS_DIM or not np.all(np.isfinite(valid)):
        raise ValueError("planner output contains nonfinite values")
    frames = int(math.floor(float(count) / PLANNER_FPS * CONTROL_FPS))
    out = np.empty((frames, QPOS_DIM))
    for f in range(frames):
        f_30 = (f / CONTROL_FPS) * PLANNER_FPS
        f0 = int(math.floor(f_30))
        f1 = min(f0 + 1, count - 1)
        w0 = 1.0 - (f_30 - f0)
        w1 = 1.0 - w0
        out[f, :3] = w0 * valid[f0, :3] + w1 * valid[f1, :3]
        out[f, 3:7] = quat_slerp_deploy(valid[f0, 3:7], valid[f1, 3:7], f_30 - f0)[0]
        out[f, 7:] = w0 * valid[f0, 7:] + w1 * valid[f1, 7:]
    return out


def cross_fade(
    old_motion: np.ndarray, current_frame: int, new_motion: np.ndarray, gen_frame: int
) -> np.ndarray | None:
    """``CurrentFrameAdvancement``: rebase at the cursor, fade in the new plan over 8 frames.

    Returns ``None`` when the plan arrived too late to contribute any frame.
    """
    old = np.asarray(old_motion, dtype=np.float64)
    new = np.asarray(new_motion, dtype=np.float64)
    length = gen_frame - current_frame + len(new)
    if length <= 0:
        return None
    blend_start = max(0, gen_frame - current_frame)
    f = np.arange(length)
    f_old = np.clip(f + current_frame, 0, len(old) - 1)
    f_new = np.clip(f + current_frame - gen_frame, 0, len(new) - 1)
    w_new = np.clip((f - blend_start) / BLEND_FRAMES, 0.0, 1.0)
    w_old = 1.0 - w_new
    out = np.empty((length, QPOS_DIM))
    out[:, :3] = w_old[:, None] * old[f_old, :3] + w_new[:, None] * new[f_new, :3]
    out[:, 7:] = w_old[:, None] * old[f_old, 7:] + w_new[:, None] * new[f_new, 7:]
    out[:, 3:7] = quat_slerp_deploy(old[f_old, 3:7], new[f_new, 3:7], w_new)
    return out


def build_inputs(
    context: np.ndarray,
    command: PlannerCommand,
    *,
    random_seed: int = INITIAL_RANDOM_SEED,
    token_mask: Sequence[int] = DEPLOY_TOKEN_MASK,
) -> dict[str, np.ndarray]:
    """The eleven graph tensors for one call, dtypes and shapes as in ``INPUT_SPEC``."""
    mask = tuple(int(v) for v in token_mask)
    if len(mask) != 11 or not set(mask) <= {0, 1} or not any(mask):
        raise ValueError("token mask must be eleven binary values, not all zero")
    context = np.asarray(context, dtype=np.float64).reshape(1, 4, QPOS_DIM)
    values = {
        "context_mujoco_qpos": context,
        "target_vel": [command.speed],
        "mode": [command.mode],
        "movement_direction": [command.movement_direction],
        "facing_direction": [command.facing_direction],
        "random_seed": [random_seed],
        "has_specific_target": [[int(command.has_specific_target)]],
        "specific_target_positions": [command.target_positions],
        "specific_target_headings": [command.target_headings],
        "allowed_pred_num_tokens": [mask],
        "height": [command.height],
    }
    result = {name: np.asarray(value, dtype=INPUT_SPEC[name][0]) for name, value in values.items()}
    for name, (_, shape) in INPUT_SPEC.items():
        if result[name].shape != shape:
            raise ValueError(f"{name} has shape {result[name].shape}, expected {shape}")
    return result


def checked_output(outputs: Mapping[str, np.ndarray], token_mask: Sequence[int]) -> tuple:
    """Return (valid 30 Hz frames, count); the count must be one the mask allows."""
    qpos = np.asarray(outputs["mujoco_qpos"])
    count = int(np.asarray(outputs["num_pred_frames"]).reshape(-1)[0])
    if qpos.shape != OUTPUT_SPEC["mujoco_qpos"][1]:
        raise ValueError("unexpected planner output shape")
    if count % 4 or not 24 <= count <= MAX_PLANNER_FRAMES:
        raise ValueError(f"planner returned {count} frames")
    if int(token_mask[count // 4 - 6]) != 1:
        raise ValueError(f"planner returned {count} frames, which the token mask disables")
    return qpos[0, :count].astype(np.float64), count


def idle_readapt_step(state: str, planned: np.ndarray, measured: np.ndarray, original: np.ndarray):
    """One tick of the IDLE double-threshold re-adaptation (lower-body joints).

    Returns the new state and the (possibly blended) planned lower-body joints.
    """
    error = float(np.mean(np.abs(planned - measured)))
    if state == "IDLE":
        if error > IDLE_ADAPT_TRIGGER:
            state = "ADAPTING"
        elif error < IDLE_RECOVER_TRIGGER:
            state = "RECOVERING"
    elif state == "ADAPTING":
        if error < IDLE_ADAPT_STOP:
            state = "IDLE"
    elif state == "RECOVERING":
        if error > IDLE_ADAPT_TRIGGER:
            state = "ADAPTING"
    if state == "ADAPTING":
        return state, 0.98 * planned + 0.02 * measured
    if state == "RECOVERING":
        return state, 0.98 * planned + 0.02 * original
    return state, planned


@dataclass
class PlannerCall:
    tick: int
    cursor: int
    gen_frame: int
    num_pred_frames: int
    frames_50hz: int
    reasons: tuple[str, ...]
    inference_s: float
    command: PlannerCommand

    def as_dict(self) -> dict:
        return {
            "tick": self.tick,
            "cursor": self.cursor,
            "gen_frame": self.gen_frame,
            "num_pred_frames": self.num_pred_frames,
            "frames_50hz": self.frames_50hz,
            "reasons": list(self.reasons),
            "inference_s": self.inference_s,
            "mode": self.command.mode,
        }


@dataclass
class _Pending:
    plan: np.ndarray
    gen_frame: int
    ready_tick: int


InferFn = Callable[[dict[str, np.ndarray]], Mapping[str, np.ndarray]]


@dataclass
class DeployPlannerRuntime:
    """Serialized planner thread + control-thread cursor, one 50 Hz tick per ``step``.

    ``infer`` maps the eleven input tensors to ``{"mujoco_qpos", "num_pred_frames"}``.
    ``measured_lower_body`` (optional) returns the 12 measured leg joints for the IDLE
    re-adaptation; kinematically the robot is the reference, which makes it a no-op.
    """

    infer: InferFn
    token_mask: Sequence[int] = DEPLOY_TOKEN_MASK
    replan_interval_scale: float = 1.0
    latency_ticks: int = 0
    random_seed: int = INITIAL_RANDOM_SEED
    measured_lower_body: Callable[[int], np.ndarray] | None = None
    calls: list[PlannerCall] = field(default_factory=list)

    def __post_init__(self):
        if self.latency_ticks < 0 or self.latency_ticks >= PLANNER_PERIOD_TICKS:
            raise ValueError("latency must be shorter than one planner period")
        if not self.replan_interval_scale > 0:
            raise ValueError("replan interval scale must be positive")
        self.motion = np.zeros((0, QPOS_DIM))
        self.current_frame = 0
        self.tick = 0
        self.initialized = False

    # -- planner thread -------------------------------------------------------------
    def reset(
        self,
        joint_positions: Sequence[float] = DEFAULT_ANGLES_MUJOCO,
        default_height: float = DEFAULT_HEIGHT,
    ) -> None:
        """``Initialize`` + the control thread's first-time copy; cursor at frame 0."""
        self.calls = []
        self.tick = 0
        self.current_frame = 0
        self.counter = np.float32(0.0)
        self.last_command = DEPLOY_INITIAL_LAST_COMMAND
        self.latest_command = DEPLOY_INITIAL_LAST_COMMAND
        self.pending = None
        self.hold_ticks = 0
        self.blends = 0
        self.late_plans = 0
        self.idle_state = "IDLE"
        self.idle_original = None
        init = PlannerCommand(IDLE, (0, 0, 0), (1, 0, 0), -1.0, -1.0)
        plan = self._plan(standing_context(joint_positions, default_height), init, 0, ("init",))
        self.motion = plan
        self.initialized = True

    def _plan(self, context, command, gen_frame, reasons) -> np.ndarray:
        inputs = build_inputs(
            context, command, random_seed=self.random_seed, token_mask=self.token_mask
        )
        started = time.perf_counter()
        outputs = self.infer(inputs)
        elapsed = time.perf_counter() - started
        valid, count = checked_output(outputs, self.token_mask)
        plan = resample_plan_50hz(valid, count)
        self.calls.append(
            PlannerCall(
                self.tick,
                self.current_frame,
                gen_frame,
                count,
                len(plan),
                tuple(reasons),
                elapsed,
                command,
            )
        )
        return plan

    def _planner_tick(self, command: PlannerCommand) -> None:
        last = self.last_command
        reasons = []
        if command.mode != last.mode:
            reasons.append("mode")
        if command.facing_direction != last.facing_direction:
            reasons.append("facing")
        if command.height != last.height:
            reasons.append("height")
        if command.waypoint_key != last.waypoint_key:
            reasons.append("waypoint")
        static = command.mode in STATIC_MODES
        self.counter = np.float32(np.float64(self.counter) + PLANNER_DT)
        time_to_replan = False
        if self.counter >= replan_interval(command.mode, self.replan_interval_scale):
            self.counter = np.float32(0.0)
            time_to_replan = True
        if not reasons and not static:
            if command.speed != last.speed:
                reasons.append("speed")
            if command.movement_direction != last.movement_direction:
                reasons.append("direction")
            if time_to_replan and command.speed != 0:
                reasons.append("timer")
        if not reasons:
            return
        self.last_command = command
        gen_frame = self.current_frame + LOOKAHEAD_FRAMES
        context = sample_context(self.motion, gen_frame)
        plan = self._plan(context, command, gen_frame, reasons)
        self.pending = _Pending(plan, gen_frame, self.tick + self.latency_ticks)

    # -- control thread -------------------------------------------------------------
    def _advance(self) -> None:
        if self.pending is not None and self.pending.ready_tick <= self.tick:
            pending, self.pending = self.pending, None
            blended = cross_fade(self.motion, self.current_frame, pending.plan, pending.gen_frame)
            if blended is None:
                self.late_plans += 1
            else:
                self.motion = blended
                self.current_frame = 0
                self.blends += 1
                self.idle_original = None
        new_frame = self.current_frame + 1
        if new_frame >= len(self.motion):
            new_frame = len(self.motion) - 1
            self.hold_ticks += 1
            if self.latest_command.mode == IDLE and self.measured_lower_body is not None:
                legs = list(LOWER_BODY_MUJOCO)
                if self.idle_original is None:
                    self.idle_original = self.motion[new_frame, 7:][legs].copy()
                    self.idle_state = "IDLE"
                measured = np.asarray(self.measured_lower_body(self.tick), dtype=np.float64)
                self.idle_state, blended_legs = idle_readapt_step(
                    self.idle_state, self.motion[new_frame, 7:][legs], measured, self.idle_original
                )
                self.motion[new_frame, 7 + np.asarray(legs)] = blended_legs
        self.current_frame = new_frame

    def step(self, command: PlannerCommand) -> np.ndarray:
        """One 50 Hz tick: planner thread (every fifth tick), observe, advance the cursor.

        Returns the reference frame the tracker observes at this tick.
        """
        if not self.initialized:
            raise RuntimeError("call reset() first")
        self.latest_command = command
        if self.tick > 0 and self.tick % PLANNER_PERIOD_TICKS == 0:
            self._planner_tick(command)
        frame = self.motion[self.current_frame].copy()
        self._advance()
        self.tick += 1
        return frame

    @property
    def current_reference(self) -> np.ndarray:
        return self.motion[self.current_frame]


def run_schedule(
    runtime: DeployPlannerRuntime,
    schedule: Sequence[tuple[float, PlannerCommand]],
    duration_s: float,
) -> np.ndarray:
    """Reset, then play a time-stamped command schedule; returns the observed 50 Hz frames."""
    events = sorted(schedule, key=lambda item: item[0])
    if not events or events[0][0] > 0:
        events = [(0.0, idle_command())] + list(events)
    runtime.reset()
    frames = []
    index = 0
    command = events[0][1]
    for tick in range(int(round(duration_s * CONTROL_FPS))):
        while index < len(events) and events[index][0] <= tick / CONTROL_FPS + 1e-9:
            command = events[index][1]
            index += 1
        frames.append(runtime.step(command))
    return np.asarray(frames)


def staged_crawl_schedule(
    final_mode: int, movement_yaw: float = 0.0, start_s: float = 0.0
) -> list[tuple[float, PlannerCommand]]:
    """GamepadManager's staged entry: kneel, then crawl after 2 s, then elbow crawl after 2 s."""
    if final_mode not in (CRAWLING, ELBOW_CRAWLING):
        raise ValueError("staged entry exists only for CRAWLING and ELBOW_CRAWLING")
    kneel_height = GAMEPAD_SPEED_HEIGHT[IDEL_KNEEL_TWO_LEGS][1]
    schedule = [(start_s, static_posture_command(IDEL_KNEEL_TWO_LEGS, kneel_height, movement_yaw))]
    schedule.append(
        (start_s + STAGED_CRAWL_DWELL_S, locomotion_command(CRAWLING, movement_yaw, movement_yaw))
    )
    if final_mode == ELBOW_CRAWLING:
        schedule.append(
            (
                start_s + 2 * STAGED_CRAWL_DWELL_S,
                locomotion_command(ELBOW_CRAWLING, movement_yaw, movement_yaw),
            )
        )
    return schedule


def yaw_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
