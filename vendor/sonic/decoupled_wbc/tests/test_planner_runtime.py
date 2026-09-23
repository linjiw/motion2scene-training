"""Deploy-faithful planner runtime logic, without ONNX Runtime, a GPU or a simulator."""

import math
import struct

import numpy as np
import pytest

from gear_sonic.dataset_generation.deployable_retiming import slerp
from gear_sonic.dataset_generation.hallucination.motion2scene_planner_adapter import INPUT_SPEC
from gear_sonic.research.planner.goal_controllers import (
    DirectionSpeedStopController,
    KinematicState,
    SpeedBand,
    WaypointController,
    goal_metrics,
)
from gear_sonic.research.planner.planner_runtime import (
    CRAWLING,
    DEFAULT_ANGLES_MUJOCO,
    DEFAULT_HEIGHT,
    DEPLOY_TOKEN_MASK,
    ELBOW_CRAWLING,
    IDEL_KNEEL_TWO_LEGS,
    IDEL_SQUAT,
    IDLE,
    SLOW_WALK,
    STEALTH_WALK,
    WALK,
    DeployPlannerRuntime,
    PlannerCommand,
    build_inputs,
    checked_output,
    cross_fade,
    encode_waypoints,
    idle_command,
    idle_readapt_step,
    locomotion_command,
    quat_slerp_deploy,
    replan_interval,
    resample_plan_50hz,
    run_schedule,
    sample_context,
    staged_crawl_schedule,
    standing_context,
    static_posture_command,
    waypoint_command,
    yaw_from_quat,
)


def yaw_quat(yaw):
    return np.stack(
        [np.cos(np.asarray(yaw) / 2), 0 * yaw, 0 * yaw, np.sin(np.asarray(yaw) / 2)], -1
    )


def ramp_motion(frames=100, fps=50.0, speed=1.0, yaw_rate=0.0):
    t = np.arange(frames) / fps
    motion = np.zeros((frames, 36))
    motion[:, 0] = speed * t
    motion[:, 2] = 0.8
    motion[:, 3:7] = yaw_quat(yaw_rate * t)
    motion[:, 7] = t
    return motion


class FakePlanner:
    """Continues the context along +x at 1 cm per 30 Hz frame; records every call."""

    def __init__(self, count=44):
        self.count = count
        self.inputs = []

    def __call__(self, inputs):
        self.inputs.append(inputs)
        context = inputs["context_mujoco_qpos"][0].astype(np.float64)
        qpos = np.repeat(context[:1], 64, axis=0)
        qpos[:, 0] += 0.01 * np.arange(64)
        qpos[self.count :] = 99.0  # padded rows are garbage in the real graph
        return {
            "mujoco_qpos": qpos[None].astype(np.float32),
            "num_pred_frames": np.array([self.count], dtype=np.int32),
        }


# -- context sampling, resampling, cross-fade --------------------------------------------
def test_context_samples_cursor_plus_two_at_thirty_hz_spacing():
    motion = ramp_motion(200)
    context = sample_context(motion, 17)
    expected_t = 17 / 50 + np.arange(4) / 30
    np.testing.assert_allclose(context[:, 0], expected_t, atol=1e-12)
    np.testing.assert_allclose(context[:, 7], expected_t, atol=1e-12)
    turning = ramp_motion(200, yaw_rate=0.5)
    context = sample_context(turning, 17)
    np.testing.assert_allclose(yaw_from_quat(context[:, 3:7]), 0.5 * expected_t, atol=1e-6)


def test_context_clamps_past_the_end_like_the_deploy():
    motion = ramp_motion(20)
    context = sample_context(motion, 18)
    np.testing.assert_allclose(context[1:, 0], motion[-1, 0])
    np.testing.assert_allclose(np.linalg.norm(context[:, 3:7], axis=1), 1.0)
    with pytest.raises(ValueError, match="not ready"):
        sample_context(np.zeros((0, 36)), 0)


def test_resample_uses_floor_count_and_linear_time():
    for count, expected in ((36, 60), (40, 66), (44, 73), (64, 106)):
        plan = ramp_motion(64, fps=30.0)
        plan[count:] = np.nan  # padding must never be read
        out = resample_plan_50hz(plan, count)
        assert len(out) == expected
        # Samples past the last valid 30 Hz frame hold it (f1 clamps), as in the C++ resampler.
        expected_t = np.minimum(np.arange(expected) / 50, (count - 1) / 30)
        np.testing.assert_allclose(out[:, 0], expected_t, atol=1e-12)
    with pytest.raises(ValueError):
        resample_plan_50hz(ramp_motion(64, fps=30.0), 65)


def test_deploy_slerp_matches_exact_slerp_and_takes_short_arc():
    q0 = yaw_quat(np.zeros(5))
    q1 = yaw_quat(np.array([0.01, 0.3, 1.0, 2.0, 3.0]))
    t = np.array([0.5, 0.25, 0.75, 0.5, 0.1])
    np.testing.assert_allclose(quat_slerp_deploy(q0, q1, t), slerp(q0, q1, t), atol=2e-7)
    flipped = quat_slerp_deploy(q0[:1], -yaw_quat(np.array([0.4])), 0.5)
    assert abs(yaw_from_quat(flipped)[0] - 0.2) < 1e-9


def test_cross_fade_rebases_and_blends_over_eight_frames():
    old = np.zeros((80, 36))
    old[:, 3] = 1
    new = np.zeros((73, 36))
    new[:, 3] = 1
    new[:, 0] = 1.0
    out = cross_fade(old, 10, new, 12)
    assert len(out) == 12 - 10 + 73
    weights = out[:, 0]
    np.testing.assert_allclose(weights[:3], 0.0)
    np.testing.assert_allclose(weights[2:11], np.arange(9) / 8)
    np.testing.assert_allclose(weights[10:], 1.0)


def test_cross_fade_late_plan_starts_immediately_or_is_dropped():
    old = np.zeros((80, 36))
    old[:, 3] = 1
    new = np.zeros((20, 36))
    new[:, 3] = 1
    new[:, 0] = np.arange(20)
    out = cross_fade(old, 15, new, 12)  # cursor already 3 frames past gen_frame
    assert len(out) == 12 - 15 + 20
    np.testing.assert_allclose(out[8:, 0], np.arange(11, 20))  # f_new = f + 3, w_new = f / 8
    assert cross_fade(old, 40, new, 12) is None


# -- commands, modes, waypoints ------------------------------------------------------------
def test_mode_whitelist_and_gamepad_defaults():
    assert locomotion_command(CRAWLING, 0.0).speed == 0.7
    assert locomotion_command(CRAWLING, 0.0).height == 0.4
    assert locomotion_command(ELBOW_CRAWLING, 0.0).height == 0.3
    assert locomotion_command(SLOW_WALK, 0.0).speed == 0.4
    assert locomotion_command(STEALTH_WALK, 0.0).speed == -1.0
    squat = static_posture_command(IDEL_SQUAT, 0.3)
    assert (
        squat.speed == 0.0 and squat.movement_direction == (0.0, 0.0, 0.0) and squat.height == 0.3
    )
    PlannerCommand(IDEL_KNEEL_TWO_LEGS)  # transition-only mode of the staged crawl entry
    for bad in (3, 6, 10, 27):
        with pytest.raises(ValueError, match="whitelist"):
            PlannerCommand(bad)
    with pytest.raises(ValueError):
        locomotion_command(IDEL_SQUAT, 0.0)
    with pytest.raises(ValueError):
        PlannerCommand(SLOW_WALK, (0.5, 0.0, 0.0))
    with pytest.raises(ValueError):
        PlannerCommand(SLOW_WALK, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_staged_crawl_schedule_follows_gamepad_manager():
    schedule = staged_crawl_schedule(ELBOW_CRAWLING)
    assert [(t, c.mode) for t, c in schedule] == [
        (0.0, IDEL_KNEEL_TWO_LEGS),
        (2.0, CRAWLING),
        (4.0, ELBOW_CRAWLING),
    ]
    assert schedule[0][1].speed == 0.0 and schedule[0][1].height == 0.4


def test_waypoint_encoding_pads_with_last_target_and_wraps_headings():
    positions, headings = encode_waypoints([(1.0, 2.0), (3.0, 4.0)], [0.5, 3 * math.pi])
    assert positions == ((1.0, 2.0, 0.0), (3.0, 4.0, 0.0), (3.0, 4.0, 0.0), (3.0, 4.0, 0.0))
    np.testing.assert_allclose(headings, [0.5, math.pi, math.pi, math.pi], atol=1e-12)
    with pytest.raises(ValueError):
        encode_waypoints([], 0.0)
    command = waypoint_command(SLOW_WALK, (2.0, -1.0), -0.4)
    inputs = build_inputs(standing_context(), command)
    assert int(inputs["has_specific_target"][0, 0]) == 1
    np.testing.assert_allclose(inputs["specific_target_positions"][0, -1], [2.0, -1.0, 0.0])
    np.testing.assert_allclose(inputs["specific_target_headings"][0], -0.4, rtol=1e-6)


def test_build_inputs_matches_graph_contract_with_deploy_defaults():
    inputs = build_inputs(standing_context(), idle_command())
    for name, (dtype, shape) in INPUT_SPEC.items():
        assert inputs[name].dtype == np.dtype(dtype) and inputs[name].shape == shape
    assert (
        tuple(inputs["allowed_pred_num_tokens"][0])
        == DEPLOY_TOKEN_MASK
        == (0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0)
    )
    assert int(inputs["random_seed"][0]) == 1234
    np.testing.assert_array_equal(inputs["movement_direction"][0], [0, 0, 0])
    np.testing.assert_allclose(inputs["context_mujoco_qpos"][0, :, 2], DEFAULT_HEIGHT, rtol=1e-6)
    np.testing.assert_allclose(
        inputs["context_mujoco_qpos"][0, 0, 7:], DEFAULT_ANGLES_MUJOCO, rtol=1e-6
    )


def test_checked_output_records_count_and_enforces_mask():
    outputs = FakePlanner(40)(build_inputs(standing_context(), idle_command()))
    valid, count = checked_output(outputs, DEPLOY_TOKEN_MASK)
    assert count == 40 and valid.shape == (40, 36)
    with pytest.raises(ValueError, match="mask"):
        checked_output(
            FakePlanner(48)(build_inputs(standing_context(), idle_command())), DEPLOY_TOKEN_MASK
        )


# -- replan triggers and intervals ---------------------------------------------------------
def planner_ticks(runtime, command, seconds):
    for _ in range(int(round(seconds * 50))):
        runtime.step(command)
    return [c.tick for c in runtime.calls]


def test_num_pred_frames_is_recorded_per_call():
    runtime = DeployPlannerRuntime(FakePlanner(40))
    runtime.reset()
    planner_ticks(runtime, locomotion_command(SLOW_WALK, 0.0), 1.0)
    assert {c.num_pred_frames for c in runtime.calls} == {40}
    assert {c.frames_50hz for c in runtime.calls} == {66}
    assert runtime.calls[0].reasons == ("init",)


def test_float32_replan_counter_intervals():
    assert replan_interval(WALK) == np.float32(1.0)
    assert replan_interval(CRAWLING) == np.float32(0.2)
    assert replan_interval(ELBOW_CRAWLING) == np.float32(1.0)  # the deploy checks CRAWLING only
    walk = DeployPlannerRuntime(FakePlanner())
    walk.reset()
    ticks = planner_ticks(walk, locomotion_command(WALK, 0.0), 4.2)
    assert ticks[1] == 5  # first planner tick: mode/height differ from the default last state
    assert [t for t, c in zip(ticks, walk.calls) if c.reasons == ("timer",)] == [50, 100, 150, 200]
    slow = DeployPlannerRuntime(FakePlanner(), replan_interval_scale=2.0)
    slow.reset()
    slow_ticks = planner_ticks(slow, locomotion_command(WALK, 0.0), 4.2)
    assert [t for t, c in zip(slow_ticks, slow.calls) if c.reasons == ("timer",)] == [100, 200]
    assert slow.hold_ticks > 0 and walk.hold_ticks == 0  # a 73-frame plan cannot bridge 2 s
    crawl = DeployPlannerRuntime(FakePlanner())
    crawl.reset()
    crawl_ticks = planner_ticks(crawl, locomotion_command(CRAWLING, 0.0), 1.0)
    # The counter is not reset by the mode-change replan at tick 5, so the 0.2 s timer fires at 10.
    assert [t for t, c in zip(crawl_ticks, crawl.calls) if c.reasons == ("timer",)] == [
        10,
        20,
        30,
        40,
    ]


def test_static_modes_and_zero_speed_never_replan_on_timer():
    runtime = DeployPlannerRuntime(FakePlanner())
    runtime.reset()
    planner_ticks(runtime, static_posture_command(IDEL_SQUAT, 0.4), 3.0)
    assert [c.reasons for c in runtime.calls] == [("init",), ("mode", "height")]
    assert runtime.hold_ticks > 0
    runtime.reset()
    stopped = PlannerCommand(CRAWLING, (0, 0, 0), (1, 0, 0), speed=0.0, height=0.4)
    planner_ticks(runtime, stopped, 3.0)
    assert [c.reasons for c in runtime.calls] == [("init",), ("mode", "height")]


def test_direction_speed_and_waypoint_changes_trigger_replans():
    runtime = DeployPlannerRuntime(FakePlanner())
    runtime.reset()
    planner_ticks(runtime, locomotion_command(SLOW_WALK, 0.0), 0.2)
    planner_ticks(runtime, locomotion_command(SLOW_WALK, 0.0, speed=0.6), 0.1)
    planner_ticks(runtime, locomotion_command(SLOW_WALK, 0.5, 0.0, speed=0.6), 0.1)
    planner_ticks(runtime, waypoint_command(SLOW_WALK, (2.0, 0.0), 0.0, speed=0.6), 0.1)
    reasons = [c.reasons for c in runtime.calls[2:]]
    assert reasons == [("speed",), ("direction",), ("waypoint",)]
    runtime.reset()
    planner_ticks(runtime, idle_command(), 0.2)
    planner_ticks(runtime, PlannerCommand(IDLE, (1, 0, 0), (1, 0, 0)), 0.2)  # static: ignored
    assert len(runtime.calls) == 2


def test_runtime_blends_each_plan_after_the_lookahead():
    planner = FakePlanner()
    runtime = DeployPlannerRuntime(planner)
    runtime.reset()
    command = locomotion_command(SLOW_WALK, 0.0)
    frames = [runtime.step(command) for _ in range(12)]
    call = runtime.calls[1]
    assert (call.tick, call.cursor, call.gen_frame) == (5, 5, 7)
    context = planner.inputs[1]["context_mujoco_qpos"][0]
    np.testing.assert_allclose(
        context[:, 0],
        np.asarray(frames[0][0]) + 0.01 * 30 * (7 / 50 + np.arange(4) / 30),
        atol=1e-6,
    )
    assert runtime.blends == 1 and runtime.current_frame == 7  # rebased at tick 5, advanced 7 times


def test_latency_ticks_shift_the_blend():
    runtime = DeployPlannerRuntime(FakePlanner(), latency_ticks=2)
    runtime.reset()
    command = locomotion_command(SLOW_WALK, 0.0)
    for _ in range(7):  # ticks 0-6; the plan computed at tick 5 is ready at tick 7
        runtime.step(command)
    assert runtime.blends == 0 and runtime.pending is not None
    runtime.step(command)
    assert runtime.blends == 1 and runtime.current_frame == 1
    with pytest.raises(ValueError):
        DeployPlannerRuntime(FakePlanner(), latency_ticks=5)


def test_run_schedule_switches_commands_on_time():
    runtime = DeployPlannerRuntime(FakePlanner())
    frames = run_schedule(runtime, staged_crawl_schedule(CRAWLING), 3.0)
    assert frames.shape == (150, 36)
    modes = [(c.tick, c.command.mode) for c in runtime.calls if "mode" in c.reasons]
    assert modes == [(5, IDEL_KNEEL_TWO_LEGS), (100, CRAWLING)]


def test_idle_readapt_state_machine():
    planned, original = np.zeros(12), np.full(12, 0.01)
    state, joints = idle_readapt_step("IDLE", planned, np.full(12, 0.2), original)
    assert state == "ADAPTING"
    np.testing.assert_allclose(joints, 0.02 * 0.2)
    state, joints = idle_readapt_step(state, planned, np.full(12, 0.07), original)
    assert state == "ADAPTING"
    state, _ = idle_readapt_step(state, planned, np.full(12, 0.03), original)
    assert state == "IDLE"
    state, joints = idle_readapt_step(state, planned, np.zeros(12), original)
    assert state == "RECOVERING"
    np.testing.assert_allclose(joints, 0.02 * 0.01)


# -- controllers and metrics ---------------------------------------------------------------
def test_goal_metrics_on_a_straight_stop():
    frames = np.zeros((200, 36))
    frames[:, 3] = 1
    frames[:100, 0] = np.linspace(0, 2.0, 100)
    frames[100:, 0] = 2.0
    metrics = goal_metrics(frames, (2.05, 0.0))
    assert metrics["success_010"] and metrics["stops"]
    assert metrics["final_error_m"] == pytest.approx(0.05)
    assert metrics["time_to_arrive_010_s"] == pytest.approx(
        np.flatnonzero(frames[:, 0] >= 1.95)[0] / 50
    )
    assert metrics["path_length_ratio_raw"] == pytest.approx(2.0 / 2.05)


def test_p0_bands_heading_deadband_and_latched_stop():
    stop = {"mode2_speed-1": 0.23, "mode1_speed0.4": 0.1, "mode1_speed0.2": 0.035}
    controller = DirectionSpeedStopController((5.0, 0.0), stop)
    state = lambda x, y=0.0: KinematicState(0.0, np.array([x, y]), 0.0, 0.0)  # noqa: E731
    assert controller(state(0.0)).mode == WALK
    assert controller(state(3.0, 0.1)).facing_direction == (1.0, 0.0, 0.0)  # 2.9 deg: no update
    command = controller(state(3.5, 0.2))  # 7.6 deg: update
    assert command.mode == SLOW_WALK and command.speed == 0.4
    assert controller(state(4.5, 0.0)).speed == 0.2
    assert controller(state(4.97, 0.0)).mode == IDLE
    assert controller(state(3.0, 0.0)).mode == IDLE  # latched
    with pytest.raises(ValueError, match="calibrated"):
        DirectionSpeedStopController((1, 0), stop, bands=(SpeedBand(0.0, SLOW_WALK, 0.3),))


def test_p1_waypoint_command_is_constant():
    controller = WaypointController((-3.0, 3.0))
    first = controller(KinematicState(0.0, np.zeros(2), 0.0, 0.0))
    later = controller(KinematicState(1.0, np.array([-1.0, 1.0]), 1.0, 1.0))
    assert first is later and first.has_specific_target
    assert first.target_headings[0] == pytest.approx(3 * math.pi / 4)


# -- geometry ------------------------------------------------------------------------------
def write_binary_stl(path, triangles):
    with open(path, "wb") as stream:
        stream.write(b"solid pretend-ascii-header".ljust(80, b" "))
        stream.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            stream.write(struct.pack("<12fH", 0, 0, 0, *np.ravel(tri), 0))


def test_stl_loader_reads_binary_with_solid_header_and_ascii(tmp_path):
    from gear_sonic.research.planner.g1_geometry import load_stl_vertices

    tri = [[(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 0, 0), (0, 1, 0), (0, 0, 2)]]
    write_binary_stl(tmp_path / "b.stl", tri)
    assert load_stl_vertices(tmp_path / "b.stl").shape == (4, 3)
    (tmp_path / "a.stl").write_text(
        "solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 3\n"
        "endloop\nendfacet\nendsolid x\n"
    )
    assert load_stl_vertices(tmp_path / "a.stl")[:, 2].max() == 3


def test_urdf_fk_standing_extents_and_mujoco_parity():
    from gear_sonic.research.planner import g1_geometry

    try:
        geometry = g1_geometry.G1Geometry()
    except FileNotFoundError:
        pytest.skip("G1 robot_description assets are not on this host")
    poses = geometry.link_poses(standing_context()[:1])
    visual = geometry.group_z_extents(poses, "visual")
    assert 0.0 < visual["left_foot"][0][0] < 0.06  # soles just above z 0 at the default height
    assert 1.25 < visual["head"][1][0] < 1.40
    assert "head" not in geometry.group_z_extents(poses, "collision")
    assert geometry.joint_limit_excess(standing_context()).max() == 0
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(geometry.root_dir / "mjcf/g1_29dof_rev_1_0.xml"))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(3)
    qpos = np.zeros((4, 36))
    qpos[:, :3] = rng.normal(size=(4, 3))
    quat = rng.normal(size=(4, 4))
    qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    qpos[:, 7:] = rng.uniform(geometry.joint_limits[:, 0], geometry.joint_limits[:, 1], (4, 29))
    poses = geometry.link_poses(qpos)
    for i in range(4):
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        for body in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            np.testing.assert_allclose(poses[name][1][i], data.xpos[body], atol=1e-5)


def test_cylinder_extents_follow_axis_tilt():
    from gear_sonic.research.planner.g1_geometry import G1Geometry, Shape

    geometry = G1Geometry.__new__(G1Geometry)
    geometry.shapes = {
        "collision": [
            Shape("pelvis", "cylinder", np.eye(3), np.zeros(3), radius=0.1, half_length=0.5)
        ]
    }
    tilt = math.pi / 2
    rotation = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]],
        ],
        float,
    )
    low, high = geometry.link_z_extents(
        {"pelvis": (rotation, np.array([[0, 0, 1.0], [0, 0, 1.0]]))}
    )["pelvis"]
    np.testing.assert_allclose(low, [0.5, 0.9], atol=1e-12)
    np.testing.assert_allclose(high, [1.5, 1.1], atol=1e-12)
