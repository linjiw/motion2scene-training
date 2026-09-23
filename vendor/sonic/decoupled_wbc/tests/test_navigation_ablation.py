"""Phase 0.4 goal/map-use ablations: actor-only views, unchanged defaults, readout rule."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import gear_sonic
from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.direct_context import task_context
from gear_sonic.research.scene_distillation.direct_scene_runtime import DirectSceneTaskCallback
from gear_sonic.research.scene_distillation.navigation_ablation import (
    ObservationAblation,
    ablation_record,
    final_position_row,
    goal_use_decision,
    observation_task,
    reference_endpoint,
    rotate_goal_about_start,
)
from gear_sonic.research.scene_distillation.navigation_motor_runtime import NavigationMotorCallback

REPO = Path(__file__).resolve().parents[4]
SCRIPTS = REPO / "scripts" / "navigation_distill"


def box(y):
    return dict(
        shape="box",
        center_xyz=[2.0, y, 0.75],
        quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        full_dimensions_xyz=[1.0, 0.2, 1.5],
    )


def corridor_task():
    return dict(
        task_id="00001-stop-corridor",
        start_xyz=[1.0, 2.0, 0.8],
        goal_xyz=[3.0, 2.0, 0.79],
        obstacles=[box(2.8), box(1.2)],
        goal_tolerance_m=0.25,
        terminal_speed_mps=0.1,
        hold_ticks=50,
    )


def fake_env(root=(1.0, 2.0, 0.8), quat=(1.0, 0.0, 0.0, 0.0)):
    return SimpleNamespace(
        device="cpu",
        motion_command=SimpleNamespace(
            robot_anchor_pos_w=torch.tensor([root]),
            robot_anchor_quat_w=torch.tensor([quat]),
        ),
        env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(1, 3))),
    )


class CaptureStudent:
    def __init__(self, use_localization=False):
        self.use_localization = use_localization
        self.actors = []

    def navigation_step(self, actor):
        self.actors.append(actor)
        return {"actions": torch.zeros(1, 29)}


def run_actor(config, task, use_localization=False, begin=True, quat=(1.0, 0.0, 0.0, 0.0)):
    callback = object.__new__(NavigationMotorCallback)
    if config is not None:
        callback.config = config
    if begin:
        callback._begin_task(None, None, task)
    student = CaptureStudent(use_localization)
    callback._student_action(
        student,
        fake_env(quat=quat),
        SimpleNamespace(running_mean_std=None),
        {"actor_obs": torch.zeros(1, 930)},
        task,
        None,
    )
    return callback, student.actors[0]


def test_goal_rotation_is_a_world_yaw_about_the_start():
    start, goal = [1.0, 2.0, 0.8], [3.0, 2.0, 0.79]
    np.testing.assert_allclose(rotate_goal_about_start(start, goal, 90), [1, 4, 0.79], atol=1e-12)
    np.testing.assert_allclose(rotate_goal_about_start(start, goal, -90), [1, 0, 0.79], atol=1e-12)
    np.testing.assert_allclose(rotate_goal_about_start(start, goal, 180), [-1, 2, 0.79], atol=1e-12)
    for degrees in (0, 360, -720):
        np.testing.assert_allclose(rotate_goal_about_start(start, goal, degrees), goal, atol=1e-12)
    for degrees in np.linspace(-270, 270, 13):
        rotated = rotate_goal_about_start(start, goal, degrees)
        assert math.isclose(math.dist(rotated[:2], start[:2]), 2.0, rel_tol=1e-12)
        assert rotated[2] == goal[2]
    for bad in ([start[:2], goal, 90], [start, [np.nan, 0, 0], 90], [start, goal, float("inf")]):
        with pytest.raises(ValueError):
            rotate_goal_about_start(*bad)


def test_ablation_flags_default_off_and_reject_ambiguous_values():
    assert not ObservationAblation.from_config({}).active
    assert not NavigationMotorCallback.ablation.active
    assert ObservationAblation.from_config(dict(goal_rotation_deg=90)).goal_rotation_deg == 90.0
    assert ObservationAblation.from_config(dict(goal_rotation_deg=0)).active
    assert ObservationAblation.from_config(dict(zero_obstacles=True)).active
    for bad in (
        dict(goal_rotation_deg=True),
        dict(goal_rotation_deg="90"),
        dict(goal_rotation_deg=float("nan")),
        dict(zero_obstacles="yes"),
        dict(zero_obstacles=1),
    ):
        with pytest.raises(ValueError):
            ObservationAblation.from_config(bad)
    task = corridor_task()
    assert observation_task(task, ObservationAblation()) is task


@pytest.mark.parametrize("use_localization", [False, True])
@pytest.mark.parametrize("config", [{}, dict(actor_profile="nav_goal_map_v1")])
def test_flags_off_actor_observation_equals_original_assembly(config, use_localization):
    task = corridor_task()
    pristine = copy.deepcopy(task)
    # The pre-ablation assembly: task_context on the task as given, at the measured pose.
    command = fake_env().motion_command
    original = task_context(
        task, command.robot_anchor_pos_w[0].numpy(), command.robot_anchor_quat_w[0].numpy()
    )
    callback, actor = run_actor(config, task, use_localization)
    # Class default, as when a caller never runs _begin_task (see test_navigation_motor.py).
    _, bare = run_actor(None, task, begin=False)
    for key, value in original.items():
        assert torch.equal(getattr(actor, key), torch.as_tensor(value)[None])
        assert torch.equal(getattr(bare, key), torch.as_tensor(value)[None])
    assert task == pristine
    if use_localization:
        assert np.array_equal(callback.localization.goal, task["goal_xyz"])
    assert callback._task_result_fields(task, [np.zeros(3)]) == {}


def test_goal_rotation_reaches_the_actor_but_not_the_task():
    task = corridor_task()
    pristine = copy.deepcopy(task)
    callback, actor = run_actor(dict(goal_rotation_deg=90.0), task, use_localization=True)
    _, baseline = run_actor({}, task)
    nav, base = actor.navigation_context[0].numpy(), baseline.navigation_context[0].numpy()
    # Robot at the start with identity yaw: body-frame goal is the rotated world offset.
    np.testing.assert_allclose(nav[3:6], [0.0, 2.0, -0.01], atol=1e-6)
    np.testing.assert_allclose(base[3:6], [2.0, 0.0, -0.01], atol=1e-6)
    np.testing.assert_array_equal(nav[:3], base[:3])
    np.testing.assert_array_equal(nav[6:], base[6:])
    assert torch.equal(actor.obstacles_body, baseline.obstacles_body)
    assert torch.equal(actor.obstacle_mask, baseline.obstacle_mask)
    np.testing.assert_allclose(callback.localization.goal, [1.0, 4.0, 0.79], atol=1e-12)
    assert task == pristine
    # A yawed robot sees the rotated goal in its own frame.
    yaw90 = (2**-0.5, 0.0, 0.0, 2**-0.5)
    _, turned = run_actor(dict(goal_rotation_deg=90.0), task, quat=yaw90)
    np.testing.assert_allclose(turned.navigation_context[0, 3:6], [2.0, 0.0, -0.01], atol=1e-6)


def test_zero_obstacles_uses_absent_primitive_padding():
    task = corridor_task()
    _, actor = run_actor(dict(zero_obstacles=True), task)
    _, baseline = run_actor({}, task)
    clear = task_context(
        dict(task, obstacles=[]), np.array([1.0, 2.0, 0.8]), np.array([1.0, 0.0, 0.0, 0.0])
    )
    assert baseline.obstacle_mask.sum() == 2
    assert not actor.obstacle_mask.any()
    assert torch.equal(actor.obstacles_body, torch.zeros(1, 5, 15))
    assert torch.equal(actor.obstacle_mask, torch.as_tensor(clear["obstacle_mask"])[None])
    assert torch.equal(actor.obstacles_body, torch.as_tensor(clear["obstacles_body"])[None])
    assert torch.equal(actor.navigation_context, baseline.navigation_context)
    assert len(task["obstacles"]) == 2


def reference_file(tmp_path, endpoint):
    path = tmp_path / "reference-with-hold.npz"
    qpos = np.zeros((4, 36))
    qpos[:, :3] = [[1.0, 2.0, 0.8], [2.0, 2.0, 0.8], [3.0, 2.0, 0.8], endpoint]
    np.savez(path, qpos=qpos)
    return dict(path=str(path), sha256=sha(path))


def test_task_result_record_scores_the_original_goal(tmp_path):
    assert DirectSceneTaskCallback._task_result_fields(object(), {}, []) == {}
    task = dict(corridor_task(), reference=reference_file(tmp_path, [3.0, 2.0, 0.79]))
    assert reference_endpoint(task) == [3.0, 2.0, 0.79]
    callback = object.__new__(NavigationMotorCallback)
    callback.config = dict(goal_rotation_deg=90.0)
    callback._begin_task(None, None, task)
    roots = [np.array([1.0, 2.0, 0.8]), np.array([2.9, 2.0, 0.79])]
    record = callback._task_result_fields(task, roots)["navigation_ablation"]
    json.dumps(record, allow_nan=False)
    assert record["scoring_goal"] == "original"
    assert record["goal_rotation_deg"] == 90.0 and record["zero_obstacles"] is False
    assert record["original_goal_xyz"] == task["goal_xyz"]
    np.testing.assert_allclose(record["rotated_goal_xyz"], [1.0, 4.0, 0.79], atol=1e-12)
    assert record["final_xy_distance_to_original_goal_m"] == pytest.approx(0.1)
    assert record["final_xy_distance_to_reference_endpoint_m"] == pytest.approx(0.1)
    assert record["final_xy_distance_to_rotated_goal_m"] == pytest.approx(math.hypot(1.9, 2.0))
    assert record["observed_obstacles"] == 2
    zero = ablation_record(task, ObservationAblation(zero_obstacles=True), roots[-1], [3, 2, 0.79])
    assert zero["rotated_goal_xyz"] is None and zero["observed_obstacles"] == 0
    changed = dict(task, reference=dict(task["reference"], sha256="0" * 64))
    with pytest.raises(ValueError, match="reference changed"):
        reference_endpoint(changed)


def test_row_recording_callbacks_reject_ablation():
    from gear_sonic.research.scene_distillation.navigation_queries import NavigationQueryCallback
    from gear_sonic.research.scene_distillation.navigation_recovery import (
        MotorRecoveryCollectionCallback,
    )
    from gear_sonic.research.scene_distillation.navigation_reentry import ReentryProbeCallback
    from gear_sonic.research.scene_distillation.navigation_takeover import (
        NavigationTakeoverCallback,
        OriginalTeacherContinuationCallback,
    )

    for cls in (
        NavigationQueryCallback,
        MotorRecoveryCollectionCallback,
        NavigationTakeoverCallback,
        OriginalTeacherContinuationCallback,
        ReentryProbeCallback,
    ):
        callback = object.__new__(cls)
        callback.config = dict(zero_obstacles=True)
        with pytest.raises(ValueError, match="evaluation-only"):
            callback._begin_task(None, None, corridor_task())


def stage_config(tmp_path, *extra):
    packet = tmp_path / "packet"
    packet.mkdir(exist_ok=True)
    (packet / "ids.json").write_text(json.dumps(dict(teacher_checkpoint="/t.pt", teacher_sha256="t")))
    task = tmp_path / "task.json"
    task.write_text(json.dumps(dict(deadline_ticks=350)))
    student = tmp_path / "nav.pt"
    student.write_bytes(b"checkpoint")
    config = tmp_path / f"config-{len(list(tmp_path.glob('config-*')))}.json"
    process = subprocess.run(
        [sys.executable, str(SCRIPTS / "stage_config.py"), "--task", str(task),
         "--output", str(tmp_path / "out"), "--config", str(config), "--student", str(student),
         *extra],
        env=dict(NAV_PACKET=str(packet), PYTHONPATH=str(Path(gear_sonic.__file__).parents[1])),
        capture_output=True,
        text=True,
    )
    return process, (json.loads(config.read_text()) if config.exists() else None)


@pytest.mark.skipif(not SCRIPTS.exists(), reason="repository scripts not present")
def test_stage_config_writes_flags_only_when_given(tmp_path):
    process, default = stage_config(tmp_path, "--mode", "nav")
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip().endswith("NavigationMotorCallback")
    assert list(default) == [
        "teacher_checkpoint", "teacher_sha256", "task_path", "output", "max_steps",
        "teacher_mode", "command_profile", "student_checkpoint", "student_sha256", "actor_profile",
    ]
    _, rotated = stage_config(tmp_path, "--mode", "nav", "--goal-rotation-deg", "90")
    assert rotated == dict(default, goal_rotation_deg=90.0)
    _, zero = stage_config(tmp_path, "--mode", "nav", "--zero-obstacles")
    assert zero == dict(default, zero_obstacles=True)
    for extra in (["--mode", "teacher", "--zero-obstacles"], ["--mode", "nav", "--goal-rotation-deg", "nan"],
                  ["--mode", "recovery", "--goal-rotation-deg", "90"]):
        process, written = stage_config(tmp_path, *extra)
        assert process.returncode == 2 and written is None


def rows(finals, tasks):
    return {
        t: final_position_row(task, final, task["goal_xyz"], 90.0) for (t, task), final in
        zip(tasks.items(), finals)
    }


def synthetic_tasks(n):
    tasks = {}
    for i in range(n):
        angle = 2 * math.pi * i / n
        start = [float(i), 0.0, 0.8]
        goal = [start[0] + 1.5 * math.cos(angle), 1.5 * math.sin(angle), 0.8]
        tasks[f"{i:05d}-stop-clear"] = dict(start_xyz=start, goal_xyz=goal)
    return tasks


def test_goal_use_rule_blind_following_and_boundary():
    tasks = synthetic_tasks(10)
    blind = rows([t["goal_xyz"] for t in tasks.values()], tasks)
    decision = goal_use_decision(blind)
    assert decision["adapter_ignores_goal"] and decision["near_reference_endpoint"] == 10
    assert decision["discriminative_tasks"] == 10
    assert decision["median_progress_along_original"] == pytest.approx(1.0)
    assert decision["median_displacement_heading_vs_original_deg"] == pytest.approx(0.0, abs=1e-9)
    follow = rows(
        [rotate_goal_about_start(t["start_xyz"], t["goal_xyz"], 90) for t in tasks.values()], tasks
    )
    assert goal_use_decision(blind)["heading_toward_rotated_goal"] == 0
    decision = goal_use_decision(follow)
    assert not decision["adapter_ignores_goal"] and decision["near_rotated_goal"] == 10
    assert decision["median_progress_along_rotated"] == pytest.approx(1.0)
    assert decision["median_displacement_heading_vs_original_deg"] == pytest.approx(90.0)
    assert decision["heading_toward_rotated_goal"] == 10
    paired = goal_use_decision(follow, blind)["sensitivity"]
    assert paired["median_heading_turn_vs_baseline_deg"] == pytest.approx(90.0)
    assert paired["heading_turned_with_goal"] == paired["heading_turn_tasks"] == 10
    assert paired["paired_rule_proposed"]["adapter_ignores_goal"] is False
    for near, expected in ((7, True), (6, False)):
        finals = [
            t["goal_xyz"] if i < near else t["start_xyz"] for i, t in enumerate(tasks.values())
        ]
        assert goal_use_decision(rows(finals, tasks))["adapter_ignores_goal"] is expected


def test_goal_use_sensitivity_flags_an_unattainable_rule():
    tasks = synthetic_tasks(10)
    stuck = rows([t["start_xyz"] for t in tasks.values()], tasks)
    baseline = rows(
        [t["goal_xyz"] if i < 4 else t["start_xyz"] for i, t in enumerate(tasks.values())], tasks
    )
    decision = goal_use_decision(stuck, baseline)
    assert decision["adapter_ignores_goal"] is False
    sensitivity = decision["sensitivity"]
    assert sensitivity["rule_attainable"] is False
    assert sensitivity["baseline_near_reference_endpoint"] == 4
    assert sensitivity["final_xy_within_radius_of_baseline"] == 6
    assert sensitivity["paired_rule_proposed"]["adapter_ignores_goal"] is False
    # A goal-blind adapter replays its baseline: the registered rule misses it at 4/10,
    # the proposed paired rule does not.
    blind = goal_use_decision(baseline, baseline)
    assert blind["adapter_ignores_goal"] is False
    assert blind["sensitivity"]["median_final_xy_shift_vs_baseline_m"] == 0
    assert blind["sensitivity"]["paired_rule_proposed"]["adapter_ignores_goal"] is True
    with pytest.raises(ValueError, match="coincides"):
        final_position_row(dict(start_xyz=[0, 0, 0], goal_xyz=[0, 0, 1]), [0, 0, 0], [0, 0, 1], 90)


def write_episode(stage, task_id, final, config_extra, tmp_path, seed=92601, success=False):
    folder = tmp_path / "tasks" / task_id
    task_path = folder / "task.json"
    if not task_path.exists():  # panels share one task file, as they share the packet
        folder.mkdir(parents=True)
        task = dict(corridor_task(), task_id=task_id, reference=reference_file(folder, [3, 2, 0.79]))
        if task_id.endswith("clear"):
            task["obstacles"] = []
        task_path.write_text(json.dumps(task))
    task = json.loads(task_path.read_text())
    out = stage / task_id
    (out / "task").mkdir(parents=True)
    config = dict(task_path=str(task_path), student_sha256="nav", **config_extra)
    (out / "config.json").write_text(json.dumps(config))
    (out / "command.json").write_text(json.dumps(["python", "x.py", f"++seed={seed}"]))
    roots = np.array([task["start_xyz"], final], dtype=np.float64)
    np.savez(out / "task" / "trace.npz", root_xyz=roots)
    result = dict(
        navigation_success=success,
        stop_reason="goal_hold" if success else "deadline",
        collision_free=True,
        fell=False,
        control_steps=2,
        task_sha256=sha(task_path),
    )
    ablation = ObservationAblation.from_config(config_extra)
    if ablation.active:
        result["navigation_ablation"] = ablation_record(task, ablation, roots[-1], [3.0, 2.0, 0.79])
    (out / "task" / "task-result.json").write_text(json.dumps(result))


@pytest.mark.skipif(not SCRIPTS.exists(), reason="repository scripts not present")
def test_analysis_script_reads_panels_and_applies_the_rule(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "analyze_goal_ablation", SCRIPTS / "analyze_goal_ablation.py"
    )
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    rotated, baseline = tmp_path / "rot", tmp_path / "base"
    ids = ["00001-stop-clear", "00001-stop-corridor", "00002-stop-clear"]
    for i, task_id in enumerate(ids):
        write_episode(baseline, task_id, [3.0, 2.1, 0.79], {}, tmp_path, success=True)
        final = [3.0, 2.0, 0.79] if i < 2 else [1.1, 3.9, 0.79]
        write_episode(rotated, task_id, final, dict(goal_rotation_deg=90.0), tmp_path)
    (rotated / "00003-stop-clear").mkdir()
    (rotated / "00003-stop-clear" / "config.json").write_text("{}")
    report = analysis.analyze(rotated, baseline)
    json.dumps(report, allow_nan=False)
    goal = report["goal_use"]
    assert report["without_result"] == ["00003-stop-clear"]
    assert report["completed_tasks"] == 3 and report["seed"] == 92601
    assert goal["applicable"] and goal["near_reference_endpoint"] == 2
    assert goal["adapter_ignores_goal"] is False  # 2/3 < 70%
    assert goal["near_rotated_goal"] == 1
    assert goal["sensitivity"]["rule_attainable"] is True
    assert report["baseline"]["successes"] == 3
    assert report["warnings"] == []
    # The baseline alone: position readout, rule marked not applicable.
    unrotated = analysis.analyze(baseline)
    assert unrotated["goal_use"]["applicable"] is False
    assert unrotated["goal_use"]["near_reference_endpoint"] == 3
    # A task-result record that disagrees with its trace is refused.
    result_path = rotated / ids[0] / "task" / "task-result.json"
    result = json.loads(result_path.read_text())
    result["navigation_ablation"]["final_root_xyz"] = [0.0, 0.0, 0.0]
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="does not match trace"):
        analysis.analyze(rotated)


@pytest.mark.skipif(not SCRIPTS.exists(), reason="repository scripts not present")
def test_zero_obstacle_readout_pairs_with_the_baseline(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "analyze_goal_ablation", SCRIPTS / "analyze_goal_ablation.py"
    )
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    zero, baseline = tmp_path / "zero", tmp_path / "base"
    write_episode(baseline, "00001-stop-clear", [3.0, 2.0, 0.79], {}, tmp_path, success=True)
    write_episode(baseline, "00001-stop-corridor", [3.0, 2.0, 0.79], {}, tmp_path, success=True)
    write_episode(zero, "00001-stop-clear", [3.0, 2.0, 0.79], dict(zero_obstacles=True), tmp_path,
                  success=True)
    write_episode(zero, "00001-stop-corridor", [2.0, 2.0, 0.79], dict(zero_obstacles=True),
                  tmp_path)
    report = analysis.analyze(zero, baseline)
    summary = report["zero_obstacles"]
    assert report["goal_use"]["applicable"] is False
    assert summary["corridor"] == dict(tasks=1, successes=0, contacts=0, falls=0)
    assert summary["baseline"]["changed_outcomes"] == ["00001-stop-corridor"]
    assert summary["baseline"]["clear_max_final_xy_shift_m"] == 0.0
    assert summary["baseline"]["corridor_median_final_xy_shift_m"] == pytest.approx(1.0)
