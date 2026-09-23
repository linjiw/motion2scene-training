"""rescore_xy.py on synthetic traces (CPU, numpy only; the legacy cross-check needs SONIC)."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load("nav_rescore_xy", ROOT / "scripts" / "navigation_distill" / "rescore_xy.py")

TASK = dict(
    start_xyz=[0.0, 0.0, 0.75],
    goal_xyz=[1.0, 0.0, 0.80],
    goal_tolerance_m=0.25,
    terminal_speed_mps=0.1,
    hold_ticks=50,
    deadline_ticks=200,
)


def trace(n=120, arrive=40, z=0.8, dx=0.0):
    """Walk 0 -> 1 m in `arrive` rows, then stand at x = 1 + dx, pelvis at height z."""
    x = np.concatenate([np.linspace(0.0, 1.0 + dx, arrive), np.full(n - arrive, 1.0 + dx)])
    root = np.stack([x, np.zeros(n), np.full(n, z)], -1).astype(np.float32)
    speed = np.concatenate([np.full(arrive, 1.0), np.full(n - arrive, 0.02)])
    return root, speed, np.zeros(n)


def test_xy_metric_ignores_pelvis_height_that_fails_the_3d_goal():
    root, speed, force = trace(z=0.52, dx=0.1)  # 0.1 m XY, 0.28 m low: 3-D error 0.297 m
    legacy = R.navigation_score(TASK, root, speed, force, fell=False, metric="xyz")
    assert not legacy["navigation_success"] and legacy["max_hold_ticks"] == 0
    ok, row, best = R.first_hold(TASK, root, speed, force, metric="xy")
    assert ok and row == 39 + 50 and best == 80
    assert R.navigation_score(TASK, root, speed, force, fell=False, metric="xy")[
        "navigation_success"
    ]
    # With the 3-D metric first_hold agrees with the legacy score.
    assert R.first_hold(TASK, root, speed, force, metric="xyz")[0] is False


def test_hold_needs_consecutive_slow_rows_before_contact_or_fall():
    root, speed, force = trace(n=121)
    speed[70] = 0.2  # breaks the run that started at row 40 (arrival row 39 is still moving)
    ok, row, best = R.first_hold(TASK, root, speed, force)
    assert ok and row == 71 + 49 and best == 50
    short = trace(n=120)
    short[1][70] = 0.2
    assert R.first_hold(TASK, *short) == (False, None, 49)  # would complete on row 120
    # Contact after the first hold: an XY-stopping evaluator had already succeeded, but the
    # legacy whole-trace semantics fail the episode.
    root, speed, force = trace()
    force[-1] = 50.0
    assert R.first_hold(TASK, root, speed, force) == (True, 89, 80)
    assert not R.navigation_score(TASK, root, speed, force, fell=False, metric="xy")[
        "navigation_success"
    ]
    force[89] = 2.0  # contact on the completion row itself fails
    assert R.first_hold(TASK, root, speed, force)[0] is False
    root, speed, force = trace()
    root[60, 2] = 0.2  # a fall before completion
    assert R.first_hold(TASK, root, speed, force)[0] is False


def test_suffix_hold_restarts_at_the_takeover_row():
    root, speed, force = trace()
    assert R.first_hold(TASK, root, speed, force, start=60) == (True, 60 + 49, 60)
    assert R.first_hold(TASK, root, speed, force, start=80)[0] is False  # 40 rows left
    force[10] = 5.0  # prefix contact still disqualifies the suffix
    assert R.first_hold(TASK, root, speed, force, start=60)[0] is False
    assert R.first_hold(TASK, root, speed, force, start=500) == (False, None, 0)


def test_xy_outcome_classes():
    base = dict(xy_success=False, collision_free=True, fell=False, first_xy_reach_row=None)
    assert R.xy_outcome(dict(base, xy_success=True, collision_free=False)) == "success"
    assert R.xy_outcome(dict(base, collision_free=False, first_xy_reach_row=3)) == (
        "contact_or_fall"
    )
    assert R.xy_outcome(dict(base, fell=True)) == "contact_or_fall"
    assert R.xy_outcome(dict(base, first_xy_reach_row=0)) == "reached_not_held"
    assert R.xy_outcome(base) == "never_reached"


def test_planar_speed_and_root_metrics():
    root, speed, force = trace()
    fd = R.planar_speed_fd(root, speed)
    assert fd[0] == speed[0]
    assert fd[1] == pytest.approx((1.0 / 39) / 0.02, rel=1e-5)
    assert fd[60] == 0.0
    m = R.root_metrics(TASK, root, speed, force)
    assert m["path_xy_m"] == pytest.approx(1.0, abs=1e-6)
    assert m["path_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert m["first_xy_reach_row"] == 30  # x = 30/39 = 0.769 m, the first row within 0.25 m
    assert m["first_xy_reach_s"] == pytest.approx(0.62)
    assert m["final_xy_err_m"] == pytest.approx(0.0, abs=1e-6)
    assert m["final_3d_err_m"] == pytest.approx(0.0, abs=1e-6)
    assert m["max_root_height_dev_m"] == pytest.approx(0.05, abs=1e-6)
    assert m["first_contact_row"] is None
    back = np.concatenate([root, root[::-1]])  # walk to the goal and back
    m = R.root_metrics(TASK, back, np.ones(len(back)), np.zeros(len(back)))
    assert m["path_ratio"] == pytest.approx(2.0, abs=1e-6)
    assert m["final_xy_err_m"] == pytest.approx(1.0, abs=1e-6)
    assert m["min_xy_err_m"] == pytest.approx(0.0, abs=1e-6)


def test_legacy_port_matches_the_vendored_scorer_on_random_traces():
    sys.path.insert(0, str(ROOT / "vendor" / "sonic"))
    try:
        direct = pytest.importorskip("gear_sonic.research.scene_distillation.direct_context")
    finally:
        sys.path.remove(str(ROOT / "vendor" / "sonic"))
    rng = np.random.default_rng(92601)
    for _ in range(200):
        n = int(rng.integers(1, 160))
        root = (TASK["goal_xyz"] + rng.normal(0, 0.15, (n, 3))).astype(np.float32)
        speed = np.abs(rng.normal(0.05, 0.05, n))
        force = np.where(rng.random(n) < 0.01, 5.0, 0.0)
        fell = bool(root[-1, 2] < 0.25)
        expected = direct.score_navigation_task(TASK, list(root), list(speed), list(force),
                                                fell=fell)
        got = R.navigation_score(TASK, root, speed, force, fell=fell, metric="xyz")
        assert not R.legacy_mismatches(expected, got)
        assert got["final_goal_distance_m"] == expected["final_goal_distance_m"]


def write_episode(stage, name, task_path, root, speed, force, config_extra=None):
    episode = stage / name
    (episode / "task").mkdir(parents=True)
    config = dict(task_path=str(task_path), max_steps=TASK["deadline_ticks"], teacher_mode=True)
    config.update(config_extra or {})
    (episode / "config.json").write_text(json.dumps(config))
    (episode / "command.json").write_text(json.dumps([
        "python", "++seed=91260",
        "++callbacks.im_eval._target_=x.stopping_teacher.StoppingTeacherCallback",
    ]))
    score = R.navigation_score(TASK, root, speed, force, fell=bool(root[-1, 2] < 0.25))
    score.update(stop_reason="goal_hold" if score["navigation_success"] else "deadline",
                 task_sha256=R.sha256(task_path))
    (episode / "task" / "task-result.json").write_text(json.dumps(score))
    np.savez_compressed(episode / "task" / "trace.npz", root_xyz=root, speed=speed,
                        undesired_force=force)
    return episode


def test_end_to_end_packet_with_symlinked_stage(tmp_path, monkeypatch):
    packet = tmp_path / "packet"
    task_path = packet / "tasks" / "t.json"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(dict(TASK, task_id="00001-stop-clear", motion_id="00001")))
    stage = packet / "eval" / "arm-91260"
    write_episode(stage, "00001-stop-clear", task_path, *trace())
    low = trace(z=0.52, dx=0.1)
    write_episode(stage, "00001-stop-low", task_path, *low)
    lost = write_episode(stage, "00001-stop-lost", task_path, *trace())
    (lost / "task" / "trace.npz").unlink()
    (packet / "eval" / "alias").symlink_to(stage, target_is_directory=True)
    episodes = R.discover(packet)
    assert len(episodes) == 3
    assert sorted(len(a) for a in episodes.values()) == [2, 2, 2]
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["rescore_xy.py", "--packet", str(packet), "--output",
                                      str(out), "--feasible-panel", "eval/arm-91260"])
    R.main()
    rows = list(csv.DictReader((out / "rescore.csv").open()))
    by = {r["episode"]: r for r in rows}
    assert by.pop("00001-stop-lost")["trace"] == "missing"
    rows = list(by.values())
    assert len(rows) == 2 and all(r["legacy_repro_match"] == "True" for r in rows)
    assert by["00001-stop-clear"]["legacy_3d_success"] == "True"
    assert by["00001-stop-low"]["legacy_3d_success"] == "False"
    assert by["00001-stop-low"]["xy_success"] == "True"
    assert by["00001-stop-low"]["task_sha256_ok"] == "True"
    # The real directory names the panel even though the symlink sorts first.
    assert {(r["panel"], r["aliases"]) for r in rows} == {("eval/arm-91260", "eval/alias")}
    summary = json.loads((out / "rescore_summary.json").read_text())
    (panel,) = summary["panels"]
    assert panel["legacy_3d"] == 1 and panel["xy"] == 2 and panel["gained"] == 1
    assert panel["kind"] == "evaluation" and panel["traces_missing"] == 1
    assert panel["xy_outcomes"] == dict(
        success=2, reached_not_held=0, never_reached=0, contact_or_fall=0
    )
    assert "Episodes whose outcome changes (1)" in (out / "rescore_summary.md").read_text()
    with pytest.raises(FileExistsError):
        R.main()
