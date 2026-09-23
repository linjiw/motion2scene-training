# SONIC kinematic planner: deploy-faithful port, goal study and posture table (September 23, 2026)

Roadmap Phase 0.6 and the CPU part of 0.7. Everything here is **kinematic**: the planner's 50 Hz reference is treated as the robot state (perfect tracking). Nothing ran in physics and no tracker ran. Tags follow the [roadmap](../../ROADMAP_20260923.md): **[M]** measured on this host (receipts in `workspace/phase0/planner/`, gitignored), **[A]** assumption.

Receipts (code commit `2c44fa4`):
- `kinematic_goal_study.{json,md}`, `posture_table.{json,md}`, `timing.json`, `p1_checks.json`;
- `clips/` and `goal_clips/`, each with a `manifest.json` and a `loader_validation.json`;
- `probes/`.

## What was built

Package `vendor/sonic/gear_sonic/research/planner/` (no ONNX Runtime, simulator or GPU import at module load):

- **`planner_runtime.py`** — a closed-loop kinematic runner ported line by line from `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref`:
  - standing context at the default height 0.78874 m;
  - context taken at cursor+2 with 30 Hz spacing, clamped at the end;
  - `floor(n/30·50)` resampling to 50 Hz, holding the endpoint;
  - linear 8-frame cross-fade after rebasing at the cursor;
  - the 10 Hz planner thread with the deploy replan triggers and float32 intervals (0.1 s run, 0.2 s CRAWLING only, 1.0 s otherwise);
  - the timer fires only in non-static modes with speed ≠ 0;
  - idle re-adaptation, which is a kinematic no-op;
  - TensorRT token mask `[0,0,0,1,1,1,0,0,0,0,0]` and seed 1234;
  - gamepad per-mode speed/height defaults and the staged crawl entry (kneel → 2 s → crawl → 2 s → elbow crawl).

  Declared extensions:
  - E1: `has_specific_target` is exposed; the deploy never sets it.
  - E2: modes {0, 1, 2, 4, 8, 14, 18, 22}, plus 5 for the staged entry.
  - E3: the two deploy threads are serialized, with `latency_ticks` (default 0).
- **`ort_session.py`** — a CPU ONNX Runtime session checked against the graph contract. Run it with `/home/robotixx/GR00T-WholeBodyControl/.venv_sim/bin/python` (ORT 1.23.2).
- **`goal_controllers.py`** — P0 (direction/speed/stop in modes 1/2, IDLE to stop, calibrated run-out) and P1 (goal as waypoint). The exploratory P1c places the waypoint 1.0 m ahead and re-places it on the replan clock. Also root-XY goal metrics.
- **`g1_geometry.py`** — numpy FK from the Isaac URDF, with per-body-group floor/top extents for collision and visual shapes. It matches MuJoCo FK of the motion-lib MJCF to 1.5e-6 m, and the SONIC CPU motion loader's FK to ≤1.4e-6 m on all 145 exported clips.
- **Scripts:**
  - `scripts/research/planner_kinematic_study.py` runs the timing, goal study, postures, clips and reports.
  - `scripts/research/planner_validate_clips.py` loads the clips with `MotionLibRobot` on CPU and asserts that no Isaac module is imported.
- **Tests** — 27 CPU tests in `decoupled_wbc/tests/test_planner_runtime.py`. They cover context sampling, resampling, cross-fade, replan triggers and float32 intervals, `num_pred_frames` recording, mode/height handling, waypoint encoding, idle re-adaptation, the controllers, metrics, STL loading and FK parity.

## Results

**Timing [M].** One call takes 28.2 ms median and 29.3 ms p90 on CPU with 4 threads (24,868 calls; ORT 1.23.2, batch 1). Session initialization takes 1.2 s. `num_pred_frames` is 36, 40 or 44 in closed loop. From standing it is 44, except CRAWLING at 36. Padded output rows are non-zero, so the valid count must be applied.

**Goal study [M]** — 200 goals (seed 70601), 1–8 m, bearing ±180°, standing start, 30 s horizon:

| Condition | ≤0.10 m | Final error median / max (m) | Arrive ≤0.10 m (median s) | Peak 0.5 s speed median / max (m/s) | Hold ticks (median of 1500) |
|---|---|---|---|---|---|
| P0 ×1 (deploy interval) | 200/200 | 0.029 / 0.070 | 9.7 | 1.05 / 1.23 | 924 (standing after IDLE) |
| P0 ×2 | 200/200 | 0.031 / 0.064 | 13.3 | 1.07 / 1.10 | 829 |
| P1 ×1 | 200/200 | 0.003 / 0.003 | 4.9 | **2.17 / 4.11** | 0 |
| P1 ×2 | 200/200 | 0.003 / 0.004 | 10.6 | 2.13 / 4.23 | 522 (frozen mid-gait) |
| P1c ×1 (exploratory) | 200/200 | 0.003 / 0.003 | 10.9 | 0.57 / 0.63 | 0 |
| P1c ×2 (exploratory) | 178/200 | 0.007 / 0.223 | 17.1 | 0.58 / 0.63 | 463 |

- **Gate (P0 or P1 within 0.10 m in ≥95%): PASS** at the deploy interval, with both P0 and P1 at 200/200 (Wilson 95% 0.981–1.000). It is a kinematic gate only.
- **P1 sprints.** With the goal as the waypoint, the planner covers far goals at a run: peak speed over 0.5 s windows reaches 2.2 m/s median and 4.1 m/s max, 122/200 episodes exceed 1.5 m/s, and brief flight phases appear. Mode and speed inputs have no effect once a waypoint is set (max qpos difference 1.6e-5). P1's millimetre precision is therefore not a usable baseline. Bounding the waypoint distance (P1c) keeps walking speed (≤0.63 m/s) and precision at the deploy interval.
- **The replan interval must stay below the plan horizon.** One plan spans 1.5 s after blending. At a 2 s interval a constant command lets the reference freeze for about 0.5 s every 2 s: 35% of the P1 ×2 episode is held. P0 is insensitive because its command changes trigger replans.
- **Waypoint semantics** (naive-chaining probe): targets are world-frame, only the last of the four slots set the endpoint in these probes, z is ignored, and the heading sets the final yaw.

**Posture table [M]** — FK of the reference; contact candidates are Isaac collision shapes below 5 cm in the last 3 s:

| Run | Steady pelvis z (m) | Head top (m) | Transit fwd (m/s) | Floor-contact candidates (duty) | Transition (head settle, s) |
|---|---|---|---|---|---|
| Squat h 0.1–0.3 (mode 4) | 0.30–0.31 (saturates) | 0.79 | 0 | knees 100% (h 0.3: right only) | 1.2 |
| Squat h 0.4 / 0.6 / 0.8 | 0.40 / 0.60 / 0.79 | 0.84 / 1.04 / 1.31 | 0 | none | 1.2–1.3 (h 0.8: none) |
| Crawl (8), direct / staged | 0.42 | 0.61 / 0.63 | 0.73 / 0.74 | hands 64–79%, knees 100% | 4.3 / 3.6 |
| Elbow crawl (14), direct / staged | 0.27 | 0.38 / 0.37 (body top 0.385) | 0.55 (−0.08 lateral) | hands, knees, forearms ~40%, right thigh 35% | 1.3 / 4.8 |
| Stealth walk (18) | 0.71 | 1.21 | 0.89 | none | 3.1 |
| Stealth walk 2 (22) | 0.50 | 0.94 | 0.60 | none | 2.0 |
| Strafe SLOW_WALK left / right | 0.76 | 1.30 / 1.29 | 0.45 / −0.32 lateral | none | – |
| Strafe WALK left | 0.76 | 1.29 | 1.09 lateral | none | – |

- **Kinematic half of the 0.7 overhang gate:** modes 8, 14 and 22 keep the head top ≤1.10 m; mode 18 does not (1.21 m). Among moving modes, only mode 22 does so with feet-only contact. Every crawl would fail today's scorer, which terminates on any non-foot floor contact. The ≥80% tracking-completion half needs the GPU run.
- **Crawl entry.** Direct CRAWLING from standing keeps walking forward while descending (x = 3.6 m at t = 5 s). The deploy's staged entry kneels in place first.
- **Mode 4 cannot move.** It is static: one plan, then the reference holds its last frame.
- **Joint-limit excess** is ≤0.071 rad, mostly ankle and waist pitch in modes 5, 8, 14 and 22.
- **Floor offset.** Walking reference feet reach up to 2.6 cm below z 0, and crawl knees and hands up to 7 cm.
- **MID-360 origin height** (for 0.9): 1.25 m standing, 0.86 m in mode 22, 0.53 m crawling, 0.28 m elbow crawling.

## Clips for the GPU tracking run (not run)

- `clips/`: 25 clips at 50 fps, the exact 50 Hz stream the deploy would play, so the motion loader does not resample. They cover the posture runs plus forward, backward, turning, diagonal and waypoint walks.
- `goal_clips/`: 120 B1-OL candidates, the first 40 goals of P0 ×1, P1 ×1 and P1c ×1, trimmed 3 s after settling.

Both directories are in the motion-lib format used by `scripts/teacher_review` (`motion_file=<dir>`).

## Caveats

- **Kinematic only.** Tracking, balance and contact forces are untested. P1's running references may be untrackable.
- **P1 and P1c rely on E1.** The deploy binary never sets `has_specific_target`.
- **P1c is exploratory.** It was added after P1's sprint was seen; its carrot length was fixed before the 200-goal run.
- **P0 tuning.** P0's heading deadband and freeze radius were chosen on 20 development goals (seed 70602).
- **Scope.** One start pose, open field, one planner seed (1234), a single deterministic run per condition.
- **Latency.** Latency is modelled as 0 ticks, TensorRT-like. The CPU call of about 28 ms would delay blends by 1–2 ticks [A: not studied].
- **Hand geometry.** FK uses the Isaac URDF, which has Dex3 hands; the motion-lib MJCF has rubber hands.

## Reproduce

```bash
cd /home/robotixx/motion2scene-training   # or a worktree
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONPATH=vendor/sonic \
  /home/robotixx/GR00T-WholeBodyControl/.venv_sim/bin/python \
  vendor/sonic/scripts/research/planner_kinematic_study.py all --out workspace/phase0/planner
PYTHONPATH=vendor/sonic /home/robotixx/GR00T-WholeBodyControl/.venv_sim/bin/python \
  vendor/sonic/scripts/research/planner_kinematic_study.py report --out workspace/phase0/planner \
  --notes workspace/phase0/planner/report_notes.json
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=vendor/sonic .venv_native/bin/python \
  vendor/sonic/scripts/research/planner_validate_clips.py workspace/phase0/planner/clips
cd vendor/sonic && PYTHONPATH=. /home/robotixx/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m pytest -q decoupled_wbc/tests/test_planner_runtime.py
```

The full run takes about 13 minutes on 4 CPU threads.
