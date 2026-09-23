# Decision-forcing navigation benchmark v1: draft specification

Roadmap Phase 1.5. **Draft, September 23, 2026: not locked.** Numbers marked *calibrate* are fixed from the Phase 1.3 smoke results on dev seeds before the lock (Phase 1.6) is written. The lock is written before any learned policy is trained and before any policy sees the test split.

The benchmark replaces the legacy stopping panel. That panel only asked for timed reproduction of training clips: its deadline and goal came from the reference, and no obstacle forced a decision. Here, tasks carry no reference clip. Every family forces a route or posture decision, and a known-map scripted baseline (B3) must be able to solve at least one family per band.

## 1. Body envelope used to design the bands

These are measured from executed release-SONIC rollouts of planner clips (Phase 0.7). Geometry comes from URDF forward kinematics of recorded joint states, using Isaac collision shapes, and the frame is yaw-aligned. Values are median / max over the steady phase (t > 2 s), all three seeds [C from M].

| Gait (planner mode) | Head top (m) | Width across motion (m) | Floor contact besides feet |
|---|---|---|---|
| Walk (1) | 1.303 / 1.316 | 0.575 / 0.616 | none |
| STEALTH_WALK_2 (22) | 1.044 / 1.081 | 0.493 / 0.530 | none |
| Crawl, staged (8) | 0.704 / 0.948 | 0.481 / 0.605 | hands, knees |
| Strafe (1, lateral) | about 1.25 | 0.55 depth across motion | none |

The margin rule adds 3 cm of clearance on each constrained side, beyond the maximum *(calibrate)*.

## 2. Task families

All tasks start at the canonical pose: origin, yaw 0, planner IDLE carrier. Each scene has 3–4 goals, including one **control goal** whose straight path the obstacle does not affect.

| Family | Geometry | Decision forced | B3 feasibility (expected) |
|---|---|---|---|
| **F0 open field** | No obstacles. Goals 1–8 m away, bearing uniform in ±180° (including behind) | Stop at a goal; turn | Walk |
| **F1 forced bypass** | A box wall (1.0–2.5 m long, 0.2 m thick, 1.5 m tall) across the straight line. **Mirrored pairs**: the wall is extended to one side so only the other side is open | Route: left vs right | Walk plus A* |
| **F1-sym** (C2 mechanism set) | A symmetric wall where both sides are open and equally long | None forced. Route entropy is used to test the imitation-gap prediction | Walk (either side) |
| **F2 gap or doorway** | A wall with one gap, offset laterally by 0–1 m. Widths in two bands: **walk-through 0.68–1.0 m**; **duck-and-narrow 0.58–0.66 m**, which needs STEALTH_WALK_2's narrower envelope | Aim and align; posture for the narrow band | Walk; stealth-2 for the narrow band |
| **F3 lintel** | A beam 0.2 m deep with flush side walls, so no detour is possible. Underside bands: **walk-under ≥1.36 m** (control); **duck 1.11–1.30 m** | Posture: duck vs walk | Stealth-2 in the duck band |
| **F3-free** | The same beam without side walls | Detour vs duck (both legal) | Either |
| **F3-crawl** | Underside **0.80–1.05 m**, flush side walls | Posture: crawl | Crawl (hands and knees on the floor; own contact whitelist and fall rule) *(provisional, pending measured contact forces)* |
| **F4 composite** | A bypass wall and a lintel, or a doorway then a lintel, 2–4 m apart | Sequence of decisions | Walk, A*, then stealth-2 |
| **E expressivity** (reported separately; pre-registered) | Gaps 0.45–0.55 m (below every planner-mode envelope). Lintels in any band that no planner mode clears but a kinematic existence proof shows passable | Postures beyond the planner vocabulary | Expected to fail. Tests whether token or chunk interfaces can express what commands cannot |

Step-over obstacles are left out of v1. They need a foot-clearance scorer.

## 3. Success, failure and deadlines

- **Success.**
  - XY pelvis distance to the goal ≤ 0.25 m.
  - Then a hold of 50 consecutive ticks at 3-D pelvis speed ≤ 0.10 m/s. Planar speed is recorded as well.
  - All before the scoring deadline.
- **Terminating failures.**
  - Obstacle contact > 1 N at 200 Hz.
  - Non-foot floor contact > 1 N, except for the family's whitelisted bodies (standing families: feet only; F3-crawl: plus hands and knees *(provisional)*).
  - A fall:
    - standing families: pelvis z < 0.25 m;
    - crawl family: pelvis z < 0.10 m, or torso roll > 60° sustained for ≥ 0.5 s *(provisional)*.
- **Deadline** *(calibrate on smoke)*.
  - Episodes run to `max_steps = 2 × deadline`. Scoring happens post hoc at the declared deadline and at two sensitivities, so time and accuracy failures stay separable.
  - The draft rule: deadline = geodesic length / 0.3 m/s + 6 s + 3 s per posture change.
  - The 6 s term covers P0cl's slow final approach at 0.166 m/s over the last metre, plus settling.
  - Sensitivities: 0.4 m/s and 0.2 m/s.
- **Secondary metrics.**
  - Success with contacts allowed.
  - Time to goal; SPL (geodesic); path ratio.
  - Minimum body clearance; the body that made contact.
  - Re-arm count (B-arms); root drift from the planned root.
  - Joint-jerk p95; foot slip.

## 4. Splits, seeds and validation

- **Layout generator seeds.** Disjoint train / dev / test.
- **Physics seeds.**
  - Dev: 92620–92699. 92610–92612 went to 0.7; smoke uses 92620 onward.
  - Confirmation: 92700–92799.
  - Test: 95000–95099, sealed.
  - Collection only from 70000–79999.
  - `experiments/seeds.yaml` is regenerated before the lock.
- **Held-out V3 panel.** The 18 LOCKED V3 overhang layouts are converted to F3 goal tasks by a dated amendment with its own sha256. The panel is opened once, for final evaluation.
- **Validation on CPU before the lock.**
  - The null policy scores 0.
  - A straight-line kinematic planner rollout collides in ≥ 95% of F1 and F3 tasks.
  - Mirrored pairs are exact mirror images.
  - Every task has a start and goal clearance ≥ 0.35 m (the Phase 0.5 assertion).
  - Every F2/F3 band respects the §1 envelope plus margin.
  - Every scene USD hash matches the task JSON.
- **Size** *(calibrate)*.
  - About 12 dev layouts per family × 3–4 goals × 3 DR seeds plus 1 nominal seed, i.e. about 900 dev episodes.
  - The confirmation and test sizes are set by simulating the registered tests with `experiments/power/gate_cmh.py`-style models.

## 5. Open items before the lock

1. Calibrate the deadline constants and the 3 cm margin on the Phase 1.3 smoke runs (dev seeds).
2. Measure contact forces for crawl (0.7 only has geometric proximity) before fixing the F3-crawl whitelist and fall rule.
3. **External anchor (Phase 1.8), decided: CAT generalist v1.** It is Apache-2.0, has public G1 weights (`Axian12138/Click-and-Traverse`, `generalist_v1` ONNX) and takes goal guidance over a known voxel map. It runs in MuJoCo and moves legs only (12 DoF); the upper body stays at its default pose. PASSAGE already reports it zero-shot (70.3% success, 14.0% collision-free). Plan:
   - evaluate it natively in MuJoCo on exported F0–F4 scenes (2 days);
   - an Isaac port is optional (3 days).

   Gallant has no weights and no licence file. MTC and PASSAGE have no released code. See the [scoop watch, 2026-09-23](scoop-watch/2026-09-23-anchor-and-scoop.md).
4. Build the Phase 1.4 multi-instance harness. Evaluating about 900+ dev episodes per arm sequentially is not practical: at the typical 290 ms/step that is days per arm.
