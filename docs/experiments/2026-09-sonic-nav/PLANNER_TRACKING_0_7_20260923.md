# Phase 0.7: release SONIC executing SONIC-planner clips open-loop

Run September 23, 2026. Release SONIC (`model_step_041550.pt`, sha `e6bdab…`) tracked the kinematic planner clips exported by the [planner study](../../sonic/motion2scene/SONIC_PLANNER_KINEMATIC_20260923.md) in Isaac Lab. Settings:
- open loop: the reference is fixed in advance, with no replanning from the robot state;
- flat plane;
- native terminations;
- three dev-block DR seeds (92610–92612).

## Results

1. **Tracking holds for every posture.** All 25 posture clips complete at all three seeds (75/75) [M]. That includes crawl, elbow crawl, squat, kneel, the two stealth walks, strafes, a back-walk and turns.
2. **Open-loop goal reaching is not precise.** The 120 goal references were each tracked at 3 seeds (P0, P1 and P1c controllers × 40 goals, 1–8 m):

   | Controller | Complete | Final XY error to the planned endpoint, median / p90 / max | Within 0.25 m |
   |---|---:|---|---:|
   | P0 direction/speed/stop | 120/120 | 0.85 / 1.13 / 1.58 m | 0/120 |
   | P1 waypoint (sprints) | 114/120 | 0.66 / 1.72 / 2.03 m | 2% |
   | P1c carrot waypoint | 120/120 | 0.66 / 1.26 / 1.65 m | 1% |

   - The robot stops cleanly: final planar speed is about 0.02 m/s.
   - It stops in the wrong place. P0 error grows only slowly with distance: 0.76 m at 1–3 m, 0.91 m at 3–5 m, 0.96 m at 5–8 m.
   - The six P1 failures are foot and end-effector tracking terminations on sprinting references.

   **Decision (roadmap 0.7 gate):** closed-loop replanning from the measured pelvis is required. The planner baseline must re-anchor its goal command on the robot's measured state (roadmap Phase 1.2). Open-loop reference reproduction cannot meet the 0.25 m goal tolerance. This is the same drift that caps release SONIC at 15/24 on the legacy stopping panel.
3. **Executed low postures are higher than planned.** The steady-phase head top is taken over t > 2 s (`head_top_steady` in the readout). Medians are averaged over 3 seeds; maxima are the maximum over all 3 seeds. It comes from URDF forward kinematics of the recorded joint states, using the visual surface for the head [C from M].

   | Mode | Head top, median / max (m) | Reference median (m) | Pelvis min (m) | Speed (m/s) | Non-foot floor contact (≤3 cm) |
   |---|---|---|---|---|---|
   | Walk (mode 1) | 1.303 / 1.316 | 1.298 | 0.75 | 0.32 | none |
   | STEALTH_WALK (18) | 1.228 / 1.250 | 1.204 | 0.66 | 0.88 | none |
   | **STEALTH_WALK_2 (22)** | **1.044 / 1.081** | 0.932 | 0.51 | 0.60 | **none** |
   | Crawl, staged (8) | 0.704 / 0.948 | 0.597 | 0.41 | 0.50 | hands 40%, knees 56–86% of ticks |
   | Elbow crawl (14) | min 0.21–0.29 | min 0.24–0.30 | 0.14–0.19 | — | forearms, hands, knees, thighs |
   | Squat, requested height ≤0.3 (4) | min 0.92–0.95 | min 0.79 | 0.43 (requested 0.30) | static | none |

   - The tracker keeps its head about 10 cm above the reference in STEALTH_WALK_2 and in deep squats, and it does not follow squat references below a pelvis height of about 0.43 m.
   - Open-loop XY drift over 6 s is 0.1–0.8 m for walking modes and 1.6–3.9 m for crawls.

## Consequences for the benchmark (Phase 1.5)

- **Walk-under.** An underside above about 1.33 m can be passed upright.
- **Duck band, feet only: about 1.11–1.30 m.** STEALTH_WALK_2 keeps the head top at or below 1.081 m (maximum over 3 seeds) at 0.6 m/s with only the feet on the floor. This is F3's B3-feasible band. The margin assumption (≥3 cm over the max) is to be fixed in the lock.
- **Below about 1.08 m, crawling is required.** Crawling puts hands and knees on the floor. It needs a crawl family with a per-body contact whitelist (hands, knees and, for elbow crawl, forearms) and a crawl fall rule. Otherwise it belongs in the pre-registered expressivity band E.
- **Closed-loop crawling.** Crawls drift by 2–4 m open-loop, so closed-loop crawl control needs the same measured-state re-anchoring.

## Protocol and evidence

- **Runner.** `scripts/teacher_review/run_eval.sh release custom`, with `SEED`, `MOTION_DIR`, `NUM_ENVS` and `OUT_DIR` overrides (commit after d2ea541). The PoseCapture callback ran 25 and 120 envs per process, taking about 42 s and 69 s wall time.
- **Readout.** `vendor/sonic/scripts/research/planner_tracking_readout.py`, with a floor-contact candidate threshold of 3 cm. Contact candidates are geometric proximity, not measured forces.
- **Evidence.** [`evidence/planner-tracking-0.7-20260923/`](evidence/planner-tracking-0.7-20260923/): readout JSONs, plus the receipt and command for each run. Raw pose files are in `workspace/phase0/tracking07/`.
- **Caveats:**
  - one planner seed;
  - standing start;
  - flat ground;
  - the kinematic clips come from the CPU planner runtime, not the deploy binary;
  - goal clips were trimmed at the kinematic arrival.
