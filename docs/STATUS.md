# Current status

Last updated **September 23, 2026, evening**. This is the single living status page. For reasoning, competitors and the plan, see the [roadmap](ROADMAP_20260923.md). Per-experiment reports stay authoritative for their own numbers.

Tags:
- **[M]** measured, with a receipt in the repo or `workspace/`.
- **[M-doc]** measured, but the receipt is doc-only because the packet lives on the other host.
- **[C]** computed from repo files.
- **[P]** proposed.

Protocol key:
- *Repaired* means grounded Kimodo references, the corpus the current stack uses.
- Denominators: /89 train clips; /20 dev clips (inspected, never a test set); /24 stopping tasks; /19 teacher-feasible stopping tasks.

## Headline (September 23)

**The goal/map navigation adapter line is closed.** Under the pre-registered held-out-seed confirmation [nav8192-confirm-v1](experiments/2026-09-sonic-nav/NAV8192_CONFIRM_V1_STAGE1_20260923.md):
- approach-DAgger c2, uniform-DAgger c2 and nav-v2-recovery each score **0/57** on fresh seeds 92601–92603 [M];
- the teacher scores 58/72 on the same tasks and seeds;
- the in-sample 7/19 came from bit-deterministic rollouts on the seed that produced the labels.

The futility rule fired. Work moves to the SONIC-planner baseline and goal-conditioned learned arms (roadmap Phases 1–3).

## Layers

| Layer | Measured result (protocol) | Status | Evidence |
|---|---|---|---|
| Tracker: release SONIC | Repaired corpus: 83–86/89 train, 17–18/20 dev [M, M-doc]. **Stopping panel 15/24 at seed 92601 [M]**; it fails only by not reaching the reference endpoint (drift). Median end-root XY error 0.50 m on train clips [C] | Best dev completion; drifts in open loop | [confirm-v1](experiments/2026-09-sonic-nav/NAV8192_CONFIRM_V1_STAGE1_20260923.md); `workspace/teacher-8192-500-review/` |
| Tracker: 8192×500 fine-tune | Train 86/88/84, dev 15/16/16 [M]. Stopping panel 19/19/20 of 24 at 92601–92603 [M] | Better train and goal-hold quality; no dev gain | [distill report](sonic/motion2scene/DISTILL_8192_TEACHER_20260913.md) |
| Motor student | Train 87/87/84, dev 11/11/9: one model, three eval seeds [M] | Full commands only | `workspace/distill-8192/results.csv` |
| Navigation adapter (known map) | In-sample 7/19 (seed 91260). **Held-out 0/57 for every arm [M]** | **Closed** (futility) | [confirm-v1 Stage 1](experiments/2026-09-sonic-nav/NAV8192_CONFIRM_V1_STAGE1_20260923.md) |
| Masked / typed interface | Best results: 79-D line ≤14/89 train and 0/20 dev; 114-D transformer 21/89 train and 0/20 dev [M] | No working masked student | [roadmap §2](ROADMAP_20260923.md#2-where-each-layer-stands) |
| SONIC kinematic planner (Phase 0.6/0.7, CPU part) | 28 ms/call on this host [M]. P0 direction/speed/stop controller: 200/200 goals within 0.10 m and stopped, **kinematic only** [M]. The waypoint mode sprints (peak 2.2 m/s median). Low postures: stealth-walk-2 head top 0.94 m with feet-only floor contact; crawl 0.61 m and elbow crawl 0.37 m both put hands, knees or forearms on the floor [C] | Not yet tracked in physics (Phase 0.7 GPU run is next) | [planner study](sonic/motion2scene/SONIC_PLANNER_KINEMATIC_20260923.md); `workspace/phase0/planner/` |
| Token action space (Phase 0.8) | ±1-bin noise changes release decoder actions by 0.095 RMS (gate <0.15). The `g1_kin` cycle residual flags 98% of ±2-bin incoherent tokens. Release and 8192 decoders differ by 0.32 RMS [M, offline] | Viable for Phase 3 | [token diagnostics](experiments/2026-09-sonic-nav/TOKEN_SPACE_DIAGNOSTICS_20260923.md) |
| MID-360 geometry (Phase 0.9) | Sensor 1.217 m above the floor standing on both mounts [C]. The Isaac URDF and Unitree's own description mount it **inverted**: the inverted mount sees the floor from 0.87 m ahead but loses overheads early, so it needs 2–5 s of registered memory [C]. The upright mount is floor-blind to 7.4 m | Mount on the real robot unverified (user question) | [sensor geometry](experiments/2026-09-sonic-nav/evidence/sensor-geometry-20260923/summary.md) |
| Legacy panel fixes (Phase 0.5) | The XY goal metric changes 0/497 legacy outcomes [M]. Spawn-clearance assertion added; 00399-corridor rebuilt as legacy-v1.1 (not yet run). 00413-corridor's goal also sits 0.313 m from a wall [C] | Done | [re-score summary](experiments/2026-09-sonic-nav/evidence/rescore-xy-20260923/rescore_summary.md) |
| Evaluation statistics (Phase 0.1) | `scripts/experiments/eval_stats.py`: Wilson, exact McNemar, exact task-stratified CMH, cluster bootstrap, Holm; reproduces the 91260 p=0.125 [M]. Gate power simulation [S] | Done | [GATE_SIM](../experiments/power/GATE_SIM.md) |
| RL (goal-conditioned) | Never run | Phase 3 | — |
| Perception on the active line | None | Phase 4 | — |
| Sim-to-real | None | Phase 5 | — |

## Next actions [P]

1. **Phase 0.7 GPU:** track the 25 planner posture clips and 120 goal clips open-loop with release SONIC at 3 DR seeds.
2. **Phase 1:** task schema v2 (reference-optional, XY goal), `PlannerBaselineCallback` (closed-loop planner → frozen release tracker) with its parity test, and the physical smoke test (idle; walk 3 m and stop; turn and walk; sidestep).
3. Then the multi-instance evaluation harness and the decision-forcing benchmark v1 (roadmap Phases 1.4–1.6).

## Decisions waiting on the user

See [roadmap §12](ROADMAP_20260923.md#12-open-questions-for-the-user). The most pressing:
- **Is the real G1's MID-360 mounted inverted, as in Unitree's description?** The inverted mount changes the perception design.
- Hardware access.
- Target venue.
