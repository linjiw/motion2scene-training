# Current status

Last updated **September 23, 2026**. This is the single living status page. Reasoning, corrections, competitors and plan are in the [roadmap](ROADMAP_20260923.md); per-experiment reports stay authoritative for their own numbers.

Tags:
- **[M]** measured, receipt in the repo or `workspace/`.
- **[M-doc]** measured, but the only receipt is a doc because the packet lives on the other host.
- **[C]** computed from repo files.
- **[P]** proposed.

Protocol key: *repaired* = grounded Kimodo references (the corpus the current stack uses). Denominators: /89 train clips, /20 dev clips (inspected, never a test set), /24 stopping tasks, /19 teacher-feasible stopping tasks.

| Layer | Measured result (protocol) | Status | Evidence |
|---|---|---|---|
| Tracker: release SONIC | Completion on the repaired corpus: 83–86/89 train, 17–18/20 dev, at seeds 91231 [M] and 91260 [M-doc]. Tracking quality at 91231 [C]: strict proxy 12/89 train and 2/20 dev; median end-root XY error 0.50 m on completed train clips | Best dev completion. Never run on the stopping panel | `workspace/teacher-8192-500-review/`; [diagnosis](sonic/motion2scene/REPAIRED_TEACHER_DIAGNOSIS_20260912.md) |
| Tracker: 8192×500 fine-tune | Repaired corpus: train 86/88/84 and dev 15/16/16 (seeds 91260–91262) [M]. Strict proxy 25/89 train [C]. Median end-root XY error 0.18 m [C]. Trained on a plane | No dev gain over release. Better train quality | [distill report](sonic/motion2scene/DISTILL_8192_TEACHER_20260913.md) |
| Motor student | Frozen SONIC encoder, FSQ and decoder plus a 1.09M-parameter forecaster. Train 87/87/84, dev 11/11/9: one trained model, three evaluation seeds [M]. The dev gap is on the student's side | Working anchor for full commands only; rejects partial commands | `workspace/distill-8192/results.csv` |
| Masked / typed interface | The 79-D line peaked at 5–6/89, 0/20 dev (14/89 once). The 114-D transformer reached 21/89, 0/20. Action-flow heads: 0/89 [M] | No working masked student | [roadmap §2](ROADMAP_20260923.md#2-where-each-layer-stands) |
| Navigation adapter (known map) | Approach-weighted DAgger cycle 2: **7/19, in-sample**. This is seed 91260, the seed that produced every label, and rollouts are bit-deterministic. Uniform DAgger cycle 2: 2/19. Out-of-sample, nav-v2-recovery at seed 91262 scores 1/19 [M]. Approach c2 solves 0/6 long tasks | Unconfirmed. The tasks are timed reference reproductions, not navigation | [DAgger report](sonic/motion2scene/NAV_8192_DAGGER_REENTRY_20260914.md); `workspace/nav-8192/eval/` |
| SONIC kinematic planner baseline | Planner on this host (27 modes, batch 1, about 28 ms per call on CPU [M]). Never tracked in physics and never run on navigation tasks | Missing baseline (roadmap Phase 0–2) | [roadmap §2](ROADMAP_20260923.md#2-where-each-layer-stands) |
| RL (goal-conditioned) | Never run. `residual_rl.py` is wired to the legacy navigator | Not started | — |
| Perception (MID-360 / depth) | None on the active line. The archived line has only an ideal pelvis ray fan | Not started | — |
| Archived Motion2Scene/ICRA line | BLOCKED since September 11 | Archived; assets reused | [audit](sonic/motion2scene/audit/20260911-pilot-boundary/README.md) |
| Sim-to-real | None | Not started | — |

## Next actions [P]

The first two weeks of [roadmap §9](ROADMAP_20260923.md#9-first-two-weeks):
1. Pre-register a held-out-seed confirmation of the DAgger arms and give release SONIC its first stopping-panel run.
2. Run a goal/map-use ablation.
3. Fix the 00399 spawn bug and re-score all traces with an XY goal metric.
4. Port the SONIC planner and measure it kinematically, including a per-posture capability table.
5. Run token-space diagnostics and compute the sensor geometry.
6. Start task schema v2.

## Decisions waiting on the user

See [roadmap §12](ROADMAP_20260923.md#12-open-questions-for-the-user): hardware access and the MID-360 mount, target venue, framing and scope, the archived line, compute, external assets, and pushing `main`.
