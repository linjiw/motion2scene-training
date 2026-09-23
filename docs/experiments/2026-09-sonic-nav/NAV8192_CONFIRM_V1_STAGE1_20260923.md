# nav8192-confirm-v1, Stage 1: the DAgger navigation adapters score 0/57 on held-out physics seeds

Run September 23, 2026 under the pre-registration [`experiments/nav8192-confirm-v1/registration.json`](../../../experiments/nav8192-confirm-v1/registration.json), committed as 661a234 before launch.

**Result: the registered futility rule fires, and the goal/map adapter line is closed.** Three navigation adapters were re-evaluated on the 19 teacher-feasible stopping tasks at fresh seeds 92601–92603. None of those seeds was used for any collection or fit.

| Arm | Seed 91260 (in-sample, the seed that produced every label) | 92601 | 92602 | 92603 | Held-out total |
|---|---:|---:|---:|---:|---:|
| approach-weighted DAgger cycle 2 | 7/19 | 0/19 | 0/19 | 0/19 | **0/57** |
| uniform DAgger cycle 2 | 2/19 | 0/19 | 0/19 | 0/19 | **0/57** |
| nav-v2-recovery | 4/24 (1/19 at 91262) | 0/19 | 0/19 | 0/19 | **0/57** |
| 8192×500 teacher (reference tracker, all 24 tasks) | 19/24 | 19/24 | 19/24 | 20/24 | 58/72 |
| release SONIC (reference tracker, all 24 tasks) | — | 15/24 | — | — | 15/24 |

The 95% Wilson upper bound for each adapter's held-out rate is 6.3%.

The registered readout (`scripts/experiments/eval_stats.py gate`) confirms the checkpoint sha256 of every arm. Its outcome:
- **Futility.** Approach mean 0 ≤ 3/19, and there are 0 discordant wins vs 0 losses against recovery, so the rule fires. Stage 2 does not run.
- **Confirmation gate.** Not met: CMH p = 1.0 against both comparators.
- The registered Phase 0.4 goal-ablation condition is moot.

## What else Stage 1 measured

- **Determinism.** Re-running approach-c2 at seed 91260 on 00908-stop-clear and 00677-stop-corridor reproduces the original traces bit for bit: the maximum difference in pelvis position, speed and contact force is 0.0. The in-sample 7/19 was therefore scored on exactly the physics instances and trajectories its training labels came from.
- **Nominal dynamics does not rescue the adapter.** With startup domain randomization removed (`eval_remove_events` = the `nominal_d1` profile), approach-c2 scores 0/2 at both 92601 and 92602 on the same two tasks. The out-of-sample failure is not only dynamics variance. Its mean final distance is 1.07 m and 0.56 m, against 0.18 m in-sample.
- **How the adapters fail out of sample.** Most episodes time out far from the goal: mean final distance 1.3–1.9 m, against 1.03 m in-sample for approach-c2. There are 1–6 corridor contacts per panel and no falls. Approach-c2 reaches the goal tolerance on only 4 of 57 held-out episodes (all at 92601; best hold 36 ticks).
- **The tasks remain feasible.** The 8192×500 teacher scores 19, 19 and 20 of 24 across the three seeds.
- **First stopping-panel number for release SONIC: 15/24 at 92601.**
  - It completes every task it reaches. It never reaches the goal on 00120, 00677 or 00711 (both variants of each).
  - Its mean final distance is 0.45 m, against 0.33 m for the fine-tune.
  - Both trackers fail 00399-corridor (spawn intersection) and 00781. Only the fine-tune fails 00707 (holds of 38 and 9 ticks).
  - This fits release's larger global drift: median end-root XY error 0.50 m vs 0.18 m on training clips. On these tasks the goal is the reference clip's endpoint, so open-loop reproduction penalizes drift. A closed-loop planner that replans from the measured pelvis should remove this failure mode, which the Phase 1 smoke test measures.

## Interpretation

The adapter learned to reproduce specific rollouts, not to navigate. On the physics instance that generated its labels it matches trajectories closely enough to hold at the goal. On any other instance it drifts off the demonstrated states and never recovers. This is consistent with the roadmap diagnosis (§5 items 1 and 3):
- the adapter must regenerate a 113-D clip-specific command (joint positions, velocities and keypoints) from goal and map alone, which is an ill-posed target;
- its labels come from a reference tracker on a nominal clock;
- DAgger labels were collected on the same physics instance used for evaluation.

The short-cycle DAgger "improvement" (4 → 7/19) was in-sample fitting. **Do not cite any of the 91260 adapter numbers as navigation results.**

Per the roadmap's stop-doing list, the 114-D full-command regression adapter is now closed:
- no more recovery sweeps, approach weighting or DAgger cycles on it;
- its checkpoints remain as historical baseline B8.

The Phase 2 planner baselines and the Phase 3 goal-conditioned learned arms replace it.

## Protocol notes

- **Scorer.** Unchanged legacy scorer: 3-D pelvis within 0.25 m and ≤0.10 m/s for 50 ticks before the reference-derived deadline; contact >1 N or pelvis <0.25 m terminates. The XY re-scorer ([`scripts/navigation_distill/rescore_xy.py`](../../../scripts/navigation_distill/rescore_xy.py)) changes 0 of 497 legacy outcomes, so the metric choice does not affect this conclusion.
- **Deviations** (logged in the registration):
  - Four Stage 1 tasks launched at 12:58–13:01 record `harness_dirty=1–2`. The cause was a new untracked module that was briefly present and is not imported by the evaluation.
  - The panels span harness commits 661a234, 000d185 and 6a094ab. The runner and every imported evaluation module are byte-identical across these (`git diff 661a234 HEAD -- scripts/navigation_distill/{run_stage.sh,scene_native.sh,stage_config.py} vendor/sonic/gear_sonic/{eval_agent_trl.py,envs,research/scene_distillation}` is empty).
  - `InstanceDumpCallback` (realized DR parameters) was deferred.
- **Evidence** is in [`evidence/nav8192-confirm-v1-20260923/`](evidence/nav8192-confirm-v1-20260923/):
  - per-panel `results.txt` copies;
  - `task-results.csv`: 273 episodes, with checkpoint sha prefix and harness commit per episode;
  - the full gate readout.

  Traces and logs remain in `workspace/confirm-v1/eval/`.
