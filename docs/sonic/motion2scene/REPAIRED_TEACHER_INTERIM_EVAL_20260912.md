# Repaired teacher: interim matched evaluation

Snapshot: 2026-09-12T16:50:38.978481+00:00.

The repaired teacher is still training and currently underperforms the previous final teacher on the same repaired development set. Do not promote this interim checkpoint.

## Live training status

- Iteration 3,453/32,000 (10.79%), 128 environments, 89 screened repaired training clips; all 89 have observed exposure.
- 10,607,616 transitions; 3.44 hours elapsed. Approximate remaining time 28.4 hours if average throughput persists. Concurrent GPU work can change this.
- Training remains active; peak allocated PyTorch memory 0.98 GiB. No training settings changed during this check.
- Previous final run: 512 environments, 8,000 iterations, 98,304,000 transitions on the original 100 training clips. At 128 environments the equivalent exposure requires 32,000 iterations; iteration counts alone are misleading.

## Fresh physical comparison

Frozen current checkpoint: **iteration 3,400**, 10,444,800 training transitions, SHA256 `f60803ad354b94bed6b67c071aae3546c1eb9cf981d62a40b1074ea9af165097`.
Previous final checkpoint: **iteration 8,000**, SHA256 `48b3a1c04cdbbcd9ffe8ad10b2591aff781c253c03c61ccb8683348d862c54ed`.

Both evaluated sequentially with the same repaired environment config, seed 91231, all 20 repaired development clips, native tracking termination criteria and 20 evaluation environments. Both subprocesses exited successfully. Snapshotting left the live training checkpoint untouched. The evaluation uses current corrected trajectory field roles. Commands, identical config copies, checkpoints, logs, metrics and body trajectories are retained in the packet.

| Outcome | Current 3,400 | Previous final 8,000 |
|---|---:|---:|
| Complete motions | 2/20 (10%) | 11/20 (55%) |
| Mean reference progress | 29.74% | 71.48% |
| Completed motion IDs | 00547, 00802 | 00047, 00117, 00150, 00187, 00311, 00547, 00714, 00744, 00802, 00835, 00981 |

This is a fair same-evaluation comparison of current versus final capability, **not equal training exposure or an isolated dataset-repair ablation**. The old teacher's previously reported 6/20 development completions came from the original data and another evaluation seed; it must not be substituted for this newly measured repaired-data baseline. Neither result measures scene navigation.

Native global position error is lower for the current checkpoint (222 vs 359 mm), but it fails much earlier. Different survival durations and trajectory drift make that aggregate insufficient to claim improved tracking; completion and progress provide the clearer result here. Local position errors are 47.3 vs 40.3 mm.

## Training signals and interpretation

Recent 100-update means for the current run: reward 2.894, episode length 73.1 control steps, body-position error 0.0703 m. Old final 100-update means: reward 6.177, length 112.3, body error 0.0867 m. Current reward/length are also below its first 100-update means. These training aggregates have different data, adaptive sampling, reset distributions and survival durations; they are diagnostics, not matched success rates.

Continue the authorized bounded run, retain the previous final policy as the stronger measured baseline, and investigate reward/episode-duration decline before asserting that longer training will resolve it. A later predeclared checkpoint evaluation can determine whether recovery occurs. Do not use this interim development observation to silently tune hyperparameters or select among many checkpoints. The predeclared final release-versus-repaired evaluation remains pending; this user-requested interim look is recorded separately.

Artifacts: `/home/linjiw/research-data/m2s-repaired-teacher-interim-comparison-20260912`. See `comparison.json`, `training-comparison.json`, `training-curves.png`, and per-arm `command.json`, `receipt.json` and `metrics/metrics_eval.json`.
