# Repaired teacher: step 12,500 evaluation

Checked 2026-09-13T00:23:51.674124+00:00.

Live snapshot: **12,572/32,000 iterations (39.29%)**, **38,621,184 transitions**, 11.01 hours elapsed. Approximate remaining time: 17.0 hours at lifetime-average throughput. The process remains in training; final evaluation has not run.

The fixed step-12,500 checkpoint was evaluated on all 20 repaired development clips using seed 91231, 20 evaluation environments and the same native tracking config bytes as the previous final-teacher evaluation. Its SHA256 is `d739673fd2c1bd4e5365f52a7b6229dd3d993341c6bed02cb573e32db1da1259`. The evaluation exited successfully. Earlier baseline metrics are reused with explicit hash bindings; training was not modified.

| Teacher | Completed motions | Mean progress |
|---|---:|---:|
| Repaired step 3,400 | 2/20 | 29.74% |
| Repaired step 4,700 | 4/20 | 35.80% |
| Repaired step 6,200 | 3/20 | 33.72% |
| **Repaired step 12,500** | **1/20** | **28.51%** |
| Previous final step 8,000, 512 envs | 11/20 | 71.48% |

Only `00835` completes in the latest evaluation. Step 12,500 represents 38,400,000 training transitions, **39.0625%** of the previous final teacher's 98,304,000 transitions. This is a same-evaluation comparison of current capability, not equal training exposure or an isolated data-repair ablation.

Measured performance has not improved with the additional training and this checkpoint is below earlier snapshots. The single-seed, repeatedly inspected 20-motion development set does not establish statistical significance or a causal explanation, but it is sufficient to avoid claiming improvement or promoting this teacher.

Training means over updates 12401–12500 versus 6101–6200: reward 3.012 vs 3.006; episode length 73.376 vs 74.740 control steps; body-position error 0.06662 vs 0.06884 m. Lower training body error can coexist with earlier failures; these aggregates do not replace the physical completion/progress comparison.

A focused diagnosis should examine per-motion failure reasons, motion exposure, reset distributions, optimizer behavior and reference/command consistency before changing settings. Do not assume remaining iterations will close the gap. Keep the previous final teacher as the stronger measured baseline. The authorized bounded run remains active and unchanged by this status check.

This is reference tracking, not scene-navigation success. The older unrepaired-data development result is not substituted for the matched repaired-data baseline. The paper draft remains a frozen earlier snapshot; this report adds later evidence without rewriting its history.

Full local packet: `/home/linjiw/research-data/m2s-repaired-teacher-step12500-comparison-20260912`. It contains the frozen checkpoint, command, config, native metrics/trajectories, receipt and comparison bindings.
