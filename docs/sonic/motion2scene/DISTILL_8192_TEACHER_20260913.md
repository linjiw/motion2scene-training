# Anticipatory motor distillation from the 8192-environment teacher

September 13, 2026, on a shared RTX 5090 host. This study repeats the [motor recovery recipe](BFM_MOTOR_RECOVERY_RESULTS_20260913.md) with the locally trained 8192-environment × 500-iteration repaired teacher. Here "motor recovery" means a frozen SONIC encoder, FSQ and decoder, a learned forecaster for the future reference, and online same-state DAgger. That recipe's source study used the 4096-environment checkpoint-1400 teacher.

The student uses more offline updates, 4× the online environments and 2.5× the online updates. It matches the recorded student on train and development completion within evaluation noise. It does not exceed it, and it does not close the gap to its own teacher on held-out motions.

![Completion versus student updates](evidence/distill-8192-20260913/distill-progress.png)

## Result

Completion means reaching the end of the reference without native termination, from nominal starts and without teacher assistance. The split is the repaired 89 train / 20 development motions. Development motions were never collected or fitted, but this benchmark has been inspected throughout the project.

| Seed | Our student s1-3200: train | dev | Recorded student: train | dev | Our teacher: train | dev |
|---|---:|---:|---:|---:|---:|---:|
| 91260 | 87/89 | 11/20 | 88/89 | 10/20 | 86/89 | 15/20 |
| 91261 | 87/89 | 11/20 | 85/89 | 10/20 | 88/89 | 16/20 |
| 91262 | 84/89 | 9/20 | 86/89 | 10/20 | 84/89 | 16/20 |
| **Mean** | **86.0** | **10.3** | **86.3** | **10.0** | **86.0** | **15.7** |

- **Three checkpoints tie on train.** Stage-1 steps 3,200, 14,400 and 16,000 each reach 87/89 at seed 91260.
- **Selection rule.** The recorded rule (train completions, then train mean progress) selects step 14,400 (0.988 progress). Its three-seed means are 85.7 train and 9.0 dev. Step 16,000 gives 85.7 and 10.0. Step 3,200 is shown above because it was the first to reach 87/89 and received repeat seeds first.
- **Noise.** The three candidates differ by at most 0.33 train and 1.33 dev motions in mean, which is inside the ±2–3 motion spread between seeds. No single checkpoint is established as better.
- **Train failures follow the teacher.** On seed 91262, s1-3200 and the teacher fail the same five train motions. On train, the student is at teacher level.
- **Persistent dev failures.** The student fails 00333 and 00623 on all nine candidate evaluations, and 00611 on seven. The teacher completes all three on all seeds. The recorded student also failed 00333, 00611 and 00623 on every seed.

## What helped and what did not

| Change | Evidence (seed 91260) | Reading |
|---|---|---|
| More offline updates (30,000 vs 6,000) | Train 86 vs 86; dev **10 vs 5** | Largest single gain; train-only selection would not have revealed it |
| Online DAgger, 2048 envs, first 3,200 updates | Train 86 → 87; dev 10 → 11 | Small gain, inside seed noise |
| Online updates 3,200 → 16,000 | Train 84–87; dev 9–11 | Plateau; no trend |
| Stage 2: teacher probability 0.2 → 0, fewer takeovers, lr 5e-5 | Train 84–86; dev 9–12 | No gain; teacher-action fraction fell from 57% to 26% |

- **Offline fit alone is strong.** It reaches 86/89 train, versus 77/89 for the recorded offline stage. Likely contributors are six nominal collection seeds (175,981 broad-support rows), a teacher that completes 86–88/89, and 5× the offline updates. None was ablated separately.
- **Stage-1 teacher-action fraction.** It was 57%, versus 33% in the recorded study. The 0.10 m takeover trigger uses global body error, and this teacher drifts more at the root (median 0.18 m on train), so takeovers fire often. Lowering intervention in stage 2 did not change completion.

## Protocol

| Stage | Setting | Budget and resources |
|---|---|---|
| Collection | `PrefixFoundationCollectionCallback`, FP32, nominal starts, 89 train motions, seeds 91400–91405 | 6 × 201 s; 534 episodes, 175,981 broad rows (77,692 strict); ~3.9 GB GPU |
| Support | Completed attempts keep every decision; terminated attempts keep the strict 0.10 m body / 0.25 m root prefix | Mirrors the recorded native-completion tier |
| Offline | Zero-initialized `AnticipatoryMotor`, public loss, token weight 0.1, future weight 0.1, batch 256, AdamW 1e-4, seed 91360 | 30,000 updates in 248 s; ~4.3 GB GPU |
| Online stage 1 | `OnlineMotorCallback` from offline 30k: 2048 envs, 250 × 32 steps, 64 updates/cycle, batch 1024 (half fresh, half replay), lr 1e-4, teacher p 0.5 → 0.05, burn-in 10, takeover 16 ticks at 0.10 m (p 0.15), 20% nominal starts, seed 91370 | 16,000 updates, 16.4M transitions, 2,302 s; ~7.7 GB GPU |
| Online stage 2 | From s1-3200: 150 cycles, lr 5e-5, teacher p 0.2 → 0, takeover at 0.15 m (p 0.05), seed 91371 | 9,600 updates, 9.8M transitions, 1,620 s |
| Evaluation | `MotorEvaluationCallback`, full commands, nominal starts; seed 91260 for every checkpoint; seeds 91261–91262 for the top three and the teacher | ~105 s per run; ~3.8 GB GPU |

- **Choices not recorded by the source study.** Future-reference loss weight, online batch size, and the ramp that makes teacher-action probability decrease over training are local choices, because the source study did not record its values.
- **Concurrency and memory.** Evaluations ran alongside online training; peak GPU for our processes was about 15 GB with two evaluations running.
- **Isolation.** No development motion, evaluation seed or evaluation output entered collection or fitting.

## Limits

- **Reference tracking only.** This is not goal/map navigation, scene qualification or robot deployment.
- **Seeds.** Evaluation seeds are repeated executions of one trained model, not independent training runs.
- **Different teachers.** The comparison with the recorded student changes the teacher, collection mix and budgets together, so it is operational rather than a matched ablation.
- **Where the gap lies.** The remaining development gap is between student and teacher: about 5–6 motions, in both this study and the recorded one. Further online updates at this setting did not reduce it. Future work should target the anticipation/generalization failure on held-out motions (00333, 00611, 00623) rather than more updates. That work needs a fresh held-out set, because development results have now been inspected repeatedly.

## Reproduction

The packet `workspace/distill-8192` holds shards, checkpoints, native logs and per-run `command.json`; it is not in Git. Compact configs, receipts, hashes and all 54 evaluation rows are in [the evidence bundle](evidence/distill-8192-20260913/). The pipeline scripts and their order are in [`scripts/distill_teacher`](../../../scripts/distill_teacher/README.md).
