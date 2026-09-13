# Full-command BFM motor recovery: protocol and implementation

**Completed readout:** [results, selected checkpoint, repeat seeds and navigation handoff](BFM_MOTOR_RECOVERY_RESULTS_20260913.md). The selected anticipatory student reaches 88/89 training and 10/20 development completions; the 114D full-command interface differs from the earlier 79D BFM.

This study prioritizes full-command control before further navigation distillation. The teacher, original SONIC motor decoder, repaired 89/20 train/development split and native evaluation termination remain fixed. The experiment packet is `/home/linjiw/research-data/m2s-bfm-motor-recovery-20260913`. Results are authoritative only when the corresponding stage receipt and native metrics exist.

## Problems being separated

The previous BFM adapts a public prior to a fixed SONIC decoder. It is not a reproduction of BFM's trainable action decoder. BFM describes current-student-state teacher queries, randomized reference starts, hard-motion sampling and a masking cold start. Our earlier training mostly used repeated fits on small, screened aggregates and mixed full/reduced commands before the full-command foundation was reliable. [BFM, Sections III–IV](https://arxiv.org/html/2509.13780v1).

This study tests four issues independently where possible: loss/masking, label support, current-command completeness, and online student-state coverage. A fifth experimental path preserves the pretrained reference encoder and predicts missing reference context from public inputs. It is deterministic reference prediction, not joint physical-state/action flow matching. Native task-qualified stopping data are prepared independently; these data do not change the full-command evaluation task.

## Immutable baseline and data tiers

The previous transformer checkpoint is `m2s-bfm-transformer-dagger-20260913/dagger-5-fit-model/step-006000.pt`, SHA-256 `e9761b7b6f7b60361b76e7578861eec4a8c802a7d872668a5fca450d4e100780`. Its original-seed completion is 5/89 train and 0/20 development. The frozen teacher hash is `afd649cfbbfd28833550e11a0f8c3b7a5f6a05ee8b4021dd0dac97a6f94733ce`.

The strict expanded-teacher set retains 35,339 rows. The explicitly different native-completion support ablation retains 75,187 rows: complete native teacher attempts retain all recorded pre-failure decisions, and failed attempts retain their original screened prefix. Original shard files and strict qualification results are preserved. This is a declared training-label support ablation, not a retroactive relaxation of evaluation or a scene/recovery qualification certificate.

Both data tiers include the same original nominal/query aggregate for offline controls. Training-only motion IDs, manifest hashes and per-shard hashes are checked. The 20 development motions are used for evaluation, never student fitting. This benchmark has been inspected repeatedly and is not a pristine final test set.

## Offline controls

Each control initializes from the same original BFM and receives 6,000 updates, batch 256, AdamW 1e-4, seed 91360, FP32, full commands only. Sampling is uniform over episodes within a randomly selected teacher/query source group. Dense GPU arrays avoid per-row Python overhead.

| Arm | Support | Objective | Input |
|---|---|---|---|
| strict-cvae | Original strict prefixes + queries | CVAE reconstruction/KL + public action + full-token loss | Legacy 79 |
| broad-cvae | Native-completion support + queries | Same | Legacy 79 |
| broad-public | Same broad support | Public decoded-action MSE + token MSE | Legacy 79 |
| current-frame-v2 | Same broad support | Public decoded-action MSE + token MSE | Legacy 79 + current joint velocity29 + relative orientation6 |

The current-command extension uses a zero-initialized path and preserves original tokens at migration. All new fields are present-time target quantities. No later reference frame, motion ID or phase is an actor input. This is a richer command interface than the historical 79D baseline, so its results must be labeled separately.

## Native reference packing audit

A native layout audit caught and invalidated the first new current-frame/horizon experiments. SONIC's `command_multi_future` concatenates all ten position frames, then all ten velocity frames, and its observation reshapes that vector into ten 58D blocks. Six relative-orientation values are appended to each block. Consequently a reshaped 640-vector is **not** ten chronological position/velocity/orientation frames.

| Representation | Layout |
|---|---|
| Native joint command | `concat(q[10,29].flatten(), qdot[10,29].flatten())` |
| Native encoder input | `concat(command.reshape(10,58), orientation[10,6], axis=-1).flatten()` |
| Physical chronological frames | `concat(q, qdot, orientation, axis=-1)` |

`reference_layout.py` provides exact packing/unpacking. Tests use independently assembled native arrays with distinct position/velocity sentinels, verify round trips, and verify that a current-frame extractor cannot read later frames even when those frames contain NaNs. Corrected teacher-horizon probes compare unpacked frame zero against native joint positions/velocities on every policy call and bound orientation differences by the configured ±0.05 observation noise. The delivered student reads only the current orientation observation slice, including that same noise; it never reads the later orientation frames. An intermediate probe rejected an over-strict noise-free parity assertion, which is preserved in `corrected-failure.json`.

`invalid-layout-experiments.json` lists rejected comparisons. Historical BFM fits treat the 640-vector as opaque posterior input, and legacy 79D online training does not unpack it; those runs are unaffected. Experimental extended checkpoints now require `reference_layout_version=native_q_then_v_v1`, preventing accidental use of the invalid versions. Recorded prior-stage files are not overwritten.

## Preserving the encoder as well as the decoder

`AnticipatoryMotor` retains the teacher's G1 encoder and exact FSQ quantizer, plus the frozen decoder. A zero-initialized predictor starts by repeating the current physical target frame, repacking it into the native format. It learns the remaining nine reference frames from measured history and current commands. The frame at the current time cannot be changed by the predictor.

The predictor uses training-derived input statistics. Its losses combine decoded action error, token error and a small normalized future-reference auxiliary loss. Future reference arrays appear only as training targets. Reconstructing recorded teacher tokens is exact, with decoded action parity checked separately. A current-reference policy initialized from the teacher is a pretrained baseline; it must not be advertised as a newly distilled BFM or goal-only navigator.

## Online DAgger and recovery probes

`OnlineMotorCallback` runs within the existing initialized Isaac Lab driver. Its initial bounded schedule uses 256 environments, 200 cycles of 32 control steps, and 16 optimizer updates per cycle. That is 1,638,400 simulator transitions and 3,200 optimizer updates. Each update mixes fresh same-state queries with teacher replay. The actor always receives full commands.

Starts use native randomized reference-state initialization with a 20% nominal-start mixture. Ten teacher-controlled burn-in ticks populate measured history after a detected reset. Teacher action probability decreases from 0.5 to 0.05, in addition to burn-in and bounded takeovers. These interventions are logged; training rollout completion is not unassisted student completion. Hard-motion sampling remains disabled in this first bounded control.

A takeover starts near a declared tracking-error threshold and lasts 16 ticks. The local recovery label requires four final consecutive ticks within the 0.10 m mean body-error threshold without a reset. Failing this conservative criterion is not synonymous with falling: the initial tracker groups boundary failures and support timeouts together. Recovery results are therefore reported as local criterion outcomes, not scene success or universal expert reliability.

Updates retain finite pre-step teacher queries under the unchanged native termination lifecycle, including states that the old 0.10 m prefix filter would discard. They never pair a pre-reset observation with a post-reset teacher target. Boundaries include native done flags, motion/start changes and cursor discontinuities. Per-motion and temporal-quarter exposure, interventions, query samples, checkpoint hashes and exact stage argv are retained. Online checkpoints are evaluated separately from nominal starts without teacher assistance.

## Successful stopping supervision

Four train motions were selected for displacement above 1 m, upright endpoints and low final-second root speed. Each receives a two-second stationary tail and two separately executed scene variants: clear and a synthetic corridor. Conversion preserves the existing MuJoCo joint-order convention. These are candidate tasks, not successful data until physical execution passes.

The independent scorer requires 3D goal distance <=0.25 m, speed <=0.10 m/s for 50 consecutive ticks, no undesired contact and no fall within the task horizon. It does not terminate for reference-pose mismatch, and it prevents reference exhaustion from teleporting the robot. Contacts are measured at 200 Hz. `StoppingTeacherCallback` retains synchronized teacher actions, tokens, measured history and public context. Only task-successful episodes are eligible in the exported collection.

A corridor-clearance demonstration is not proof of selecting an alternate route or ducking under a beam. Changed-goal and avoidance-required counterfactuals still need matching successful continuations. This study prepares supervision for the next navigation phase; it does not silently substitute stopping-task success for full-command motor success.

## Entry points

```bash
PYTHONPATH=. .venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.motor_training \
  --config /path/to/recorded-config.json --output /path/to/new-fit-directory

.venv_isaaclab/bin/python -m pytest -q decoupled_wbc/tests/test_motor_recovery.py
```

Native callbacks are `motor_runtime.MotorEvaluationCallback`, `motor_runtime.TeacherHorizonCallback`, `online_motor.OnlineMotorCallback`, and `stopping_teacher.StoppingTeacherCallback`. Use recorded packet argv to reproduce the environment and teacher settings. Output directories must be new. These modules require the existing Isaac Lab, SONIC checkpoint and repaired datasets; they are not standalone asset downloads.
