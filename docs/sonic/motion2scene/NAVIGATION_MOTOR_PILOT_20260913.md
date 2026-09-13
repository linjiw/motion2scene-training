# Goal/map navigation through the preserved motor: first pilot

**Completed result:** the preserved full-command student succeeds on **8/8 stopping/corridor tasks**, while the new goal/map-only command-completion student succeeds on **0/8 of the same tasks**. The latter enters the goal region on 5/8 and remains upright without prohibited contact on all eight. This is useful progress in isolating the remaining failure, not a successful navigation model.

The previous teacher also passed these eight tasks. Thus the existing motor student can execute the requested terminal behavior with detailed current commands; the new public task-conditioning path does not yet infer and maintain sufficiently accurate commands in closed loop. The control does not prove which combination of prediction error, observation/history limitations and distribution shift causes failure. It makes another motor-decoder replacement a poorly supported immediate priority.

## Implementation

- `navigation_motor.py`: typed `NavigationInput`, goal/map-only command-completion model, fixed motor retention path, bounded offline training and checkpoint loading.
- `navigation_data.py`: hash-bound import of independently successful stopping episodes. Task outcome, teacher, scene/continuation, train split, ancestry, shard shape and target masks are checked. Original decision indices remain intact.
- `navigation_motor_runtime.py`: separate native goal/map and full-command task callbacks using the same independent scorer.
- `direct_scene_runtime.py`: reusable student loading/action hooks; contact collection, termination, goal hold and reference-cursor clamping retain their existing semantics.
- `test_navigation_motor.py`: information-boundary, masked-padding, complete-map, frozen-motor, actor-runtime and evidence-admission tests.

The navigation actor sees normalized measured history (930 values), task context (10 values), and a set of up to five 15D obstacle primitives with validity masks. Task context contains body-relative start and goal, tolerance, terminal speed, hold duration and complete-map status. The actor receives **no current target commands, motion ID, reference clock, future reference or critic state**. The runtime test deliberately supplies an environment with only measured pose and no target-command fields.

A two-layer 512-wide MLP predicts the 114D detailed command internally. An obstacle set encoder supplies pooled context. The predicted command goes through the frozen selected anticipatory motor, its encoder/quantizer and SONIC decoder. Teacher actions/tokens and recorded detailed commands supervise training; they are not actor features. This is deterministic internal command completion, not action-residual PPO, direct reference flow or arbitrary masked BFM.

All inherited motor parameters are frozen. The optimizer updates only the navigation predictor and obstacle encoder. Real-checkpoint training verifies every inherited tensor is unchanged and verifies exact full-command action identity on an anchor batch. The full-command task comparator then provides physical evidence that the unchanged parent can complete these tasks.

## Dataset and budget

The training manifest is `m2s-bfm-motor-recovery-20260913/successful-stopping-context.json`: eight independently scored teacher executions, 2,739 rows, four motion IDs. All four IDs share ancestry group `34a712255f13f7ba08a92bcc9c01af140142e40866d80561bd671dcb49673e16` in the recorded catalog. This is **one ancestry group**, not four independent families. All eight evaluated tasks are training tasks; clear/corridor replicas and row splits do not create held-out generalization evidence.

The pilot uses 32 smoke updates followed by a separately initialized 6,000-update fit, batch 256, AdamW at 1e-4, FP32, training seed 91370. Sampling chooses a motion, a scene replica, and a valid decision; no sequence model is trained on collapsed valid rows. There is no navigation DAgger in this pilot. The terminal checkpoint is predeclared, with no task-based checkpoint selection.

Eight goal/map evaluations were registered first. Their failure motivated a separately recorded eight-task full-command control with zero optimization. Each native task uses one environment and evaluation seed 91260, no teacher intervention, the original deadline, at most 800 ticks, and the same collision geometry and initial state. Actual execution totals are 3,900 goal/map and 3,076 full-command control ticks. The planned maximums and per-stage wall ceilings are retained in the evidence bundle.

## Task outcomes

Success requires goal distance at most 0.25 m and speed at most 0.10 m/s for 50 consecutive 20 ms control ticks, no fall, and no prohibited contact. Pair-resolved contacts are measured every 5 ms. Goal entry alone is not success. Full-command reference completion is also a different metric.

| Motion / scene | Full-command success | Goal/map success | Goal/map entered goal | Best goal/map hold |
|---|---:|---:|---:|---:|
| 00908 clear | Yes | No | Yes | 0/50 |
| 00908 corridor | Yes | No | Yes | 42/50 |
| 00413 clear | Yes | No | No | 0/50 |
| 00413 corridor | Yes | No | No | 0/50 |
| 00976 clear | Yes | No | Yes | 0/50 |
| 00976 corridor | Yes | No | Yes | 0/50 |
| 00265 clear | Yes | No | No | 0/50 |
| 00265 corridor | Yes | No | Yes | 38/50 |

Every goal/map attempt ended at its unchanged deadline, without a recorded fall or prohibited contact. The last logged training minibatch action MSE is 0.000875, compared with 0.249 in the first minibatch. These are different training minibatches, not a held-out error estimate. The small training loss did not produce task success, so simply extending the same offline fit is not justified by this result.

![Stopping diagnostic](evidence/navigation-motor-20260913/stopping-diagnostic.png)

The 00908 corridor trace reaches 42 consecutive valid hold ticks and subsequently leaves the goal region. [The supporting CSV](evidence/navigation-motor-20260913/stopping-diagnostic.csv) retains the full time, distance, speed and hold series. This supports investigating terminal drift; it does not prove a single causal explanation or justify shortening the success criterion.

## Next experiment informed by the failure

Preserve the full-command checkpoint and its successful matched-task results. Compare the present command-completion branch against direct context-to-desired-reference prediction through the same frozen encoder/decoder. The direct branch may avoid unnecessary reconstruction of all detailed commands, but that advantage must be measured on the same task/query budget.

Add **causal navigation history** before making the model much larger. The current context is instantaneous, and the legacy930D history does not contain a separately measured translational-velocity channel. For arrival, recent localized pose/goal changes can help infer motion relative to the destination. Extend the collector with timestamped measured pose, orientation and task context, then compare recurrent context with velocity estimated from that causal localization history. Do not introduce simulator world velocity as an undeclared sensor shortcut. Store full sequences and reset markers; do not reconstruct false temporal adjacency after filtering.

Collect navigation-student states during approach, deceleration and hold, with compatible teacher continuations. First validate short teacher takeovers under the same goal/contact requirements, preserving failed queries separately. Continue successful teacher recordings beyond the minimum success instant under a new collection mode so sustained terminal behavior is represented; retain the existing evaluation threshold and deadlines. Goal/map DAgger must run the actual goal/map student, not reuse full-command rollouts and call them task-profile coverage.

A next bounded proposal is 12 short cycles over the four existing clip tasks, 200 ticks per episode (9,600 student/teacher-control decisions), with separately capped recovery takeovers, fixed task replay and periodic unassisted evaluation. This is a proposal, not an implemented or launched navigation DAgger run. New ancestry/task families and avoidance-required pairs must be acquired before scaling this positive control into a navigation-generalization study. Flow and residual-RL branches remain comparisons after the information and teaching contracts are addressed.

## Evidence, commands and validation

Local packet: `/home/linjiw/research-data/m2s-nav-motor-bridge-20260913`. The research checkpoint is `fit/training/step-006000.pt`; it is an unsuccessful task candidate, not a replacement for the selected full-command checkpoint.

The committed [evidence bundle](evidence/navigation-motor-20260913/README.md) includes the selected checkpoint hash, fit config, task metrics, matched control, diagnostic data, native command/config examples and validation. Absolute paths in recorded configs refer to external research assets. Checkpoint/data files are not included in Git.

```bash
PYTHONPATH=. .venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.navigation_motor \
  --config docs/motion2scene/evidence/navigation-motor-20260913/fit-config.json \
  --output /path/to/new-navigation-fit

.venv_isaaclab/bin/python -m pytest -q \
  decoupled_wbc/tests/test_navigation_motor.py \
  decoupled_wbc/tests/test_motor_recovery.py
```

The broader affected suite passes **95 tests**. Ruff and Black pass across the distillation module and required teacher helpers. All 16 native task evaluations completed successfully as software runs; eight goal/map policy attempts failed their task objective. Those outcomes remain distinct from technical failures. The [contribution report](MOTOR_TO_NAVIGATION_CONTRIBUTION_20260913.md) explains how this result revises the next research priorities.
