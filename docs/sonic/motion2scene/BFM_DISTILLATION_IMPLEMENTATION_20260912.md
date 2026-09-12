> 2026-09-12 update: [100-motion experiment, scene implementation, and trajectory-role correction](BFM_SCENE_NAVIGATION_COVERAGE_20260912.md). The final student is not navigation-qualified; use the corrected video/scene receipts linked there.

# BFM distillation implementation and first training rounds

September 12, 2026. Follow-up to [teacher evaluation](SONIC_TRACKING_EVALUATION_20260912.md).

The BFM-style training path now runs against the native SONIC teacher. Two teacher episodes were collected, a foundation was fitted, its public prior was evaluated in physics, and one student-state DAgger round was collected and fitted. The scene/navigation adapter, recurrent imitation trainer, bounded action residual and live residual-PPO loop are implemented and tested at their available boundaries. **Physical scene-navigation and residual-PPO training have not started:** scene-teacher labels and a command-qualified foundation are still missing.

The full architecture, interfaces, objectives, reward definition, runtime commands and gates are in [TRAINING_PIPELINE.md](../../gear_sonic/research/scene_distillation/TRAINING_PIPELINE.md).

## What changed

- A 79-value masked root/keypoint/joint foundation interface, with a scene-independent public prior, privileged posterior and SONIC token adapter.
- Native teacher and student-state collectors with same-decision encoder/decoder hooks, checkpoint binding, failure retention and explicit distinctions between executed labels and exploratory queries.
- Hash-bound dataset aggregation; episode-balanced fitting; separate weights-only initialization and exact offline optimizer/RNG continuation.
- Recurrent scene/navigation fitting through the frozen foundation and decoder, with complete public-prefix reconstruction and qualified-query masking.
- Zero-initialized action residuals bounded per native action coordinate; magnitude/smoothness penalties; optional latent residual retained separately.
- Bounded tanh-Gaussian residual PPO, GAE/reset handling, KL early stopping, physical rollout accounting and checkpointing. The frozen base policy is tested for exact preservation during residual updates.
- A single-environment native scene adapter, exact-task qualified continuation registry, same-state query ledger checks, pre-reset goal/contact reward measurements, and launch/readiness checks. A general route planner, batched articulated scene training and automatic physical qualification of new routes are not supplied by this implementation.

## Executed experiment

Packet: `/home/linjiw/research-data/m2s-bfm-pilot-20260912`.

Teacher checkpoint SHA-256: `48b3a1c04cdbbcd9ffe8ad10b2591aff781c253c03c61ccb8683348d862c54ed` (the final 8,000-additional-iteration teacher). Its weights were not changed.

An explicit posthoc engineering screen of the previous **training** evaluation found only motions `00185` and `00490` with native completion, mean global body error ≤0.10 m and maximum root XY error ≤0.25 m. This tiny support set is suitable for an integration pilot, not a reusable-foundation claim. No development or reserved-layout data was fitted.

Fresh native teacher collection reproduced completion on both motions:

| Motion | Retained teacher rows | Mean global body error | Maximum root XY error |
|---|---:|---:|---:|
| 00185 | 248 | 92.2 mm | 192.5 mm |
| 00490 | 298 | 98.6 mm | 117.2 mm |

There were 299 control calls across two environments: 598 observed environment transitions in the collection callback. The two episodes provide 546 retained training examples after the native end-frame convention. The collection scope excludes simulator initialization. These are obstacle-free tracking screens, not environment-contact/scene qualifications.

Initial fitting used five command-mask profiles, batch 32, learning rate 0.0003, posterior reconstruction + 0.001 KL + 0.1 public-prior action MSE. The surviving initial checkpoint contains 192 optimizer updates from a fresh model. Its final sampled prior action MSE was 0.01318.

The first public-prior evaluation failed both motions under both full and four-command masks. The student-state collector then executed 38 control calls / 76 environment transitions and retained 58 queries within the declared numerical support screen (32 and 26). Both failed episodes remain in the ledger. These labels are teacher predictions on the student's actual states, **not demonstrated recovery actions**. The fitting config explicitly enables exploratory queries.

The aggregate contains four episodes and 604 retained examples. One new, bounded 200-update round initialized from the initial foundation weights, with a fresh optimizer. Its final sampled prior action MSE was 0.003657. These are training-batch losses on changing datasets, not held-out performance estimates.

## Public-prior physical results

Each evaluation used the native scorer, the same two training motions, deterministic prior means, native teacher decoder and fixed full/native-four-command profile. Neither evaluation used the privileged posterior. The four-command profile receives explicit reference-derived velocity/yaw-rate/height commands; **it is not scene/goal navigation**.

| Motion | Initial full-command progress | After DAgger full-command progress | Initial four-command progress | After DAgger four-command progress |
|---|---:|---:|---:|---:|
| 00185 | 15.3%, failed | 26.5%, failed | 14.9%, failed | 26.1%, failed |
| 00490 | 8.0%, failed | 46.8%, failed | 8.7%, failed | 100%, complete |

Four-command completion improved from 0/2 to 1/2; full-command completion remained 0/2. This supports continuing a bounded on-policy experiment. It does not qualify command expressibility or establish generalization. Native completion still lacks the contact, drift and terminal-hold guarantees needed for scene teaching. No statistical significance claim is appropriate for two training motions and one execution each.

Current selected experimental checkpoint: `foundation-round-1-v2/step-000200.pt`. Its weights descend from 192 initial updates plus 200 round-1 updates; the filename counts updates within this fresh-optimizer round. This is an experimental checkpoint, not a deployed or qualified BFM.

## Failure and cost accounting

All failed attempts were preserved:

1. The first offline attempt rejected teacher/decoder parity before an optimizer update. Its historical receipt incorrectly counted the attempted step as an update; `decoder-numerics.json` records the correction to zero. The trainer now increments its counter only after `optimizer.step()`.
2. A first numerical amendment consumed eight updates before encountering a larger discrepancy in the second episode. Inspection of **all** rows found a maximum TF32/full-precision discrepancy of 0.003684 native units. The audited 0.0045 absolute + 0.0002 relative tolerance passed all 604 teacher/queried rows before round-1 fitting. No physics replay was used to replace labels.
3. The successful initial fit consumed the remaining 192 updates of its 200-update budget. The eight earlier updates are consumed cost and are not ancestors of this checkpoint.
4. The first mixed-source round-1 fit rejected optional query-mask batching before an update. The sampler now selects explicit training fields, and the regression test combines teacher and queried episodes. Its replacement consumed the separately locked 200-update round-1 budget.

Total observed optimizer cost: **400 updates** (8 discarded + 192 initial + 200 round 1). No optimizer update took place during evaluation or collection. No unbounded job or background training loop was launched. The initial checkpoint lacks CUDA RNG persistence, so round 1 is explicitly weights-only initialization. New checkpoints record CUDA RNG; exact offline continuation is tested against uninterrupted CPU fitting. Simulator-state continuation remains unsupported.

## Scene/navigation and residual readiness

`navigation-training-proposed.json` enables action residuals at 0.05 native units per joint, latent residual scale zero, sequence length 16, and residual magnitude/smoothness weights 0.1. The command ranges in that proposal are measured extrema of the two teacher episodes; they are not qualified controllability limits. `navigation-readiness.json` reports:

- Missing executed scene-label manifest.
- Missing public-prior command qualification.

Those conditions prevent launch; elapsed time or a low loss cannot satisfy them. The teacher-only scene collector can run once an exact scene/task/continuation has genuine physical qualification. Student-driven scene collection additionally requires the public-prior command gate. The finite task registry deliberately rejects unqualified changed-scene or changed-goal tasks rather than inventing a route or teacher action.

The next substantial training experiment should expand reliable motion support and run explicitly bounded foundation/DAgger rounds, then qualify the intended command profiles. After that: qualify discriminating scene continuations, collect teacher-only labels, fit the scene/goal head and action residual, collect same-state DAgger labels, and run residual PPO against the declared physical reward. Compare against command-only under matched budgets. The current 240 geometry proposals still do not contain executed scene labels, and the eighteen reserved layouts remain unopened.

## Validation and reproduction

All 64 focused tests passed; Black and Ruff E/F/I checks passed. The packet's source archive is a post-run snapshot. A complete contemporaneous source snapshot was not captured for every initial pilot attempt; do not treat the final snapshot hash as an immutable launch-version record for those earlier attempts.

Native commands and stage configurations are preserved in the packet: teacher collection, initial/full/four-command evaluations, DAgger collection, aggregation and round-1 fitting/evaluation. The native collector/evaluator processes exited 0. Check the per-stage exit files and `metrics/metrics_eval.json`; an exit code alone is not tracking success.

```bash
.venv_isaaclab/bin/python -m pytest -q decoupled_wbc/tests/test_bfm_pipeline.py decoupled_wbc/tests/test_foundation_navigation.py decoupled_wbc/tests/test_scene_distillation.py decoupled_wbc/tests/test_hindsight_training.py decoupled_wbc/tests/test_hindsight_long_tracking.py decoupled_wbc/tests/test_hindsight_qualification.py
.venv_research/bin/python -m black --check gear_sonic/research/scene_distillation decoupled_wbc/tests/test_bfm_pipeline.py
.venv_research/bin/python -m ruff check --select E,F,I gear_sonic/research/scene_distillation decoupled_wbc/tests/test_bfm_pipeline.py
.venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.preflight /home/linjiw/research-data/m2s-bfm-pilot-20260912/navigation-training-proposed.json
```

The last command intentionally exits 2 with the missing-evidence report. Contract tests cover masks, frozen gradients, residual bounds, reset-aware smoothness, GAE boundaries, PPO integration, exact task/goal bindings, exploratory-label gating, mixed-source batches and optimizer continuation. Synthetic residual/scene tests are not physical navigation results.
