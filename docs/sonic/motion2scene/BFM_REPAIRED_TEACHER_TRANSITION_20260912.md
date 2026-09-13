# Teacher stopping decision and repaired BFM preparation

## Teacher result and stopping decision

The 4096-environment teacher was stopped at observed iteration **1984**, preserving checkpoint **1900**. It did not finish its planned 8000 iterations. We stop this recipe for practical research allocation, not because mathematical convergence or monotonic deterioration has been established.

| Checkpoint | Train completion | Development completion | Train strict-qualified | Development strict-qualified |
|---|---:|---:|---:|---:|
| Released baseline | 83/89 | 17/20 | 10/89 | 2/20 |
| Current 700 | 84/89 | 15/20 | not audited here | not audited here |
| Current 1400 | 88/89 | 14/20 | 28/89 | 2/20 |
| Current 1900 | 85/89 | 16/20 | 19/89 | 0/20 |

All completion comparisons use the same repaired 109 IDs, seed 91260 and native evaluation criteria. Strict qualification additionally requires whole-episode mean tracked-body error <=0.10 m and maximum pelvis XY error <=0.25 m. These screens do not certify scene/contact/navigation behavior. The strict audit uses saved whole-episode arrays only for nonterminated clips. Failed clips cannot qualify regardless of their reset-tail statistics.

Step 1900 **did improve completion over 1400**, so it would be incorrect to claim no improvement in every metric. However, it remains below the released baseline in development completion and loses strict tracking-quality coverage, which is the relevant requirement for distillation labels. We retain release as the completion baseline and select 1400 as a **foundation-label candidate** for its better strict training coverage. This is an explicit multi-metric choice, not a claim that 1400 is universally the best teacher. Development strict counts are very small and do not establish generalization.

Latest evaluation evidence: ../m2s-repaired-teacher-4096-step1900-20260912/; stop receipts: ../m2s-sonic-repaired-teacher-4096env-20260912/USER-STOPPED.json.

## Materialized preparation

- Immutable checkpoint-1400 teacher and matching config; hash afd649cfbbfd28833550e11a0f8c3b7a5f6a05ee8b4021dd0dac97a6f94733ce.
- Fresh FP32 collection of all 89 repaired training clips, retaining failures and exclusions. No development motion is collected for training.
- **21 whole-qualified episodes / 6433 labels**, filtered with the unchanged 0.10 m body / 0.25 m root criteria. The collection has 89 environments and explicitly FP32 inference, unlike the 109-environment diagnostic; its exact qualification count differs from the earlier 28. Train exclusively on this new collection's verified 21 episodes.
- Same-state teacher action/token/proprioception alignment, privileged posterior fields, 79-value commands with masks, complete recorded phase quarters, and per-file hashes. Every admitted row passed frozen-decoder replay (maximum absolute action difference 1.526e-5, tolerance 2e-4 plus relative 2e-4).
- Ready 3000-update foundation configuration: fresh student, batch 128, learning rate 3e-4, posterior KL weight .001, direct public-prior action weight 1, phase-balanced sampling, staged full/root -> navigation/mixed command masking. No old student or old-teacher labels are reused. The higher prior weight and curriculum are proposals, not proven improvements.
- A **32-update GPU smoke** completed, validating the training path and decoder bindings. This is not physical student qualification. The 3000-update fit has **not** been launched.
- Twenty schema/hash-validated task proposals from five qualified training motions: clear, corridor, low-beam and changed-goal variants. Each binds start/goal, robot frame, repaired reference, a separately exported 1.2-second terminal-pose hold, collision USD and obstacle encoding. These are synthetic research tasks, not outputs certified by the learned scene generator. Existing generator scenes remain a separate source for later repaired-motion requalification.
- Prepared native scene qualification commands, plus command-only and bounded action-residual navigation configuration templates. Both navigation preflights correctly report missing trained/qualified foundation and scene labels. No fake hashes or successful navigation receipts are supplied.
- Existing foundation/navigation/coverage tests: **49 passed**. All 20 task records pass actual task schema and file binding checks.

## Navigation/traversal design and execution order

1. Fit and physically evaluate the public foundation under full, root and navigation command masks. Keep the privileged future/reference only in the posterior or explicit full-command teacher interface. Evaluate the 20 development motions without training on them. Test command changes, stopping, turning and height control; MSE alone cannot qualify control.
2. For navigation, the recurrent actor consumes measured robot history, start/goal in the measured robot frame, known obstacle geometry and masks. It predicts bounded body velocity, yaw rate and height commands. Freeze the foundation and its checkpoint-matched SONIC decoder. Future trajectory, phase and motion ID remain training-only; command ranges measured from labels are not certified physical limits.
3. Physically qualify the clear/scene continuations, goal arrival and a 50-tick terminal hold, retaining 200 Hz contact evidence. Corridors/low beams and the appended hold require new execution. A changed-goal reference is intentionally an unqualified candidate and cannot label that new goal until the expert supplies and executes a matching continuation.
4. Start navigation imitation with qualified nominal teacher rollouts. Add DAgger only when student command control is qualified; labels from student-drift states need an explicit supported recovery contract. A scene-independent tracking teacher is not by itself a route planner.
5. Compare command-only against a zero-initialized bounded **action residual** (0.05 native action units per coordinate, magnitude and temporal penalties 0.1), with no simultaneous latent residual. These are experiment bounds, not hardware safety limits. Residual PPO follows successful imitation/physical qualification; it is not launched here.
6. Low-beam traversal may require posture/keypoint/skill-conditioned motor commands beyond the four navigation controls. The 79-value foundation interface can represent posture targets, but the current navigation head emits only four values. Treat low-beam failures as an interface/expert question before assuming that a larger navigation network or longer training will solve them. Do not claim a crawling/traversal controller exists.

Use matched changed-map and changed-goal evaluations, falls, contacts, goal/hold completion and failure-censored tracking errors for promotion. These development motions have been inspected repeatedly; reserve fresh motion/scene groups for a final research generalization claim.

## Commands and artifacts

From /home/linjiw/groot-wbc-sonic-sim-trackb:

```bash
.venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.preflight /home/linjiw/research-data/m2s-bfm-repaired-navigation-prep-20260912/foundation-config.json
.venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.train --config /home/linjiw/research-data/m2s-bfm-repaired-navigation-prep-20260912/foundation-config.json --output /home/linjiw/research-data/m2s-bfm-repaired-navigation-prep-20260912/foundation
```

Key evidence: collection/collection.json (all attempts), qualified-collection.json (admitted episodes), parity.json, foundation-preflight.json, smoke/receipt.json, command-profile.json, scene-tasks/manifest.json, scene-qualification-commands.json, navigation-preflight.json. Scripts retain absolute paths and are preparation records, not portable CLI replacements. The incomplete scene preparation is preserved separately and excluded from the active manifest.
