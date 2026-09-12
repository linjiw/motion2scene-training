# Research guide: motion → scene → teacher → student

## Scientific objective

Motion2Scene asks whether a physically executable humanoid motion identifies a useful distribution of environmental constraints, and whether those constraints can produce training data for scene-conditioned control. A compatible scene is easy to construct: an empty room is compatible with many motions. A useful inverse model should identify when a behavior is necessary or preferable to a meaningful alternative, and that preference must survive physical execution.

Our downstream research asks whether a compact student can use public scene and navigation requests to reproduce a composite expert's useful behavior, then improve task performance through bounded residual learning. This is not yet demonstrated. The [paper draft](../paper/main.pdf) reports the current exploratory evidence and negative results.

## Read in this order

1. [README](../README.md): package contents and supported workflows.
2. [Setup](SETUP.md) and [readiness audit](READINESS_AUDIT.md): environment and evidence limits.
3. [Dataset coverage](DATASETS.md): what is bundled and what remains external.
4. [Design map](WORKFLOWS.md): modules and historical plans.
5. [Research runbook](RESEARCH_RUNBOOK.md): fresh fits, continuation and evaluation boundaries.
6. [Paper source and evidence map](../paper/README.md): claims supported by the current record.

## Research layers and interfaces

| Layer | Inputs | Outputs | Evidence needed before downstream use |
|---|---|---|---|
| Reference data | root/joints, frame timing, motion group | canonical motion and repair ledger | finite/native loader checks and split integrity |
| Inverse scene proposals | body-aware motion summaries, alternatives | candidate shapes/poses and probabilities | geometric predicates with defined uncertainty |
| Physical qualification | teacher, exact reference, exact scene/task | trajectory, contacts, completion/hold receipt | all declared task/contact gates pass |
| Foundation distillation | same-state teacher labels, masked commands | token/prior model for a frozen decoder | teacher/decoder parity and standalone physical checks |
| Scene-conditioned distillation | public history, start/goal, obstacle map | bounded foundation commands and optional residual | qualified composite-expert continuation data |
| Residual task learning | qualified base plus public observations | bounded action corrections | matched-budget task gains without hidden teacher intervention |

A failed layer does not produce qualified data for the next layer merely because the relevant file exists.

## Representations

**Motion:** preserve the original frame rate, root frame, native joint order and split/group identity. A repaired pose stream and a successful policy rollout are different artifacts. Keep original, repaired and excluded candidates separately.

**Obstacle:** store primitive type, dimensions/radius, world pose, body-relative encoding, availability mask and exact collision-scene binding. Scene proposal geometry and the collision USD used by Isaac must refer to the same intended objects. Units are metres; the current task schema uses world-z-up and wxyz quaternions.

**Task:** store start, goal, deadline, stopping radius/speed, required hold duration and scene hash. Current tasks provide a known map; do not describe them as perception-based navigation through unknown obstacles. Future trajectory/reference information is a training label, not an implicit public actor input.

**Foundation command:** the 79-component interface includes root signals, 14 body targets and 29 joint targets. Missing values are indicated by masks. Root/heading and sparse-command modes expose fewer components. Four sparse commands do not uniquely determine gait, arm posture or detailed reference phase.

**Actor:** current public navigation observation has proprioceptive history, body-frame start/goal and up to five obstacle encodings. The recurrent actor must not consume motion identity, future reference, privileged critic state or a path selected with unavailable information.

**Teacher label:** identify the checkpoint/decoder hash, observed state, selected task, query validity and whether a teacher intervention executed the action. An action from a nominal state is not a valid label for a different perturbed student state.

## Proposed method and current status

### Inverse scene generation

Use motion to generate low-dimensional candidate constraints, then separate geometric compatibility from contrast with alternatives. The implemented hindsight study uses rule-derived anchors and a learned local candidate scorer. The alternatives are geometric probes, not physically qualified counterfactuals. Current contrast supervision is sparse and the hindsight objective did not improve the measured geometric baseline.

The next meaningful experiment is physical preference reversal: a target and a simpler qualified alternative share start/task conditions; a critical obstacle should favor the target, and removing or moving that obstacle should remove/reverse the preference. Report all zero-candidate motions. An analytic label oracle is diagnostic, not a learned baseline available at deployment.

### Tracking teacher

The repaired fit uses 89 screened training clips, 128 environments and a planned 32,000 iterations. The previous final fit used 512 environments and 8,000 iterations. Both budgets target 98,304,000 transitions at 24 rollout steps per environment. Compare actual transitions and wall time, not iteration counts alone.

At the packaged step-6,200 evaluation, the repaired teacher completes 3/20 repaired development motions versus the previous teacher's 11/20 under the same evaluation. This is an interim unequal-exposure result. Do not infer improvement from repaired kinematics alone, or silently replace this baseline with the previous evaluation on unrepaired data.

### Foundation student

The model learns posterior reconstruction, posterior/prior alignment and a public-prior action objective through a frozen SONIC decoder. The deployed path uses the public prior. DAgger adds same-state teacher queries on student-induced states; teacher assistance improves collection coverage but mixed rollouts are not student successes.

A key defect was late-phase starvation. Use the original reference fraction before support-mask filtering, retain complete qualified episodes, and report empty quarters. The implemented quarter sampler balances episodes and occupied quarters; aggregates containing multiple sources/episodes per motion still need explicit motion/source balancing.

The existing full → root → sparse curriculum did not outperform uniform masking in the small matched diagnostic. Keep uniform masking as a baseline and match cumulative profile exposure when testing ordering effects. Evaluate command sensitivity with meaningful same-state expert responses; simply changing commands while retaining the old action label creates invalid supervision.

### Composite scene/navigation expert

A scene-aware guidance module should choose feasible task commands or qualified continuations, and the tracking controller should supply dynamically consistent actions. The current recorded-continuation registry is not a general planner. Collect different goals in one scene and obstacle changes at fixed goals to make public conditioning identifiable.

Two current scene probes fail the qualification conjunction. They therefore supply failure evidence, not qualified scene-navigation demonstrations. Complete scene/student training remains gated on obtaining valid data.

### Residual improvement

Freeze the qualified base initially; bound residual actions per joint and initialize near zero. Optimize progress, stable stopping and collision costs with an imitation anchor. Compare residual-enabled versus residual-disabled policies under equal interaction budgets. Report teacher interventions separately and retain cases where a residual worsens stability.

No physical residual-learning improvement has been measured in the packaged study.

## Evaluation matrix

| Question | Required comparison | Primary outcomes |
|---|---|---|
| Does repair help tracking? | original/repaired references with matched policies, plus matched-budget training | completion, progress, drift, error and failure reason |
| Does inverse learning help? | geometry, hindsight, no-motion-feature and privileged-oracle diagnostics | candidate coverage, clearance/contrast mass; then physical preference |
| Does curriculum help? | same data, initialization, updates and cumulative profile exposure | standalone command tracking and complete durations |
| Does scene input matter? | full input, goal-only, shuffled scene, changed goal/obstacles | arrival, stopping, collision and stability |
| Do residuals help? | frozen base vs bounded residual, matched transitions/seeds | task success and contact costs, not imitation loss alone |
| Does it generalize? | disjoint motion groups and held-out scene families | the same metrics with all attempts in denominator |

Keep exact-reference tracking separate from command following and goal navigation. A controller can match velocity with a different valid gait and fail exact foot-position tracking; conversely, a short low-error surviving prefix does not establish stable navigation.

## Experimental record

For each new experiment, create a new output directory and record:

- question, falsifier and primary metric before running;
- Git commit and any working-tree patch;
- teacher/model/data/scene hashes and original split membership;
- seed set, environment count, rollout horizon, optimizer updates and transition cap;
- actual queries, interventions, executed steps and censored/failed attempts;
- reference and tracked roles, coordinate frames and time bases;
- qualification thresholds, contact sampling rate and valid evaluation duration;
- final checkpoint and full run receipt, with incomplete attempts retained.

Repeated inspection of the existing development set makes it exploratory. Use a separate declared held-out protocol for confirmatory claims. Do not claim statistical significance or independent replication without defining experimental units and the seed/group structure.

## Next milestones

1. Establish a clean isolated native installation and bounded physical smoke.
2. Finish and qualify the repaired teacher; keep the old teacher as the measured baseline until the comparison supports promotion.
3. Collect complete, command-diverse walking/turning/stopping support and qualified recoveries.
4. Validate a sparse motor prior using task-appropriate stability/command metrics.
5. Demonstrate physical scene preference with qualified alternatives.
6. Distill a scene/goal-aware composite expert and test input ablations.
7. Evaluate residual learning under a matched budget.
8. Update the paper with results only after these measurements exist.

Do not reinterpret this list as already completed work. The repository provides a reproducible starting point and an evidence ledger, not a guarantee that the proposed research will succeed.
