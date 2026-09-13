# Direct scene/navigation distillation: next bounded experiment

Status: prepared research protocol; direct context transformer and its native rollout interface are not yet implemented or trained. The completed 40,000-update transformer is the motion-command baseline. Its navigation mask is not a scene/goal policy.

## Question and decision

Can one masked student use paired motion/scene/navigation annotations to execute a task with only measured robot history, a requested destination, and scene geometry? A separate planner/director is optional. Our primary experiment is direct conditioning, as described in [the agreed design](DIRECT_MASKED_SCENE_NAVIGATION_20260913.md).

BFM's masked online CVAE distillation motivates the shared command interface ([BFM](https://arxiv.org/abs/2509.13780)). MaskedMimic supplies a related precedent for conditioning physics-based character control on partial motion and object descriptions ([MaskedMimic](https://arxiv.org/abs/2409.14393)); transferring that idea to this robot/teacher is the experiment, not an established result. DAgger motivates querying the expert on student-visited states ([DAgger](https://arxiv.org/abs/1011.0686)); our tracking teacher's competence still depends on the requested continuation.

Keep round 5 as the latest baseline and round 4 as a checkpoint comparison. Do not freeze either as a proven navigation motor. The immediate bottleneck includes poor full-command closed-loop tracking. Context distillation can proceed as a research baseline while that bottleneck is investigated; it does not require declaring the current motor reliable.

## Data contract

| Stream | Recorded contents | Public actor access |
|---|---|---|
| Robot state | Existing normalized 930D measured history; measured anchor pose for frame conversion; episode reset | Yes, sensed at each decision |
| Navigation | Requested start and goal poses in the registered map, optional terminal heading, arrival tolerance, required hold duration, terminal speed bound | Yes; recompute relative quantities from current measured pose |
| Scene | Exact geometry asset/hash, map frame, obstacle pose and full dimensions, shape, floor geometry, map bounds/completeness | Yes in known-map mode |
| Visibility | Object presence/validity separately from observed/unknown status, timestamp and sensor frame | Explicit known-map status now; camera-derived belief later |
| Detailed motion | 79D commands and availability | Only in full or partial command modes |
| Optional route | Ordered externally requested path with a separate availability mask | Only in an explicitly named path-input mode |
| Supervision | Same-state teacher actions/tokens, privileged critic state and future reference | Actions/tokens are targets; critic/future are posterior-only |
| Provenance | Original motion family, split, scene hash, teacher/student hash, interventions, termination and qualification evidence | Metadata only; never motion IDs, reference phase or scene IDs as actor features |

Initial implementation: retain existing history/motion tokens and add projected navigation tokens and a padded obstacle set (15 geometric features per object, explicit validity/observed masks). Encode geometry relative to the measured anchor at every control step, including obstacle rotations. Use the entire bounded small scene for this pilot; record an explicit overflow error rather than silently dropping obstacles. Floor height alone cannot describe a low beam: preserve vertical obstacle extent and free space below it. Add a scene-summary token carrying map bounds and observation completeness.

Do not relabel existing normalized history as a known world pose. Current foundation NPZ files contain no explicit measured anchor pose, start/goal, or scene fields. Recollect or recover those only from verified synchronized native traces before forming body-relative labels. Reference position is not measured robot position after tracking drift.

Existing per-motion paired annotations are valid starting conditions. Distinguish (1) physically executed scene/task demonstrations, (2) scene-compatible geometry/reference candidates, and (3) exploratory same-state expert queries. Tier 2 supports nominal context pretraining and weaker trajectory/compatibility objectives, but does not prove collision-free action labels. Changed goals or obstructed routes need matching executed continuations; never copy the unchanged reference's action labels onto an incompatible task.

## Model and masks

`history + masked detailed commands + navigation + scene -> transformer prior -> behavior latent -> frozen SONIC decoder -> actions`

The posterior additionally sees privileged state and future reference. Keep the direct context path trainable; initialize shared weights from the chosen motion baseline, zero-initialize the new context contribution, and first verify that full-command outputs are preserved before adaptation. A weight migration must check every shared tensor and explicitly list new parameters. The initial static known-map experiment can use existing history; add recurrent scene memory in a separate comparison for partial observability rather than conflating it with the first context result.

Train masks at the episode level (and keep the selected mode during collection):

- 30% full motion commands plus navigation/scene.
- 20% partial root/posture commands plus navigation/scene.
- 50% navigation/scene only, with all detailed commands, future trajectory, phase and IDs hidden from the actor.

Required goal and map information remain available in the principal context mode. A missing-map task is a separate observation condition. Teacher action imitation is primary; retain posterior reconstruction, KL and full-mode token supervision as baseline losses. Test whether these terms cause a prior/posterior bottleneck before adding a larger generative model. Sampling a new independent behavior latent every tick is not the initial navigation execution policy.

## Three matched arms and compute bounds

A. Existing motion-command transformer, continued with the same fresh demonstrations and same-state query budget.

B. Direct context transformer, always given full detailed commands during fitting: tests whether merely appending context is enough. Its context-only evaluation is deliberately an unseen mask condition.

C. Same direct context transformer with the 30/20/50 masking recipe above: principal proposed method.

Use the same frozen teacher, train motion/scene partitions, initial shared weights and action-label provenance across arms. Each arm: 10,000 initial optimizer updates, then three DAgger rounds with 2,000 updates each; three training seeds (91320, 91321, 91322), batch 128, AdamW 1e-4, FP32. Total planned maximum: 144,000 updates across nine fits/chains. Keep full-command and context-only collection quotas explicit for C; same interaction ceiling for all arms (600 ticks per episode, 89 train motions maximum per round). Record actual valid rows, reset transitions and interventions rather than treating the ceiling as useful data volume. Set a six-hour total wall cap and stop on nonfinite losses, provenance violations or evaluator failures. These are planned bounds, not a launched training job.

First run a 32-update shape/gradient smoke and a small physical pilot. For the existing 20 corrected synthetic scene candidates from five source motions, hold out complete source-motion families before fitting and keep scene variants of one motion in one partition. This is only a pipeline pilot; expand the number of independent motion/scene families before making generalization claims. Retain the existing 20 development motions as a separately reported, previously inspected benchmark, not a pristine test set.

## Improve motor and DAgger coverage in the same research cycle

Only 21 complete nominal teacher episodes were originally qualified. Supplement with separately labeled, locally supported teacher prefixes across the broader train set. Preserve strict whole-motion qualification as its own metric; do not loosen it silently. Audit support by motion, temporal quarter, locomotion type and teacher-intervention level. Uniform sampling across episodes does not repair missing late-motion or recovery states.

Compare the current query screen with expert-validated recovery segments. At drift states, run a bounded teacher takeover and record whether it actually recovers in the requested scene. Maintain unsupported/failure examples for analysis instead of turning them into action targets. Use a mixture of nominal starts and physically valid recorded intermediate states to gain late-motion coverage, with history burn-in and reset validation. A fixed reference teacher can supply local recoveries along a compatible task; arbitrary new-goal routing needs a matching expert continuation.

Before a larger model or residual, test a direct action-head diagnostic on the same full-command data against the frozen-token-decoder route. This distinguishes a motor representation/quantization bottleneck from context information and coverage limits. It is an ablation, not a replacement of the intended SONIC-based architecture. Current decoder parity is good; that verifies wiring, not the adequacy of the learned token predictor.

## Evaluator and go/no-go evidence

Maintain two separate scorers:

1. Reference tracking: original completion, duration progress, joint/body error and uncensored duration. Keep this for BFM comparability.
2. Scene task: fixed task horizon, goal arrival, stable hold, terminal speed/heading, fall and undesired contacts, clearance, route efficiency and timeout. Do not terminate solely because a valid task solution differs from the reference pose. Count failures in the denominator and distinguish support foot contact from obstacle/body contact.

Start with the same recorded task tolerances as the current scene qualification files. Add native scene-task termination before claiming navigation rollout success; do not simply rename the tracking scorer. Existing corrected teacher probes are 0/4 fully qualified: three reach the goal but fail terminal hold; the beam case also contacts the obstacle. Appending a stationary reference pose is not evidence of a physically stable stop. Repair or synthesize a demonstrated deceleration/standing continuation, execute it, and requalify.

For causal context tests, use the same initial robot state with different requested goals and the same start/goal with obstacles requiring different routes or postures. Correct context should improve task success against hidden/swapped context on held-out families. An action change under shuffled context alone is insufficient evidence. Do not require exact teacher joint trajectories when multiple routes/gaits are valid.

Required implementation checks: actor outputs invariant to hidden reference/ID/phase changes; finite masked geometry handling; correct moving-frame transforms and rotations; attention padding does not treat unknown space as free; reset/history handling; same-state teacher/decoder parity; exact checkpoint/schema roundtrip; scorer fixtures for arrival-without-hold, contact, timeout and valid alternate motion. Then run bounded native pilot episodes before the larger three-seed study.

Residual learning remains a later controlled ablation: zero-initialized bounded action residual (initial limit 0.05), nominal replay and explicit magnitude/smoothness penalties. Enable it only after demonstrating a reproducible task-control baseline so its effect can be distinguished from basic motor failure. Camera extension replaces known-map tokens with a timestamped visibility-aware belief encoder and memory; goal coordinates still require registration/localization.


## Follow-up experiment completed

The [flow, DAgger, residual and direct-context pilot](FLOW_DISTILLATION_RESULTS_20260913.md) has now completed, including an independent scene-task evaluator and a context-conditioned BFM that preserves the SONIC decoder. The [research review](FLOW_DISTILLATION_RESEARCH_20260913.md) explains the implementation and paper comparisons. This bounded pilot is distinct from the larger three-seed study proposed above; the new results identify missing successful stop/obstacle supervision before that study should be scaled.
