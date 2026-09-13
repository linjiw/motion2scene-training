# Preserving motor competence while distilling navigation context

Research contribution report and working manuscript argument, September 13, 2026. This document connects the completed [motor recovery study](BFM_MOTOR_RECOVERY_RESULTS_20260913.md) to the new [navigation pilot](NAVIGATION_MOTOR_PILOT_20260913.md), incorporating the framework recommendations in [distill-research.md](../../distill-research.md). It is an engineering research report, not a claim of established novelty or a completed navigation paper.

## Working abstract

Distilling a privileged humanoid tracker into a policy with reduced task information can fail even when the teacher and action decoder are individually competent. We study this failure in a SONIC-based G1 control stack using a fixed repaired-motion teacher, native termination conditions, and a frozen motor decoder. Controls separate training-label support, public command completeness, preservation of the teacher's reference encoder, and online teacher supervision at student-visited states. A student that predicts missing desired-reference context from measured history and current commands reaches 77/89 training-motion completions; online DAgger increases this to 88/89, compared with 5/89 for the historical 79D command student. The selected model completes 10/20 development motions on three evaluation seeds, leaving a gap to the teacher's 15/20 on the original seed. We then collect eight physically successful goal-hold/corridor demonstrations and implement a goal/map-only sibling policy through the unchanged motor model. The resulting study distinguishes motor transfer, task abstraction and observation transfer, and provides executable contracts for testing navigation without sacrificing the existing motor specialist.

## Contribution and supporting evidence

| Contribution | Implemented mechanism | Evidence and scope |
|---|---|---|
| Diagnose a failed distillation setup | Matched support/loss controls, current-target extension, native teacher-horizon probes | More rows and full-only fitting alone remain weak; correct anticipation and preserved representation provide the largest gains |
| Preserve pretrained motor structure | Frozen teacher encoder, FSQ and action decoder; predict missing reference frames | Tensor-exact weight retention, teacher-token/action parity, native full-command evaluations |
| Improve online coverage | Same-state teacher queries, randomized starts, explicit burn-in/takeovers, fresh/replay mixture | 3,276,800 main transitions across two bounded stages; separate unassisted checkpoint evaluations |
| Separate supervision evidence | Hash-bound task-success adapter with preserved decision indices and lineage | Task positives require actual matching task execution; generic prefix eligibility is insufficient |
| Make navigation input separation executable | Typed history/goal/map actor view and a separate trainable command-completion branch | Tests exclude reference inputs; training cannot update the motor specialist; native goal-only pilot |

The result supports an effective adaptation of a pretrained SONIC motor path. It does not show that a larger generic transformer, flow objective, residual PPO, or arbitrary-mask BFM has been solved. The 114D student has a richer current-command interface and more preserved pretrained structure than the historical 79D baseline. Those differences are explicitly part of the method, rather than hidden confounds in an equal-architecture claim.

## Method and design decisions

Let `h_t` denote measured proprioceptive history, `c_t` the current detailed target, `g_t` the body-frame task request, and `S_t` the available geometry. The teacher uses a ten-frame reference `r_t` and produces actions through a reference encoder `E`, quantizer `Q`, and history-conditioned decoder `D`.

The successful motor specialist predicts a desired reference:

```text
r_hat_t = F_theta(h_t, c_t)
a_t = D(Q(E(pack_native(r_hat_t))), h_t)
```

`E`, `Q` and `D` remain fixed. The current physical target frame is fixed by the command; `F_theta` predicts residuals for the remaining nine frames. Native packing and physical frame order are separately defined and tested. The actor receives current target quantities, not future labels or a reference clock. Future-reference targets appear only in training. This is desired-reference anticipation, not an executed-state dynamics model.

The first navigation comparator predicts the missing command internally:

```text
c_hat_t = I_phi(h_t, g_t, SetEncoder(S_t))
a_t = D(Q(E(pack_native(F_theta(h_t, c_hat_t)))), h_t)
```

Only `I_phi` and its obstacle encoder train. The public API contains no detailed commands; the complete motor specialist remains unchanged. The internal 114D command interface provides an inspectable comparator with greater posture expressivity than a four-command director. It is not claimed to be the final architecture. A direct context-to-reference or context-to-token prior remains an important comparison if this intermediate representation limits task learning.

This updates the earlier report's default suggestion to initialize from `ContextTokenFoundation`: that prototype remains a historical baseline, while the newly validated anticipatory motor is the stronger preservation anchor. The overall recommendation—shared data/runtime infrastructure with specialized deployment profiles—remains intact.

BFM motivates masked low-level behavior specifications and online distillation, but its published interface and learned decoder differ from this adaptation. [BFM, Sections IV–V](https://arxiv.org/html/2509.13780v1). BeyondMimic motivates modeling future state/action trajectories so that task objectives can guide generation; our current desired-reference forecaster does not yet provide calibrated executed-state predictions for such guidance. [BeyondMimic, v4](https://arxiv.org/html/2508.08241v4).

## What the negative results teach

The original 79D online student improved to 14/89 on its selection seed, but repeated at 8/89 and 6/89. Offline support expansion alone improved 5/89 to 6/89. These findings argue against simply increasing repeated fitting on the same representation. The corrected 114D transformer reached 21/89; preserving and completing the reference representation reached 77/89 before online refinement.

The initial new reference parser had a physical/native packing error. Early affected probes are rejected and retained as failed experimental work. The historical 79D student was unaffected. The audit is a contribution to implementation reliability, not an explanation falsely assigned to every earlier failure.

The motor result also does not guarantee task abstraction. Removing current target pose, velocity and orientation removes information that can distinguish multiple behaviors under the same goal. Navigation therefore requires learning the deployed input mode on matching task demonstrations and correcting its own visited states. A decreasing offline action loss is insufficient evidence of that capability.

The new matched pilot confirms this distinction: the preserved full-command student passes all eight stopping/corridor tasks; the goal/map command-completion branch passes none, despite entering five goals without falls or prohibited contacts. Its best holds are 42 and 38 of 50 required ticks. The next research target is the public task-conditioning and visited-state supervision path, with causal context history for arrival-speed estimation as a concrete observation improvement.

The successful stopping dataset contains **four distinct clips, eight scene executions, but one recorded ancestry group**. Earlier shorthand describing four source families was too strong. Clear/corridor replicas cannot create independent families, and a row split would exaggerate generalization. These data support an in-sample learning test and positive teacher controls. New ancestry/task families are required for a stronger evaluation.

## Next discriminating experiments

1. **Terminal behavior and local recovery.** Retain full successful approach/turn/deceleration/hold attempts, continue teacher recording beyond the minimum success instant in a separately declared collection mode, and collect navigation-student states near arrival. Keep evaluation at the original 50-tick criterion. Separate failure to enter the goal, excessive arrival speed, leaving the goal, prohibited contact and falls. Measure teacher support before turning queries into positive recovery targets.
2. **Short navigation DAgger cycles.** Freeze the driver for each chunk, capture synchronized teacher/student/executed actions, preserve decision indices and continuation identity, and fit a declared fresh/task replay mixture. A compatible tracking continuation supplies labels only within its tested support. Teacher-assisted progress and independent student success must remain separate metrics. Start with the existing single-scene runtime; validate isolated scene cloning before scaling environment counts.
3. **Representation comparison.** Compare the implemented detailed-command completion branch with a direct desired-reference branch through the same fixed encoder/decoder, using identical task episodes, queried states and interaction budgets. Hold the complete full-command specialist fixed in both. Add a direct token prior if reference prediction is the measured bottleneck.
4. **Context dependence and task families.** Acquire same-start/goal pairs requiring different routes or postures when obstacles change, and changed-goal pairs with compatible continuations. Include irrelevant-obstacle variants. Track recorded ancestry, task family and repair lineage; no development motion should migrate into training because it failed evaluation.
5. **Flow only after the right targets exist.** Compare deterministic prediction with persistent, receding-horizon reference flow when multiple valid continuations exist for one context. For physical-state guidance, store contiguous actual transitions and calibrate prediction error/clearance against execution. Do not interpret reference clearance as executed contact safety.
6. **Observation transfer later.** Introduce a sensor profile with timestamps, visibility, unknown-space semantics and memory after known-map task control works. Exact map geometry becomes training-only supervision in that profile. Camera support is not implemented by adding noise to a complete-map token.

Navigation retention is currently guaranteed by architectural separation: the navigation optimizer cannot change the complete motor path. If later shared-layer adaptation is proposed, add explicit full-command replay and action-preservation tests, then re-run native tracking. Residual action learning is a later controlled option, not a replacement for missing successful stopping/avoidance labels.

## Reproducibility and reporting

Code, tests, compact metrics and research reports are committed; large checkpoints, repaired motions, scenes and raw traces remain in their existing research-data packets. The evidence bundles record hashes and local provenance without pretending the assets are downloaded with a Git clone. The [motor results](BFM_MOTOR_RECOVERY_RESULTS_20260913.md) contain the selected-checkpoint evaluation command. The navigation pilot report records its own bounded budget, config, failures and public-input contract.

The current claim is strong motor recovery in this bounded simulation study plus an executable, evaluated navigation extension. A publishable navigation contribution still needs task-generalization evidence, matched intervention budgets and causal responses to valid changed-goal/obstacle pairs.
