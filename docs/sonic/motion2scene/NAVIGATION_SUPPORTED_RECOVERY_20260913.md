# Exact-motor demonstrations and supported navigation recovery

This study follows the [terminal-control audit](NAVIGATION_TERMINAL_CONTROL_20260913.md).
That audit found low teacher-state fitting error, unstable 1/8 task performance,
and different recovery outcomes for late arrival versus position drift. The motor
specialist and the original navigation baseline remain unchanged.

## Hypothesis and bounded experiment

The navigation adapter previously learned on states visited by the original
reference-conditioned teacher, but executes through a different, frozen motor
student. Even a target computed as `M(h, c*)` is evaluated at the supplied history
`h`; original-teacher histories do not necessarily cover the motor student's own
closed-loop state distribution. This motivates an **exact-motor demonstration**
control before claiming that the remaining failures only need more navigation
DAgger or a new architecture.

The fixed study uses the existing eight train tasks, four clips and one ancestry
group. All original task deadlines, 0.25 m 3D goal tolerance, 0.10 m/s 3D speed
limit, 50 consecutive hold ticks and prohibited-contact/fall conditions remain
unchanged. No held-out layout, new motion ancestry or camera claim is tested.

1. Re-execute the frozen full-command motor on all eight tasks and retain its
   actual pre-action observations, reconstructed 114D commands and motor actions.
2. Fit the localized command adapter for 6,000 updates on those demonstrations;
   evaluate all eight tasks with no externally supplied commands.
3. Run that **current** adapter up to a predeclared switch on each task: half the
   original teacher's first-completion tick, rounded down. From the actual state,
   execute the frozen full-command motor with the nominal continuation. No phase
   jump, reference translation or synthetic command modification is performed.
4. Admit only motor-executed suffixes whose own physics trace completes a fresh
   50-tick hold before the original deadline, with no prohibited contact anywhere
   in the attempt. Failed attempts remain recorded but supply no training targets.
5. Fork the same 6,000-update adapter into two 3,000-update continuations: replay
   only, and 50% qualified recovery / 50% motor demonstration replay. Both use
   batch 256, AdamW 1e-4, training seed 91370 and evaluation seed 91260. The optimizer
   is restarted identically in both forks; adapter normalization remains inherited.

Both forks retain the causal localization interface and structured motor objective
from the previous study. The comparison between forks controls extra optimization
budget. It adds recovery-query simulation cost in the recovery condition; that
cost and the number of successful/failed attempts must be reported explicitly.
The comparison to the prior teacher-state pilot also changes the number and
composition of rows, so it is not a perfectly matched data-volume ablation.

## Implemented collection and evidence contract

`MotorRecoveryCollectionCallback` handles two cases: switch tick zero is an
unassisted full-command motor demonstration; a positive switch tick is a
navigation prefix followed by an explicit privileged motor intervention.
The same-state command is reconstructed before each action. The original
teacher is queried separately for action/token quality diagnostics only, after
choosing the executed action and while preserving RNG state.

All eight motor demonstrations complete and reproduce the archived full-command
root, speed and prohibited-contact traces exactly. They provide **3,076 executed
rows**. This verifies that adding diagnostic queries did not change the motor's
physical behavior. Pre-action pose history agrees with the preceding physical
post-step pose to 1e-6 m; small native floating-point recomputation differences
prevent claiming bit identity for that separate timing check.

The new `executed_motor_navigation_recovery_v1` receipt binds the behavior
checkpoint, frozen motor, original teacher, original task, shard and physical
score. The aggregate additionally binds the raw trace. The loader independently
recomputes whole-attempt and suffix scores, enforces the original task allowlist,
checks ancestry and actor profiles, rejects unavailable current commands, verifies
that supported motor actions were actually executed, reconstructs causal features
before masking rows, and rejects changed or duplicate artifacts.

A recovery suffix must earn its **own** 50 consecutive ticks. A hold accumulated
by the navigation prefix cannot silently qualify a shorter motor suffix. The
shared evaluator now has an overridable goal-stop hook; ordinary task evaluation
keeps its existing rule, while recovery collection waits for independently scored
suffix completion or the original deadline/contact/fall stop.

The first admitted suffix row is one verified learner-state query. Later rows are
expert-executed recovery data. They must not be counted as hundreds of additional
learner queries. This study is a bounded intervention-based DAgger experiment,
not dense expert labeling of every learner-visited state or a test of DAgger's
full theoretical guarantees. Prefix candidate commands remain excluded even if
the final recovery succeeds.

`navigation_motor.fit` supports the separate recovery dataset view, a hash-bound
navigation warm start, and an explicit recovery/replay sampling fraction. All
inherited motor and decoder tensors remain frozen. First-batch checks compare
stored executed motor actions with recomputed frozen-motor actions and check
original teacher-token decoding. Final checks preserve every inherited tensor
and full-command anchor action exactly.

## Why this follows the research evidence

DAgger addresses the distribution of states induced by the learner, motivating
current-adapter prefixes rather than recycling full-command specialist rollouts
as though they were navigation DAgger. The exact-motor demonstration control
addresses a separate, earlier mismatch between the supervised state distribution
and the deployed backend. [DAgger](https://proceedings.mlr.press/v15/ross11a.html)

DART studies recovery demonstrations induced by perturbing the supervisor. That
provides a useful alternative if learner-prefix takeover support is too narrow:
collect controlled perturbations from states where the expert can still recover.
The present experiment uses actual navigation prefixes and does not implement
DART's optimized noise-injection method or inherit its reported performance.
[DART](https://proceedings.mlr.press/v78/laskey17a.html)

The inherited nominal continuation is only considered suitable at a queried
state after its motor execution succeeds. This establishes local support for the
observed suffix, not a task-aware expert for every possible state. In particular,
late goal-position drift may require a new braking/goal-correcting continuation
rather than labels from a clock that has already reached the standing tail.

## Original-teacher information boundary

The selected teacher's **saved checkpoint configuration**, rather than a newer
repository experiment profile, lists gravity direction, angular velocity, joint
position, joint velocity and previous actions in its 930D history. Its G1 encoder
uses future joint position/velocity and relative anchor orientation. Root-position
error and base linear velocity appear in the privileged critic, not directly in
this actor path.

Consequently, an ideal horizontal rigid translation of the robot, preserving
joint state, orientation, angular velocity and action history, leaves the original
teacher actor's inputs unchanged. This is a structural observation, not a claim
that the teacher cannot stop or that all its training experiences are identical.
It means a nominal reference-tracking teacher is not automatically an expert for
correcting arbitrary global goal-position errors.

The added 114D motor forecaster **does** receive relative keypoints and desired
velocity; it is not subject to exactly the same input invariance. A CPU sensitivity
probe on three recorded switch states found action RMS changes of 0–0.0428 for
±0.2 m body-frame keypoint translations and 0.0102–0.0556 for ±0.2 m/s desired
velocity changes. These are unexecuted input-sensitivity probes in model action
units, not physical controllability or validated command-generation results.
No perturbed commands enter the recovery training set. Some zero changes are
consistent with the quantized interface; they do not establish a global dead zone.

This distinction supports a bounded next route if supported recovery imitation
stalls: construct and execute task-aware braking/goal-correcting continuations,
or test command-space task feedback with imitation replay, while retaining the
motor backend. Simply querying the original nominal teacher more often does not
create missing goal-correction instructions.

## Results

The motor-data fit completes **2/8** tasks on the initial seed; adding 3,000
replay-only updates reaches **3/8**. The equal-update recovery branch reaches
**4/8**. The registered full-panel confirmation falls to **1/8** on seed 91261.
This candidate is not promoted as a reliable replacement for the archived model.

| Task | Motor data, 6k updates | +3k replay | +3k recovery/replay |
| --- | --- | --- | --- |
| 00908 clear | Pass | Pass | Fail; final hold 35 ticks |
| 00908 corridor | Pass | Fail | Pass |
| 00413 clear | Fail | Fail | Pass |
| 00413 corridor | Fail | Fail | Fail |
| 00976 clear | Fail | Pass | Pass |
| 00976 corridor | Fail | Pass | Pass |
| 00265 clear | Fail | Fail | Fail |
| 00265 corridor | Fail | Fail | Fail |

All 24 main navigation evaluations remain contact-free and above the declared
fall threshold. Recovery training gains two cases over replay-only and loses one,
for one net additional completion. This small, single-training-seed result is not
statistical evidence that recovery data reliably improves the whole task class.
The two branches share the same initial checkpoint and inherited normalization.

The failed 00908-clear recovery-trained attempt is inside the goal at the end
(distance 0.060 m) and retains its longest 35-tick hold until the deadline. The
last earlier speed reset is at tick 309 (0.10091 m/s), after a shorter 15-tick run;
it must not be described as breaking the later 35-tick hold. This is now a timing
or earlier-braking problem rather than a failure to locate that endpoint.
Only final scheduled checkpoints are evaluated; intermediate checkpoints are
not searched for a favorable task result.

All eight learner-prefix attempts are retained. **Six qualify**, providing 1,250
motor-executed recovery rows and six verified learner-state queries. The two
00976 attempts fail to hold and contribute zero training rows. Their nominal
motor continuations enter the goal but do not stabilize, so the successful
nominal 8/8 control cannot be generalized to those arrival states.

All eight prefixes match the corresponding unassisted motor-data student root,
speed and prohibited-contact traces exactly. This checks that diagnostic target
queries and the collection wrapper do not alter learner behavior before takeover.
The combined admitted training view contains 3,076 demonstration rows plus 1,250
recovery rows; unsuccessful attempts are retained in the bound aggregate for
coverage/cost accounting.

A confirmation plan was registered before the recovery fit: select the greatest
main-seed stable-completion count, breaking ties in favor of motor-data, then
replay-control, then recovery. Evaluate that checkpoint on **all eight** original
tasks with seed 91261. This is a selected-checkpoint evaluation-seed check on
training tasks, not an independently constructed layout benchmark or a repeat
across training seeds.

The confirmation succeeds only on 00908 clear, which failed on the main seed.
All four main-seed successes fail on the confirmation seed. The eight confirmation
attempts remain contact-free with no falls; six never enter the goal and 00908
corridor reaches only ten hold ticks. Evaluation seed changes both initialization
and observation randomness; this result does not isolate sensor noise as the cause.

Four adaptively selected confirmation failures were then rerun with privileged
full commands: one per motion family. `NavigationFullCommandControlCallback` loads
the same navigation checkpoint and model construction as the matched navigation
run, then bypasses the adapter. This avoids changing the model-construction RNG
consumption when comparing initial conditions and observations.

| Seed 91261 task | Navigation | Same-checkpoint full-command control |
| --- | --- | --- |
| 00908 corridor | Fail | Pass |
| 00413 clear | Fail | Pass |
| 00976 clear | Fail | Fail; maximum hold 31 ticks |
| 00265 clear | Fail | Pass |

These are four selected diagnostic controls, not an eight-task oracle panel.
The three successful controls localize a command-inference problem in those
executions. The failed 00976 control enters the goal but ends at 0.2514 m distance,
just outside the unchanged 0.25 m tolerance. Together with both unsupported 00976
takeovers, it shows that nominal motor execution also has a terminal robustness
limit. No falls or prohibited contacts occur in these four controls.

## Next priority and decision gates

1. Broaden the **collection** distribution across declared initialization and
   observation seeds, approach speeds and headings. Keep the recovered motor
   frozen and admit only physically supported continuations. Compare against
   equal-update replay with the same training seeds; reserve evaluation seeds
   before collecting additional data. The present one-seed study cannot establish
   a repeatable benefit from six supported entries.
2. Diagnose 00976 braking separately using original-teacher and frozen-motor
   controls from matched arrival states. Construct and execute earlier braking
   or goal-correcting continuations if the nominal clock lacks local support.
   Failed queries remain excluded. Do not shorten the 50-tick hold or extend
   official deadlines to manufacture completions.
3. Retain the eight tasks as an engineering regression panel. Require repeatable
   unassisted completion before expanding to valid goal-switch, route-switch and
   posture-switch pairs and independently constructed evaluation layouts. The
   current clear/corridor variants do not establish necessary scene reasoning or
   the learning value of motion-to-scene generation.

These findings prioritize supported coverage and terminal control over increasing
model size or unfreezing the motor. A direct-reference or command-space task-reward
comparison remains conditional on persistent failure with adequate observations
and demonstrated expert support; it is not implemented or claimed by this study.

## Reproduction and evidence

External experiment packet:
`/home/linjiw/research-data/m2s-nav-supported-recovery-20260913/`.
`plan.json`, `collect-motor.py` and `run-study.py` record the bounded protocol;
per-stage `command.json`, config, source hashes and logs reproduce each run.
Motor/combined manifests retain unsuccessful recovery attempts as well as
supported ones. Collection costs are 3,076 motor-demonstration steps plus 3,235
learner-prefix/recovery steps. The latter include 1,367 candidate prefix queries
and eight attempted switch queries; only six switch entries have verified
successful continuations. There are 6,311 motor target forwards and 6,311 original
teacher diagnostic forwards across collection. These costs are not reduced to
the count of admitted labels. Wall times are recorded, but a separate teacher
training job began sharing the GPU during the study, so they are not a controlled
throughput comparison. All native simulations run headless at 50 Hz with 200 Hz
pair-resolved contacts.

Training uses the existing entry point with the selected condition config and
an unused output directory:

```bash
.venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.navigation_motor \
  --config /home/linjiw/research-data/m2s-nav-supported-recovery-20260913/motor-data-config.json \
  --output /tmp/m2s-motor-data-fit
```

This packet extends the current training pipeline; it does not require changes
to the selected motor checkpoint or a new simulator installation.

The [compact evidence packet](evidence/navigation-supported-recovery-20260913/README.md)
contains plans, training receipts, checkpoint hashes, all 52 native run scores and
commands, manifests, costs and diagnostics. Those runs total 22,318 control steps
across collection, unassisted evaluations and privileged controls; they must not be
pooled into one navigation success rate. The four fits use 12,016 updates in total,
including the 16-update smoke run; each final continuation inherits 6,000 updates
and adds 3,000. Raw arrays and large checkpoints remain external.

Validation: **106 tests passed** across the affected recovery, motor, navigation,
scene-distillation, BFM, flow and qualification modules. Ruff and Black checks pass
on the six affected implementation/test files. The recovery tests cover suffix-only
qualification, forged admissions, executed-action validation and stopping an already
successful learner before an unnecessary planned intervention. Exact commands are
recorded in the evidence packet's `validation.json`.
