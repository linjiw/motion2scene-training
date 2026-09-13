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

## Results

The complete registered panel and equal-update recovery comparison are reported
here after native evaluation finishes. Preliminary task outcomes are not used to
select among intermediate checkpoints.

## Reproduction and evidence

External experiment packet:
`/home/linjiw/research-data/m2s-nav-supported-recovery-20260913/`.
`plan.json`, `collect-motor.py` and `run-study.py` record the bounded protocol;
per-stage `command.json`, config, source hashes and logs reproduce each run.
Motor/combined manifests retain unsuccessful recovery attempts as well as
supported ones. All native simulations run headless at 50 Hz with 200 Hz
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
