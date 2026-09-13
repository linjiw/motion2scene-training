# Navigation terminal control: preserved motor, measured context and same-state targets

This increment keeps the selected full-command specialist unchanged and studies
why the navigation adapter cannot access its demonstrated approach/stop capability.
The previous matched result remains **8/8 full-command versus 0/8 goal/map**. Those
are eight training tasks, four clips and one motion ancestry group; neither the
videos nor that contrast establishes new-layout scene reasoning.

## What the hold audit actually found

The scorer requires 3D pelvis distance ≤0.25 m and 3D speed ≤0.10 m/s for 50
consecutive 50-Hz control decisions, no prohibited contact above 1 N, and no fall
below the declared 0.25 m pelvis-height guard. There is no separate posture test
inside the hold counter. This scorer and all eight original deadlines remain fixed.

- **00908 corridor:** the 42-tick run ends at measurement tick 243 (4.86 s).
  Distance is 0.184695 m, still inside the goal, but speed rises to 0.114575 m/s.
  This is a speed violation, followed later by position drift.
- **00265 corridor:** the 38-tick run is still intact at the 500-tick deadline.
  It is a late-arrival failure, not a broken 38-tick hold.
- The other goal entries do not accumulate even one joint position-and-speed-valid
  tick. Goal entry alone is not stable completion.

The previous summary of 42/38 ticks therefore concealed two different mechanisms.

## Implemented observation and objective changes

`NavigationMotorStudent` remains a 512-wide command adapter with the same masked
permutation-invariant obstacle encoder. Its frozen backend includes the recovered
forecaster, pretrained reference encoder, FSQ, decoder and normalization buffers.
The legacy checkpoint profile remains loadable without changing its input shape.

The optional `nav_goal_map_localization_v2` profile appends four public values:
three components of backward estimated body velocity and a validity bit. For
measurement times t and t−5 control intervals, velocity is the world-position
difference divided by actual elapsed measurement time, rotated using the current
measured body orientation. The first five decisions return invalid/zero velocity;
each new episode or changed goal resets the estimator. No differences of rotating
body-frame goal vectors, reference clock, motion identifier, future poses, or
simulator velocity enter this actor. The 930D motor history retains its native
layout and is not reshaped into an invented sequence.

This first profile assumes a complete registered map and zero-delay localized
poses. Using simulator pose measurements supplies an ideal localization stream;
it is not a camera implementation or evidence of robustness to localization noise
or delay. The five-step backward window is causal but smooths recent acceleration.
A delayed pose estimator will require an explicit arrival-time/measurement-time
contract and an age input before deployment.

Teacher collection now records **pre-action** position, quaternion and measurement
time alongside its targets. Loading reconstructs the velocity over the complete
episode before target masking and verifies equality to the stored feature. This
prevents using a post-action state to label a pre-action observation or erasing
gaps in causal history.

The opt-in `structured_motor` objective averages normalized errors over eight
semantic groups: heading (0:2), velocity (2:5), height (5:6), yaw rate (6:7),
keypoints (8:50), joint position (50:79), joint velocity (79:108), and orientation
(108:114). Arrival coordinate 7 is unavailable and excluded. Equal group weights
and the fixed 0.05 standard-deviation floor are deliberate initial choices.

The principal action target is `M(h, c_expert)` at the same measured history,
computed without target gradients. The predicted command passes through the
frozen motor with input gradients enabled. The original teacher action is retained
as a separate quality diagnostic rather than silently requiring the command
adapter to compensate for the frozen motor's error. The new loss is motor-action
MSE + 0.1 × structured command MSE, with no extra teacher-token term. The legacy
loss remains available to reproduce the archived pilot.

A smoke check through the **actual encoder, FSQ and decoder** produces nonzero
action-only gradient at the navigation command head while all inherited parameter
gradients remain absent. Every fit checks exact equality of all inherited tensors
and an unchanged full-command action anchor batch. These checks preserve the
backend; they do not prove the adapter selects valid commands in closed loop.

## Executed stopping-tail coverage

All eight teacher attempts were repeated through their original task deadlines,
adding 1,161 executed rows: **3,900 total**, compared with 2,739 in the first pilot.
All original prefixes match archived root, speed and contact traces exactly.
All eight remain successful and contact-free. Six remain continuously terminal
following first completion; both 00413 tails briefly lose the terminal condition,
then regain a final 154-tick hold. Their tails are retained as executed behavior,
not described as uninterrupted standing.

The independent scorer is unchanged. Continuing collection after first success
is explicitly teacher-only. No padded or imagined frames are added to the data.
Positive-task admission still requires a bound successful physical receipt;
unsuccessful navigation queries are not promoted into this view.

## Same-state continuation probes

Four predeclared probes begin with the archived navigation policy and switch in
the same physical episode. All pre-switch root, speed and contact prefixes match
the baseline exactly. Robot state, observation history and controller history
are preserved. A separately loaded full-command motor reconstructs commands at
each actual state with the nominal reference clock, without a phase jump. Loading
the diagnostic motor preserves RNG state.

| Task | Switch time | Distance / speed at switch | Post-switch max hold | Stable completion by original deadline |
| --- | ---: | --- | ---: | --- |
| 00908 corridor | 3.60 s | 0.143 m / 0.116 m/s | 50 | Yes |
| 00908 corridor | 4.86 s | 0.185 m / 0.115 m/s | 48 | No |
| 00265 corridor | 8.00 s | 0.352 m / 0.433 m/s | 23 | No |
| 00265 corridor | 9.24 s | 0.150 m / 0.103 m/s | 25 | No; only 38 ticks remain |

These are privileged intervention diagnostics, not navigation-policy successes.
The early success establishes one locally recoverable navigation arrival state.
The failures show that nominal task executability does not imply timely recovery
from every student state. In particular the final probe is deadline-censored and
cannot diagnose inability to hold for 50 ticks. A broader task-aware continuation
bank has not been validated by these four probes.

## Bounded comparison

The registered comparison trains two new adapters from scratch, each for 6,000
updates, batch 256, AdamW 1e-4, seed 91370 and the same 3,900 executed rows:
structured motor supervision without velocity, and the identical objective with
causal localization velocity. Both use the same frozen motor, exact known map,
evaluation seed 91260, eight original tasks and original deadlines. No task-based
checkpoint selection is performed: the final registered update is evaluated.

The difference between these two conditions isolates the added feature interface
within this small pilot. Comparing either against the original 0/8 run also changes
the objective and tail data, so it cannot isolate either change by itself. One
training seed cannot establish a robust improvement.

| Condition | Stable completion | Goal entry | Best hold | Prohibited-contact attempts |
| --- | ---: | ---: | ---: | ---: |
| Archived adapter, original 2,739 rows | 0/8 | 5/8 | 42 ticks | 0/8 |
| Structured motor loss, 3,900 rows | 0/8 | 4/8 | 5 ticks | 1/8 |
| Same loss + causal localization | **1/8** | 3/8 | **50 ticks** | 0/8 |

The localized adapter completes 00976 clear at tick 332 (6.64 s), with final
3D goal distance 0.131 m. It achieves a 44-tick maximum hold in 00976 corridor.
The corridor hold breaks at tick 325: distance remains 0.228 m but speed rises
to 0.10994 m/s. The remaining six attempts never accumulate a hold tick. All attempts in both
new conditions remain above the declared fall threshold. The structured-only
00265 corridor attempt ends on prohibited contact at tick 488.

Two adaptive follow-up evaluations of the **selected successful task only**, at
seeds 91261 and 91262, both fail with zero hold ticks. This is not a repeated
eight-task panel. Across the three evaluated seeds that task succeeds once.
The original 1/8 panel result therefore does **not** establish reliable improvement.
The localized checkpoint is an experimental candidate and is not promoted over
the archived navigation baseline. The archived motor
and navigation checkpoints remain intact. No new full-command checkpoint is
trained or selected.

On the same 3,900 recorded teacher states, the archived adapter has motor-action
MSE 0.004055, versus 0.000296 for the structured adapter and 0.000299 for the
localized adapter (about 13.5–13.7× lower). All eight physical command-group MSEs
also improve. Despite this, task execution remains poor. This is direct evidence
that fitting these teacher states is an inadequate model-selection signal.
It motivates navigation-induced state coverage; it does not by itself prove
which source of closed-loop error dominates.

Both new fits retain exact motor tensor and full-command anchor-action identity.
Compact results include the complete per-task scores, training receipts and
checkpoint SHA-256 hashes.

![Offline fitting and physical task outcomes](evidence/navigation-terminal-20260913/offline-versus-task.png)

## Navigation-induced query capture and deadline diagnostics

`NavigationQueryCallback` runs the navigation actor unassisted and records its
public observation, predicted command and action. Only after choosing that action
does it reconstruct a candidate full command at the same physical state and
query the frozen motor. Those privileged targets never enter the actor. Two
archived-policy attempts produce **850 candidate rows**, with exact root, speed
and contact parity to the original unassisted runs.

All `supported_query_mask` values remain false. The new candidate-query schema is
separate from successful-task data and explicitly ineligible for the positive
loader. A success receipt for a whole attempt is not sufficient to self-qualify
its candidate advice. No navigation DAgger fit has been run on these rows.
Candidate motor-action disagreement is 0.06476 on 00908 corridor and 0.08951 on
00265 corridor, but these nominal-clock targets have not all been shown to be
suitable task continuations. Do not interpret this as validated expert regret.

Two further **adaptive diagnostics**, each allowing 100 additional ticks, preserve
their original rollout prefix and are kept outside the official task benchmark:

- The 00265 takeover at tick 462 achieves a 50-tick post-switch hold after 63
  continuation steps (global tick 525). The original 500-tick task still fails.
  This resolves its deadline censoring: that arrival state is locally stabilizable
  with the tested full-command continuation if sufficient time is available.
- The 00908 takeover at tick 243 still achieves only 48 hold ticks by the extended
  deadline; final distance drifts to 0.452 m. For this state, simply waiting longer
  with the nominal continuation does not solve position regulation.

These observations motivate two concrete coverage targets: earlier braking/arrival
for 00265-like approaches, and a goal-correcting recovery continuation for the late
00908-like state. Qualify such continuations before running larger navigation
DAgger. More optimization of the present nominal teacher rows is not justified
by their already low fitting error. A bounded continuation bank, replay mixtures,
sequence burn-in and task-RL remain next experiments, not implemented results of
this increment.

## Research direction and discriminating experiments

The modular deployment boundary is compatible with internally generated detailed
commands: SONIC separates a kinematic planner and universal motion-control
representation. Our navigation adapter remains a distinct learned component;
SONIC does not establish its performance. [SONIC](https://arxiv.org/html/2511.07820v1)

An insufficient observation interface can make privileged expert decisions
ambiguous to an imitator. This motivates the explicit velocity comparison, but a
positive result would not identify every source of partial observability.
[Robust Asymmetric Learning in POMDPs](https://proceedings.mlr.press/v139/warrington21a.html)

The next data distribution must include states induced by navigation command
errors. Motion-specialist DAgger under supplied correct commands is a different
rollout distribution. Use navigation-specific aggregation, retaining complete
attempts and a separate supported query/recovery view. Validate continuation
entry conditions using takeover execution before treating late-phase targets as
expert advice; never advance the reference across an obstacle merely to match a
pose. [DAgger](https://proceedings.mlr.press/v15/ross11a.html)

Per-step imitation may underweight brief actions essential to completing a task.
PHP reports this issue for dynamic parkour and combines DAgger with RL. That
supports a later command-space task-reward comparator if imitation improves fit
without fixing outcomes; it does not demonstrate that our stopping problem needs
RL. Keep the motor frozen, retain imitation replay, and compute log probabilities
on the actual sampled command variables.
[Perceptive Humanoid Parkour](https://arxiv.org/html/2602.15827v1)

Light-Loco-Parkour distinguishes merging individually executed skills from
learning within-episode transitions, adding a transition training stage. Its
terrain/command tests motivate approach–prepare–traverse–exit–brake–hold sequences
and goal-away controls in our data; isolated compatible traversal clips are
insufficient transition evidence.
[Light-Loco-Parkour](https://arxiv.org/html/2608.02653v1)

Indoor humanoid traversal already includes crouching, hurdle crossing and narrow
side passages. Our contribution should be measured learning/acquisition value of
generated motion–scene supervision, not the mere existence of obstacle traversal.
[Collision-Free Humanoid Traversal](https://arxiv.org/html/2601.16035v1)

The next experimental gates are:

1. Establish reliable approach and stopping on the eight fixed tasks and under
   limited, supported approach perturbations. Separate late arrival, speed drift,
   goal drift, contact and falls; retain 50 ticks. Check longer diagnostic holds
   separately from deadline-based task success.
2. Collect navigation-induced states with a task-compatible expert continuation.
   Version supported queries separately from successful demonstrations. Use short
   aggregation cycles and evaluate unassisted rollouts after every predeclared
   update budget. Recurrent sequence training, if added, must include contiguous
   windows, reset masks and burn-in under the actual previous-command convention.
3. Make context necessary using physically executed matched goal-switch,
   route-switch and posture-switch tasks, plus irrelevant-geometry controls.
   Keep initial state and goal fixed for route-switch pairs. Never attach the old
   action label to newly incompatible geometry. Report success on **both** members.
4. Freeze independent layouts and compare generated-pair supervision with a valid
   procedural/seed-data source at matched executed-demonstration and expert-query
   budgets. Separately measure acquisition yield per generation/simulation budget,
   including rejected attempts. Report oracle-command coverage on every task,
   with supported-task transfer as a separate denominator.
5. Add a single direct-reference comparator or task-RL refinement only when the
   observed bottleneck motivates it. Keep camera transfer, dynamic scenes and
   broader motion ancestry as distinct later generalization axes.

The bounded target remains known-map local whole-body traversal followed by
stable stopping. Current clear/corridor scenes do not require alternate route
selection or low-clearance posture change; successful performance here alone
would not establish scene reasoning or generated-data value.

## Artifacts and reproduction

External packet: `/home/linjiw/research-data/m2s-nav-terminal-control-20260913/`.
It contains registered plan, source hashes, per-stage exact commands, configs,
training metrics/checkpoints, physical traces, takeovers and causal pose-labelled
teacher shards. `collect.py`, `audit.py` and `run.py` retain the executed experiment
orchestration. Each output directory is created exclusively to prevent overwrites.
Compact receipts are copied into `docs/motion2scene/evidence/navigation-terminal-20260913/`.

Given a generated condition config, the training entry point is:

```bash
.venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.navigation_motor \
  --config /home/linjiw/research-data/m2s-nav-terminal-control-20260913/localized-config.json \
  --output /tmp/m2s-localized-navigation-fit
```

Validation: 99 impacted tests pass; nine changed Python files pass Ruff and
Black. Native evidence comprises eight teacher collections, 16 new adapter task
evaluations, four original-deadline takeovers, two candidate-query rollouts, two
extended-deadline diagnostics and two selected-task repeat evaluations. Real
encoder/FSQ/decoder gradient smoke, exact frozen tensor/action retention, causal
feature reconstruction, archived-prefix parity and candidate-label rejection
provide additional checks. No training job is left running by this increment.

Native evaluation uses the saved per-stage `command.json`, bound to its student,
teacher, collision USD and task hashes. It runs headless Isaac physics, with
200-Hz pair-resolved contacts and the 50-Hz motor loop. No renderer is needed.
