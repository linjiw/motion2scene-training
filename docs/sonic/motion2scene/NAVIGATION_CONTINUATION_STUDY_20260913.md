# Executed goal-regulating continuations and broader recovery coverage

This increment follows the [supported-recovery study](NAVIGATION_SUPPORTED_RECOVERY_20260913.md):
2/8 motor-data navigation completions, 3/8 with more replay, 4/8 with qualified
recovery, but only 1/8 for that selected checkpoint on another evaluation seed.
Three of four selected failed cases succeeded under same-checkpoint full commands;
00976 clear also failed through the motor interface. The recovered motor remains
frozen and the candidate is not a reliable replacement for the archived baseline.

The supplied manuscript guidance describes the preceding terminal-control increment.
Its recommendation to qualify continuation advice remains applicable; its statement
that no supported navigation aggregation has run is superseded by the six qualified
entries in the linked study. The supplied sandbox manuscript files are unavailable
on this machine; this report uses the pasted guidance and recorded repository data,
without claiming to have inspected the PDF, editable package or illustrations.

## Question and fixed protocol

Can a bounded goal-regulating continuation improve local expert support, and can
broader supported collection improve unassisted navigation under the same motor,
observation contract and fitting objective? These are separate questions. A useful
privileged intervention is not a successful deployed navigation episode.

The initial screen branches the current recovery-trained navigation policy at four
actual failure states: 00976 clear / tick 191, 00265 clear / tick 187 and 00908
corridor / tick 230 on seed 91261, plus 00908 clear / tick 280 on seed 91260.
Each receives three continuation candidates, totaling twelve attempts:

- Nominal current commands, reconstructed at the measured state.
- Goal-regulating desired horizontal velocity.
- The same desired velocity plus a bounded translation of nominal body keypoints.

The velocity candidate uses world-horizontal `0.8 * goal_error - 0.5 * causal_velocity`,
clamped to 0.35 m/s and rotated into the measured body frame. Its spatial blend is
`clip(2 * (1 - horizontal_distance / 0.6), 0, 1)`. The keypoint candidate adds the
blended horizontal goal-minus-reference-anchor offset, limited to 0.2 m. Joint
positions, joint velocities, heading, root height and relative orientation remain
nominal. These modifications do not establish dynamic consistency by construction;
only executed support justifies including their labels. No reference phase jumps,
spliced motions or changed robot states are used.

The regulator uses causal localization, resets at episode/goal changes and is
inactive before velocity becomes available. It uses neither a deadline nor a
reference clock to infer urgency. The nominal pose path remains privileged teaching
information. This is a command-space continuation experiment, not a reference-free
handwritten navigation policy, validated planner or learned residual motor.

The provider with the most timely supported screen cases is selected, with ties
favoring nominal, then velocity, then velocity plus keypoints. It receives eight
additional early takeover trials on seed 91261. All attempted suffixes, including
failures, remain in the receipt ledger. Two equal 3,000-update forks from the same
current navigation checkpoint compare existing replay against half newly qualified
recovery / half existing replay. Training seed 91470, architecture, normalization,
loss and frozen motor are held fixed. Both receive the complete eight-task panel
on seed 91260. The better final checkpoint receives the complete panel on seed
91262, reserved before collection; ties favor replay. No intermediate checkpoint
selection is allowed. This is a bounded training-task study, not new layouts or
independent training-seed replication.

Seed 91262 is reserved from **this round's collection**, not an uninspected final
test: the earlier localized pilot evaluated 00976 clear at that seed. The original
plan's `held_out_evaluation_seed` field denotes this round's collection exclusion.
All eight layouts remain training tasks, and the new forks share the parent's
earlier training history. The additional fitting seed is not an independent
from-scratch training replicate.

The replay view already contains the preceding study's supported recoveries.
Therefore the new comparison concerns additional current-policy coverage and the
selected teaching package, not a pure comparison of offline data versus any DAgger.
Generation and recovery selection costs must remain visible.

## Evidence and implementation contract

`navigation_continuation.py` defines the bounded provider and a dual-outcome scorer.
Each receipt separates observed local stabilization from original-deadline success
and timely suffix support. The scorer can analyze extended diagnostic traces, but
late stabilization never makes their rows eligible for original-deadline training.
A short observed suffix is censored evidence, not proof of unstabilizability.
The present native screen preserves the original deadlines and does not extend them.

The collector stores nominal commands, reference anchor, measured poses and causal
velocity. The recovery loader reconstructs candidate commands and support labels
from these bound records, in addition to its existing execution, ancestry, task,
checkpoint, mask and frozen-motor checks. Candidate rows before takeover are not
admitted. Successful expert suffixes still provide only one verified learner-state
entry each; expert-executed tails are not additional learner queries.

## Research interpretation

Learner-induced state collection follows the distribution-shift motivation of
[DAgger](https://proceedings.mlr.press/v15/ross11a.html), but requires an expert that
can supply useful behavior at those states. Our bounded screening procedure does
not inherit DAgger's theoretical guarantees.

The observation contract follows the caution in
[Robust Asymmetric Learning in POMDPs](https://proceedings.mlr.press/v139/warrington21a.html):
privileged experts can prescribe decisions the student cannot distinguish under its
own inputs. We avoid adding hidden deadline-dependent urgency to local regulation;
this is not an implementation of that paper's adaptive asymmetric algorithm.

[Perceptive Humanoid Parkour](https://arxiv.org/html/2602.15827v2) combines DAgger
and task-outcome optimization and studies approach-condition coverage. That motivates
retaining command-space task feedback as a contingency. Its highly dynamic skill
results do not establish that our terminal-control problem requires PPO. The next
comparison keeps the current supervised objective fixed to study teaching quality.

External packet: `/home/linjiw/research-data/m2s-nav-continuation-20260913/`.
The pre-execution `plan.json` records cases, parameters, budgets and selection rules.

## Completed continuation screen

| Arrival state | Nominal motor | Velocity feedback | Velocity + keypoints | Original teacher |
| --- | --- | --- | --- | --- |
| 00976 clear, seed 91261, tick 191 | Fail; 0 hold ticks | Fail; 0 | Fail; 0 | Fail; 0 |
| 00265 clear, seed 91261, tick 187 | Fail; 0 | Fail; 0 | Fail; 0 | Fail; 0 |
| 00908 corridor, seed 91261, tick 230 | Fail; 37 | Fail; 39 | Fail; 40 | Pass; 50 |
| 00908 clear, seed 91260, tick 280 | Pass; 50 | Pass; 50 | Pass; 50 | Fail; 49 |

All twelve motor probes and four original-teacher controls reproduce the archived
unassisted learner prefix exactly in root position, speed and prohibited contacts.
The teacher controls are diagnostic only and emit no motor-recovery training receipt.
None of these successes is counted as unassisted navigation.

Each motor provider succeeds on the same one of four cases. The predeclared tie
rule therefore selects **nominal continuation** for the follow-up collection.
The feedback candidates are not credited with broader support or trained into the
next adapter merely because their maximum hold rises from 37 to 39–40 ticks.

In 00976, horizontal distance never drops below 0.857 m: the 0.6 m regulator gate
never activates, so its three motor traces are identical. This case tests the
candidate's limited coverage, not active goal-feedback efficacy. In 00265, velocity
feedback does activate but still stops outside the goal. The closer corridor case
is a speed-regulation failure through the tested motor commands: it remains inside
the positional tolerance, while the original teacher can complete the hold.
Conversely, the motor completes late 00908 clear where the teacher reaches 49 ticks.
That 49-tick run is intact at the deadline; it is not a broken 49-tick hold.
These results support state-specific continuation qualification rather than treating
either backend as uniformly superior.

![Executed continuation traces](evidence/navigation-continuation-20260913/continuation-traces.png)

The figure shows post-takeover 3D goal distance and root speed, with original
remaining deadlines at the right edge. All conditions share the measured prefix.
Curves stop at successful suffix completion or the original deadline. Different
vertical scales help inspect each failure mechanism; this is not a pooled success
or effect-size chart.

The frozen anticipatory motor's current encoded frame is built from target joint
positions, joint velocities and relative orientation. Root velocity and keypoints
influence its *future-frame forecaster*, rather than directly changing that current
frame. Our bounded feedback leaves the direct pose channels unchanged. Its limited
result therefore does not establish that the complete 114D interface cannot express
goal correction. A coherent additional walking/braking reference remains a distinct
teaching candidate, requiring its own measured-state execution evidence.

The dual-support scorer also reproduces the archived extended diagnostics: late
00265 stabilizes locally after its original deadline; late 00908 does not stabilize
within its extended diagnostic horizon. Neither becomes timely supported training
data. Those are reanalyses of archived traces, not new successful executions.

One attempted concurrent launch exhausted GPU memory during checkpoint loading,
before any task step. Its failed logs are retained; the unchanged condition completed
in a fresh retry directory after returning to sequential simulation. The separate
training process was not stopped. Native execution counts exclude that startup
failure; resource accounting reports it separately. Wall times under shared GPU load
are not a controlled throughput comparison.

## Follow-up comparison registration

Before follow-up collection or fitting, an additional confirmation control was
registered: evaluate the unchanged parent navigation checkpoint on all eight tasks
at seed 91262 as well. This permits a matched comparison with the selected new fork
on the reserved evaluation seed. It does not alter provider or checkpoint selection.
Old qualified recoveries are replay data in this new round; the new sampler assigns
fresh recovery by the bound driver-checkpoint hash, preserving their original
qualification and ancestry. The replay-only fork thus retains all 4,326 previously
admitted rows rather than accidentally dropping the old recovery stratum.

## Matched fitting results on seed 91260

The eight early follow-up attempts qualify only the two 00908 variants. Together
with the selected late 00908-clear screen continuation, the selected-provider view
contains **three fresh supported entries and 294 rows from twelve attempts**.
Across the entire screen and follow-up, twenty motor continuation attempts were
executed; nonselected provider trials remain in acquisition accounting. The twelve
selected-provider attempts are a subset, not the total collection cost.

All twenty motor attempts preserve the corresponding learner prefix exactly.
Both 3,000-update fits complete with exact inherited motor tensor and full-command
anchor-action identity. Replay uses 4,326 rows in fourteen episodes, including the
six previously qualified recoveries. The new mixture uses 4,620 rows in seventeen
episodes. Its three fresh recoveries all belong to 00908.

| Task | Unchanged parent | +3k old replay | +3k fresh recovery/replay |
| --- | --- | --- | --- |
| 00908 clear | Fail | Pass | Pass |
| 00908 corridor | Pass | Fail; 24 ticks | Pass |
| 00413 clear | Pass | Fail; 7 ticks | Pass |
| 00413 corridor | Fail | Pass | Fail; 45 ticks |
| 00976 clear | Pass | Pass | Fail; 26 ticks |
| 00976 corridor | Pass | Fail; 31 ticks | Pass |
| 00265 clear | Fail | Fail | Fail |
| 00265 corridor | Fail | Fail | Fail |
| **Completions** | **4/8** | **3/8** | **4/8** |

Fresh recovery gains three cases and loses two relative to the equal-update replay
control, for one net completion. Relative to the parent, it gains 00908 clear and
loses 00976 clear, leaving the aggregate unchanged. Both new panels remain free of
prohibited contacts and falls. The failed fresh-recovery 00413 corridor attempt
breaks its hold and eventually leaves the goal; its 45 ticks are not an intact
hold censored at the deadline.

The comparison changes the sampling distribution as well as the state examples.
Role/motion/episode-balanced sampling assigns 62.5% expected probability to 00908
in the fresh branch, versus 25% under old replay. Each other motion receives 12.5%
versus 25%. Thus the one-task advantage over replay does not isolate fresh-state
quality from motion reweighting. A motion-weight-matched replay control would be
needed for that narrower attribution. The current study compares the declared
teaching package at equal update budgets, with its acquisition costs disclosed.

The replay receipt's zero `qualified_recovery_episodes` counts the fresh sampling
role in this round; it does not mean the six old qualified recovery episodes were
dropped. `sampling-audit.json` records both fresh and replayed recovery counts.

## Matched confirmation and current checkpoint status

| Navigation checkpoint | Seed 91260 | Seed 91262 |
| --- | --- | --- |
| Unchanged parent | 4/8 | 1/8 |
| Additional old replay | 3/8 | Not evaluated |
| Additional fresh recovery/replay | 4/8 | 2/8 |

The selected fresh-recovery checkpoint completes **00908 clear and 00265 clear**
on seed 91262. The parent completes only 00265 clear. Thus fresh recovery adds one
completion on the matched confirmation seed while retaining the first-seed total.
The clearest task-level gain is 00908 clear: the new checkpoint succeeds on both
tested seeds where the parent fails. This is a modest observed improvement, not
reliable navigation, an independent training-seed result or held-out-layout evidence.
The first-seed loss on 00976 clear and the sampling confound remain part of the result.

**Both confirmation panels have a prohibited-contact failure on 00413 corridor**:
the new checkpoint stops at tick 187 with peak undesired force 19.47 N; the parent
stops at tick 166 with 30.52 N. Both fail the same unchanged 1 N criterion. Neither
panel has a fall. The selected checkpoint has two completions, one contact failure
and five timeouts; the parent has one completion, one contact failure and six
timeouts. Lower contact magnitude is not treated as task success or a demonstrated
safety improvement. The sixteen first-seed evaluations are contact-free.

Retain the new checkpoint as an **experimental navigation candidate**, alongside
its parent and the historical baseline:

```text
/home/linjiw/research-data/m2s-nav-continuation-20260913/recovery-fit/training/step-003000.pt
SHA256 d378947413cdd167e71e3271440e9a18fbfd5a609836aa0be7d98cefe3a6e9cd
```

The selected motor is unchanged. No model is promoted as a reliable scene-navigation
solution, and the two successes do not establish meaningful scene reasoning. The
operational hold criterion is one second, not proof of indefinite stabilization;
there is no separate terminal-posture test in this scorer.

## Next priority: coherent locomotion re-entry and braking

The screen narrows the required teaching behavior. When the robot stops well short
of its goal, replaying the nominal standing tail and changing the tested root
command fields does not produce an adequate correction. A bounded next provider
should choose **coherent joint-position/velocity continuations** that initiate
locomotion again, approach the goal and brake, using the preserved motor interface.

1. Build a small bank from already motor-supported walking, turning and stopping
   clips. Record source hashes, ancestry, entry joint/velocity support and contact
   phase. Start with the two distant **clear-scene** failure states in this study.
   Match measured state to a supported entry and generate a continuous transition;
   reference translation, phase changes and foot placement are candidates requiring
   execution evidence, not automatically valid labels.
2. Gate the provider by measured goal error and causal speed. Do not add hidden
   deadline urgency. Reconstruct all 114 commands at each measured state and retain
   the motor weights. A bank switch needs a separately versioned, explicitly bound
   continuation contract; the present nominal-phase loader must not silently admit
   arbitrary reference jumps or skips past obstacles.
3. Require successful approach–brake–hold execution from displaced states before
   using those commands as corrective teaching. Preserve separate local and timely
   support labels. Extend to corridor recovery only after clearance-aware bank
   continuations actually succeed; the confirmation contacts identify a real missing
   behavior, not merely a scorer inconvenience.
4. Once new support spans multiple motion families, compare additional replay,
   motion-weight-matched replay, and fresh qualified corrections at fixed budgets.
   Keep whole-panel, paired-seed readouts and include failed expert attempts in cost.
   If supported commands remain hard to infer, test contiguous command-sequence
   training or one direct-reference comparator under the same public inputs and
   preserved motor components. Do not reopen multiple action backends at once.
5. After repeatable completion of the regression panel, build executed goal-switch,
   route-switch and posture-switch pairs plus independently constructed layouts.
   That study must isolate learning/acquisition value from the motion-to-scene
   generator. Current corridor variants and two evaluation seeds cannot earn that
   claim, and camera transfer remains a later observation-contract extension.

These are next experiments, not completed bank construction or validated goal
planning. The current study implements and tests the bounded command-feedback
provider, qualification/replay infrastructure and matched training comparison.

## Validation, cost and reproduction

The [evidence packet](evidence/navigation-continuation-20260913/README.md) includes
plans, source/command bindings, all scores, checkpoint hashes, training receipts,
provider reconstruction audits and original PNG/PDF trace figures. Its archived
scripts require the external raw arrays and checkpoints at the recorded paths or
explicitly adapted asset paths; it is not a self-contained dataset download.

There are **56 completed native attempts and 24,061 control steps**: twenty motor
continuation attempts, four original-teacher controls and thirty-two unassisted
navigation evaluations. Collection uses 8,732 physical steps, 8,732 frozen-motor
target forwards and 8,732 original-teacher diagnostic forwards. The separate
original-teacher controls execute 754 teacher actions after takeover. One failed
checkpoint-loading launch adds no task steps and is recorded separately. Neither
interventions nor failed startup jobs are pooled into navigation success rates.

Both new fits use 3,000 updates, for 6,000 updates of additional optimization in
total. Each fork inherits the parent's 9,000-update navigation history. GPU wall
times differ under the separate ongoing training job and are not efficiency claims.

**115 affected tests pass**, with Ruff and Black checks passing on the six affected
implementation/test files. Native execution verifies all twenty motor and four
teacher prefixes exactly. The all-probes loader audit also admits the three
successful candidate executions under their declared command-reconstruction
contracts; that audit includes nonselected providers and is not the training view.
Training receipts verify full inherited-motor tensor and anchor-action identity.
Exact validation commands are recorded in `validation.json`.
