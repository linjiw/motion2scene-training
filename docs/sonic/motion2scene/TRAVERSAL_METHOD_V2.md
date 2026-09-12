# Motion2Scene: execution-aware curricula for humanoid traversal

> This is the evolving design and development record. For the current seven-schedule, 114-dimensional implementation and dataset entry points, start with [Project guide](PROJECT_GUIDE.md). The [working manuscript](submission/traversal_method_v2.tex) is being reorganized around that implementation. Earlier binary, four-second, and 106-dimensional experiments below retain their original scope; they do not define the primary runtime.

Research and implementation specification, 2026-09-08. The mechanism below is a
proposed revision; each implementation and experiment must earn its own evidence
status. The existing manuscript and its archived experiments remain development
evidence. This document does not claim that the redesigned system has improved
held-out traversal, established robustness, or demonstrated hardware transfer.

The component implementation in
[`motion2scene_curriculum.py`](../../gear_sonic/dataset_generation/hallucination/motion2scene_curriculum.py)
now provides the revised finite geometric screen, matched-branch teacher choice,
verified cost/passage gap, uniform/coverage/regret replay, and synchronized
mechanical-work integration. Component tests do not establish integrated
closed-loop performance. Sensor integration, qualified option outcomes, generated
dataset counts, and final experiment status must be read from their actual run
records rather than inferred from these APIs.

The current integration also supplies causal floor/ceiling and occupancy history,
100 common sensor/state features, and a guarded binary linear readout through
`motion2scene_observation_history.py`, `motion2scene_closed_loop_policy.py`, and
`motion2scene_history_imitation.py`. The imitation teacher ranks passage first,
then measured passage time; relative time regret weights the cross-entropy loss.
Both-fail or unadmitted pairs receive no invented positive target. The expanded
five-reference adapter now has 19 recorded teacher/smoke episodes with 110D
inputs and six matched finite-schedule target tables. A separate forced-d085
development course passes two beams during one authored adaptation. These
component records do not establish collected DAgger iterations, learned-course
performance, or curriculum superiority.

The original four-second interface expands reference and entry time with a
3.30 s return request and 3.30--3.50 s guard. A separately qualified six-second
extension now adds short and sustained authored adaptations, entered at reference
phase 0.30 s and returned at 5.30/5.10 s. Nine measured development branches
demonstrate a useful duration distinction: both adaptations pass a 0.10 m beam,
while only sustained passes a 1.00 m passage. These are complete finite reference
schedules; arbitrary holding/looping, stopping, steering, adaptation-to-adaptation
switches and repeated enter/recover cycles remain unqualified.

The [16-episode six-second dataset](TIMED_DATASET_V1.md) preserves qualification,
contact diagnostics and all nine scene branches, with 4,768 aligned physical/sensor
rows, 19,072 physics steps and three exact 106D teacher targets. The
[measured duration figure](evidence/traversal-v2/duration_extension.pdf) shows
executed body envelopes, geometric construction witnesses and nominal physical
outcomes separately. A separate [15-episode comparison](TIMED_POLICY_DEVELOPMENT_V1.md)
now executes the 106D student on those three layouts at seed 8732. Learned and
scripted policies both pass 3/3 with identical choices and times. The student
misses 0.04 s available from short adaptation on the short beam; this establishes
policy integration without duration-decision or curriculum superiority.

Command entry/return times and sensor timestamps below use **reference phase**.
The driver records the physical state, advances the 50 Hz reference cursor by
0.02 s, then captures the sensor/command packet. Traversal costs use the physical
recording clock. The aligned dataset masks identify post-recording command resets
and exclude subsequent first-episode sensor history while retaining all raw
packets and physical rows. Equal row counts alone are not an alignment check.
The newer timed packets also provide explicit capture elapsed time; their entry
at command tick 15/reference phase 0.30 s uses physical state at elapsed 0.28 s.

The [four-branch sensor acquisition](../../../research-data/groot-wbc/m2s-history-d085-acquisition-v1/result.json)
now supplies actual 65-ray measurements and exact 100-dimensional decision
inputs. At 0.30 s, walking and d085 branches have matched physical prefixes,
states, and inputs within each scene. Both pass the absent scene in 2.44/2.86 s;
only d085 passes the designed contrast, in 2.96 s. Thus the two admitted teacher
targets differ for physically measured reasons. This is a two-context development
integration corpus, not a held-out or dataset-aggregation result.

The [eight-episode policy evaluation](../../../research-data/groot-wbc/m2s-history-d085-policy-evaluation-v2/result.json)
executes the frozen binary model on those same two layouts under seed 8732,
following acquisition at seed 8731. Learned and scripted policies each pass 2/2
in 2.46 s (absent) and 2.98 s (contrast); always-walk passes 1/2, and
always-adapt passes 2/2 in 2.88/2.98 s. The learned policy matches the strong
scripted baseline. Its 0.42 s saving on the absent condition relative to constant
adaptation is a measured local cost difference, not an arbitrary request penalty
or demonstrated curriculum advantage. The eight episodes use 6,368 physics steps.
This fit does not establish an optimal choice to wait before adaptation.

## Research objective and operational scope

A motion prior supplies candidate behavior. The frozen controller determines which
transitions can actually execute. Sensor history determines when the robot can
select them. Motion2Scene should connect these three facts to training environments
that improve closed-loop traversal under a measured acquisition budget.

The initial task is traversing static overhead constraints along a specified
approach, including short beams, extended passages, and sequences of obstacles.
The policy chooses a qualified motion option at legal decision points. Goal-directed
steering, route selection, dynamic obstacles, jumping, and hardware transfer are
separate extensions. A dataset can support later navigation research without the
present experiment establishing general navigation capability.

**Intended methodological claim.** Controller-executed motion alternatives can
identify training scenes in which an observable decision preserves an executable
continuation; a curriculum of physically verified decisions can teach a high-level
policy when to enter an adaptation, how long to maintain it, and when to recover.
Whether this mechanism improves passage or acquisition efficiency is a hypothesis
to test on new courses.

The three intended contributions are:

1. A transition-conditioned option representation connecting executed body
   envelopes, command legality, continuation, and an empirical sensing deadline.
2. A scene-construction and curriculum algorithm with two explicit admission
   conditions: a feasible positive option and information available before its
   entry deadline. Learned proposals and regret prioritization are optional,
   separately ablated components of this algorithm.
3. A reproducible humanoid traversal dataset and evaluation protocol separating
   capability, policy decisions, sensor representation, and acquisition cost.

These are contribution targets, not completed performance claims.

## What the development evidence establishes

The completed command outcomes are directly available in
[`submission/evidence/matched-commands.csv`](submission/evidence/matched-commands.csv).
The comparator table is
[`submission/tables/comparators.tex`](submission/tables/comparators.tex); the
interpretation and development overlap are retained in
[`submission/paper.tex`](submission/paper.tex).

| Outcome under the original interface | Conditions |
| --- | ---: |
| Both commands pass | 14 |
| Only d040 passes | 10 |
| Only walk passes | 0 |
| Both commands fail | 12 |
| Total | 36 |

For these commands, entry time, return window, and deterministic execution settings,
the outcome ceiling is `(14 + 10) / 36 = 66.7%`. Always-d040, the analytic-trained
selector, and scripted rays reach 24/36. The learned-constructor selector reaches
22/36. Avoided requests have not established lower mechanical work, traversal time,
or instability. Solving any of the twelve both-fail cases requires a changed
execution alternative or route; retraining the existing binary selector cannot
raise this ceiling.

The old paper also reports conflicting walking labels for exactly equal complete
214-dimensional inputs. This is a representation constraint, not evidence that a
larger deterministic classifier can solve those labels. The original finite
perturbation contrast contract failed for most nominal witnesses. The revised
screen below must therefore receive a new protocol identifier; it does not
retroactively satisfy the old contract.

## Problem formulation

Let `x_t` be the simulated robot state, `h_t` the low-level controller and command
history, `q` a scene, and `nu` a specified sensor configuration. The frozen tracker
`K` executes a legal option `a` from `(x_t, h_t)` and produces trajectory
`tau = Execute(K, x_t, h_t, a, q, seed)`. The teacher may use privileged state for
simulation branching and scoring. The deployed student receives only sensor
history, observable robot state, active option/phase, and the command legality
mask. Ground-truth scene parameters and branch outcomes are never student inputs.

The student acts at repeated legal decision times `t_k`:

```text
a_k ~ pi_theta(sensor_history_0:t_k, proprioception_t_k, legal_options_t_k).
```

The objective is lexicographic: maximize full-course completion first, then reduce
declared measured costs. Report time, positive/absolute mechanical work, and
switching overhead separately even if training uses a weighted sum. Define work
using actual applied actuator torque and joint velocity on a common time base:

```text
P_abs[t] = sum_j abs(tau_applied[t,j] * dq[t,j])
W_abs = sum_t (P_abs[t+1] + P_abs[t]) * (time[t+1] - time[t]) / 2.
```

This is the implemented trapezoidal estimate of absolute mechanical work in
joules, not battery energy. Positive work may be added as a separately defined
metric; it is not currently part of this integration API. Missing torque
records produce a missing cost field, not a zero or a position-error substitute.
Failure-conditioned elapsed time and work are retained; successful-run averages
state their denominator. Weights, units, cost normalizers, timeout, and failure
rules are fixed on development data before evaluating the new method.

For a registered evaluation set `E`, report two quantities:

```text
capability(A) = mean_e[exists physically verified successful continuation using A]
decision_gap = capability(A) - actual closed_loop_passage(pi, A).
```

The capability quantity is only a lower bound on attainable capability unless the
allowed branches are exhaustively searched. A finite teacher is the best verified
teacher under its declared branch budget, not a globally optimal oracle.

## A. Qualify executable options before acquiring training scenes

Represent each option by its source clip/ancestry, controller version, legal entry
phase or condition, playback/transition settings, supported termination or return,
and duration/speed parameters only when the installed interface supports them.
Start by varying legal entry timing for existing references. Next qualify additional
low-height references and longer supported adaptation windows. Repeating frames or
retiming a clip is a new execution intervention that needs qualification.

The current d040/d055/d070/d085 library consists of **authored `local_crouch`
edits of a Kimodo neutral reference**, executed by the frozen SONIC tracker.
These are not separately generated low-height skills from the learned prior.
Their useful capability is established by the recorded controller executions;
transfer to a broader prior-generated option library remains a future test.
The [candidate lineage](../../../research-data/groot-wbc/cg-wbc-v2-shared-seed-confirmatory/e1_controlled_duck_ladders_v1/candidates.json)
binds the Kimodo source CSV, per-level operator report, and converted CSV/PKL
hashes. Its [original registration](../../../motion2scene/experiments/registrations/E1_CONTROLLED_DUCK_LADDERS_V1.json)
sets `local_crouch` at route station 0.55/window 0.30 with requested drops
0.040/0.055/0.070/0.085 m. Original reference-only candidate status is distinct
from the physical qualification reported below. See the
[installed generation capability audit](INSTALLED_GENERATION_CAPABILITY_V1.md).

Qualification replays the entire approach, entry, maintenance, exit, and recovery.
Record achieved robot geometry, contact, progress, stability, command events,
transition delay, and measured cost. Group forecasts by approach state and history;
do not paste a reference envelope or a different approach's executed envelope onto
a branch. Store failures alongside qualified candidates.

For option `a`, let `E_a(x_t,h_t)` be its time-indexed executed body geometry and
`T_enter(a,x_t,h_t)` its measured delay to attain the clearance envelope required
by a particular passage. A nominal minimum pelvis height is insufficient:
head/torso/arms and recovery can be the limiting geometry. An option is admitted
only for the tested initialization/transition regime. A conservative delay bound
must specify its finite samples and uncertainty margin.

**Deliverable gate A:** publish the option-by-condition physical success matrix
and original-versus-expanded union of successful conditions. More candidate
options is not greater capability. If no new command succeeds on an original
failure and no measured cost improves, investigate execution before attributing
benefit to the generator.

**Completed qualification.** The
[option qualification record](../../../research-data/groot-wbc/m2s-option-qualification-v2/result.json)
executes seven alternatives on development source 41002, physics seed 8731:
walking; d040 entered at 0.20, 0.30, and 0.40 seconds; and d055/d070/d085 entered
at 0.30 seconds. All seven complete the empty-scene qualification, including legal
entry and return, no fall/reset/refusal, contact synchronization, and a matched
recorded approach prefix. The experiment consumes 5,572 physics steps. Empty-scene
crossing times are 2.44 seconds for walking, 2.50 for each d040 entry, and
2.78/2.82/2.86 for d055/d070/d085, respectively; these are one-run descriptive
measurements, not a statistically established cost ranking.

All seven fail the
[low-beam check](../../../research-data/groot-wbc/m2s-option-lowbeam-v1/result.json)
at route progress 0.65 and underside 1.18 m, also consuming 5,572 physics steps.
The d055 run resets; its later frames cannot rescue its failed first episode.
Thus the expanded interface is executable in the qualified empty setting, but
has not solved that earlier capability failure. Actual mechanical work remains
unavailable: the installed implicit PhysX actuator exposes an estimated effort,
which is not accepted as measured applied torque. Work values stay null.

**First measured capability extension.** An executed-geometry proposal at route
progress 0.55 and underside 1.249 m was registered before its physical trial.
The completed
[five-option experiment](../../../research-data/groot-wbc/m2s-option-contrast-v1/result.json)
on source 41002, seed 8731 yields:

| Option | Passage | Maximum recorded beam force | Passage time |
| --- | --- | ---: | ---: |
| walk | Fail | 2,731.24 N | — |
| d040 at 0.30 s | Fail | 864.16 N | — |
| d055 at 0.30 s | Pass | 0 N | 2.94 s |
| d070 at 0.30 s | Pass | 0 N | 2.94 s |
| d085 at 0.30 s | Pass | 0 N | 2.96 s |

All five recorded approaches match their qualification prefix; the three passing
adaptations execute legal entry/return without a fall, reset, or refusal. This
demonstrates one development condition that the original walk/d040 responses
cannot pass and additional qualified references can pass. It is a capability
extension at that condition, not a learned-policy gain, source transfer, or a
change to the original 36-condition result. The five branches cost 3,980 physics
steps; perturbation executions remain unverified in this nominal study.
The additional [entry-time study](../../../research-data/groot-wbc/m2s-option-entry-contrast-v1/result.json)
records d085 at 0.20 and 0.40 s passing the same condition in 2.96 s each,
with legal return and no fall/reset/refusal; its walking branch fails. This adds
2,388 physics steps. The final tested entry time is not a demonstrated physical
deadline: later entry times have not been exhaustively evaluated.

The subsequent 15-branch height check uses source 41002 and the second physics
seed 8732. It executes the same five references at three specified heights:

| Underside (m) | Walk | d040 | d055 | d070 | d085 |
| --- | --- | --- | --- | --- | --- |
| 1.239 | Fail | Fail | Fail | Pass | Pass |
| 1.249 | Fail | Fail | Pass | Pass | Pass |
| 1.259 | Fail | Fail | Pass | Pass | Pass |

All successful branches have 0 N recorded beam force and no reset. d070/d085
times are 2.96/2.98 s at each height; successful d055 branches take 2.94 s.
The [lower](../../../research-data/groot-wbc/m2s-option-robustness-lower-v1/result.json),
[nominal](../../../research-data/groot-wbc/m2s-option-robustness-nominal-v1/result.json),
and [higher](../../../research-data/groot-wbc/m2s-option-robustness-higher-v1/result.json)
records total 11,940 physics steps. This is finite local sensitivity evidence
at three heights, not verification of all 113 geometric perturbations, continuous
robustness, or held-out layouts.

## B. Construct scenes that preserve a solution

Construct an analytic proposer first. For positive and contrast alternatives
`a+`, `a-`, use their executed geometries and common transition context. A learned
proposal can later amortize the same search:

```text
q ~ g_phi(q | E_a+, E_a-, x_t, h_t, nu).
```

For signed geometric clearance `c_a` (positive means separation), use the revised
finite screen

```text
min_delta_in_U c_a+(q + delta) >= m
c_a-(q) <= -m.
```

`U` is an explicitly enumerated perturbation set in specified coordinates and
units. It is not a continuum. The positive option retains clearance throughout
that set; the contrast is required at nominal placement. A perturbed scene may
correctly become both-pass. Geometric eligibility is a proposal label only;
execute every admitted branch to obtain its own physical passage outcome.

For efficiency examples where both alternatives pass, replace interference by a
verified, development-defined cost difference. The feasible alternative and the
relative cost are measured before assigning a preference. Preserve neutral
backgrounds and both-fail challenges in the dataset with appropriate roles.

The screen includes the complete option and its recovery. Multi-obstacle
composition must be checked with the resulting state and a verified continuation;
concatenating individually successful clips or clearance envelopes is insufficient.

## C. Couple sensing to the latest feasible entry

The proposed representation is a temporally accumulated egocentric map of floor
and ceiling surfaces with separate observed/unknown masks, measurement age, and
sensor calibration. A conventional single-surface elevation map can lose the free
space below an overhead obstacle. Floor/ceiling pairs cover the current overhead
task; general multi-layer geometry may require local occupancy instead.

Build the map from depth or ray returns generated at the configured sensor pose.
Publish pose source, extrinsics, field of view, near/far planes, rate, noise,
dropout, fusion resolution, retention horizon, and egomotion assumption. An ideal
simulator ray caster must be described as such. Rendering a recorded state is a
sensor simulation, not a new physical trajectory. A map made from true beam
coordinates is privileged and is not this representation.

For each candidate scene and executed approach, determine the first qualifying
observation time `t_seen` using a fixed sensor-evidence rule. Then require

```text
t_seen + T_sensing + T_enter + T_margin <= t_obstacle.
```

For a nonuniform approach, derive `t_obstacle` from the achieved approach and
obstacle footprint, rather than distance divided by a nominal speed. The accepted
frame must actually be delivered before the legal decision point; teacher-only
visibility cannot satisfy this condition. This is a measured timing eligibility
screen, not a guarantee of perfect scene recognition.

Retain late-visible and unobserved challenges in a separate evaluation stratum.
Compute observation-equivalence groups on the student inputs/history. When a common
verified passing action exists across a group, use that conservative action as a
feasible shared target. Do not prioritize contradictory optional cost labels as
unlimited learning opportunity. If no shared feasible response exists, mark the
example unresolved by this sensor/action interface. Slowing and stopping count as
responses only after their actual execution is supported and qualified; the old
refusal that continues walking is not a protective stop.

## D. Obtain teacher targets and aggregate student states

At a matched decision state, search legal options and a finite continuation
horizon. Rank successful continuations first, then measured cost. Store all tested
branches, including unsuccessful ones, the teacher branch budget, and its selected
target. Branch restore must include simulator state, controller observations and
hidden state, reference phase, prior actions, and random state. When restoration
cannot reproduce this, replay the identical prefix and verify state/history
agreement before comparing branches. An unmatched restart is not a counterfactual.

The teacher target is

```text
a_star = first_action(argmin_verified_passing_continuation C(continuation)).
```

If there is no verified successful continuation, do not invent a successful
teacher action. Keep the failure for feasibility learning or challenge evaluation;
use a recovery target only if physically verified. The teacher's continuation
search and its sensing constraints must be explicit, because a privileged target
can be impossible for the student to infer.

Begin with imitation on teacher decisions, then collect student-visited states,
label legal counterfactuals, and aggregate them with the original corpus. This is
the DAgger-style intervention motivated by action-dependent observation
distributions. It does not inherit a realizability or no-regret guarantee in a
partially observed task with a privileged finite-search teacher.

## E. Learn proposals and prioritize verified learning opportunities

LfLH motivates learning the inverse map from useful motion to scene distributions.
Here the positive supervision should be physically verified scenes conditioned
on both executed alternatives and transition/sensor context. Begin with weighted
maximum likelihood or a low-dimensional conditional density over scene parameters;
record architecture, input units, bounds, seed, weights, and rejected proposals.
Use a mode-preserving mixture or coverage bins before introducing a larger model.

After a frozen student rollout and matched teacher verification, the implemented
empirical gap keeps passage before successful-branch cost:

```text
gap(q) = unknown, if no passing teacher or no admitted/legal student branch
         1,       if teacher passes and verified student fails
         max(0, (C_student - C_teacher) / max(C_student, C_teacher, 1e-12)),
                  if both pass
coverage(q) = inverse count of its registered option/scene coverage bin
p_k(q) = epsilon_uniform * Uniform(pool)
         + epsilon_coverage * Normalize(coverage)
         + (1-epsilon_uniform-epsilon_coverage) * Normalize(verified_gap).
```

Unknown gaps receive zero regret weight. Such examples remain available for
uniform/coverage diagnostics; downstream training must still require an admissible
target and preserve their role. If every gap is zero, the regret component becomes
uniform. The API defaults to `epsilon_uniform = 0.2` and
`epsilon_coverage = 0.2`; these are implementation defaults, not experimentally
selected values. Final values are fixed on development data. Scores name the student checkpoint,
teacher checkpoint/search, observations, and rollouts that produced them. Recheck
stale priorities after policy updates. Prioritize only information-feasible targets;
keep a separately reported stream of unresolved challenges without treating their
raw BCE as reducible regret. Small measured cost differences require a declared
practical threshold, so rollout noise does not become artificial difficulty.

An adversarial variant mutates obstacle height, position, length, and course spacing
inside registered bounds, but must reapply solution and timing screens and verify
the resulting teacher solution. This adapts the solvable-gap idea of PAIRED and
the mutation/replay idea of ACCEL. It is not a reproduction of either algorithm,
a proof of minimax regret, or a new claim to adversarial generation itself.

The proposed learned component earns a contribution only by improving held-out
policy learning, verified coverage, or acquisition cost against the same analytic
objective. Report proposal training, rejected samples, all geometry queries,
teacher/student branches, simulation steps, and wall time. Existing development
measurements favor neither learned proposal efficiency nor learned passage over
the strong analytic constructor.

## Completed six-second duration extension

The [measured result](/home/linjiw/research-data/groot-wbc/m2s-timed-duration-teachers-v1/result.json)
contains nine complete branches on three development scenes, with a separately
generated and qualified six-second Kimodo neutral carrier and two authored
low-height profiles. The frozen tracker executes each complete schedule, including
entry at reference phase 0.30 s and return at 5.30 s (short) or 5.10 s (sustained).
This expands finite supported duration choices without introducing arbitrary
holding, looping or repeated adaptation.

| Scene | Neutral | Short | Sustained | Verified teacher |
| --- | --- | --- | --- | --- |
| Empty | Pass, 5.36 s | Pass, 5.86 s | Pass, 5.92 s | Neutral |
| 0.10 m beam, underside 1.265 m | Fail | Pass, 4.04 s | Pass, 4.06 s | Short |
| 1.00 m passage, underside 1.275 m | Fail | Fail | Pass, 4.64 s | Sustained |

Passage time uses the physical clock and is compared within a scene because
finish planes differ. All nine outcomes are measurement-admitted, including
three failures. Successful branches satisfy full-horizon stability and actual
return, not only a short collision-free prefix. Each scene has an exact three-way
matched physical prefix, sensor history and 106D entry input. The resulting three
teacher decisions rank passage first, then measured time. These nine branches
use 10,728 physics steps, 2,682 aligned sensor rows and 174,330 ray measurements.

The analytic screen selected the two constraints before their physical outcomes.
Short has a minimum outer-envelope clearance of 34.58 mm over 81 finite offsets
against a −33.67 mm nominal neutral inner witness. Sustained has 30.08 mm minimum
clearance against a −27.08 mm nominal short witness on the long passage. The
[figure and source receipt](evidence/traversal-v2/duration_extension.json) separate
these geometric proposals from nominal physical labels. The figure's height
curves use measured native outer-body geometry at physical elapsed time
0–5.94 s; target poses and root height do not replace the body envelope.

The screen used original executed trajectories that failed a strict nonfoot net
contact predicate. Those failures remain unchanged. A counterpart diagnostic
identifies inherited hip–wrist self-contact. Three separately registered new
captures qualify the options under an explicit environment-contact criterion:
all 30 robot bodies against 36 exact counterparts at 200 Hz, foot–floor support
allowed, no other environment normal force above 1 N, and complete synchronized
streams with legal entry/recovery and no fall/reset/refusal. Self-contact pairs
are retained separately; the new criterion does not retroactively satisfy the old
strict predicate or imply friction-force measurements.

Both actual beam surfaces are measured at physical elapsed 0.00 s. The short
beam's underside first appears at 0.42 s, later than the entry state at 0.28 s;
the long passage's underside appears at 0.00 s. These records establish early
surface evidence, not universal early underside visibility. Privileged box
geometry is used only in the endpoint association audit. The student receives
65 ideal rays, known-normal masks, 0.5 s/26-frame causal history, robot state
and action legality through the exact 106D feature schema.

The [five-policy comparison](TIMED_POLICY_DEVELOPMENT_V1.md) at seed 8732 is now
complete: learned and scripted policies pass 3/3, constant short and constant
sustained pass 2/3, and always walk passes 1/3. Learned/scripted times are
5.40/4.10/4.68 s on empty/short/long. The short constant takes 4.06 s on the short
beam, exposing a 0.04 s learned cost error. Constant sustained crosses the empty
finish plane without fall/contact/refusal, but its 0.30 s hold does not finish
within the 5.94 s recorded horizon. That task failure remains in the denominator
with null success time. No scene is held out and no curriculum advantage is shown.
The separate 15-episode archive adds 4,470 aligned 106D rows and 17,880 steps, with
no invented teacher targets. Later entry choices, learned timing, repeated
adaptations and held-out curriculum learning remain outside this comparison.

Seven further empty-scene schedules now qualify under the declared environment
criterion, including early/late schedules and an explicitly projected/spliced
prior-derived reference. Their raw prior rejection remains preserved. These
separate qualifications do not supply the authored duration performance above
or establish obstacle passage for the prior-derived candidate.

## Dataset: Motion2Scene traversal episodes

The latest [six-second release](TIMED_DATASET_V1.md) adds 16 actual recordings:
three original strict qualification failures, one counterpart diagnostic, three
new environment qualification episodes and the nine branches above. Its
[210.4 MiB archive](/home/linjiw/research-data/groot-wbc/m2s-six-second-development-dataset-v1-aligned.tar.gz)
retains 4,768 aligned physical/sensor rows and 19,072 physics steps. All 1,619
package hashes pass an independent portable-data audit. The nine exact 106D
records provide three teacher targets; the other seven retain sparse rays and
measured state without invented feature vectors. Raw priors and weights are
excluded. Original strict assessments, corrected analysis snapshots and the
analysis-correction registration are retained; no raw data or physical threshold
was changed to correct analysis. A separate [policy increment](TIMED_POLICY_DEVELOPMENT_V1.md) preserves all 15
new-seed policy recordings, 4,470 aligned rows and 17,880 steps, with 3,990 verified
file hashes and no inferred teacher targets. Earlier releases below remain immutable.

### Completed local development package

The exporter
[`motion2scene_export_traversal_dataset.py`](../../scripts/research/motion2scene_export_traversal_dataset.py)
has produced a local package at
[`m2s-traversal-development-dataset-v3-aligned`](../../../research-data/groot-wbc/m2s-traversal-development-dataset-v3-aligned/manifest.json).
Its verified manifest contains **72 paired groups and 144 physical-simulation
branches**, from development sources 41001, 41002, and 41003:

| Package role | Both fail | d040-only pass | Both pass | Groups |
| --- | ---: | ---: | ---: | ---: |
| Canonical regression pairs | 12 | 10 | 14 | 36 |
| Nominal generated acquisition pairs | 6 | 18 | 12 | 36 |
| Total | 18 | 28 | 26 | 72 |

No walk-only pair occurs. The shared training backgrounds are not separately
included. Counts describe a repackaging of recorded development executions,
not newly acquired labels or a new held-out dataset. The package retains failures,
reset-spanning trajectories, ideal ray history, contact arrays, decision features,
and branch provenance, with pickle-free arrays and relative data paths. The clean export strips simulator
object identity from sensor records. Historical
packets have no measured surface normals or torque measurements; no work labels,
course continuations, steering labels, real sensor noise, or hardware trials are
supplied. These records support option-outcome learning, sensor-memory development,
and overhead-traversal regression. They do not support a learned-course or
navigation claim. The aligned v3 preserves all 28,656 physical rows and raw
packets, with 27,358 eligible and 1,298 ineligible sensor packets. Historical
pose fields are unavailable, so verification confidence is explicitly phase-only.
Failed-branch traversal times are null; recorded crossing frames remain available.
The original clean v2 remains immutable, with a separate
[erratum](../../../research-data/groot-wbc/m2s-traversal-development-dataset-v2-clean.ERRATA.md).

### Newly executed option and history package

The second [dataset manifest](../../../research-data/groot-wbc/m2s-executed-options-development-dataset-v2-aligned/manifest.json)
and [35.5 MiB archive](../../../research-data/groot-wbc/m2s-executed-options-development-dataset-v2-aligned.tar.gz)
package **49 newly recorded episodes across nine development studies**, source
41002, with 30 passing and 19 failing episodes and 39,004 recorded physics steps.
The [exporter and audit](../../scripts/research/motion2scene_export_option_dataset.py)
verify 501 file hashes and preserve 9,751 raw sensor packets: 9,661 are eligible
for first-episode sensor history and 90 are excluded. All requested completed
rows, reset-spanning trajectories, and nested tracking errors are preserved.

| Study | Episodes | Pass | Fail |
| --- | ---: | ---: | ---: |
| Empty qualification | 7 | 7 | 0 |
| Earlier low beam | 7 | 0 | 7 |
| Designed contrast | 5 | 3 | 2 |
| d085 entry-time comparison | 3 | 2 | 1 |
| Sensor-history teacher acquisition | 4 | 3 | 1 |
| Lower / nominal / higher height checks | 15 | 8 | 7 |
| Binary policy execution | 8 | 7 | 1 |

Thirty-seven episodes contain the legacy 214D single-decision schema; twelve
contain actual 100D repeated sensor/state inputs, with names and action-mask
dimensions recorded. Never concatenate these schemas. Sensor packets are
whitelisted to measurements and poses; requested actions, policy logits, teacher
targets, scene geometry, source IDs, and outcomes live separately. The package
retains two explicitly matched teacher groups. Its one failed startup attempt
is a separate 10.329 s infrastructure record, with unknown startup physics/query
counts; it supplies no physical failure label. Applied torque remains unavailable.
No course continuation, new motion ancestry, or navigation evaluation is added.

Each episode includes `sensor_alignment.npz` and JSON reasons/confidence.
The twelve modern histories verify recorded state and pose; legacy histories
carry weaker phase-only confidence. Apply the per-packet mask instead of
truncating a fixed tail. The original v1 release and archive remain immutable,
with an [erratum](../../../research-data/groot-wbc/m2s-executed-options-development-dataset-v1.ERRATA.md).
Outcome, cost, reset, acquisition, physical-array and student-input fields match
the original export; the correction adds chronology metadata.

The manifest SHA-256 is
`b3e7cc2e2dcda4c7723abdbd756a8a58946abb4a1b6a062cccbec0053f60004d`;
the archive SHA-256 is
`1ec9b907a6c94d5c4224c45a92b253a9508ff9299064da8c8d133efa9abf2723`.

### Multi-option finite-schedule package

The separate [19-episode manifest](../../../research-data/groot-wbc/m2s-multi-option-development-dataset-v1-aligned/manifest.json)
and [22.3 MiB archive](../../../research-data/groot-wbc/m2s-multi-option-development-dataset-v1-aligned.tar.gz)
contain 18 teacher schedules and one distinct smoke episode, without deduplicating
actual executions. All episodes use source 41002 and the five registered options
`neutral`, `d040`, `d055`, `d070`, `d085`. There are 15 passing and four failing
episodes, 15,124 physics steps, and 3,781 raw sensor packets; 3,762 packets align
exactly to recorded first-episode state. Each episode's final post-recording
reset packet is retained with an ineligible mask.

The named 110D inputs contain the common 100D sensor/state schema plus five active
option bits and five legality bits. Configured preference is distinct from the
actually executed index: actual entry and recovery are derived from logged
switches, and a learner that remains neutral retains a no-entry execution.
Measurement admission, option IDs and command provenance remain separate from
student features.

Six portable finite-schedule targets map every continuation to an archived
episode ID and every exact matched input to an eligible NPZ row. Action zero
means **wait at this decision**, with the selected future continuation explicitly
recorded; it does not mean commit to walking through the encounter. Targets are
complete only for the registered finite continuation set. They do not establish
optimality outside that set or an actual learned-policy result. The manifest
SHA-256 is `de48a5edc39aab52f3f532aa6cb076cfbe89877b3f82301e7de058cee903e2fb`.

### Separate development course evidence

The [two-beam smoke](../../../research-data/groot-wbc/m2s-course-two-beam-smoke-v4/result.json)
executes one d085 entry at reference phase 0.30 s and recovery at 3.30 s on a fresh
development course. It completes both constraints in 3.04 s of physical recording
time, with zero measured beam force across 792 substeps, two beams and 30 robot
bodies. Both authored beam lengths (0.12/0.18 m) are audited, and all 198 sensor
packets match recorded physical state. Successful acquisition costs 48.94 s wall
time; two earlier infrastructure attempts add 30.074 s and remain linked in
`prior_attempts.json`. This record is separate from the 49/19-episode packages.
It demonstrates continued passage during one authored adaptation; repeated
adaptation, learned-course performance, route choice and held-out generalization
remain unmeasured. See [course contract](DEVELOPMENT_COURSE_V1.md).
The separate [portable demonstration](../../../research-data/groot-wbc/m2s-two-beam-development-demonstration-v1/DATASET_CARD.md)
and [1.04 MiB archive](../../../research-data/groot-wbc/m2s-two-beam-development-demonstration-v1.tar.gz)
retain both geometries and force streams, all 198 physical/state/110D rows,
actual switches, raw whitelisted measurements, exact alignment masks and prior
infrastructure costs. The audit reproduces every course score from portable
arrays, including contact from the second beam. Manifest SHA-256:
`7d1a32e40e9b82e109c13a50ceaa7e04734205f132d6a9eb2d96bb34d8c41af6`.

### Multi-option policy development package

A fourth [option-policy package](../../../research-data/groot-wbc/m2s-multi-policy-development-dataset-v1-aligned/manifest.json)
contains 18 actual policy episodes, 16 pass/two fail, with 14,256 physics steps.
All 3,564 recorded sensor packets are exactly aligned; the 110D schema, actual
executed option, configured preference and source/model provenance remain separate.
The [comparison](MULTI_POLICY_DEVELOPMENT_V1.md) uses softmax and ridge-value
students trained from the same original 18 teacher branches. Softmax selects
d040 on empty (2.50 s), versus neutral for the value student and scripted policy
(2.44 s). On nominal/lower scenes the value student takes 2.94/2.96 s, versus
2.94/2.94 s for scripted d070. The value fitter repairs two local cost errors;
the lower-scene cost gap remains. A subsequent nine-branch lower-scene acquisition
costs 7,128 physics steps and supplies matched student-visited neutral prefixes.
The same value fitter trained on all 27 branches then passes three actual episodes
in 2.50/2.94/2.94 s, adding 2,376 physics steps. Lower improves by 0.02 s, empty
regresses by 0.06 s and nominal ties. The sum worsens by 0.04 s; this measured
aggregation comparison supplies no net benefit. All layouts and sources remain
development evidence. Its records are preserved separately from the original
18-policy archive. A separate phase-specific fitter comparison on the same
18/27-branch data pair now passes all six physical episodes. Its times change
from 2.44/2.94/2.96 to 2.44/2.94/2.94 s, matching scripted d070 and removing one
50 Hz tick on lower without the shared-model empty regression. This bounded
same-fitter data update is not a held-out or constructor comparison. The
[incremental 18-episode package](../../../research-data/groot-wbc/m2s-aggregation-increment-development-dataset-v1-aligned/manifest.json)
retains lower9, shared-new3 and phase6 (13 pass/five fail), 14,256 physics steps,
3,564 aligned packets and three new portable teacher decisions. See the
[offline baseline reuse interface](DATASET_BASELINE_V1.md).

The [proposed implementation amendment](TRAVERSAL_IMPLEMENTATION_AMENDMENT_V2_PROPOSED.md)
identifies the unresolved source, finite-horizon and course-interface mismatch
with the original 72-episode recipe. It does not adopt or execute a restricted
benchmark and does not modify the locked JSON.

The separate [six-second geometry lock V3](TRAVERSAL_EVALUATION_LOCK_V3.md) reserves
12 single layouts and six two-beam compositions on the exact newly qualified
six-second neutral before short/sustained option physics. Both nominal and
stress boxes are fixed in world space; no future reference can silently move
them. All layouts remain, without clearance or outcome filtering. All 162
nominal/stress layouts and 216 beams are materialized and natively audited.
Common timed policy implementation, course sequences and finite-horizon terminal policy remain pending;
V3 does not complete the original V2 evaluation.

### Dataset contract for the expanded method

The release unit is an encounter/course episode with its motion ancestry and
matched branch group, not an isolated frame. Provide a manifest, versioned schema,
checksums, data card, loader, validation command, fixed splits, and small runnable
examples. A code/schema release is not a collected dataset; report actual counts
by evidence tier and completion status.

| Record | Required content |
| --- | --- |
| Provenance | Episode/group IDs; source ancestry; reference, robot, controller and code hashes; simulator version and seed; proposal/checkpoint IDs |
| Option | Reference ID and legal entry/return settings; qualification regime; full command-event log; success/failure of qualification |
| Scene | Explicit geometry and units; scene/layout/course composition ancestry; perturbation ID; generation bounds and construction role |
| Observations | Timestamped raw sensor measurements or reproducible rendering recipe; calibration and pose source; masks; causal history/map; observable state and legal mask |
| Verification | Paired/multi-option branch linkage; prefix/snapshot identity; screen results; contact and progress; complete success/failure reason; full-course continuation status |
| Costs | Applied torque/velocity provenance; work convention and units; elapsed time; switch events and overhead; missing-value masks |
| Supervision | Teacher action and tested alternatives; student checkpoint/action; feasible set; timing class; ambiguity/common-response class; teacher budget |
| Acquisition | Proposed, rejected, executed, and admitted counts; all branch steps; geometry queries; proposer/teacher/learner wall time |
| Split | Development/train/validation/test role; ancestry component; course family; lock time; manifest digest; prior-inspection flag |

Split by connected components of motion ancestry, scene template/layout ancestry,
and course composition reuse where those axes are claimed held out. Keep paired
branches, perturbations, adjacent frames, and duplicated source clips together.
Create and hash the evaluation manifest before tuning, and keep evaluation outcomes
out of curriculum updates. A generated manifest is only a proposed split until
ancestry exclusions and source availability are audited. A downstream motion-source
holdout does not imply absence from the inherited motion prior's pretraining.

Use separate datasets or explicit roles for: option qualification; teacher-labeled
training; student-state aggregation; validation/tuning; locked evaluation; and
historical development. Preserve attempted failures, screen rejections, corrupt
runs, and exclusions with reasons; never manufacture labels for unexecuted branches.
The release can initially distribute derived records and scripts with pointers to
licensed source assets. Verify redistribution terms before claiming that the
inherited motion clips or controller weights are part of an open dataset.

Downstream supported uses should follow stored evidence: offline option feasibility,
timing prediction, floor/ceiling perception, imitation, and actual simulator
closed-loop traversal. Navigation and route planning require destination/steering
interfaces, alternate routes, and corresponding goal-completion evaluation.

## Evaluation matrix and resource accounting

Freeze task bounds, split hashes, scoring, budgets, command set, sensors, learner,
teacher horizon, acquisition seeds, tuning rules, and stopping conditions before
new held-out execution. Reuse the old bank only for regression/development.

The new [locked recipe](TRAVERSAL_EVALUATION_LOCK_V2.md) now fixes exact geometry
for 24 single-beam layouts and 12 two-beam compositions, two physics seeds, the
comparison contract, perturbation sets, and acquisition budgets. Its hash-linked
runtime implementation lock is pending. These are new layout instances on
development carriers, not held-out motion sources; course feasibility is untested.

| Question | Controlled comparison | Primary evidence | Confound ruled out |
| --- | --- | --- | --- |
| More executable capability? | Original commands vs qualified expanded options, same conditions/scoring | Union of physically passing continuations; cost on shared successes | Selector changes cannot create new physical options |
| Better curriculum? | Uniform, target-only, strong analytic contrast, execution+observation-aware construction; same improved policy/options/sensors | New-course completion vs total acquired rollout steps | Improved sensor or action set alone |
| Useful observation timing? | Same proposer and sensor with timing admission on/off | Avoidable late-entry failures and new-course completion | Easier sensor or larger network |
| Useful temporal floor/ceiling map? | Current-frame rays vs temporal floor/ceiling, fixed corpus/options/training | Completion and ambiguity/deadline error strata | Changed scene corpus |
| Useful learned proposal? | Analytic and learned proposals under the same acceptance objective | Verified options/scenes per acquisition cost; downstream learning curves | Different screens or privileged inputs |
| Useful adversarial curriculum? | Uniform verified-pool sampling vs feasible-gap/mutation mixture | Course completion over matched acquisition budgets and corpus seeds | More teacher branches or endless hard-case replay |
| Useful sequential learning? | Initial imitation vs aggregate student-state labels | Actual complete unseen-course rollouts | Offline lookup based on fixed earlier actions |

Keep always-walk, constant-adaptation with the applicable qualified interface, a
development-tuned scripted policy, and the finite privileged teacher. Give simple
baselines supported durations/entry options when that is their defined interface;
do not force their old return window while upgrading only the learned policy.

Partition evaluation into short beams, sustained passages, repeated transitions,
and late/unobserved challenges. Define full-course success as goal-region arrival
within the registered timeout, without reset, disqualifying contact, or fall.
Contact thresholds, body sets, stability criteria, goal geometry, and invalid-run
rules must be fixed in the executable scoring configuration. Report exhaustive
outcomes: success, disqualifying contact, fall, timeout/no progress, illegal command,
and infrastructure-invalid. Specify precedence where failures coincide; retain
all underlying flags. Infrastructure-invalid runs are shown in attempted counts.

Report capability and policy completion separately, plus time/work/switch metrics
and late decision counts. Show numerator/denominator for each stratum. Course
layouts/compositions and independently acquired corpora are experimental units;
frames, branches, physics seeds on one course, and neighboring perturbations are
not independent new environments. Report paired course-level differences and
uncertainty grouped at the course/corpus level, without presenting correlated
condition-level tests as motion-source evidence.

Use total simulator steps across all branches as the primary acquisition budget,
with proposal/teacher/learner computation and wall time separately reported.
Different branch lengths and counts make equal labels insufficient. Register
multiple budgets after measuring throughput on development, and fit separate
corpora with independent acquisition seeds. If resources permit only one corpus,
report that limitation; refitting one corpus does not measure acquisition variance.

## Claim--evidence map and completion criteria

### Completed geometric proposal pilot

The strengthened development pilot at
[`m2s-solution-curriculum-v3/result.json`](../../../research-data/groot-wbc/m2s-solution-curriculum-v3/result.json)
applies the revised screen to recorded empty-scene executions, conditions on the
actual 15-frame approach history, and fits a conditional proposal to accepted
**geometric** examples. This is a weaker supervision stage than the intended
physically verified proposal corpus. It includes nonlearned local resampling of
the same analytic witnesses as a strong comparator.

| Development context | Grid scenes | Nominal contrasts | Revised-screen accepted | Also all-offset contrast | Analytic resample accepted / sampled | Learned accepted / sampled | Uniform accepted / sampled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41001 | 1,380 | 28 | 10 | 1 | 29/64 | 11/64 | 0/64 |
| 41002 | 1,380 | 19 | 1 | 0 | 52/64 | 10/64 | 0/64 |
| 41003 | 1,380 | 43 | 18 | 3 | 42/64 | 20/64 | 0/64 |
| Total | 4,140 | 90 | 29 | 4 | 123/192 | 41/192 | 0/192 |

The 29 accepted candidates versus four satisfying all-offset contrast demonstrate
the effect of changing the finite geometric contract. They do not demonstrate
physical robustness or more traversable scenes. The nonlearned resampler accepts
123/192 proposals (64.1%), versus 41/192 (21.4%) for the learned proposal. This
experiment supports retaining analytic construction as the primary method; it
does not establish a neural sampling advantage.

Coverage uses bins of **0.02 normalized route progress and 0.01 m underside**.
The raw result key mistakenly names the first dimension `2cm_station`; the
recorded computation divides normalized progress by 0.02, so it is not a metric
2 cm bin. Per-source accepted-bin counts are 11/2/20 for analytic resampling and
6/5/14 for learned proposals. The learned arm covers more bins on source 41002,
which supplied only one analytic exemplar, but less on the other two contexts.
These are declared coarse bins in three fitting contexts, not evidence of
broader scene populations or downstream utility.

Analytic search consumed 28,620 clearance queries and 8.727 seconds. Fitting took
0.919 seconds. Each of the three sampling arms consumed 43,392 clearance queries;
audit time was 12.238 / 12.553 / 12.698 seconds for uniform / analytic resampling /
learned proposals, respectively, with proposal time reported separately. The
full recorded run took 47.144 seconds. Search and fit costs cannot be removed
from a learned acquisition-efficiency claim. This pilot acquires zero physical
rollout steps and does not check observation timing, train a traversal student,
or evaluate unseen contexts. The earlier v2 pilot remains archived and is
superseded for this comparison by the full-history, stronger-comparator v3 run.

### Evidence ledger

| Claim | Mechanism/comparator | Status and exact evidence | Required next evidence / boundary |
| --- | --- | --- | --- |
| Original command ceiling is 24/36 | Exhaustive completed two-command table | Measured development; `submission/evidence/matched-commands.csv` | Bound to original interface and inspected bank |
| Expanded options increase capability at one development condition | Qualified extra execution alternatives vs original options | Measured: walk/d040 fail while d055/d070/d085 pass the registered source-41002 contrast; 5 matched branches | No new held-out gain; all seven options still fail the earlier low beam |
| Finite duration options expose a useful capability distinction | Complete short/sustained executions, solution-preserving scene screen and physical labels | Nine measured branches: both adaptations pass short beam; only sustained passes long passage; three exact 106D teacher targets | Development geometry only; 15 further policy episodes match the script and retain a 0.04 s short-beam cost error; physical stress remains pending |
| Proposed screen preserves geometric clearance for a positive alternative | Finite executed-envelope clearance test | Implemented; 29/4,140 accepted proposals in the development geometric pilot | Physical verification at every tested perturbation; no continuous guarantee |
| Sensor history retains actionable evidence in the development contrast | Actual 65-ray history and qualified entry times | Four acquired branches; upper evidence at 0.02 s, ceiling at 0.22 s; d085 entries at 0.20/0.30/0.40 s pass | Last tested entry is not the physical deadline; timing on/off ablation and unseen layouts pending |
| A perceptive binary student uses the useful option | Exact-input matched teacher labels and actual repeated-decision execution | Learned/scripted 2/2, always-adapt 2/2, always-walk 1/2 on two reused layouts under a second seed; 0.42 s absent-scene saving vs constant adaptation | Learner matches scripted; no curriculum superiority, optimal timing or held-out claim |
| Curriculum improves traversal | Same learner/options/sensors, alternative corpora | Unmeasured | Locked complete-course comparison across acquisition budgets/corpora |
| Learned/adversarial generation adds downstream value | Same screen, feasible gap with uniform mixture | Unmeasured downstream; analytic resampling accepts 123/192 versus learned 41/192 on fitting contexts | Strong analytic-resampling comparison and downstream learning, all branches counted |
| Development dataset supports reproducible option studies | Episode records, arrays, labels and provenance separated from student inputs | Immutable aligned historical, option, policy and aggregation packages; six-second release adds 16 episodes/19,072 steps/three 106D targets; separate policy increment adds 15 episodes/17,880 steps with no teacher targets; all development-only | Actual torque, source transfer and locked evaluation remain unavailable |
| One authored adaptation traverses two constraints | Forced d085, fresh two-beam development layout | One admitted physical episode, 3.04 s course time, zero recorded beam force and 198 exactly aligned sensor packets | No learned-course gain, repeated adaptation or generalization claim |
| General navigation or hardware transfer | Goal-directed route/steering or physical deployment | Outside present evidence | Separate supported interface and task evaluation |

The next completion decision is empirical: qualify useful execution options first;
then admit and physically verify one integrated analytic constructor; then train
and execute the common student on the locked courses. Rewrite abstract/results
as completed contributions only after those outcomes exist. If the expanded
interface fails to add passage capability, state that result and revise execution
instead of relabeling selector improvements as a capability advance.

## Primary literature checked for this revision

The following are the sources for related-work statements, not evidence for
Motion2Scene's proposed performance. Versions/links checked 2026-09-08.

- **LfH:** constructs obstacle configurations around existing open-space motion
  plans. The inverse scene-from-motion idea is established prior work.
  [Project and paper links](https://www.cs.utexas.edu/~xiao/Research/LfH/LfH.html).
- **LfLH:** learns obstacle distributions with an encoder and fixed motion-planner
  decoder, then trains a reactive planner on rendered observations. It motivates
  an amortized proposal, not a claim that our learned generator is inherently
  better. [IROS 2021 paper](https://www.cs.utexas.edu/~xiao/papers/lflh.pdf).
- **LfH-CP:** learns critical spatial/temporal obstacle constraints and procedurally
  expands them into dynamic trajectories. Diversity must be tied to downstream
  utility. [Author-hosted paper](https://people.cs.gmu.edu/~xiao/papers/lfh_cp.pdf).
- **PAIRED:** uses a protagonist--antagonist return gap to avoid the unsolvable
  extremes of naive minimax generation. Our verified finite teacher is a different
  implementation and supplies no inherited equilibrium guarantee.
  [Paper](https://arxiv.org/abs/2012.02096).
- **ACCEL:** combines regret-based curricula with level editing. Scene mutation
  plus prioritized replay is therefore an established ingredient.
  [ICML 2022 paper](https://proceedings.mlr.press/v162/parker-holder22a.html).
- **SONIC:** provides motion tracking and a kinematic planner using history and
  target keyframes, including skill-dependent keyframes. Published interfaces
  must be qualified in this installation before use.
  [Version 3](https://arxiv.org/html/2511.07820v3).
- **DAgger:** addresses policy-induced observation distributions through iterative
  imitation-data aggregation. [AISTATS 2011 paper](https://proceedings.mlr.press/v15/ross11a.html).
- **Perceptive Humanoid Parkour:** composes motion via matching, trains tracking
  experts, and distills a depth-based student with DAgger and RL. Motion composition
  plus perceptive teacher--student learning is already demonstrated.
  [Version 1](https://arxiv.org/html/2602.15827v1).
- **HumanoidPF:** represents humanoid--obstacle relations and mixes realistic
  scene crops with procedural obstacles for traversal. General indoor traversal
  plus hybrid scene generation is not a distinct novelty claim.
  [Version 2](https://arxiv.org/abs/2601.16035v2).

The defensible difference to investigate is the mechanism joining matched
controller execution, legal transition timing, causal perceptual availability,
and physically verified continuation in curriculum construction. Establishing that
difference requires the controlled comparisons above; this bounded literature
check does not establish an exhaustive novelty claim.
