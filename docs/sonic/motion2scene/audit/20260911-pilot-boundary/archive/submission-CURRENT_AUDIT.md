# Motion2Scene evidence reconciliation — 8 September 2026

**The frozen results reproduce. The scientific interpretation changes.** All 854
recorded executions were rescored from trajectories and synchronized 200 Hz beam
contacts, with no scorer or pin mismatch. This includes 838 original executions
and 16 separately registered post-hoc command-audit executions. All four nominal
fits reproduce bitwise. The nominal analytic selector and frozen scripted rule
make identical decisions on all 36 conditions. Always-d040 also passes 24/36.
The contribution is contrast-targeted training data and measured selectivity on
this development bank, not superior learned generation or traversal capability.

## Protocol and deviation table

| Item | Perturbation study | Nominal study | Evidence / disposition |
|---|---|---|---|
| Registrations | `M2S_ICRA_V1`, `M2S_ICRA_EXECUTION_V1`; construction and learning JSON registrations | `M2S_ICRA_NOMINAL_V1`; separate construction and learning JSON registrations | Markdown versions and hashes in `diagnostics.provenance.protocol_git_versions`; JSON hashes in `analysis.studies.*.registrations` |
| Run families | `m2s-icra-v1` labels; `m2s-icra-learning-v1` evaluation | `m2s-icra-nominal-v1` labels; `m2s-icra-nominal-learning-v1` evaluation | 82+540 and 72+144 physical executions; all admitted, none excluded |
| Geometry | 113 fixed offsets, ±20 mm x/y, ±10 mm z, ±0.02 rad yaw; 10 mm contrast margins | Nominal pose with same margins; 113 offsets retained as a diagnostic | Finite-grid minimum, not continuous worst-case robustness or probabilistic acceptance |
| Quota and order | First six outputs/carrier/arm assigned before acceptance; 18 generated slots + six backgrounds = 24 requested groups/arm | First three eligible outputs/carrier/arm selected; nine generated + same six backgrounds = 15/arm | Fixed cap 16 outputs/carrier/arm, 48/arm; no new draws, no post-label replacement or visibility filter |
| Realized groups (U/A/T/M) | 24 / 7 / 22 / 6 | 15 / 15 / 15 / 15 | Robust shortages retained, not replenished |
| Evaluation | 12 layouts × 2 seeds × 3 carriers = 72 traversal conditions; six policies = 432 runs, plus 108 backgrounds | 12 layouts × seed 8511 × 3 carriers = 36 conditions; four fits = 144 runs | Nominal uses one seed, **not** an average of repeats; seed 8512 belongs to the parent panel |
| Comparators | Scripted rays and privileged geometry physically evaluated with parent models | Same seed-8511 comparator executions reused after full matching audit | Same conditions; not new independent replicates or nominal-model background evaluations |
| Metadata deviations | Equal requested quotas do not imply equal acquired corpora | Stale JSON fields retain six slots/carrier and 72 requested generated groups; operative quota is three and actual selected groups total 36 | Protocol, implementation and executed selection agree; obsolete fields remain in original files |
| Scope deviation | — | Acceptance, selection order, quota and physics-repeat count differ | Original “changed and only this” wording is too strong; nominal is a separately specified, outcome-informed revision |
| Test exposure | 366-run snapshot covers 61 traversal conditions before nominal registration | 31/36 nominal conditions appear in that snapshot | Development-bank inference, not an untouched confirmatory test |

U/A/T/M denote uniform, analytic, target-only (`no_contrast` in records), and
learned contrast (`motion2scene`). The six shared label groups are C1 absent/raised,
C2 absent/blocked, C3 raised/blocked, at station 0.60; each background type occurs
twice. Their twelve command executions are acquired once and reused by every fit.

The parent and nominal panels overlap in all twelve layouts, all three carriers,
seed 8511, controller, generator checkpoints, and shared labels. Nineteen nominal
generated scenes match parent scenes at label seed 8722: nine uniform, nine
target-only, one analytic. Those scenes were reexecuted, not independently drawn.
Primary learned selector weights differ between studies and are separately pinned.

## Units, labels, scorer, and freeze boundary

- **Proposal:** one output scene/beam placement after the arm's internal search;
  it may embody many clearance queries. There are 48 per arm in each study.
- **Eligible/accepted scene:** passes the arm's registered finite geometric and
  shared guards. Eligibility is not a physical label. “Accepted selected” in the
  exported table additionally requires assignment to the finite label quota.
- **Encounter group:** one carrier, scene, label seed, pre-decision history and
  timing, with **both** frozen command outcomes; descendants/repeats remain grouped.
- **Command outcome:** one obstacle-present simulator execution of walk or d040,
  scored under the unchanged first-episode passage contract.
- **Useful physical contrast:** paired outcomes `(walk fail, d040 pass)`. The four
  labels `(11,01,10,00)` are all retained; no walk-only groups occurred here.
- **Critical scene:** geometric target-clear/walk-blocked forecast at nominal pose.
  Counts 47 analytic and 34 learned include all outputs before common exclusions;
  they do not count selected groups or physical successes. Their ray-hit counts
  are 44 and 24, respectively.
- **Decision-time observation:** one full 214-value input at 0.30 s, comprising
  144 ray values and 70 robot/timing values. Fifteen groups have thirty command
  labels but fifteen paired input rows; different groups can share identical inputs.
- **Condition:** carrier × layout × physics seed. Thus 36 and 72 are condition
  denominators, not numbers of independent motion sources. Nine is the selected
  generated-group count per nominal arm; fifteen includes six shared backgrounds.
- **Adaptation request:** action d040 selected at the one decision time. A
  **successful adaptation** is an executed d040 request achieving passage; it is
  neither every request nor every correctly classified outcome. Nominal analytic
  and learned models both achieve eighteen successful d040 passages.
- **Refusal:** both predicted commands fail threshold; runtime continues walking.
  It does not stop, protect, or establish safety.

The scorer uses all achieved body origins crossing the beam exit face plus 0.10 m,
remaining upright for sixteen 50 Hz samples (0.30 s), with no per-body beam-force
norm above 1 N through completion. If passage never completes, contact is checked
through first-episode end. The requested 200-step full-pass run records 199 frames
through 3.96 s when no reset occurs. Reset truncates the first episode; later
recoveries cannot turn its failure into a pass. Upright means root height at least
0.50 m and negative projected-gravity z at least 0.50. Four 200 Hz force samples
per control interval supply the measured maxima. This endpoint is not a continuous
collision guarantee or full-reference tracking acceptance.

Controller, commands, timing, original feature function, primary fitting procedure,
generator checkpoint, layouts, seeds, and scorer were not modified. The driver,
models, motion banks, physics configuration, and per-cell configuration hashes are
in the machine manifest. An inherited `source_commit` in original manifests is
lineage metadata, not proof of the exact execution checkout. We preserve it and
use the actual file/configuration hashes rather than invent an execution commit.

## Primary outcomes and simple-controller audit

| Policy | Passage / 36 | Requests / 36 | Unnecessary requests / 14 both-pass | Requests / 12 both-fail |
|---|---:|---:|---:|---:|
| Uniform | 14 | 0 | 0 | 0 |
| Analytic | 24 | 24 | 8 | 6 |
| Target-only | 14 | 0 | 0 | 0 |
| Learned contrast | 22 | 22 | 6 | 8 |
| Frozen scripted rays | 24 | 24 | 8 | 6 |
| Registered privileged geometry | 20 | 7 | 1 | 0 |
| Always walk, post hoc | 14 | 0 | 0 | 0 |
| Always d040, post hoc | 24 | 36 | 14 | 12 |
| Physical-outcome hindsight oracle, post hoc | 24 | 10 | 0 | 0 |

The registered privileged geometric forecast is not the outcome hindsight oracle.
The completed command table has fourteen both-pass, ten d040-only, twelve both-fail,
zero walk-only conditions. Analytic avoids six unnecessary adaptations relative to
always-d040 without losing passage. Learned avoids eight but misses two rescues.
Its additional selectivity is not evidence of a superior overall tradeoff.

Original nominal and parent seed-8511 records supplied both commands on 28/36
conditions. The separately frozen `SUBMISSION_COMMAND_AUDIT_V1` executed matched
walk/d040 pairs on the eight missing conditions (C1 layouts 02,03,06,08,09 and
C2 layouts 02,09,11; zero-based layout names). All sixteen were admitted. Repeated
walk outcomes and recorded pre-decision prefixes matched the originals exactly.
All 304 available same-command comparisons agree in passage and full captured
state/action/token trajectories. Full decision inputs, seed, banks, beam and
physical configuration also match. No later policy-dependent decisions occur.
Hidden simulator state was not snapshotted; the lookup is conditional on the
checked deterministic runtime, not justified for arbitrary sequential policies.

| Registered nominal pair (first / second) | Both pass | First only | Second only | Both fail | Exact descriptive p |
|---|---:|---:|---:|---:|---:|
| Learned / analytic | 22 | 0 | 2 | 12 | 0.5 |
| Learned / uniform | 14 | 8 | 0 | 14 | 0.0078125 |
| Analytic / uniform | 14 | 10 | 0 | 12 | 0.001953125 |
| Learned / target-only | 14 | 8 | 0 | 14 | 0.0078125 |

All registered parent comparisons and predictions, including failures, are retained
in the paper and machine outputs. Parent traversal counts (U/A/T/M/script/oracle)
are 28/39/28/28/46/39 out of 72. Analytic/uniform cells are `(28,11,0,33)`;
learned/analytic `(28,0,11,33)`; learned/uniform and learned/target `(28,0,0,44)`.
The eleven-discordance p is 0.0009765625. Original zero-discordance p is null;
there is no evidence of equivalence. Parent prediction flags are false, true,
true, true, false; all six nominal flags are true. Exact predictions, counts,
thresholds and scopes are in `diagnostics.registered_predictions`.

All registered parent background endpoints are exported. All six policies pass
all six absent and all six raised conditions, without requests. All six blocked
conditions fail per policy, with six resets and zero requests. The four parent
learners and privileged geometry record six refusals; scripted rays record none.
These are not physical background evaluations of the new nominal fits. Their
nominal-bank both-pass and both-fail behavior is measured directly above.

## Dependence and estimand

The estimand is balanced mean finite-bank passage on three inspected development
carriers, conditional on acquired corpora. The same layouts cross carriers and
the parent has two repeats per carrier/layout. Descriptive condition-level exact
tests assume independence that this design does not establish. No episode-based
population confidence interval or three-carrier transfer claim is reported.
Results by carrier and layout, preserving seed-level rows and grouping physics
repeats, are exported as CSV. Carrier passage counts C1/C2/C3 are 4/5/5 out of
twelve for U/T, 8/8/8 for analytic, and 7/7/8 for learned construction.

The 24 registered parent leave-four-assignment-out fits remain intact; removing
the sole analytic contrast yields the background-only learned model exactly.
On the complete ninety parent condition inputs it changes requests from 29 to
zero. The old 61-condition analysis must not be confused with this post-hoc
complete-input replay. Thirty separate post-hoc nominal CPU fits remove either
all groups from one carrier (including backgrounds) or one full contrast group.
Removed-carrier passage is analytic 8/6/8 and learned 8/7/7 out of twelve;
leave-one-contrast results are analytic 21–24/36, learned 22/36. These are corpus
sensitivity checks via validated command lookup, not new training draws. Generator
training still includes development sources. No generator-level source holdout
or necessity-of-paired-labels experiment is claimed.

## Mechanism and acquisition

Per-arm/per-carrier funnel counts in `acquisition.csv` distinguish proposed,
geometrically eligible, assigned, physically labeled, useful, and visible groups.
Selection precedes physical verification in the actual workflow, so the columns
must not imply a retrospective useful-or-visible selection rule. Nominal eligible
outputs are 48/41/46/33 (U/A/T/M), with nine generated groups selected per arm.
The labels over all fifteen groups are `(both pass,d040 only,walk only,both fail)`:
U `(7,0,0,8)`, A `(4,9,0,2)`, T `(13,0,0,2)`, M `(4,9,0,2)`.

The robust envelope is a 113-offset finite-grid minimum. The envelope diagnostic
uses 81×141 = 11,421 station/height centers, correcting the protocol's arithmetic
typo 7,181 without changing its registered axes or grid. Nominal witnesses per
carrier are 200/168/337; at ±20 mm x/y, ±10 mm z, ±0.02 rad yaw they become 2/0/32.
The numbers 20.476–26.993 mm are **best joint margins**, not measured interval
widths. A connected one-dimensional useful-height interval `[L,U]` subject to
every vertical displacement in `[-epsilon,+epsilon]` erodes to
`[L+epsilon,U-epsilon]`, with length `max(0,U-L-2*epsilon)`. This is a conditional
explanation, not a theorem for the actual pose-space witness set. Inclusive equal
endpoints give a singleton of zero length; a zero length does not imply emptiness.

Fitting uses mean BCE over thirty labels plus `0.01*mean(W**2)` during optimization;
reported BCE excludes that penalty. All groups count once, with two heads each;
there are no class weights. Both contrast arms have positive-label counts 4 and
13. Full features exclude direct IDs/beam parameters, but measured geometry and
source-correlated proprioception remain legitimate inputs and potential shortcuts.
Decision-time visibility is audited using all original groups, not just rays.

Analytic BCE is 0.000541579385753721; learned is 0.13900436460971832. Penalties
are 0.001360423630103469 and 0.001004991470836103. Two exact 214-value equivalence
classes in the learned corpus mix labels: C1 has two both-pass and one d040-only
groups; C3 has one both-pass and three d040-only. The entropy infimum over all
thirty labels is 0.138629436..., leaving about 0.000375 above that floor in the
measured fit. Two head errors equal the minimum permitted by the conflicting
labels. Analytic has no conflicting exact-input classes and zero head errors.
This establishes an irreducible outcome-fitting component, not that invisibility
caused all generator differences or the two downstream missed rescues. All members
of these classes permit d040, so passage does not require disambiguating them.
Exact fixed-step reproduction does not certify optimizer convergence.

## Time and cost boundary

| Study | Physical executions | Recorded rollout-process hours |
|---|---:|---:|
| Perturbation labels + evaluation | 82 + 540 | 5.13507249300749 |
| Nominal labels + evaluation | 72 + 144 | 1.4421235287686196 |
| Historical “6.6 GPU-hour” subtotal | 838 | 6.57719602177611 |
| New post-hoc matched audit | 16 | 0.13722260269247477 |
| Shared empty-transition bank, outside subtotal | 12 | 0.105115 |

Process-hours include startup and contention during admitted rollout processes.
They are not measured GPU utilization, scheduling span, or total method training
cost. Idle waiting, CPU search, inherited SONIC and motion/generator training,
and earlier experiments are excluded. The target-only generator and learned
proposal were pretrained on CPU in earlier apparatus; their costs are inherited,
not nominal acquisition. Original selector-fit wall time was not recorded.
Total end-to-end method cost remains unresolved and is not claimed.

Nominal proposal preparation takes 38.819750783 CPU seconds. Recorded per-arm
loops take 3.490806/9.420203/8.955175/15.984056 s. Search-clearance query counts
are 0/13,872/13,872/27,744, before common verification counters. These counts
are logged search calls, not comprehensive FLOPs or GPU time. Equal-label
comparison does not imply equal acquisition effort. Earlier fresh-source search
speedups remain attached to their reference-only apparatus; they are excluded
from this paper's selector-transfer and efficiency claims.

## Provenance limits and claim ladder

The 366-run result is committed at `05e318a` before the nominal protocol at
`ef048a2`. Local nominal registration/proposal times are 2026-09-07 11:46:53 /
11:47:39 UTC. These and run timestamps show documented availability of earlier
outcomes, not an external timestamping service or a complete human-inspection
log. Any undocumented earlier inspection remains unresolved. The honest scope
already treats the bank as development data; a stronger untouched-test claim
cannot be recovered through prose.

The strongest supported claim is: **at fifteen complete groups per arm in this
nominal task, targeting executed contrasts supplies adaptation-relevant examples
missing from uniform/target-only corpora and increases measured finite-bank
passage; analytic matches simple-policy passage with fewer unnecessary requests.**

No statistically significant learned-versus-analytic difference was detected by
the registered test; equivalence was not established. The observed difference
is 2/36. Shared both-pass/both-fail examples may still be useful. Paired-label
necessity, learned-generator superiority, protective refusal, hardware readiness,
robust transfer, and source-held-out generator/selector transfer are not supported.

The current paper reports only the two frozen ICRA studies and directly related
diagnostics. Historical documents are inventoried and pinned, not rewritten or
silently promoted into independent replications. The row-level manifest is the
reproduction record; this summary is not a substitute for it.
