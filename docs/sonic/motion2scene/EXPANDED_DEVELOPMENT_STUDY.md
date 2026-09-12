# Execution-conditioned construction across broader teaching budgets

The next experiment asks whether measured execution geometry acquires more useful
teaching encounters than commanded reference geometry, and whether that benefit
persists beyond four encounters. It retains the seven-schedule repertoire and
the common passage-regret ridge learner. Replay remains a separate comparison.

## Completed common-pool comparison

We screened 3,840 candidates: 256 draws in each of five strata, for each of three
seeds. The sampling domain, random streams, native collision shapes, 1 cm margin,
and 81 placement offsets are unchanged. The first 160 candidates in each seed
retain their original scene parameters and future teacher branch orders.

Reference envelopes use forward kinematics of the captured commanded full
schedules, including transitions and recovery. Applying the same kinematics to
measured joints and root poses reproduces native body positions within
approximately 2 micrometres. Reference and executed constructors therefore use
the same bodies and shapes; their difference is commanded versus measured poses.

| Corpus seed | Executed contrast scenes | Reference contrast scenes | Reference-selected pairs retaining executed contrast |
| --- | ---: | ---: | ---: |
| 93201 | 77 | 206 | 59/206 |
| 93202 | 89 | 241 | 67/241 |
| 93203 | 65 | 198 | 43/198 |

Only 169/645 reference-selected positive/negative pairs satisfy the same contrast
criterion under measured execution geometry. The main discrepancy concerns the
chosen positive: 182/645 retain the robust-clearance criterion, whereas 624/645
chosen negatives retain nominal intersection. However, 207 reference-selected
scenes still have an executed contrast using some schedule pair. These results
motivate complete physical labeling: a rejected chosen pair is not evidence
that the scene is physically impossible, and geometric clearance is not passage.

[Common-pool results](/home/linjiw/research-data/groot-wbc/m2s-construction-comparison-20260909-v1/result.json)
and [reference-envelope results](/home/linjiw/research-data/groot-wbc/m2s-reference-candidate-pools-20260909-v2/result.json)
contain the per-candidate measurements and input bindings. The preliminary v1
reference output has incomplete queue references; v2 is the usable result.
The exact executed reference-screen source is preserved as
`m2s-reference-candidate-pools-20260909-v2/source_snapshot.py`; the working copy
only moves lint-suppression comments. Its snapshot hash matches the experiment.

## Matched physical learning curves

The first primary checkpoint is complete for all twelve original corpora. Each
contains one acquired scene after bootstrap, with a pre-update student followed
by seven complete teacher executions. Across the three corpus seeds:

| Constructor | Solvable acquired tasks | Tasks requiring adaptation | Pre-update student passages |
| --- | ---: | ---: | ---: |
| Feasibility-screened uniform | 3/3 | 0/3 | 3/3 |
| Target-only | 3/3 | 3/3 | 0/3 |
| Executed contrast | 3/3 | 3/3 | 0/3 |
| Same executed scenes, observation replay | 3/3 | 3/3 | 0/3 |

Here “requiring adaptation” means the measured neutral branch fails and at least
one supported adaptation passes. All seven schedules pass each uniform scene;
such scenes can still provide measured time distinctions. The other constructors
acquire passage contrasts in all three seeds. Target-only ties executed contrast
at this budget. The student outcomes occur on each constructor's acquired tasks,
so they measure pre-update difficulty, not comparative held-out policy quality.
Analytic and observation-curriculum tasks and teacher outcomes match by design.

The [complete M1 acquisition-yield result](/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-yield-M1-20260909-v1/result.json)
contains all 96 encounter episodes and 114,432 physics steps, excluding bootstrap.
Every outcome is known. This first point has three independently drawn corpus
seeds but only one acquired task per corpus; M2/M4 and broader curves remain
necessary to establish persistent construction and learning effects.

### Completed M2 acquisition

All twelve M2 corpora completed on September 9 at 18:56 UTC. Each now contains
two nonempty teaching encounters, one short and one sustained, after bootstrap.
The [M2 task-yield result](/home/linjiw/research-data/groot-wbc/m2s-primary-checkpoint-readouts-20260909-v1/M2/task_yield/result.json)
contains 192 acquired-scene episodes and 228,864 measured physics steps,
excluding bootstrap. Every arm uses 57,216 of these steps across its three corpora.

| Constructor | Solvable acquired tasks | Tasks requiring adaptation | Pre-update student passages |
| --- | ---: | ---: | ---: |
| Feasibility-screened uniform | 6/6 | 0/6 | 6/6 |
| Target-only | 6/6 | 3/6 | 3/6 |
| Executed contrast | 6/6 | 6/6 | 3/6 |
| Same executed scenes, observation replay | 6/6 | 6/6 | 3/6 |

The contrast requirement now distinguishes the proposed constructor from
target-only: every acquired sustained-stratum target-only scene is also passable
by neutral locomotion, whereas all three executed-contrast scenes require an
adaptation. The first-round short scenes required adaptation for both methods.
This is a completed difference in useful-task acquisition at matched cost, not
evidence that one learned policy generalizes better. In particular, pre-update
passage counts compare policies on different constructors' task distributions.
The paired executed/observation arms have the same task identities and outcomes.

All three M2 observation-corpus replay-control comparisons also completed.
Every one of the seven controls gives 5/6 recorded six-context passage proxies
within each corpus. Seeds 93201 and 93203 always select the prior at 1.00 s;
seed 93202 always selects it at 0.30 s. All miss the complementary passage.
Thus the completed M2 replay comparison shows no passage improvement or
context-dependent option selection on this validation panel. The broader
cross-corpus M2 readout has also completed; M3 acquisition continues.

### Completed M2 cross-corpus readout

The [M2 cross-corpus result](/home/linjiw/research-data/groot-wbc/m2s-primary-checkpoint-readouts-20260909-v1/M2/cross_corpus/result.json)
contains 17 unique geometry/execution-seed pairs: eight first-round and nine
second-round tasks. Excluding each model's training seed produces 11, 12 and
11 validation contexts for seeds 93201, 93202 and 93203, respectively.

| Training constructor | Seed 93201 | Seed 93202 | Seed 93203 |
| --- | ---: | ---: | ---: |
| Feasibility-screened uniform | 11/11 | 11/12 | 10/11 |
| Target-only | 10/11 | 11/12 | 10/11 |
| Executed contrast | 11/11 | 11/12 | 10/11 |
| Same executed scenes, observation replay | 11/11 | 11/12 | 10/11 |

These are recorded-branch passage proxies, not new closed-loop executions.
All three tied arms choose the same prior schedule within each corpus: late
prior for seeds 93201/93203 and early prior for seed 93202. The target-only model
at seed 93201 chooses neutral on three tasks and late sustained on eight, giving
one fewer passage. A fixed late sustained schedule passes all 17 unique tasks.
Consequently, the measured construction-yield advantage has not produced better
selection than uniform or a strong fixed response at M2. The changing validation
pool precludes interpreting the M1-to-M2 readout as a fixed-test learning curve.
The result SHA-256 is
`9f5bfe7a9e19cfa5d199d19db51cacaf9e285c66ba3f863f7e7acc073ef7f03b`.

## Cross-corpus decision learning at M1

The completed [cross-corpus readout](/home/linjiw/research-data/groot-wbc/m2s-cross-corpus-readout-M1-20260909-v1/result.json)
tests each acquired model on measured encounters from the other acquisition
seeds. For each training seed, all four construction arms use exactly the same
validation set. Validation draws from uniform, target-only and executed-contrast
acquisition, excluding the paired observation-replay source and duplicate
geometry/seed pairs. Training-geometry overlaps are also excluded jointly across
the four arms. The target-only and executed constructors share one seed-93202
scene, leaving eight unique validation geometries overall. All are first-round
short-stratum tasks, not the six manually developed contexts.

| Model / fixed-schedule reference | Training seed 93201: 5 contexts | Training seed 93202: 6 contexts | Training seed 93203: 5 contexts |
| --- | ---: | ---: | ---: |
| Feasibility-screened uniform | 2/5 | 5/6 | 4/5 |
| Target-only | 4/5 | 5/6 | 4/5 |
| Executed contrast | 5/5 | 5/6 | 4/5 |
| Same executed scenes, observation replay | 5/5 | 5/6 | 4/5 |
| Fixed prior at 0.30 s | 5/5 | 5/6 | 4/5 |
| Fixed sustained at 1.40 s | 5/5 | 6/6 | 5/5 |
| Any schedule in the bank | 5/5 | 6/6 | 5/5 |

These are selected complete-branch outcomes from recorded neutral observations,
not new policy rollouts. The 16 validation assignments per arm reuse eight
geometries across folds and do not constitute 16 independent layouts. The
acquired model for each seed has only bootstrap and one nonempty teaching
encounter. No fitting or checkpoint selection uses the validation readouts.

The executed-construction gain over uniform is concentrated in seed 93201;
target-only recovers two of its three additional passages. All other seed-wise
passage comparisons tie, and observation replay adds no gain. The contrast-trained
models match the fixed prior's passage counts by selecting a prior schedule in
every validation context. Inspection of all seven fixed schedules shows that
the late sustained response covers all eight unique scenes. It is a fixed-bank
reference identified from these development tables, not a separately executed
learned-policy baseline. Thus M1 shows a data-dependent change in decisions but
does not yet teach useful selection beyond a strong constant response.

The seed-93201 target-only encounter requires sustained adaptation: both prior
schedules fail there. The other-seed models select a prior and miss it. This
provides an independently acquired test of a selection limitation, while the
early-constraint and multi-beam acquisition strata remain necessary to test
whether a sensor-dependent choice beats constant sustained adaptation.

The [checkpoint readout worker](/home/linjiw/research-data/groot-wbc/m2s-primary-checkpoint-readouts-20260909-v1/experiment.json)
is running. It has completed M2 and now waits for all twelve M4 corpora, then repeats both
acquired-task yield and the cross-corpus readout. These later readouts include
all acquired encounters through their budget, so their validation pool changes;
they must not be plotted as a fixed-test-set learning curve. The worker performs
CPU analysis only and leaves physical acquisition and reserved evaluation
unchanged.

## Expanded physical learning curves

The [expanded plan](/home/linjiw/research-data/groot-wbc/m2s-expanded-acquisition-plan-20260909-v1/plan.json)
contains five arms and three independent corpus seeds:

- Feasibility-screened uniform construction.
- Target-only construction.
- Executed contrast construction, uniform fitting.
- The same executed encounters, historical observation-gated replay.
- Reference contrast construction, uniform fitting.

The original four-arm M0–M4 prefixes remain unchanged. The reference arm receives
its own physical bootstrap and first four encounters. Subsequent selection keeps
stratum counts matched across all five arms, takes the next eligible candidate
in each arm's existing queue, and never uses acquired physical outcomes to
replace a scene. Analytic and observation-curriculum scene identities and order
remain identical. All 15 corpora reach the same 32-scene composition: seven
short, seven sustained, seven early-constraint, six short–short, and five
short–sustained encounters. The last stratum exhausts its common geometric
support at five; those counts are explicit rather than hidden by resampling.

| Encounters after bootstrap | Assigned episodes per corpus | Maximum assigned physics steps |
| --- | ---: | ---: |
| 2 | 23 | 27,416 |
| 4 | 39 | 46,488 |
| 8 | 71 | 84,632 |
| 16 | 135 | 160,920 |
| 32 | 263 | 313,496 |

The full extension adds at most 3,477 assigned episodes to the original 468.
Each encounter runs its pre-update student before seven teacher branches. Known
physical failures remain data; unknown outcomes stop advancement. Each corpus
has a 320,000-step charged ceiling, distinct from its assigned maximum. Actual
recorded steps and corpus-level outcomes will determine the learning curves.

The [staged continuation worker](/home/linjiw/research-data/groot-wbc/m2s-expanded-and-capability-stages-20260909-v3/sequence.json)
waits for all original M4 corpora, then advances all five arms through M8. At that
natural checkpoint it executes a matched development policy panel, followed by
the independently fixed extension capability panel. It next waits for the existing
CPU worker's six equivalent-teaching models and executes their 24 matched
held-center policy trials, then continues the unchanged acquisition plan through
M16 and M32. This prioritizes broader training and
closed-loop construction comparisons before the secondary option study.
It waits for GPU/disk capacity without resampling or retrying an unknown
physical attempt. Byte-identical completed collision inventories share storage
through hard links; all paths and bytes remain available. Reserved evaluation is
not launched by this worker.

One disclosed exception to that no-retry sentence exists. A clean host reboot on
2026-09-10 at 15:40 UTC killed the coordinator and the development evaluator two
seconds after episode_113's recorder logged "all data saved" and before the eval
driver wrote its success manifest, leaving a capture that the frozen collector
cannot score: six required inputs were never written, and this two-beam scene is
scored by a routine whose only beam-force stream is among them. That capture
therefore carries no outcome, so nothing was resampled or replaced. Under the
project's existing host-reboot precedent, and with the user's explicit
authorization, the interrupted files were archived intact and the one assigned
slot was repeated. See the
[recovery receipt](/home/linjiw/research-data/groot-wbc/m2s-M8-panel-reboot-recovery-20260910-v1/decision.json).

The [M8 closed-loop panel](/home/linjiw/research-data/groot-wbc/m2s-M8-native-development-panel-20260909-v1/study.json)
contains 138 assigned episodes: 15 acquired policies × 6 development contexts,
the unchanged strong script × 6 contexts, and all 7 forced schedules × 6 contexts.
Every execution uses physics seed 8732, the same bank, ideal observations and
physical scorer. The assigned allocation is 164,496 physics steps, a 1,192-step
ceiling per assignment. Measured physics across the 138 scored assignments is
164,060: 137 complete captures plus one forced comparator that terminated early
at 756 steps. Executed physics is 165,252, adding the reboot-interrupted
episode_113 capture, which really ran and whose 1,192 steps are charged even
though they carry no outcome; its repeat is one of the 138. That puts the panel
756 steps over its declared allocation, as an interruption cost on one
assignment rather than a 139th assignment. Any total for this panel must be
taken from the recovery receipt and its erratum rather than from the panel's own
artifacts, and the 164,496 ceiling must never be summed as though it were a
measurement. No model is
chosen using this panel: all five arms and three corpus seeds are included at
the specified M8 checkpoint. Each model is bound to its exact nine-encounter
training prefix, including bootstrap, before preparation. The 42 forced branches
provide capability on the same conditions as the 96 policy episodes; the old
seed-8731 development capability table is not substituted. All assignments stay
in their denominators, and time comparisons use mutually successful contexts.
These six contexts support development comparisons, not reserved generalization.
Twenty-three focused tests cover the panel, unchanged collector and attempt
executor; actual panel execution awaits M8 acquisition.
Separate native command-preparation checks also pass for one fixed schedule,
the strong script and an existing M1 policy. These use a development scene and
produce no physical execution; the M1 model is only a preparation check and is
not substituted for any assigned M8 policy.

The [panel statistics implementation](/home/linjiw/groot-wbc-sonic-sim-trackb/scripts/research/motion2scene_development_panel_statistics.py)
retains all assigned outcomes and reports capability separately from policy
passage. Missing schedule measurements produce lower/upper capability bounds;
they do not become measured failures. Selection-failure counts use conditions
with a complete matched bank and known policy outcome. Any policy-only success
against an all-failing matched bank is reported explicitly as a repeat-level
disagreement, rather than clipped from the comparison. Passage-time differences
use only mutually successful matched contexts, with their counts. Construction
effects are paired by corpus seed for executed versus uniform, target-only and
reference construction, and replay versus uniform fitting of executed scenes.
Three-seed means and sample standard deviations describe corpus variation on
the fixed six contexts; they are not confidence intervals over 18 independent
layouts. The implementation reproduces the archived 42-episode capability counts
without mixing those seed-8731 records with seed-8732 policy measurements.
Thirty-two combined panel, collector, executor and statistics tests pass,
including exhaustive completions of small partially measured capability tables.

After the M8 panel runs, its analysis command is:

```bash
.venv_isaaclab/bin/python scripts/research/motion2scene_development_panel_statistics.py \
  --panel /home/linjiw/research-data/groot-wbc/m2s-M8-native-development-panel-20260909-v1 \
  --out /home/linjiw/research-data/groot-wbc/m2s-M8-native-development-statistics-20260909-v1
```

## Implementation checks and remaining science

The expanded fitter reproduces the existing six-context model to floating-point
precision (maximum coefficient difference below 6e-17). Its historical gaps,
coverage keys, deadlines and six phase weights exactly match the completed first
observation-curriculum encounter. Both teacher and learned-student collection
preparations match the original frozen runtime dependencies. These checks validate
the handoff; they are not new policy-performance measurements.

Forty-two focused tests pass for expansion, construction, storage, original
controller behavior, replay, decision learning and observation-consistent
teaching. Another 40 checks pass for the original schedule learner, curriculum,
candidate pools and new CPU controls. The sets overlap and are not an 82-test
unique total.

The literature check also sharpens the teacher's positioning. The general
privileged-expert mismatch is established in
[Robust Asymmetric Learning in POMDPs](https://proceedings.mlr.press/v139/warrington21a.html),
[Student-Informed Teacher Training](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a8223b0ad64007423ffb308b0dd92298-Abstract-Conference.html),
and [Guided Policy Optimization](https://proceedings.iclr.cc/paper_files/paper/2026/hash/398753ddd2e10ce39d0c620e85fab922-Abstract-Conference.html).
Our design choice is a finite observation-group continuation solver backed by
measured humanoid schedules, not a claim to originate observation-aware teaching.

Next, evaluate development learning curves and controlled replay fits, select a
common final learner from development closed-loop outcomes, and only then bind
the final reserved evaluation. The original 972-episode low-budget proposal is
not a substitute for the expanded method comparison or its matched finite-bank
capability measurements. Strong scripts and constants remain comparators.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_expanded_acquisition.py \
  --plan /home/linjiw/research-data/groot-wbc/m2s-expanded-acquisition-plan-20260909-v1/plan.json \
  --adoption /home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-adoption-v3/adoption.json \
  --until-budget 32 --watch
```

Only one continuation worker should run; the experiment lock prevents concurrent
physical acquisition. Do not edit its three plan-bound implementation files
mid-experiment. Development analysis and paper writing can continue separately.
