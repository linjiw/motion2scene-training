# Repeatable option extraction from a motion prior

The current seven-schedule bank contains one useful generated-and-repaired
reference. That establishes the value of that option, but not the repeatability
of the procedure that produced it. This experiment compares two small extension
procedures with a shared neutral carrier and equal candidate/qualification
budgets. It does not change the bank used by the running curriculum experiment.

## Controlled comparison

The generated arm draws four new Kimodo samples with seeds 96001–96004. It uses
the existing six-second prompt, planned-neutral route and endpoint conditioning,
and the same four low-pelvis constraints. The authored arm uses the existing
local-crouch operator at four fixed duration-window half-widths: 0.18, 0.28, 0.38
and 0.48 of route progress. Both retain the existing requested lowering magnitude
of 0.085 m. These targets belong to different operators: the prior constrains
pelvis height, while the authored operator controls its geometric lowering
objective. This is a matched-budget procedure comparison, not an identical
distribution of motion targets.

Every candidate receives the same native hinge-box projection and neutral
prefix/tail splice. The neutral prefix ends at 1.00 s; the entry blend ends at
1.40 s; the return blend spans 4.50–5.00 s. The authored operator retains its
original model description; the shared repair uses the actual native robot's
joint intervals. No candidate receives outcome-dependent extra repair.

Each of the eight candidates has two assigned schedules: entry at 0.30 or 1.00 s,
return at 5.30 s, and recovery through tick 297. A fresh neutral execution and all
16 candidate schedules load the same nine-reference bank and use physics seed
97000. The allocation is eight candidate episodes per arm plus one shared neutral
episode, totaling at most 20,264 physics steps. Every candidate and failed
schedule remains in its assigned denominator.

The physical criteria are the existing complete-horizon stability, measured
external-contact, matched-prefix, legal-transition and recovery requirements.
Report both passing schedules out of eight and candidates with at least one
passing schedule out of four; the two entry times are not independent candidates.

## Fixed task coverage test

Before new outputs were generated, the study fixed 12 beam geometries: longitudinal
centers 1.70/2.25/2.80 m, lengths 0.10/0.75 m and undersides 1.24/1.30 m. Lateral
center is −0.10 m, yaw zero, width 1.20 m and thickness 0.10 m. The task execution
seed is 97001. These geometry choices do not use any newly generated motion
envelope or task outcome. They are a separate fixed procedure-development panel,
not the reserved primary evaluation.

The next comparisons measure incremental task coverage over the shared neutral
and downstream policy behavior after equivalent teaching. Qualification alone
does not establish either result. Neural draws and a deterministic authored grid
also have different compute costs, which should be reported alongside their
equal proposal and qualification allocations.

All 12 scenes and their complete 17-schedule collection commands are now
[materialized and checked](/home/linjiw/research-data/groot-wbc/m2s-extension-fixed-task-coverage-20260909-v1/prepared.json).
The panel assigns 204 episodes and at most 243,168 physics steps, retaining the
shared full reference bank in every execution. No new candidate envelope filters
these tasks. This panel is prepared, not yet physically executed. The
[staged worker](/home/linjiw/research-data/groot-wbc/m2s-expanded-and-capability-stages-20260909-v3/sequence.json)
executes it after all five construction arms reach their natural M8 checkpoint
and the matched development policy panel finishes. It then waits for the six
equivalent-teaching fits, executes the 24 matched arm-policy trials, and continues
the same acquisition plan through M16/M32. No active acquisition
rollout is interrupted by this ordering.

The capability scorer reports each arm's union of passing schedules separately
from the neutral baseline and from learned selection. Unfinished outcomes stay
in the full task denominator; lower/upper capability bounds distinguish them
from measured failures. These bounds express missing outcomes, not confidence
intervals. Best-schedule time comparisons require both banks to be solvable with
complete branch outcomes. Exhaustive tests cover every three-valued outcome
pattern in a five-schedule two-arm bank. The scorer also reproduces the existing
42-episode six-context result: each arm covers five tasks, with one exclusively
solved by each.

The separate [candidate-level scorer](../../scripts/research/motion2scene_extension_candidate_coverage.py)
groups both entry schedules by their actual motion reference. For each of the
eight candidates it reports standalone coverage, coverage with neutral,
incremental coverage over neutral, and unique coverage after removing that
candidate from its own arm. The last quantity retains neutral and every other
candidate; it measures redundancy, not a generated-versus-authored difference.
All candidates remain visible, including those that add no capability. Time is
compared with neutral only on complete, mutually passing task conditions.
Neither two entry times nor twelve task outcomes create additional independently
generated candidates. Missing-outcome bounds are not sampling intervals.

After the assigned capability panel completes, run:

```bash
.venv_isaaclab/bin/python scripts/research/motion2scene_extension_candidate_coverage.py \
  --coverage /home/linjiw/research-data/groot-wbc/m2s-extension-fixed-task-coverage-20260909-v1/coverage.json \
  --output /home/linjiw/research-data/groot-wbc/m2s-extension-fixed-task-coverage-20260909-v1/candidate_coverage.json
```

Fourteen focused bank/candidate tests pass, including exhaustive missing-outcome
completions. An allocation-only smoke check loads the actual registry and all
twelve prepared tasks: eight candidates, 204 assigned branches, all still
unmeasured. This scorer does not modify the queued execution or add physics.

## Equivalent-teaching comparison

The [arm-specific teaching implementation](../../scripts/research/motion2scene_extension_teaching.py)
restricts each learner to neutral plus its eight qualified entry schedules.
It recomputes complete WAIT continuations from original physical branches;
slicing the full-bank teacher table would be incorrect because WAIT could
inherit a passing motion from the excluded arm. Original matched-prefix tests,
physical outcome labels and missing-outcome treatment remain unchanged.

Both arms use the common phase-wise ridge equations with penalty 10. The physical
simulator continues to load all nine references. The logical policy menu has nine
schedules and 118 features: the same 98 sensing/state channels, two recomputed
aggregate legality channels, and the arm's active-option and legal-option fields.
The projection changes the action interface, not the measured scene information.
It retains actual schedule names on output and the existing mandatory recovery.
The logical menu is a subset of the qualified bank, not a separately qualified
registry. The [isolated simulator adapter](../../scripts/research/motion2scene_extension_execution.py)
and [bound policy loader](../../scripts/research/motion2scene_extension_policy.py)
are now implemented. They retain the full 134-channel recording while separately
recording the 118-channel arm-policy input and its actual value predictions.
The loader checks the full physical bank, logical action menu, assigned training
fold and ridge archive. The runtime uses the existing legal-transition and
mandatory-return rules, and keeps nominal sensing.

The [episode wrapper](../../scripts/research/motion2scene_extension_episode.py)
now prepares an arm-policy episode from that held task's original forced-neutral
template. It verifies the model's held-task assignment, physical seed, full bank
and unchanged command components before execution. Complete captures reconstruct
every selected action and projected input exactly; ridge values use an absolute
tolerance of 1e-12. Physical passage uses the same contact, stability, legal
schedule and recovery scorer as the other development studies. Incomplete
attempts retain their physical outcome separately from complete policy-trace
verification. Timeouts are charged once, and resource waits remain unlaunched.
The [24-episode batch declaration](/home/linjiw/research-data/groot-wbc/m2s-extension-native-policy-panel-20260909-v1/study.json)
now fixes every arm/held-task assignment and its order. The
[serial runner](../../scripts/research/motion2scene_extension_policy_panel.py)
requires all six fitted models before preparing commands, retains known failures,
and stops on unresolved outcomes. The batch is queued in the staged continuation
after capability and fitting; its commands are not prepared and no episode has
executed yet. The declaration binds 168 source dependencies, and all 64 combined
focused tests pass.

The [fitting study](/home/linjiw/research-data/groot-wbc/m2s-extension-equivalent-teaching-20260909-v1/study.json)
now binds the three folds, both arms, fixed learner and all 75 source dependencies.
Its [CPU fitter](../../scripts/research/motion2scene_fit_extension_teaching.py)
independently checks completed physical recordings, reconstructs each arm's
continuations, and saves six policies with training-only constant selection.
Its worker is live, waiting for the complete twelve-task capability panel.
It adds no simulation and does not change the existing execution sequence.
A fitting failure is retained without producing a substitute policy or labeling
it as a traversal failure.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_fit_extension_teaching.py watch \
  --out /home/linjiw/research-data/groot-wbc/m2s-extension-equivalent-teaching-20260909-v1
```

The worker above is already running; the command documents its invocation, not
a request to start a duplicate. No equivalent-teaching model has yet been fitted.

Before the fixed-task outcomes are available, the comparison uses three folds
that hold out an entire longitudinal center (1.70, 2.25 or 2.80 m). Each arm trains
on the other eight tasks and evaluates all four length/height combinations at
the held center. Each fold therefore supplies 72 teacher branches per arm,
including eight shared neutral branches. The two arms jointly use 136 unique
training branches; reusing these recordings across folds does not create more
physical acquisitions. No held-center teacher outcomes enter fitting, replay,
penalty selection or constant selection.

For each arm/fold, select a constant using training passage count first, then
summed successful passage time, then its original schedule order. Compare the
learned selector with that constant and its own finite bank on the four held
tasks. The existing fixed-schedule panel supplies matched reference outcomes at
seed 97001; the learned comparison needs 24 new closed-loop episodes, at most
28,608 physics steps. These episodes are queued but not yet prepared. Report
passage and capability separately, along with matched successful-time differences
and their counts. The three folds are not independent trained corpora: training
sets overlap, and each arm has one four-candidate bank. This is a controlled
procedure-development comparison, not the primary reserved evaluation or a
population-level estimate over motion generators.

Sixty-four combined batch/episode/runtime/loading/fitting/teacher/projection/capability tests pass.
Runtime tests use simulator doubles: they check full-bank retention, projected
decision recording, rejection of mismatched models and nominal sensor settings.
They are not physical execution evidence. The fitting
tests reject held-task inputs, reproduce the common ridge coefficients, and
confirm that changing the other arm's successful costs changes neither this
arm's fit nor its selected constant. The teacher test
includes a full-bank WAIT that succeeds only via the other arm, missing own-arm
outcomes, and mismatched history. On the existing seven-schedule recordings,
all 3,576 projections across six complete neutral episodes preserve the 98
sensing/state channels exactly. Reconstructing branches from the stored audited
groups also reproduces all 18 original teacher tables field for field. This does
not substitute for the seventeen-
schedule feature and native-runtime checks. The completed qualification captures
do not contain policy feature vectors, so those checks must use the new task
recordings when available.

## Matched held-center policy analysis

The held-center [policy analysis](../../scripts/research/motion2scene_extension_policy_statistics.py)
is implemented. It reports each arm's measured capability separately from learned
passage, missed selections, and matched-repeat policy-only successes. Its constant
baseline is the option already selected from that fold's eight training tasks;
it never selects a constant using held-center outcomes. Cross-arm time differences
use only mutually successful tasks and retain their counts. Unexecuted or missing
policy outcomes stay in the assigned denominator. Fold summaries are descriptive:
their overlapping training data and one candidate bank per procedure do not
provide three independent estimates of generator quality.

Fourteen analysis tests and the existing 64 extension tests pass together.
No new physical results are claimed. After the fixed-task coverage and policy
episodes complete, run:

```bash
.venv_isaaclab/bin/python scripts/research/motion2scene_extension_policy_statistics.py \
  --panel /home/linjiw/research-data/groot-wbc/m2s-extension-native-policy-panel-20260909-v1 \
  --coverage /home/linjiw/research-data/groot-wbc/m2s-extension-fixed-task-coverage-20260909-v1/coverage.json \
  --output /home/linjiw/research-data/groot-wbc/m2s-extension-native-policy-panel-20260909-v1/comparison.json
```

## Completed candidate generation

All four new neural samples completed, with no replacement draws. Four authored
alternatives and all eight common repaired references are saved. Each repaired
reference has 299 native-interpolated frames, no source hinge-limit violation,
and passing position guards at ticks 15, 50 and 265. The largest checked join
discrepancy is 0.001013 rad, below the unchanged 0.05 rad criterion. These are
reference checks, not physical qualification.

The [candidate results](/home/linjiw/research-data/groot-wbc/m2s-option-extension-study-20260909-v2/candidate_results.json)
bind all raw generations, authored settings, repair stages and reference files.
## Completed physical qualification

All 17 assigned executions completed: the shared neutral and both entry schedules
for all eight candidates. Under the existing external-contact, stability,
matched-prefix, transition and recovery criteria, both arms qualify 8/8 schedules
and 4/4 candidates. The test records 5,066 control rows and 20,264 physics steps
at the common seed 97000. No sample was replaced and no candidate received extra
repair after execution. The complete nine-reference bank is qualified separately
from the unchanged bank used by primary acquisition.

This establishes repeatable extraction of executable options in this small
experiment. It does not establish an advantage over authored construction: both
procedures have the same observed qualification yield. The fixed-task coverage
and equivalent-teaching comparisons must establish any downstream advantage.

[Qualification results](/home/linjiw/research-data/groot-wbc/m2s-option-extension-qualification-20260909-v1/yield_result.json)
and the [qualified bank](/home/linjiw/research-data/groot-wbc/m2s-option-extension-qualification-20260909-v1/registry.json)
bind every assigned candidate and physical outcome. The dispatcher resumed after
qualification; a separate six-context learner comparison follows between intact
acquisition batches.

An initial conversion attempt stopped before writing any repaired candidate
because the authoring XML was passed to a native-only parser. The corrected
conversion separates those model descriptions and reuses all four original raw
samples. The old source snapshot, plan, raw outputs and failed attempt directory
remain available; the corrected conversion adds no inference or physics.

Implementation:
[candidate study](../../scripts/research/motion2scene_option_extension_study.py) and
[physical qualification](../../scripts/research/motion2scene_qualify_extension_study.py).
