# Motion2Scene persistent research state

Updated 2026-09-10 UTC. Resume the current experiment from this file and its receipts;
do not restart a general project diagnosis. The central question is whether
executed-motion scene construction plus complete-continuation teaching improves
perceptive traversal at matched measured acquisition cost.

**Current resume point, September 10 at 16:50 UTC: the M8 common-set comparison is
COMPLETE. It is a null result for the central claim and a positive result for a
different, sharper one.** All 138 panel assignments are measured — 90 learned M8
executions, 42 forced schedule branches, 6 script branches — with 105 passes, 33
failures and no unknown or unmeasured outcome. `summary.complete` is true.
[Statistics receipt](/home/linjiw/research-data/groot-wbc/m2s-M8-native-development-statistics-20260910-v1/result.json)
SHA `15a883e07f701c2dd731531fcf40d7ed0533b6e5a52a03104f38e9a54cff4087`;
[rendered report](/home/linjiw/research-data/groot-wbc/m2s-M8-native-development-statistics-20260910-v1/report.md).
Do not re-run the panel or the statistics; both are written once and hash-bound.

| Construction method | Passages by corpus (93201 / 93202 / 93203) | Mean | Corpora whose policy conditions on the scene |
|---|---|---:|---:|
| Feasibility-screened uniform | 5/6, 5/6, 5/6 | 5.00/6 | 0/3 |
| Target-only | 4/6, 4/6, 5/6 | 4.33/6 | 1/3 |
| Reference-envelope contrast | 6/6, 5/6, 4/6 | 5.00/6 | 1/3 |
| Executed-envelope contrast | 5/6, 5/6, 5/6 | 5.00/6 | 0/3 |
| Executed contrast with replay | 5/6, 5/6, 5/6 | 5.00/6 | 0/3 |

**Executed contrast does not beat screened uniform.** The two arms are identical in
every measured cell: same passage count, same single failing context
(`complementary_late_development_0100`), same −0.204 s paired time against the script
on five mutually successful contexts. The executed constructor's acquisition advantage
in adaptation-required task yield produced no downstream difference at all.

**Replay is a clean null.** `observation_curriculum` matches `analytic_contrast`
cell for cell in all three corpora, including the identical selected schedule. On this
panel the replay module is unsupported and should be reported as a tested component
with a null result, not as a contribution.

**The mechanism is response collapse.** 13 of 15 learned policies select the *same*
schedule on all six contexts — they are constant functions, not perceptive selectors.
Their −0.204 s / n=5 signature is exactly `fixed_prior_splice`'s, and the target-only
policies that collapsed onto sustained carry exactly `fixed_sustained`'s +0.115 / n=4.
No learned policy passed anywhere the bank fails (`policy_only_successes` = 0 for all 15).

**The one thing that predicts a selective policy is within-corpus complementarity, and
it predicts it perfectly.** The acquisition receipt's
`corpora_without_one_passing_fixed_schedule` is exactly `["seed93201_reference_contrast",
"seed93203_target_only"]`. Those are exactly the two corpora whose M8 policies use more
than one schedule, and all 13 corpora coverable by a single fixed schedule trained
constant policies — **15/15 agreement**. `seed93201_reference_contrast` is the only
learned policy to reach 6/6, matching the script's ceiling. This is a correspondence
across 15 corpora, not a controlled manipulation: nothing varied complementarity while
holding the construction method fixed. That is the experiment still to run.

**Decision.** The evidence does not support the current central claim that executed
contrast trains a better policy; it supports simplifying the unsupported replay module
and re-aiming collection at within-corpus complementarity. Do not continue to M16/M32
on the assumption that more of the same corpora will help — 13/15 corpora at M8 train a
constant policy, and a larger budget of single-response-coverable encounters is the one
thing this result predicts will not fix it. The declared M16/M32 trajectory is preserved
and unchanged, but deliberately not started. The full decision, including what the next
experiment must not do, is [recorded separately](POST_M8_DECISION_V1.md).

**The envelope screen has now been calibrated against measured outcomes, and it closes
the standing negative-margin hypothesis.** Over the 92 already-executed encounters, the
constructor's targeting predicate finds the measured property — every covering schedule
fails, some other schedule passes — with **precision 2/2 but recall 2/7**.
[Calibration receipt](/home/linjiw/research-data/groot-wbc/m2s-envelope-predictor-calibration-20260910-v1/result.json)
SHA `85fbd6f8f5fd93f5b6bd37f313d7b08457b5f404aa76275fccec75dc2f3f3f38`; zero physics,
zero fits, zero new candidate searches. Relaxing the negative margin from −10 mm to 0 mm
raises recall only to 3/7, so threshold tuning cannot fix it: two missed encounters have
covering-schedule clearance predicted **positive** yet measured **failing**
(`primary_candidate_93203_0148` at +15.80/+14.46 mm and `primary_candidate_93203_0847`
at +3.70/+2.70 mm) and are unreachable by any margin, and a third,
`primary_candidate_93201_0417`, is rejected by the **positive** robustness bar — it is
the round-8 encounter of `seed93201_reference_contrast`, the corpus that produced the
only 6/6 policy. The two encounters the screen does select landed in the two corpora that
collapsed to a constant sustained policy at 4/6, the worst scores on the panel. The
screen is a high-precision, low-recall proxy, which is precisely what would produce the
M8 null. The next intervention should therefore select on **measured** response, not
predicted geometry.

**One correction to the correspondence, from the same evidence.** Corpus complementarity
is necessary but *not* sufficient. `seed93203_target_only` is non-constant — it uses three
schedules — yet on the single discriminating context it selects `short_e070_r265` and
fails, where the bank's passers are the two sustained schedules. It scores 5/6, the same
as the collapsed policies. Non-constancy is not the outcome; correct selection is. Any
success criterion phrased as "the policy stops being constant" would score that corpus a
win for no gain, and must not be used.

**One declared deviation.** A clean host reboot at 15:40 UTC killed the v8 coordinator
(PID 2474326) and its evaluator (PID 2863023) mid-capture, leaving `episode_113`
unscoreable: six collector inputs were never written and this two-beam scene is scored
by a routine whose only beam-force stream is among them. Under the project's own
[host-reboot precedent](/home/linjiw/research-data/groot-wbc/m2s-primary-reboot-recovery-20260909-v1/decision.json)
and with the user's explicit authorization, the interrupted files were archived intact
and the slot repeated once. See the
[recovery receipt](/home/linjiw/research-data/groot-wbc/m2s-M8-panel-reboot-recovery-20260910-v1/decision.json)
and its [accounting erratum](/home/linjiw/research-data/groot-wbc/m2s-M8-panel-reboot-recovery-20260910-v1/decision.ERRATA.json).
Both attempts are charged. Measured physics across the 138 scored assignments is
**164,060** — 137 complete captures plus one forced comparator, `episode_055`, that
terminated early at 756 steps — and executed physics is **165,252** once the archived
interrupted capture's 1,192 recorded steps are added. That is 756 steps over the 164,496
declared allocation, an interruption cost on one assignment and not a 139th assignment.
The receipt's first accounting block said 165,688; it had added the repeat to the
declared ceiling instead of to measured physics, and the erratum corrects it. No outcome,
capture or hash is affected. Two
declarations are deviated from and recorded rather than concealed — `readiness.json`'s
write-once "one charged attempt per assignment", and `EXPANDED_DEVELOPMENT_STUDY.md`'s
published allocation, which is amended. The repeat was declared unconditional before
launch: the completed execution is `episode_113`'s measurement whatever it shows, and the
archive can never acquire an outcome. It passed.

The text below this line predates that completion and is retained as history. Where it
describes the coordinator or the evaluator as live, it is superseded: both are gone, the
panel is finished, and `stages_finished.json` is written.

**Superseded resume point, September 10 at 14:41 UTC:** the new user guidance in
`/home/linjiw/.codex/attachments/9c8c5ea0-000d-419c-8b81-f348b1aad963/pasted-text.txt`
prioritizes needed physical execution and one bounded complementarity pilot;
defer further export formats and manuscript polishing. All twelve original M4
fits, dispatcher boundaries, the independent raw-record audit and primary
cross-corpus readouts are **complete**. The original dispatcher (161884), final
worker (2248207), M4 auditor (2259482) and its watcher (2174226) finished normally;
do not restart them. Tool session 91946 returned exit code 0 and is closed.
The five-arm M1 raw audit completed successfully at 01:15 UTC. Auditor PID
2304521 ended and tool session 10270 returned exit code 0; do not poll or restart
it. [Archived evidence](evidence/research-progress-20260909/five_arm_M1/result.json)
has SHA `f3e125a2389cdab1a0ebc5c003812043f543fde1d177de5a299f50861c084c8d`.
It covers 225 episodes and 268,200 measured physics steps, including the three
independent reference prefixes (45 episodes, 53,640 steps total). The twelve
original M1 histories match their prior audit exactly. Reference contrast has
2/3 bank-solvable, adaptation-required tasks versus 3/3 for executed contrast
at equal cost. Seed 93202's reference encounter has seven verified failures,
all with measured undesired environment contact; it stays in the denominator.
There are no unknown outcomes in this completed prefix. These are early task
yield measurements, not post-update policy performance. The audit adds no
physics or fitting. Its launch receipt preserves the corrected Path/string
invocation error in the initial read-only receipt check.

**All 15 original M8 acquisition corpora and the 48 frozen comparators are complete.**
`scripts/research/motion2scene_development_ready.py` removes only the all-model
preparation barrier and supports verified reuse of existing assignments. All
21 focused readiness/original-panel tests pass; all 48 native comparator
preparations and 174 pinned sources were checked before any comparator outcome.
The original 138 assignments, strong script, seven schedules, seed 8732,
geometries, collector and scorer are unchanged. `readiness.json` in the
original M8 panel declares the order amendment. The full `prepared.json` may
only be finalized after all fifteen M8 models exist; a partial panel never
produces the original final result.

The live coordinator is PID **2474326**, executing
`m2s-expanded-and-capability-stages-20260910-v8/run.py` under the research-data
root. M8 acquisition child **2474361 exited successfully**; its authoritative
`expanded_M8_resume_exit.json` records status 0. Do not poll or restart it.
Development preparation **PID 2861112 also exited 0**. The coordinator has
started **the M8 native development evaluator, PID 2863023**; its PID/start
ticks match and it is live at the 14:41:26 UTC snapshot. The persistent launch
and stage-process receipts identify subsequent children; no interactive tool
session owns this detached coordinator. Follow v8 `coordinator.log`,
`development_M8.log` and stage exits. Do not restart either completed child.
Do not duplicate it. The same M8/M16/M32 queues, adoption, runtime and original
execution directories remain in force. All fifteen M4 checkpoints and the
five-arm raw-record audit are complete. **All fifteen M5 corpora are complete.**
**All fifteen M6 corpora and the bound M6 corpus readout are complete.**
**All fifteen M7 corpora and the bound M7 corpus summary are complete.**
**All fifteen M8 corpora and the bound M8 acquisition summary are complete.**
**The actual learned M8 development evaluation has reached 15/90 assignments
verified complete at 14:41:26 UTC**, with 11 passes, 4 failures and
**17,880 measured physics steps**. Episode 018 is launched and pending in
that snapshot. Do not mistake other prepared directories for executions.
The original 48 comparators are reused. No reserved outcome exists.

The [second native M8 progress receipt](evidence/research-progress-20260909/M8_development_second_snapshot/second_M8_development_progress.json)
adds **12 executions and 14,304 measured steps** to the original three-result
snapshot: **9 new passes and 3 new failures**. The new failures are on
`complementary_late_development_0100`: seed-93203 uniform selects late prior,
seed-93203 target-only selects late short, and seed-93202 uniform selects early
prior; each records measured environment contact in a complete 1,192-step
capture. The earlier empty-scene finite-horizon failure remains unchanged.
All earlier result bindings, all 15 M8 policy/training bindings and all 48
comparator result hashes match. The snapshot includes full original outcome
definitions and generating-policy/scene/attempt/sensor references. It adds
no fit, retry or raw-record re-audit. The partial, unbalanced assignment
prefix does not establish construction-arm performance or uncertainty.

The [first native M8 development receipt](evidence/research-progress-20260909/M8_development_started/first_M8_development_progress.json)
verifies all 15 prepared model/training-result bindings against the completed
M8 acquisition summary and all 48 unchanged comparator result hashes. For
each completed learned result it checks the original assignment, scene,
manifest, M8 policy, sensor and actual attempt bindings. The prepared panel
has all **138 original assignments**, with SHA
`3f63b64fc28e9934f9e7cbe8f1a2f255bbc2ae96a5558901cbe2c8951d52a151`.

| Assignment | M8 policy | Development scene | Selected schedule | Outcome / passage time |
|---|---|---|---|---|
| 000 | seed93202 uniform | early_prior_development_0284 | early prior | pass, 2.72 s |
| 001 | seed93201 target-only | long_neutral_empty | late sustained | failure |
| 002 | seed93202 replay | long_neutral_empty | early prior | pass, 5.38 s |

Each has 1,192 measured physics steps. These are new closed-loop executions
of M8 policies, not teacher-table lookups or generating-M7 acquisition
outcomes. The partial panel does not support an arm-level performance claim.
Episode 001 is a finite-horizon completion failure: its last recorded physical
phase is 5.94 s, before the task completes. Later unrecorded continuation is
unknown; this result does not measure a collision or fall.
Passage times measure crossing plus stabilization. The receipt uses original
collector scoring; it does not constitute another raw-record audit.
Acquisition and preparation exit receipts both record 0; the coordinator and
evaluation worker are live. No policy, baseline, selection rule, queue,
runtime, scorer or reserved assignment changed. Finish the same 90 learned
assignments before interpreting the checkpoint. The coordinator then follows
the predeclared original acquisition trajectories through M16/M32.

After the full M8 panel completes, use the existing
`scripts/research/motion2scene_development_panel_statistics.py` with
`--panel /home/linjiw/research-data/groot-wbc/m2s-M8-native-development-panel-20260909-v1`
and a fresh output directory. It preserves the full 138-assignment denominator,
separates capability from selection, reports paired time on mutually successful
conditions, and shows individual corpus results on the reused six development
layouts. The full-panel statistics command remains **unexecuted** while this
panel is incomplete. Do not substitute a partial result for the checkpoint.

The [complete M8 acquisition summary](evidence/research-progress-20260909/M8_complete/all_M8_bound_response_tables.json)
accounts for **1,065 recorded captures**, **1,264,988 measured physics steps**
and **120 postbootstrap task assignments**. It verifies all M7 teacher
prefixes, ridge lambda 10, the declared weighting rules and all three
executed-contrast/replay candidate, scene and branch orders through M8.
It uses original hash-bound collection scoring, without another all-capture
raw-record audit or a new policy execution.

| Arm | Bank-solvable / assigned | Adaptation-required / assigned | Cumulative measured physics steps |
|---|---:|---:|---:|
| Feasibility-screened uniform | 24/24 | 6/24 | 253,896 |
| Target-only | 24/24 | 10/24 | 253,896 |
| Reference contrast | 14/24 | 14/24 | 250,276 |
| Executed contrast | 24/24 | 24/24 | 253,460 |
| Executed contrast with replay | 24/24 | 24/24 | 253,460 |

Task counts exclude bootstrap; physical costs include bootstrap, every
pre-update student and teacher branch, partial failures and the reconciled
known-cost student unknown. Proposal computation remains separately recorded
in the original proposal/adoption receipts and is not amortized here.
The ten reference bank-unsolvable tasks remain in the denominator. Teacher
outcomes have no unknowns. Only reference seed 93201 and target-only seed
93203 require two schedules to cover their solvable acquired tasks. In each
executed/replay arm, either fixed prior still covers all 24 task assignments.
These reuse three training corpora and paired scenes, not independent layouts.
The current encounter students are generated by **M7**: 13 pass and 2 fail
on their own acquired scenes. This is not an M8 common-set result.

[Final reference seed-93203 M8](evidence/research-progress-20260909/M8_complete/seed93203_reference_M8_bound_readout.json)
uses candidate `primary_candidate_93203_0427`. The generating M7 student
chooses late sustained and passes in **3.48 s**; both sustained teachers
pass in **3.48 s** and all five other schedules fail. Each capture has
1,192 measured steps: **9,536** for the encounter and **84,240** cumulative.
Bank and fixed sustained coverage are **4/8**, retaining four unsolvable
tasks. Passage times measure crossing plus stabilization.

The [M8 completion progress receipt](evidence/research-progress-20260909/M8_complete/all_M8_complete_progress.json)
adds **8 executions and 9,536 measured steps** to the 589-assessment snapshot:
**3 passes and 5 failures**. All earlier assessment hashes match, including
the reconciled unknown. Original acquisition exited 0; the coordinator and
development-preparation child are verified live. Frozen study/readiness
bindings and all 48 comparator result files remain present, with no learned
launch receipt or result yet. The summary command completed with exit 0
(tool session 23858 is closed); Black and Ruff E/F/I checks passed before
execution. The archived final-reference command passes shell syntax checking.
No queue, runtime, collector, teacher, learner, scorer or reserved assignment
changed. Continue the original development stage, then its predeclared
M16/M32 acquisition continuation.

[Target-only and uniform seed-93203 M8 are complete](evidence/research-progress-20260909/M8_target_uniform93203/seed93203_target_uniform_M8_bound_readout.json).
On `primary_candidate_93203_1097`, the actual generating M7 target student
chooses late short at 1.40 s and fails. Its hash-bound original outcome has
a complete 1,192-step measurement with environment contact (maximum measured
force **1,752.5430908203125 N**). Both prior teachers pass (**3.28/3.30 s**),
as do both sustained teachers (**3.38 s**); neutral and both shorts fail.
Target bank coverage is **8/8**, best fixed coverage **7/8**, and minimum
cover size **2**, retaining earlier corpus complementarity. **4/8** tasks
require adaptation. The failure is not relabeled unknown or removed.

On its different acquired scene, `primary_candidate_93203_0027`, the actual
generating M7 uniform student chooses late prior and passes in **2.78 s**.
All seven teachers pass: priors in **2.78 s**, others in **2.82 s**. The bank
and fixed priors or sustained cover **8/8** tasks, with **2/8** requiring
adaptation. Each encounter costs **9,536 measured steps** and each cumulative
prefix **84,632**. Old M7 teacher prefixes and uniform-per-phase ridge lambda
10 remain unchanged. These outcomes concern different acquired scenes and
generating M7 policies, not common-set comparisons of the new M8 fits. Times
measure crossing plus stabilization, not full adaptation/recovery completion.

The [14:08 progress receipt](evidence/research-progress-20260909/M8_target_uniform93203/seed93203_target_uniform_M8_progress.json)
records **16 new executions and 19,072 measured physics steps** since the
573-assessment checkpoint: **12 passes and 4 failures**. All previous
assessment hashes match, including the reconciled known-cost unknown.
Frozen M8 panel bindings match: **48 comparator results present**, **90
learned episode directories absent**, and no full preparation. The bound
readout adds no physics or fitting and does not repeat original raw scoring;
the archived reproduction command passes shell syntax checking. No queue,
runtime, collector, teacher, learner, scorer or reserved assignment changed.

[The third M8 matched pair is complete](evidence/research-progress-20260909/M8_third_pair/seed93203_paired_M8_bound_readout.json).
On `primary_candidate_93203_0192`, both actual generating M7 students choose
late prior and pass in **3.38 s**. The complete teacher tables match: both
priors pass in **3.38 s**, both sustained in **3.44 s**, and neutral and
both shorts fail. Original candidate, scene and branch order through M8,
generating-policy bindings and M7 teacher prefixes are verified. The learner
remains ridge lambda 10, with uniform-per-phase versus historical-observation-
gap weighting. Each encounter costs **9,536 measured steps**, and each
cumulative prefix **84,196**, preserving the earlier partial neutral failures.
Both banks and either fixed prior cover **8/8** acquired tasks. These are
pre-update M7 outcomes, not M8 common-set policy results. Times measure
crossing plus stabilization, not full adaptation/recovery completion.

All three completed M8 pair readouts were checked against their saved hashes.
For **each** of executed contrast and replay, all **24/24** postbootstrap
task assignments require adaptation, yet either fixed prior passes **24/24**.
Thus passage on these corpora still does not require choosing among different
responses. The three matched current-student passage times also tie
(3.42, 3.00 and 3.38 s); this does not establish population equality or
common-set replay benefit. Counts reuse paired scenes and three training
corpora; they are not 48 independent tasks. The frozen queues stay unchanged.
The M8 common-set development result must precede any new acquisition pilot.

The [13:47 progress receipt](evidence/research-progress-20260909/M8_third_pair/seed93203_pair_M8_progress.json)
records **16 new executions and 19,072 measured physics steps** since the
557-assessment checkpoint: **10 passes and 6 failures**. All prior assessment
hashes match, including the reconciled known-cost unknown. Frozen M8 panel
bindings still match, with **48 comparator results present**, **90 learned
episode directories absent**, and no full preparation. The bound readout
adds no physics or fitting and does not repeat original raw scoring.
Its archived shell command passed syntax checking. No queue, native runtime,
collector, teacher, learner, scorer or reserved assignment changed.

[Reference seed-93202 M8 is complete](evidence/research-progress-20260909/M8_reference93202/seed93202_reference_M8_bound_readout.json).
On `primary_candidate_93202_0592`, the actual generating M7 student selects
early prior and passes in **3.38 s**. Both prior teachers pass in **3.38 s**,
both sustained teachers in **3.44 s**, and neutral and both shorts fail.
All eight captures have 1,192 measured steps: **9,536** for this encounter
and **83,848** for the cumulative prefix. Finite-bank coverage is **4/8**;
either fixed prior covers the same four tasks. All four bank-unsolvable
tasks remain in the denominator. Teacher outcomes are known on all branches.
The old M7 teacher prefix, original scene-wise complete-continuation teacher
and uniform-per-phase ridge lambda 10 remain unchanged. This is an acquired
scene outcome from the generating M7 student, not an evaluation of the new
M8 fit. Passage times measure crossing plus stabilization.

The [13:28 progress receipt](evidence/research-progress-20260909/M8_reference93202/seed93202_reference_M8_progress.json)
records **8 new executions and 9,536 measured physics steps** since the
549-assessment checkpoint: **5 passes and 3 failures**. All previous assessment
hashes match, including the reconciled known-cost unknown. The frozen M8 panel
still has **48 comparator results present**, **90 learned-policy episode
directories absent**, and no full preparation. The bound readout adds no
physics or fitting and does not repeat original raw scoring. No queue,
runtime, teacher, learner, scorer or reserved assignment changed. Continue
the original worker; the fifteen-policy M8 common-set development evaluation
remains the next decisive experiment.

The existing completed-checkpoint summarizer was adapted as
`m2s-expanded-and-capability-stages-20260910-v8/summarize_completed_M8.py`
under the research-data root and **completed successfully** on all 15 M8
corpora. Its result and implementation are archived in `M8_complete` above.
It does not execute or score common-set policies; do not rerun it or overwrite
the report. The separate full raw-record M8 audit remains unexecuted.

[Target-only and uniform seed-93202 M8 are complete](evidence/research-progress-20260909/M8_target_uniform93202/seed93202_target_uniform_M8_bound_readout.json).
On `primary_candidate_93202_0422`, the actual generating M7 target student
selects late sustained and passes in **2.88 s**. All seven teachers pass:
early prior in **2.86 s**, late prior and both sustained in **2.88 s**, and
neutral and both shorts in **2.90 s**. On its different acquired scene,
`primary_candidate_93202_0017`, the generating M7 uniform student selects
early prior and passes in **2.76 s**. Both priors pass in **2.76 s**, and
all other schedules fail. Thus the target scene does not require adaptation,
while the uniform scene is prior-only. This latter classification uses the
completed seven-branch table, including the final late-short failure.

Target bank and fixed sustained coverage are **8/8**, with **3/8** adaptation-
required tasks; uniform bank and fixed prior coverage are **8/8**, with **1/8**
adaptation-required tasks. Neither corpus requires multiple fixed schedules
to cover its acquired tasks. Each encounter costs **9,536 measured steps**,
and each cumulative prefix **84,632 steps**. The old M7 teacher prefixes and
uniform-per-phase ridge lambda 10 are verified unchanged. These are pre-update
M7 student outcomes on different scenes, not common-set constructor comparisons
or evaluations of the new M8 fits. Times measure crossing plus stabilization,
not full adaptation/recovery. The bound readout adds no physics or fitting and
does not repeat original raw scoring.

The [13:14 progress receipt](evidence/research-progress-20260909/M8_target_uniform93202/seed93202_target_uniform_M8_progress.json)
records **16 new executions and 19,072 measured physics steps** since the
533-assessment checkpoint: **11 passes and 5 failures**. All earlier assessment
hashes match, including the reconciled unknown. Frozen M8 panel bindings still
match: **48 comparator results present**, **90 learned-policy episode directories
absent**, and no full panel preparation. No queue, runtime, teacher, learner,
scorer or reserved assignment changed. The archived readout command passed
shell syntax checking. Continue the original worker without duplication; the
fifteen-policy M8 common-set development evaluation remains the next decisive
experiment.

[The second M8 matched pair is complete](evidence/research-progress-20260909/M8_second_pair/seed93202_paired_M8_bound_readout.json).
On `primary_candidate_93202_0927`, both actual generating M7 students select
early prior and pass in **3.00 s**. The teacher tables match: early prior passes
in **3.00 s**, late prior in **3.02 s**, both sustained in **3.38 s**, and
neutral and both shorts fail. Candidate, scene and branch order through M8,
actual generating-policy bindings and old M7 teacher prefixes are verified.
Both learners remain ridge lambda 10, with uniform-per-phase versus historical
observation-gap weighting. Each encounter costs **9,536 measured steps**,
and each cumulative prefix **84,632 steps**. Both banks and fixed prior
schedules cover **8/8** acquired tasks. This supplies another observed
pre-update tie, without establishing population equality, added response
complementarity or common-set performance. The newly fitted M8 policies are
not evaluated here. Passage times measure crossing plus stabilization, not
full adaptation/recovery. The bound readout adds no physics or fitting and
does not repeat original raw scoring.

The [12:52 progress receipt](evidence/research-progress-20260909/M8_second_pair/seed93202_pair_M8_progress.json)
records **16 new executions and 19,072 measured physics steps** since the
517-assessment checkpoint: **10 passes and 6 failures**. It includes the two
seed-93202 executions already observed at the previous resume point; neither
is rerun or double-counted. All earlier assessment hashes match, including
the reconciled unknown. Frozen common-panel bindings still match: **48 comparator
results present**, **90 learned-policy episode directories absent**, and no
full panel preparation. No queue, runtime, teacher, learner, scorer or reserved
assignment changed. The archived readout command passed shell syntax checking.
Continue the original worker without duplication. The fifteen-policy M8
common-set development evaluation remains the next decisive experiment.

[Reference seed-93201 M8 is complete](evidence/research-progress-20260909/M8_reference93201/seed93201_reference_M8_bound_readout.json).
On `primary_candidate_93201_0417`, the actual generating M7 student commits to
late prior at tick 50 and fails with measured torso–beam contact up to
**2,301.016 N**. Both sustained teachers pass in **3.48 s**; neutral, both priors
and both shorts fail. This adds measured corpus complementarity: bank coverage
is **6/8**, best fixed prior coverage **5/8**, and minimum cover **2**. The two
bank-unsolvable assignments remain in the denominator, and the teacher table
has no unknown branch outcomes. The earlier M7 teacher prefix and uniform-
per-phase ridge lambda 10 remain unchanged. This is training-bank capability,
not sensor realizability or downstream superiority. The new M8 fit has not
been evaluated on the common panel.

A [bounded commitment check](evidence/research-progress-20260909/M8_reference93201/seed93201_reference_M8_commitment_readout.json)
recomputes the raw pre-action history digest at tick 50. The failing student's
recorded history, causal features and legal mask match those of the passing
late-sustained teacher and the stored complete teacher target. The student
chooses late prior; the teacher selects legal **WAIT**, with all **three**
expected WAIT-continuation outcomes admitted and late sustained explicitly
bound as its passing continuation. Here WAIT continues neutral execution and
retains the later adaptation choice. Early sustained also passes, so this
does not isolate a WAIT information benefit from choosing a complete schedule
initially. The earlier tick-15 WAIT is not charged the later commitment failure;
the student's committed tick-70 state is not treated as a matched neutral
decision. The digest binds recorded history, not unrecorded simulator state.

[Three partial-capture rechecks](evidence/research-progress-20260909/M8_reference93201/seed93201_reference_M8_partial_reaudit.json)
exactly reproduce the original known contact failures and their measured costs:
neutral **628 steps**, each short **644 steps**. Their synchronized contact
streams record maximum undesired forces of **1,776.470 N** and **1,649.539 N**,
respectively. Each log ends at the non-repeated/in-horizon command-tick guard;
that capture termination is distinct from the measured physical failure.
Later uncaptured fall, passage and recovery outcomes remain unknown. The
encounter costs **7,876 measured steps**, and its cumulative prefix **82,188**.
The rechecks add no physics, retries or fitting. Tool session 73377 completed
with exit code 0 and is closed. Reproduction commands are archived alongside
the reports and each passed a separate shell syntax check.

The [12:32 progress receipt](evidence/research-progress-20260909/M8_reference93201/seed93201_reference_M8_progress.json)
records **8 new executions and 7,876 measured physics steps** since the
509-assessment checkpoint: **2 passes and 6 failures**. All earlier assessment
hashes match, including the reconciled student unknown attached to its M4
generating policy. Frozen M8 panel bindings still match: **48 comparator
results present**, **90 learned-policy episode directories absent**, no full
preparation yet. No queue, runtime, teacher, learner, scorer or reserved
assignment changed. Continue the original worker without duplication. The
fifteen-policy M8 common-set development evaluation remains the next decisive
experiment; the new reference complementarity is not a policy result.

[Target-only and uniform seed-93201 M8 are complete](evidence/research-progress-20260909/M8_target_uniform93201/seed93201_target_uniform_M8_bound_readout.json).
On `primary_candidate_93201_0212`, the actual generating M7 target student
selects late sustained and passes in **2.98 s**. Both prior teachers pass in
**2.96 s** and all other schedules in **2.98 s**. On its different acquired
scene, `primary_candidate_93201_0007`, the generating M7 uniform student
selects late prior and passes in **2.46 s**. Both prior teachers pass in
**2.46 s** and all other schedules in **2.74 s**. Neither new scene requires
adaptation. Each encounter costs **9,536 measured steps**, with **84,632 steps**
in each cumulative prefix. Target bank and fixed sustained coverage are **8/8**;
uniform bank and fixed prior coverage are **8/8**. Each different corpus has
**3/8 adaptation-required tasks**. Both retain a fixed response covering their
acquired tasks; these encounters do not add required response complementarity.
The old M7 teacher prefixes and uniform-per-phase ridge lambda 10 are unchanged.
These are pre-update M7 student outcomes, not new M8 policy evaluations or a
common-set comparison between constructors. Times measure crossing plus
stabilization. The bound readout adds no physics or fitting and does not repeat
original raw scoring.

The [12:18 progress receipt](evidence/research-progress-20260909/M8_target_uniform93201/seed93201_target_uniform_M8_progress.json)
records **16 new executions and 19,072 measured physics steps** since the
493-assessment checkpoint, all passing. All prior assessment hashes match,
including earlier failures and the reconciled unknown. Frozen M8 panel bindings
still match: **48 comparator result files present**, **90 learned-policy episode
directories absent**, and no full preparation yet. No queue, runtime, teacher,
learner, scoring or reserved assignment changed. The archived readout command
passed shell syntax checking. Continue the original acquisition without
duplication; the fifteen-policy M8 common-set development evaluation remains
the next decisive experiment.

[The first M8 matched pair is complete](evidence/research-progress-20260909/M8_first_pair/seed93201_paired_M8_bound_readout.json).
On `primary_candidate_93201_0817`, both actual generating M7 students select
late prior and pass in **3.42 s**. Their teacher tables match: both priors pass
in **3.42 s**, both sustained in **3.48 s**, and neutral and both shorts fail.
Candidate, scene and branch ordering through M8, old M7 teacher prefixes and
ridge lambda 10 remain unchanged. Weighting is uniform-per-phase versus
historical observation-gap. Each encounter costs **9,536 measured steps**,
and each cumulative prefix **84,632 steps**. Both banks and fixed prior
schedules cover **8/8** acquired tasks. This supplies no observed replay
benefit or additional required response on this encounter, without establishing
population equality or evaluating the newly fitted M8 models. Passage times
measure crossing plus stabilization, not full adaptation/recovery. The bound
readout adds no physics or fitting and does not repeat original raw scoring.

The [11:58 progress receipt](evidence/research-progress-20260909/M8_first_pair/seed93201_pair_M8_progress.json)
records **16 new executions and 19,072 measured physics steps** since the
477-assessment checkpoint: **10 passes and 6 failures**. All prior assessment
hashes match, including the reconciled unknown. A read-only check of the
frozen M8 study/readiness bindings confirms that all **48 comparator result
files remain present**, all **90 learned-policy episode directories are
absent**, and full panel preparation has not occurred. No queue, runtime,
teacher, learner, scoring or reserved assignment changed. The archived paired
readout command passed shell syntax checking. Continue the original worker
without duplication; the fifteen-policy M8 common-set development evaluation
remains the next decisive experiment.

[The complete M7 bound response tables](evidence/research-progress-20260909/M7_complete/all_M7_bound_response_tables.json)
cover **945 recorded captures**, **1,123,608 measured physics steps** and
**105 post-bootstrap encounter assignments** across fifteen corpora. Tasks
and conditions are deliberately reused; these are not 105 independent layouts
or a common-set policy comparison. The summary verifies all M6 teacher prefixes,
ridge lambda 10, and all three contrast/replay candidate and branch orders.
It uses bound original scoring, not a second raw audit of all captures.
The summary script completed with exit code 0; tool session 55243 is closed.
Black and Ruff E/F/I pass. No new physics or fitting comes from this readout.

| M7 arm | Bank-solvable / assigned | Adaptation-required / assigned | Cumulative measured physics steps |
| --- | ---: | ---: | ---: |
| Feasibility-screened uniform | 21/21 | 5/21 | 225,288 |
| Target-only | 21/21 | 9/21 | 225,288 |
| Reference-envelope contrast | 11/21 | 11/21 | 223,328 |
| Executed-envelope contrast | 21/21 | 21/21 | 224,852 |
| Executed contrast with replay | 21/21 | 21/21 | 224,852 |

Step totals include bootstrap and recorded partial failures; task counts exclude
bootstrap. Proposal computation remains separately recorded in the original
manifests. All **10 reference bank-unsolvable assignments** remain included,
with no unknown teacher-branch outcomes. The reconciled older student unknown
remains attached to its generating M4 policy and included in measured cost.
Every executed/replay corpus still admits a fixed prior covering all **7/7**
tasks. Only target-only seed 93203 has bank **7/7**, best fixed **6/7**, minimum
cover **2**. Thus high adaptation-required yield has not yet produced required
response diversity in the contrast corpora. This does not establish a downstream
advantage for target-only or justify changing the ongoing acquisition queue.
The fifteen current M7-encounter student outcomes belong to generating M6
policies; the new M7 fits are not evaluated here.

[Reference seed-93203 M7](evidence/research-progress-20260909/M7_complete/seed93203_reference_M7_bound_readout.json)
finishes with a late-sustained student passage of **4.78 s**. Both prior
teachers pass in **4.66 s**, shorts in **4.74 s**, sustained in **4.78 s**, and
neutral fails. Its encounter costs **9,536 steps** and prefix **74,704 steps**.
Bank and fixed sustained coverage are **3/7**; four unsolvable assignments
remain. Times measure crossing plus stabilization, not full recovery.

The [11:36 progress receipt](evidence/research-progress-20260909/M7_complete/all_M7_complete_progress.json)
records **8 new executions and 9,536 measured physics steps** since the
469-assessment checkpoint: **7 passes and 1 failure**. Together with the
target/uniform receipt below, this turn completed **24 executions and 28,608
steps** with **20 passes and 4 failures**. All earlier assessment hashes match.
No queue, teacher, learner, scorer, native runtime or reserved assignment changed.
Continue the original M8 worker without duplication. The fifteen-policy M8
common-set development evaluation, reusing the 48 completed comparators, remains
the next decisive experiment. The bounded complementarity-pilot revision is
still conditional on that development gate and has not been adopted.

[Target-only and uniform seed-93203 M7 are complete](evidence/research-progress-20260909/M7_target_uniform93203/seed93203_target_uniform_M7_bound_readout.json).
The actual generating M6 target student selects late sustained and passes in
**3.58 s**. All seven teachers pass on this new scene: priors in **3.48 s**,
neutral and shorts in **3.54 s**, and sustained in **3.58 s**. Its acquired
corpus preserves earlier complementarity, with bank coverage **7/7**, best
fixed **6/7**, and minimum cover **2**; this new all-pass scene does not itself
add complementarity. The earlier complementary records remain unchanged.
On its different scene, the generating M6 uniform student selects late prior
and passes in **4.78 s**. Both priors pass in **4.78 s**, sustained in **5.30 s**,
and neutral and shorts fail. Uniform bank and fixed prior/sustained coverage
are **7/7**. Each encounter costs **9,536 measured physics steps**, with
**75,096 steps** in each cumulative prefix. Old M6 teacher prefixes and
uniform-per-phase ridge lambda 10 are verified unchanged. These are acquired-
scene pre-update outcomes, not evaluation of the M7 fits or a common-set
comparison. Times measure crossing plus stabilization; the target student's
0.10 s difference from passing priors is not an energy estimate. No new physics,
fitting or second raw audit comes from the readout.

The [11:22 progress receipt](evidence/research-progress-20260909/M7_target_uniform93203/seed93203_target_uniform_M7_progress.json)
records **16 new executions and 19,072 measured physics steps** since the
453-assessment checkpoint: **13 passes and 3 failures**. All prior assessment
hashes match, including the reconciled unknown. No queue, runtime, teacher,
learner, scorer or reserved assignment changed. The existing bound M6 summary
calculation was advanced to `v8/summarize_completed_M7.py`; Black and
Ruff E/F/I pass. It was **prepared but unexecuted at this earlier snapshot**;
its successful execution is recorded above. Continue the original worker
toward M8 without duplication. The fifteen-policy
M8 common-set development evaluation with reused comparators remains the
next decisive experiment.

[The third matched M7 pair is complete](evidence/research-progress-20260909/M7_third_pair/seed93203_paired_M7_bound_readout.json).
On `primary_candidate_93203_0586` at seed 93203, both actual generating M6
students select late prior and pass in **4.68 s**. Complete teacher tables
match: both priors pass in **4.68 s**, both shorts in **4.76 s**, both sustained
in **4.80 s**, and neutral fails. Candidate, scene and branch order through
M7, actual policy bindings, unchanged old M6 teacher prefixes and ridge lambda
10 are verified. Weighting remains uniform-per-phase versus historical
observation-gap. Both banks and both fixed prior schedules cover **7/7** tasks.
This pair supplies no observed replay benefit or additional required fixed
response, without establishing population equality or common-set performance.
The new M7 fits are not evaluated here. Passage times measure crossing plus
stabilization, not full adaptation/recovery.

Each neutral teacher capture is partial, with **756 measured physics steps**.
Separate bounded raw rechecks of
[executed contrast](evidence/research-progress-20260909/M7_third_pair/seed93203_executed_M7_partial_neutral_reaudit.json)
and [replay](evidence/research-progress-20260909/M7_third_pair/seed93203_replay_M7_partial_neutral_reaudit.json)
exactly reproduce their original rows: synchronized contact streams, torso–beam
force up to **701.912 N**, and known task failure. Both logs terminate at the
non-repeated/in-horizon command-tick guard. That termination is recorded
separately from the measured physical failure; later uncaptured fall, passage
and recovery remain unknown. No retry, imputed full duration or unrelated raw
re-audit was added. Each encounter costs **9,100 steps**, and each cumulative
prefix **74,660 steps**. The archived raw-check script ran successfully for
both captures and passed Black plus Ruff E/F/I with a declared E402 exception
for its repository-path bootstrap import. The paired readout command passed
shell syntax validation.

The [11:02 progress receipt](evidence/research-progress-20260909/M7_third_pair/seed93203_pair_M7_progress.json)
records **16 new executions and 18,200 measured physics steps** since the
437-assessment checkpoint: **14 passes and 2 failures**. All earlier assessment
hashes match, including the reconciled unknown. No queue, runtime, collector,
teacher, learner, scoring or reserved assignment changed. Continue the same
worker without duplication. The fifteen-policy M8 common-set development
evaluation, reusing the 48 completed comparators, remains the next decisive
experiment.

[Reference seed-93202 M7 is complete](evidence/research-progress-20260909/M7_reference93202/seed93202_reference_M7_bound_readout.json).
On `primary_candidate_93202_1016`, the actual generating M6 student selects
early prior and passes in **4.74 s**. Both prior teachers pass in **4.74 s**,
both sustained in **5.26 s**, and neutral and both shorts fail. The encounter
costs **9,536 measured physics steps**, with **74,312 steps** in its cumulative
prefix. Bank and best fixed prior coverage are **3/7**; all four bank-unsolvable
assignments remain in the denominator. The previous M6 teacher prefix and
uniform-per-phase ridge lambda 10 are unchanged. The readout verifies bound
completed records without repeating the original raw outcome scoring, adding
physics or fitting. These are acquired-scene outcomes from the M6 student,
not common-set performance or an evaluation of the newly fitted M7 model.
Times measure crossing plus stabilization, not full adaptation/recovery.

The [10:43 progress receipt](evidence/research-progress-20260909/M7_reference93202/seed93202_reference_M7_progress.json)
records **8 new executions and 9,536 measured physics steps** since the
429-assessment checkpoint: **5 passes and 3 failures**. Combined with the
target/uniform receipt immediately below, this is **24 executions and 28,608
steps** since the 413-assessment checkpoint, with **17 passes and 7 failures**.
All earlier assessment hashes match, including the reconciled unknown. The
original coordinator and acquisition worker remain live with matching process
identities and no stage exit. No queue, teacher, learner, runtime, scoring or
reserved assignment changed. Exact readout commands are archived beside these
receipts and passed shell syntax checks. Continue the existing worker without
duplication. The next decisive experiment remains the fifteen-policy M8
common-set development evaluation with the 48 completed comparators reused.

[Target-only and uniform seed-93202 M7 are complete](evidence/research-progress-20260909/M7_target_uniform93202/seed93202_target_uniform_M7_bound_readout.json).
The actual generating M6 target student waits at ticks 15 and 50, then selects
late short at tick 70 and fails with measured undesired environment contact
(maximum **1,084.690 N**). Both prior teachers pass in **5.24 s** and both
sustained teachers in **5.76 s**; neutral and both shorts fail. A
[bounded final-choice check](evidence/research-progress-20260909/M7_target_uniform93202/seed93202_target_M7_final_choice_readout.json)
recomputed the student and passing late-sustained teacher's raw pre-action
history digests at tick 70. Recorded histories, causal features, legal masks
and the stored complete teacher prefix match. Late sustained is the only
passing immediate action at that final choice and is legal. This establishes
an acquired-scene selection failure relative to measured bank capability;
it does not establish perceptual realizability across scenes or a WAIT
information benefit. The digest binds recorded history, not unrecorded
simulator state. Earlier WAIT choices must not be charged this complete-future
failure as separate immediate errors.

On its different acquired scene, the generating M6 uniform student selects
early prior and passes in **5.32 s**. All seven teacher schedules pass there;
early and late prior times differ (**5.32 and 5.34 s**). Each encounter costs
**9,536 steps**, and each cumulative prefix **75,096 steps**. Fixed sustained
covers all **7/7** target tasks; every fixed schedule covers all **7/7** uniform
tasks. Old M6 teacher prefixes and uniform-per-phase ridge lambda 10 are
unchanged. These are pre-update student outcomes, not common-set comparisons
or evaluations of the newly fitted M7 policies. Readouts add no physics or
fitting and do not repeat full original outcome scoring. Times measure
crossing plus stabilization, not full adaptation/recovery.

The [10:32 progress receipt](evidence/research-progress-20260909/M7_target_uniform93202/seed93202_target_uniform_M7_progress.json)
records **16 new executions and 19,072 measured physics steps** since the
413-assessment checkpoint: **12 passes and 4 failures**. Every earlier
assessment hash matches, including the reconciled unknown. Both original v8
process identities remain live with no stage exit. No queue, runtime, teacher,
learner, scoring or reserved assignment changed. Continue the original worker
without duplication. The fifteen-policy M8 common-set development evaluation,
reusing the 48 completed comparators, remains the next decisive experiment.

[The second matched M7 pair is complete](evidence/research-progress-20260909/M7_second_pair/seed93202_paired_M7_bound_readout.json).
On `primary_candidate_93202_0521` at seed 93202, both actual generating M6
students choose early prior-splice and pass in **4.14 s**. Their complete
teacher tables match: both priors pass in **4.14 s**, both sustained schedules
in **4.60 s**, and neutral and both shorts fail. Candidate, scene and branch
ordering through M7, actual policy bindings and the old teacher prefixes are
verified. Both learners remain ridge lambda 10, with uniform-per-phase versus
historical-observation-gap weighting. Each encounter costs **9,536 steps**,
and each cumulative prefix **75,096 steps**. Both banks and their fixed prior
schedules cover **7/7 acquired tasks**. This readout uses bound completed
records without a second raw audit, new physics or fitting. The observed tie
supplies no replay benefit on this condition, without establishing population
equality or common-set performance; the new M7 fits are not evaluated here.
Times measure crossing plus stabilization, not full adaptation/recovery.

The [10:10 progress receipt](evidence/research-progress-20260909/M7_second_pair/seed93202_pair_M7_progress.json)
records **16 new executions and 19,072 measured physics steps** since the
397-assessment checkpoint: **10 passes and 6 failures**. All earlier assessment
hashes match, including failures and the reconciled unknown. Both original
v8 process identities remain live with no stage exit. A read-only storage
snapshot found **390,850,863,104 free bytes** and about **391 MB** per sampled
eight-execution encounter, with no immediate storage constraint on M8. It is
a current snapshot, not a reservation for later runs. No queue, runtime,
teacher, learner, scoring or reserved assignment changed. Continue the
original worker without duplication. The fifteen-policy M8 common-set
development evaluation, reusing the 48 completed comparators, remains the
next decisive experiment.

[Reference seed-93201 M7 is complete](evidence/research-progress-20260909/M7_reference93201/seed93201_reference_M7_bound_readout.json).
On `primary_candidate_93201_0051`, the actual generating M6 student selects late
prior and passes in **4.58 s**. Both prior teachers pass in **4.58 s**, shorts
in **4.68 s**, sustained in **4.72 s**, and neutral fails. The encounter costs
**9,536 steps**, with **74,312 steps** in its cumulative prefix. Bank and fixed
prior coverage are **5/7**; both bank-unsolvable assignments remain included.
The old M6 teacher prefix and uniform-per-phase ridge lambda 10 are unchanged,
and the reconciled round-5 student unknown, generated by its M4 policy,
remains in history and cost. The readout
verifies bound completed records without a second raw audit, new physics or
fitting. These are acquired-scene M6 student results, not M7 model evaluation
or common-set performance. Times measure crossing plus stabilization.

A [bounded proposal readout](evidence/research-progress-20260909/M7_reference93201/seed93201_target_M7_proposal_readout.json)
connects the already acquired, sustained-only target scene
`primary_candidate_93201_1131` to its saved geometric measurements. Early and
late sustained retain minimum outer clearances of **11.238 and 11.680 mm** over
all 81 declared offsets. Prior nominal inner clearances are **−2.651 and
−4.019 mm**: both predict shallow interference and both schedules physically
fail, but neither meets the targeted selector's **−10 mm** negative margin.
Recomputing the original selector for this one candidate reproduces that
rejection. Ordinary executed contrast already admits the scene using neutral
as its negative; its saved pool rank is **75/77**, versus **10/438** for target-
only. It is absent from the adopted executed/replay M32 queues and acquired
at target-only M7. Eligibility alone therefore does not ensure that this
useful response enters a finite corpus. Target-only also uses a preassigned
positive and different ordering, so arm differences do not isolate the
negative constraint alone.

This one-scene readout used saved arrays, **zero new proposals, clearance
queries or physical executions**, and **0.000233 s** of selector computation.
It does not explain the entire pilot null or establish physical robustness
over the geometric offset set. It supports only a conditional **negative-margin
hypothesis**, not a method change: after the M8 development gate, consider
varying that margin alone while preserving the positive margin, fixed-schedule
selection, exclusions, common proposal draws, ranking rule and maximum budget.
Any such version still needs physical useful-task yield and common-set policy
evidence. No threshold, revised pilot or queue change has been adopted, and
the already acquired scene remains excluded from a fresh pilot.

The [09:50 progress receipt](evidence/research-progress-20260909/M7_reference93201/seed93201_reference_M7_progress.json)
records **8 new executions and 9,536 measured physics steps** since the
389-assessment checkpoint: **7 passes and 1 failure**. All earlier assessment
hashes match, including failures and the reconciled unknown; pending seed-93202
execution is excluded. Both original v8 process identities remain live with
no stage exit. No queue, runtime, teacher, learner, scoring or reserved
assignment changed. Continue the original worker without duplication. The
fifteen-policy M8 common-set development evaluation, reusing the 48 completed
comparators, remains the next decisive experiment.

[Target-only and uniform seed-93201 M7 are complete](evidence/research-progress-20260909/M7_target_uniform93201/seed93201_target_uniform_M7_bound_readout.json).
On target-only `primary_candidate_93201_1131`, the actual generating M6 student
selects late sustained at tick 70 and passes in **4.10 s**. Only the two
sustained teachers pass, both in **4.10 s**; neutral, both priors and both shorts
fail. The student selects a passing response on this acquired scene, but both
fixed sustained schedules still cover **7/7 corpus tasks**. Thus this does not
establish new within-corpus complementarity. An early sustained option also
passes, so the late choice does not isolate WAIT's information benefit.

On uniform `primary_candidate_93201_0016`, the generating M6 student selects
late prior and passes in **4.76 s**. Both priors pass in **4.76 s**, sustained
in **5.28 s**, and neutral and shorts fail. Fixed prior schedules cover
**7/7 corpus tasks**. These are different acquired scenes, not a common-set
constructor comparison. Each encounter costs **9,536 measured steps**, each
prefix **75,096 steps**, with unchanged M6 teacher prefixes and uniform-per-
phase ridge lambda 10. The readout verifies bound completed collection,
manifest, training and actual policy records without repeating raw scoring
or adding physics or fits. It does not evaluate the newly fitted M7 models.
Times measure crossing plus stabilization, not full adaptation/recovery.

The [09:36 progress receipt](evidence/research-progress-20260909/M7_target_uniform93201/seed93201_target_uniform_M7_progress.json)
records **16 new executions and 19,072 measured physics steps** since the
373-assessment checkpoint: **8 passes and 8 failures**. It completes both
encounters. All earlier assessment hashes match, including failures and the
reconciled unknown. Both original v8 process identities remain live with no
stage exit. No queue, runtime, teacher, learner, scoring or reserved assignment
changed. Continue the original worker without duplication. The fifteen-policy
M8 common-set development evaluation, reusing the 48 completed comparators,
remains the next decisive experiment.

[The first matched M7 pair is complete](evidence/research-progress-20260909/M7_first_pair/seed93201_paired_M7_bound_readout.json).
On `primary_candidate_93201_1186` at seed 93201, both actual generating M6
students choose late prior-splice and pass in **4.16 s**. Their complete teacher
tables match: neutral fails, both priors pass in **4.16 s**, both shorts in
**4.56 s**, and both sustained schedules in **4.62 s**. Candidate, scene and
branch ordering through M7, actual policy bindings and the old teacher prefixes
are verified. Both learners remain ridge lambda 10, with uniform-per-phase
versus historical-observation-gap weighting. Each encounter costs **9,536
steps**, and each cumulative prefix **75,096 steps**. Both banks and their
fixed prior schedules cover **7/7 acquired tasks**; this encounter does not
establish within-corpus passage complementarity. The readout uses bound
completed records without a second raw audit, new physics or fitting. It is
not an evaluation of the new M7 fits or a common-set comparison. The observed
student tie supplies no replay benefit on this condition, without establishing
population equality. Times measure crossing plus stabilization.

The [09:16 progress receipt](evidence/research-progress-20260909/M7_first_pair/seed93201_pair_M7_progress.json)
records **16 new executions and 19,072 measured physics steps** since the
357-assessment checkpoint: **14 passes and 2 failures**. All earlier assessment
hashes match, including failures and the reconciled unknown. Pending target-only
execution is excluded. Both v8 process identities remain live with no stage
exit. A bounded read-only selector inspection confirms positive clearance over
the declared offset set and **nominal** interference for the chosen fixed
schedule; it does not require negative clearance over all offsets. No predicate
correction, new pilot search or physical robustness test was performed.
No queue, native runtime, collector, teacher, learner, scoring or reserved
assignment changed. Continue the original worker without duplication. The
fifteen-policy M8 common-set development evaluation, reusing the 48 completed
comparators, remains the next decisive experiment.

[The complete M6 acquisition readout](evidence/research-progress-20260909/M6_complete/all_M6_bound_response_tables.json)
covers **825 recorded captures, 981,440 measured physics steps and 90 post-
bootstrap encounter assignments across fifteen corpora**, with deliberately
reused scene identities. These are not 825 independent training tasks. The
[executed readout source](evidence/research-progress-20260909/M6_complete/summarize_completed_M6.py)
uses the existing finite-bank response statistic and bound completed collection,
manifest and training records. All M5 teacher prefixes are unchanged, as are
ridge lambda 10 and the three contrast/replay candidate orders. The paired
M6 teacher tables match. This is not a second all-capture raw audit or a common-
set policy evaluation. The script returned 0 in tool session 20583, which is
closed; Black and Ruff E/F/I checks pass. It adds no physics or fitting.

| Construction arm | Bank-solvable / assigned | Adaptation-required / assigned | Cumulative measured physics steps |
| --- | ---: | ---: | ---: |
| Feasibility-screened uniform | 18/18 | 3/18 | 196,680 |
| Target-only | 18/18 | 7/18 | 196,680 |
| Reference-envelope contrast | 8/18 | 8/18 | 194,720 |
| Executed-envelope contrast | 18/18 | 18/18 | 196,680 |
| Executed contrast with replay | 18/18 | 18/18 | 196,680 |

These physical task-yield measurements do not establish downstream performance
or negative-contrast superiority over target-only. Every executed/replay corpus
still admits one fixed schedule covering all six acquired tasks. **Only target-
only seed 93203 is complementary:** bank coverage **6/6**, best-fixed coverage
**5/6**, minimum cover **two**, retaining its independently verified disjoint
M3/M5 passing sets. This does not establish perceptual realizability or learned
selection on common tests. All ten reference bank-unsolvable assignments stay
in the denominator. Teacher unknowns are zero in these completed tables;
the reconciled earlier reference student unknown remains in its history and
measured acquisition cost. Proposal computation remains separately accounted
in the adopted proposal manifests; this table reports physical steps only.

The last target-only M6 scene is all-passing: its generating M5 student chooses
late short and passes in **3.58 s**, versus both priors at **3.52 s**, neutral
and both shorts at **3.58 s**, and sustained at **3.62 s**. The uniform M6 scene
is also all-passing: student and priors **4.00 s**, neutral **4.06 s**, shorts
**4.08 s**, sustained **4.12 s**. These are different acquired scenes and are
not a common-set comparison. Times measure crossing plus stabilization.
The last reference scene, `primary_candidate_93203_0905`, has seven failing
teachers and a failing student, all with measured environment contact. Each
capture records **1,192 steps**; the encounter costs **9,536 steps** and its
prefix **65,168 steps**. Both short rows also retain the finite-reference-
horizon-before-completion event. Their contact failures are known, while later
unrecorded completion is not imputed. Neither this all-failing scene nor the
reference arm's ten unsolvable assignments create response complementarity.

The [08:57 progress receipt](evidence/research-progress-20260909/M6_complete/all_M6_complete_progress.json)
records **23 new executions and 27,416 measured physics steps** since the
334-assessment checkpoint: **15 passes and 8 failures**. It completes the last
target teachers, uniform and reference M6 encounters. All earlier assessment
hashes match, including failures and the reconciled unknown; pending M7
execution is excluded. Both v8 process identities remain live with no stage
exit. No queue, runtime, teacher, learner, scoring or reserved assignment
changed. Continue the original worker without duplication. The next decisive
experiment remains the fifteen-policy M8 common-set development evaluation,
reusing the 48 completed comparators.

[The third matched M6 pair is complete](evidence/research-progress-20260909/M6_third_pair/seed93203_paired_M6_bound_readout.json).
On `primary_candidate_93203_0230` at seed 93203, both actual generating M5
students choose late prior-splice and pass in **4.18 s**. The complete teacher
tables match: neutral fails, both priors pass in **4.18 s**, both shorts in
**4.58 s**, and both sustained schedules in **4.62 s**. Candidate, scene and
branch order through M6, actual student policy bindings, and earlier teacher
prefixes are verified unchanged. Both learners remain ridge lambda 10, with
uniform-per-phase versus historical-observation-gap weighting. Each encounter
costs **9,536 steps**, each prefix **65,560 steps**; both banks and their fixed
prior schedules cover **6/6 acquired tasks**. This readout verifies bound
completed records without a second raw audit, new physics or fitting. These
are generating M5 student outcomes, not evaluations of the new M6 models.
Times measure crossing plus stabilization, not full adaptation/recovery.

The [08:25 progress receipt](evidence/research-progress-20260909/M6_third_pair/seed93203_pair_M6_progress.json)
records **14 new executions and 16,688 measured physics steps** since the
320-assessment checkpoint: **12 passes and 2 failures**. It completes the third
M6 pair and includes the next target-only student. All earlier assessment hashes
match, including failures and the reconciled unknown; pending branches are
excluded. It also binds all three completed M6 paired readouts: their actual
generating M5 students tie on their respective acquired conditions at **4.12,
4.68 and 4.18 s**, with each of the six corpora still admitting a fixed schedule
covering all six tasks. This supplies no observed replay benefit on these three
conditions, without establishing population equality or common-set performance.
Both original v8 process identities remain live with no stage exit. No queue,
runtime, teacher, learner, scoring or reserved assignment changed. Continue
the existing worker without duplication. The original fifteen-policy M8 common-
set development comparison, reusing the 48 completed comparators, remains
the next decisive experiment.

[Reference seed-93202 M6 is complete](evidence/research-progress-20260909/M6_reference93202/seed93202_reference_M6_bound_readout.json).
All seven teachers and the generating M5 student fail with measured undesired
environment contact. The student chooses early prior-splice, with maximum
measured contact **1,945.865 N**. Both short captures contain **996 physics
steps** and establish contact failure (maximum **660.271 N**); their later
unrecorded behavior remains unknown. The other six captures each contain
1,192 steps. The encounter costs **9,144 measured steps**, and its full prefix
**64,776 steps**. The acquired bank now covers **2/6 tasks**; a fixed schedule
also covers both solvable tasks. The four bank-unsolvable assignments stay in
the denominator. This encounter establishes neither an avoidable selection
error within the bank nor response complementarity. The readout verifies bound
collection, assessment, manifest, training and prior-prefix records, without
repeating raw-record scoring. The old teacher prefix and uniform-per-phase
ridge lambda 10 are unchanged; the readout adds no physics or fitting and
does not evaluate the new M6 policy.

The [08:08 progress receipt](evidence/research-progress-20260909/M6_reference93202/seed93202_reference_M6_progress.json)
records **11 new executions and 12,720 measured physics steps** since the
309-assessment checkpoint: **3 passes and 8 failures**. It includes the whole
reference encounter and the first three seed-93203 executed-contrast captures.
All previous assessment hashes match, including the reconciled unknown;
pending branches are excluded. Both v8 process identities remain live with
no stage exit. No queue, runtime, teacher, learner, scoring or reserved
assignment changed. Continue the existing worker, without duplication. The
original fifteen-policy M8 common-set development comparison, reusing the
48 completed comparators, remains the next decisive experiment.

[Target-only seed-93202 M6 has a verified initial-choice error](evidence/research-progress-20260909/M6_target_uniform93202/seed93202_target_M6_selection_bound_readout.json).
The generating M5 student commits to early prior-splice at tick 15 and fails
with measured environment contact (maximum **1,630.714 N**). Only the two
sustained teacher schedules pass, both in **3.58 s**; neutral, both priors and
both shorts fail. Recomputing the first pre-action raw history digest exactly
matches the completed teacher prefix, as do causal features and legality.
At that decision, early sustained and WAIT-to-late-sustained are passing
alternatives. Later neutral teacher decisions are not available after the
student's early commitment. This establishes an acquired-scene selection error,
but does not isolate WAIT's information benefit because an early passing option
also exists. Surface visibility does not establish semantic distinguishability.
The readout uses bound completed scoring records and a recomputed initial raw
history; it does not repeat full-episode raw scoring. The old teacher prefix
and uniform-per-phase ridge lambda 10 are unchanged. The encounter costs
**9,536 steps**, its prefix **65,560 steps**, and both sustained schedules
still cover **6/6 acquired tasks**. No new within-corpus passage complementarity
is established. The readout adds no physics or fits and does not evaluate M6.

The [07:54 progress receipt](evidence/research-progress-20260909/M6_target_uniform93202/seed93202_target_uniform_M6_progress.json)
records **15 new executions and 17,880 measured physics steps** since the prior
294-assessment checkpoint: **10 passes and 5 failures**. It completes all seven
target teachers and all eight uniform executions. The bound uniform M6 records
show its generating M5 student chooses early prior and passes in **3.62 s**.
All seven teachers pass: both priors in 3.62 s, neutral in **4.04 s**, both shorts
in **4.06 s**, and both sustained in **4.10 s**. Thus this all-passing scene
still provides measured passage-time preferences. These times are crossing
plus stabilization, not full adaptation/recovery completion. The uniform
encounter also costs **9,536 steps**, its prefix **65,560 steps**, and its old
teacher prefix and uniform ridge lambda 10 are unchanged. Uniform uses
completed bound records without a second raw audit. The arms acquired different
scenes, so these student outcomes are not a common-set constructor comparison.
Every earlier assessment hash matches, including failures and the reconciled
unknown. Pending branches are excluded; both v8 process identities remain live
with no stage exit. No native, collector, scoring, learner, queue or reserved-
assignment source changed. The original fifteen-policy M8 common-set development
comparison, reusing the 48 completed comparators, remains the next decisive
experiment.

[The second matched M6 pair is complete](evidence/research-progress-20260909/M6_second_pair/seed93202_paired_M6_bound_readout.json).
On `primary_candidate_93202_0600` at seed 93202, both generating M5 students
choose early prior-splice and pass in **4.68 s**. Their complete teacher tables
match: neutral fails, early prior passes in **4.68 s**, late prior in **4.70 s**,
both shorts in **4.80 s**, and both sustained schedules in **5.20 s**. The
candidate/scene/branch order is identical through M6, including both actual
teacher preparations. Previous teacher prefixes are unchanged; both learners
remain ridge lambda 10 with uniform-per-phase versus historical-observation-gap
weighting. Each encounter costs **9,536 steps** and each prefix **65,560 steps**.
Both banks and their best fixed schedules cover **6/6** acquired tasks. This
readout verifies completed bound collection/training records without a second
raw-record audit; it adds no physics or fits. The actual student tie is not
an evaluation of the newly fitted M6 models, a common-set comparison, or a
held-out result. It supplies no observed replay benefit on this encounter,
without establishing population equality. Passage time remains crossing plus
stabilization, not full adaptation/recovery completion.

The [07:36 progress receipt](evidence/research-progress-20260909/M6_second_pair/seed93202_pair_M6_progress.json)
records **15 new executions and 17,880 measured physics steps** since the prior
279-assessment checkpoint: **12 passes and 3 failures**. It completes the
second M6 pair and includes the next target-only student. Every earlier
assessment hash matches, including failures and the reconciled unknown;
pending branches are excluded. Both v8 process identities remain live with
no stage exit. No native, collector, scoring, learner, queue or reserved-
assignment source changed. The original fifteen-policy M8 common-set
development comparison, reusing the 48 completed comparators, remains the
next decisive experiment.

[Reference seed-93201 M6 is independently verified](evidence/research-progress-20260909/M6_reference93201/seed93201_reference_M6_verified.json).
The new scene has **no passing finite-bank schedule**: all seven teachers and
the generating M5 student fail with measured undesired environment contact.
Both short captures contain **996 physics steps**, 249 recorded physical rows
and 248 sensor packets; they terminate with process exit 1 after verified
contact (maximum **816.020 N**). Their task failures are established, while
later unrecorded behavior remains unknown. The other six captures contain
1,192 steps each. The encounter therefore costs **9,144 measured steps**, and
the complete prefix **64,776 steps**. The original raw teacher/student auditors
reproduce the new targets and outcomes, including both partial captures.
The earlier M5 teacher prefix is unchanged; the reconciled unknown student
remains in its history and accounting. Uniform-per-phase ridge lambda 10 is
unchanged. This corpus is now **4/6 bank-solvable**, with both prior schedules
covering those four tasks and minimum cover one. The all-failing scene remains
assigned and does not create response complementarity. Audit session 46261
returned 0 and is closed. The audit adds no physics or fitting, and the new
M6 policy is not evaluated here.

The [07:18 progress receipt](evidence/research-progress-20260909/M6_reference93201/seed93201_reference_M6_progress.json)
records **10 new executions and 11,528 measured physics steps** since the prior
269-assessment checkpoint: **8 failures and 2 passes**. This includes the full
reference encounter and two captures from seed-93202 executed contrast. Every
earlier assessment hash matches; pending branches are excluded. Both v8 process
identities remain live with no stage exit. No native, collector, scoring,
learner, queue or reserved-assignment source changed. The receipt also derives
the M8 acquisition count from the adopted plan: **1,065 assigned episodes**,
with **745 captures and 886,472 steps in completed checkpoint boundaries**.
The two active-encounter captures are separate from those boundary totals.
These acquisition counts are not policy-evaluation counts. The original
fifteen-policy M8 common-set development comparison, reusing the 48 completed
comparators, remains the next decisive experiment.

[The target-only M6 selection failure is verified](evidence/research-progress-20260909/M6_target_uniform93201/seed93201_target_M6_selection_verified.json).
For seed 93201 on `primary_candidate_93201_0420`, the generating M5 student
continues neutral at ticks 15, 50 and 70 and fails with measured environment
contact (maximum **1,108.068 N**). The original raw student auditor reproduces
the outcome; its recorded history, causal features and legal mask exactly match
the completed, bound teacher targets at every decision tick. WAIT has passing
continuations at ticks 15 and 50, although committing to late prior at tick 50
is faster. At tick 70, both late short and late sustained pass; neutral fails
and retains no later adaptation choice. Historical gap one at all phases
describes the generating policy's complete failed continuation, not three
isolated immediate-action failures. Surface visibility does not establish
semantic distinguishability. Both prior teachers pass in **5.28 s**, both short
in **5.44 s**, and both sustained in **5.82 s**. This is a student raw re-audit
against completed teacher records, without a second teacher raw audit. The
old teacher prefix and uniform-per-phase ridge lambda 10 remain unchanged.
The encounter costs **9,536 steps** and its prefix **65,560 steps**. Both
sustained schedules still cover all **6/6** assigned tasks in this corpus.
Audit session 79722 returned 0 and is closed. No new fit or physics was added
by the verification, and the newly fitted M6 policy is not evaluated here.

The [07:04 progress receipt](evidence/research-progress-20260909/M6_target_uniform93201/seed93201_target_uniform_M6_progress.json)
records **15 new executions and 17,880 measured physics steps** since the prior
254-assessment checkpoint: **11 passes and 4 failures**. It completes the seven
remaining target teachers and all eight uniform executions. The bound uniform
M6 records show its generating M5 student selects late prior and passes in
**5.24 s**; both prior teachers pass in 5.24 s, both sustained in **5.72 s**, and
neutral plus both shorts fail. Its encounter also costs **9,536 steps** and
prefix **65,560 steps**; its previous teacher prefix and uniform ridge lambda
10 are unchanged. These uniform results use completed bound records without a
second raw audit. The two arms acquired different scenes, so these student
outcomes are not a common-set constructor comparison. Every earlier assessment
hash matches, including failures and the reconciled unknown. Pending branches
are excluded; both v8 process identities remain live with no stage exit.
No native, collector, scoring, learner, queue or reserved-assignment source
changed. The original fifteen-policy M8 common-set development comparison,
reusing the 48 completed comparators, remains the next decisive experiment.

[The first matched M6 pair is complete](evidence/research-progress-20260909/M6_first_pair/seed93201_paired_M6_bound_readout.json).
On `primary_candidate_93201_0570` at seed 93201, both generating M5 students
select late prior-splice and pass in **4.12 s**. The seven teacher outcome/time
tables match: neutral fails; both priors pass in 4.12 s, both shorts in 4.48 s,
and both sustained schedules in 4.60 s. The original candidate/scene/branch
order agrees through M6, including both actual teacher manifests. Previous
teacher prefixes are unchanged; the learners remain ridge lambda 10 with
uniform-per-phase versus historical-observation-gap weighting. Each encounter
costs **9,536 steps**, and each prefix **65,560 steps**. Both finite banks and
their best fixed schedules cover **6/6 acquired tasks**. This readout verifies
bound completed collection/training records without a second raw-record audit.
It adds no physics or fitting, and does not evaluate the newly fitted M6
models. The measured tie on one shared acquired scene/seed supplies no observed
replay benefit there; it does not establish population or held-out equality.
Passage time remains crossing plus stabilization, not full recovery completion.

The [06:47 progress receipt](evidence/research-progress-20260909/M6_first_pair/seed93201_pair_M6_progress.json)
records **13 new executions and 15,496 measured physics steps** since the prior
241-assessment checkpoint: **10 passes and 3 failures**. It completes the first
M6 pair and includes the next target-only student. All earlier assessment hashes
match, including the retained failures and reconciled unknown; pending branches
are excluded. Both v8 process identities remain live with no stage exit. No
native, collector, scoring, learner, queue or reserved-assignment source changed.
The original fifteen-policy M8 common-set development comparison, reusing the
48 completed comparators, remains the next decisive experiment. Manuscript
fixed-schedule statements remain explicitly scoped to the completed M4 data;
no manuscript edit was made during this acquisition turn.

[The completed M5 bound-table readout](evidence/research-progress-20260909/M5_complete/all_M5_bound_response_tables.json)
uses the existing finite-bank response statistic, verifies all fifteen completed
model/teacher bindings, and matches every earlier M4 table to its original raw
audit. It is not a new all-capture raw audit. These prefixes contain **705
recorded captures and 839,184 measured physics steps**, covering **75 assigned
post-bootstrap encounters** plus the bootstraps. Reused scene identities are
not independent layouts. Reference contrast has **8/15 bank-solvable** tasks;
each other arm has **15/15**. All seven bank-unsolvable reference tasks remain
assigned. The previously reconciled unknown student remains in history and
physical accounting; no teacher branch outcome is unknown at M5.

**The response-complementarity finding has changed:** seed-93203 target-only
now has bank coverage **5/5**, best-fixed coverage **4/5**, and minimum schedule
cover **two**. Round 3 (`primary_candidate_93203_0148`) passes only with sustained
schedules; round 5 (`primary_candidate_93203_0612`) passes only with prior-splice.
Their passing sets are disjoint, and both tables have independent raw-record
verification through the M4 audit and the target M5 selection audit. Every
other individual M5 corpus still has one fixed schedule covering all its
solvable tasks, including all executed-contrast and replay corpora. Thus it is
no longer correct to describe every acquired corpus as admitting one fixed
response, or to imply that explicit negative contrast is necessary for this
observed complementarity. This is finite-bank capability on acquired tasks;
sensor realizability and downstream policy benefit remain unproven. No queue
or method intervention follows from this interim readout. Its executed script
is archived beside the receipt; it adds no physics or fits.

[The final reference M5 encounter is independently verified](evidence/research-progress-20260909/M5_complete/seed93203_reference_M5_verified.json).
For seed 93203, both sustained schedules pass in **3.46 s** and the other five
teacher schedules fail. The generating M4 student commits to early prior-splice
at tick 15 and fails with measured undesired environment contact (maximum
**1,875.590 N**). Its tick-15 neutral history, causal features and legality match
the teacher, and reconstructing the frozen model reproduces the recorded
choice. The historical physical gap is one there. At later ticks the student
has already committed: those histories do not match neutral teacher prefixes,
and their gaps remain null. Surface visibility does not prove semantic
distinguishability. The original raw auditors reproduce the new teacher targets
and student outcome; the old M4 teacher prefix is unchanged. The fit retains
uniform-per-phase ridge lambda 10. The encounter costs **9,536 steps**, and its
prefix **55,632 steps**. This corpus is **2/5 bank-solvable**, with both sustained
schedules covering its two solvable tasks. Audit session 96087 returned 0 and
is closed; no new physics or fitting was performed by the audit.

The [06:30 progress receipt](evidence/research-progress-20260909/M5_complete/all_M5_complete_progress.json)
records **11 new executions and 13,112 measured physics steps** since the prior
230-assessment checkpoint: **6 passes and 5 failures**. This completes the seven
remaining reference teachers and includes four M6 captures. Every earlier
assessment hash matches; pending branches are excluded. Both v8 process
identities remain live with no stage exit. No native, collector, scoring,
learner, queue or reserved-assignment source changed. The next decisive
experiment remains the original fifteen-policy M8 common-set development
comparison, reusing the 48 completed frozen comparators.

[The seed-93203 target-only M5 selection failure is independently verified](evidence/research-progress-20260909/M5_target93203/seed93203_target_M5_selection_verified.json).
The original raw auditors reproduce all seven teacher branches and the actual
generating M4 student. Both prior schedules pass in **2.44 s**; all other
teachers fail. The student waits at ticks 15 and 50, then selects short at tick
70 and fails with measured undesired environment contact (maximum **731.572 N**).
Actual neutral histories, causal features and legal masks match the teacher
prefixes at all three ticks. Reconstructing the frozen generating policy also
reproduces the recorded values and choices. Waiting at tick 15 was valid;
at tick 50, late prior-splice is the only passing legal choice. After that
entry opportunity, every remaining schedule fails. The historical physical
gap is one at ticks 15 and 50, but the tick-15 gap describes the failed later
policy continuation, not an incorrect immediate WAIT. Tick 70 has no feasible
teacher and its gap remains null. Surface visibility does not establish
semantic distinguishability. The original M4 teacher prefix is unchanged;
M5 fitting retains uniform-per-phase ridge lambda 10. Verification adds no
physics or fits, and does not evaluate the new M5 policy. Tool session 64214
returned 0 and is closed; its executed verification script is archived beside
the receipt. This acquired-scene failure does not establish an arm comparison.

The [06:18 progress receipt](evidence/research-progress-20260909/M5_target93203/seed93203_target_selection_progress.json)
records **13 new executions and 15,496 measured physics steps** since the prior
217-assessment checkpoint: **8 passes and 5 failures**. Every earlier hash
matches; the reconciled unknown remains preserved. Pending branches are
excluded. The receipt also binds the completed seed-93203 uniform encounter:
all seven teachers pass, and its generating M4 student passes in **3.00 s**,
matching the two prior teachers. This encounter costs **9,536 steps**, with
**56,024 steps** in its prefix; those uniform results are read from completed
bound collections, without a second raw audit. Both v8 process identities
remain live with no stage exit. No source, queue, learner, scoring or reserved
assignment changed. The original fifteen-policy M8 common-set development
comparison remains the next decisive experiment.

[The third paired M5 encounter is independently verified](evidence/research-progress-20260909/M5_third_pair/seed93203_paired_M5_verified.json).
Both arms preserve identical candidate/scene/branch order through M5 and the
original M4 teacher prefixes. The original raw teacher/student auditors
reproduce the new branches and saved targets. Both generating M4 students select
late prior-splice and pass in **2.78 s**; only the two prior schedules pass this
scene. Every capture spans 1,192 physics steps, with a full recorded horizon of
5.96 s. Passage time is crossing plus stabilization, not full adaptation/recovery
completion. Both eight-capture encounters cost **9,536 steps**, and each prefix
costs **56,024 steps**. The learners remain ridge lambda 10, with uniform-per-
phase versus historical-observation-gap weighting. The audit adds no physics
or fits; tool session 99289 returned 0 and is closed.

The [06:00 progress receipt](evidence/research-progress-20260909/M5_third_pair/seed93203_pair_progress.json)
records **16 additional full executions and 19,072 steps** since the prior
201-assessment checkpoint: **6 passes and 10 failures**. All earlier hashes,
failures and the reconciled unknown remain preserved. Pending branches are
excluded, and both v8 process identities remain live with no stage exit.
The receipt also binds all three paired M5 student results: both generating M4
policies pass in **4.84, 4.82 and 2.78 s** for seeds 93201, 93202 and 93203,
respectively. This descriptive equality on three acquired scene/seed conditions
does not establish population/held-out equality or evaluate the new M5 fits.
It supplies no observed replay benefit on those encounters. No source, queue,
learner, scoring or reserved assignment changed; the original M8 common-set
development comparison remains the next decisive experiment.

[Reference seed-93202 M5 is independently verified](evidence/research-progress-20260909/M5_reference93202/seed93202_reference_M5_verified.json).
All seven teacher schedules and the generating M4 student fail with complete
1,192-step recordings and measured undesired environment contact. The encounter
costs **9,536 steps** and remains assigned; the 47-capture prefix costs
**55,632 steps**. The original raw teacher/student audits reproduce the saved
outcomes and new teacher targets. The M4 teacher prefix is unchanged, and the
fit retains ridge lambda 10 with uniform-per-phase weighting. Within this
corpus's five assigned encounters, **2/5 are bank-solvable and 3/5 bank-unsolvable**;
both prior schedules cover the two solvable tasks. Minimum cover remains one:
the all-failing encounter does not falsely establish response complementarity.
This is matched finite-bank capability on acquired development conditions,
not held-out policy performance. The verification adds no physics or fits;
tool session 50508 returned 0 and is closed.

The [05:42 progress receipt](evidence/research-progress-20260909/M5_reference93202/seed93202_reference_progress.json)
records **12 additional executions and 14,304 measured steps** since the prior
189-assessment checkpoint: **2 passes and 10 failures**, with no new unknown.
This includes the complete reference encounter and the start of seed-93203
executed contrast. All prior assessment hashes match; the older reconciled
unknown remains retained. Pending branches are excluded. Both v8 process
identities remain live with no stage exit. No source, queue, learner, scoring
rule or reserved assignment changed. The next decisive experiment remains the
original M8 common-set development comparison after acquisition completes.

The [05:27 uniform completion and progress receipt](evidence/research-progress-20260909/M5_uniform93202/seed93202_uniform_progress.json)
records **6 additional full executions and 7,152 steps**, all passes, since
the prior 183-assessment checkpoint. All earlier hashes remain unchanged,
including the preserved unknown. The completed seed-93202 uniform M5 encounter
has seven passing teacher schedules: neutral and both priors take **5.36 s**,
both short schedules **5.86 s**, and both sustained schedules **5.94 s**.
The generating M4 student passes in 5.36 s. Times mean crossing plus
stabilization, not full adaptation/recovery completion. This encounter requires
no adaptation but has measured time differences; no comparative policy gain is
established. It costs 9,536 steps across eight captures, with 56,024 cumulative
steps. Bound controller/collector/model-prefix checks confirm the preserved M4
teacher prefix and ridge lambda 10 with uniform-per-phase weighting. No second
raw rescore or fit was performed. Both v8 process identities remain live; the
original M8 acquisition and 90-learned/48-comparator development panel remain
the next decisive dependency. No source, queue, scoring or reserved split changed.

[Target-only seed-93202 M5 completion](evidence/research-progress-20260909/M5_target93202/seed93202_target_M5_complete.json)
binds all seven passing teacher records and the unchanged earlier teacher
prefix. The new fit retains ridge lambda 10 and uniform-per-phase weighting.
Neutral and both prior schedules pass in **5.28 s**, both short schedules in
**5.42 s**, and both sustained schedules in **5.84 s**. These are crossing-plus-
stabilization times; every full recording lasts 5.96 s. The generating M4
student also passes in 5.28 s. Thus the encounter requires no adaptation but
contains a measured time-choice signal; it is not evidence of construction or
post-update policy superiority. The eight-capture encounter costs 9,536 steps
and its prefix 56,024 steps. This check verified bound controller/collector,
invocation and model-prefix receipts; it did not repeat the raw scoring or fit.

The [05:20 progress receipt](evidence/research-progress-20260909/M5_target93202/seed93202_target_progress.json)
records **8 additional full executions and 9,536 steps**, all passes, since the
175-assessment checkpoint. These include completion of target-only teachers and
the start of uniform acquisition; earlier target-only student/teacher captures
are not counted again. All 175 prior hashes remain unchanged, including the
historical unknown. Both v8 process identities remain live. No native source,
queue, scoring rule, learner, or held-out assignment changed this turn.

[The second paired M5 encounter is independently verified](evidence/research-progress-20260909/M5_second_pair/seed93202_paired_M5_verified.json).
Both arms retain identical candidate/scene/branch order through M5, including
`primary_candidate_93202_0428`. The original raw-record auditor reproduces both
new teacher tables and their saved targets. Prior M4 teacher prefixes are
unchanged; both learners retain ridge lambda 10. Executed contrast uses
`uniform_per_phase`; replay uses `historical_observation_gap` weighting.
The **generating M4 students** both select early prior-splice and pass in
**4.82 seconds**. Both prior schedules pass in 4.82 seconds, both sustained
schedules pass in 5.36 seconds, and neutral plus both short schedules fail.
Each eight-capture M5 encounter costs **9,536 steps**, and each 47-capture prefix
costs **56,024 steps**. This one matched acquisition encounter shows no replay
passage/time advantage. It does not evaluate the new M5 fits or establish a
common-set or held-out result. The read-only verification adds no physics or
fits; tool session 11506 returned 0 and is closed.

The [05:11 progress receipt](evidence/research-progress-20260909/M5_second_pair/seed93202_pair_progress.json)
records **14 additional executions and 16,688 measured steps** since the prior
161-assessment checkpoint: **10 passes, 4 failures**, and no new unknown outcome.
All 161 earlier assessment hashes remain unchanged, including the earlier
reconciled unknown student. Pending executions are excluded. Both v8 process
identities remain live with no stage exit; no source/queue/scoring change was
made in this progress turn. The next decisive experiment remains acquisition
to M8 followed by the original 90 learned plus 48 comparator development panel.

V7 coordinator **2411988** and child **2412056** ended unexpectedly. Tool session
**50750 returned 143 and is closed**; no acquisition-child exit receipt exists,
and the cause is unknown. The native reference M5 student finished saving its
recording at 04:26 UTC after its parent ended. At recovery, no native simulator
or GPU process remained. Its new reconciliation receipt explicitly stores
`exit_status: null`, not an invented success/failure exit code.

[Recorded-student reconciliation](RECORDED_STUDENT_RECONCILIATION.md) preserves
all **1,192 measured steps** and the original scorer's **unknown task outcome**.
There was no retry, erased attempt, or imputed passage. The separately declared
`motion2scene_resume_recorded_student.py` adapter bypasses only this one
reference-arm student collection's admission pause. The reference learner uses
uniform complete teacher targets; the student outcome cannot change its fit.
The original replay auditor independently accepted one recorded student and
its measured cost, while retaining `task_outcome_admitted: false`. All seven
teacher branches remain required. Native/controller/scorer/learner sources and
the existing path-repair adapter are unchanged. Fifteen focused tests pass;
prelaunch verification preserved all 120 earlier assessment hashes and passed
the same 222-artifact native inventory.

The later raw-audit reader's rejection of a known-cost unknown student was
reproduced on the actual recording and repaired through
`scripts/research/motion2scene_response_diversity_recorded_student.py`.
This separate reader accepts only the declared reconciled student, retains its
unknown outcome and cost, uses the original reader for all other sources, and
binds both implementations in its output. The original M4 audit/source remains
unchanged. Fourteen reader/response tests pass; the actual-record check confirms
unchanged rows and 1,192 steps. **Full M8 analysis has not run.** See
[reader validation](evidence/research-progress-20260909/recorded_student_recovery/unknown_reader_validation.json)
and the documented command in the reconciliation note.

[Reference M5 is complete and independently verified](evidence/research-progress-20260909/recorded_student_recovery/reference_M5_complete.json).
Its M4 prefix is unchanged; all seven new teacher branches reproduce the saved
targets, and the new fit remains ridge lambda 10 with `uniform_per_phase`
weighting. Both prior-splice schedules pass; neutral, both short schedules and
both sustained schedules fail. The pre-update student remains unknown. This
encounter costs **9,144 measured steps** (two 996-step contact-established short
failures, six 1,192-step captures); its 47-capture prefix costs **55,632 steps**.
The re-audit adds no physics or fitting. This is an acquisition result, not
common-set M5 policy performance.

The [04:55 progress receipt](evidence/research-progress-20260909/recorded_student_recovery/reference_M5_and_next_corpus_progress.json)
verifies **10 additional completed executions and 11,724 measured steps** since
the preceding 151-assessment checkpoint: four passes and six failures. The
earlier unknown student is preserved and not counted again. All 151 previous
assessment hashes match. Both v8 process identities remain live, with no stage
exit receipt; the next seed-93202 executed-contrast teacher branch is active.

At 04:44 UTC the resumed first teacher (`short_e070_r265`) has completed with
a measured contact-established failure and 996 steps; the next neutral branch
is running. Its missing tail remains unobserved. The
[current progress receipt](evidence/research-progress-20260909/recorded_student_recovery/M4_audit_and_M5_recovery_progress.json)
records **31 additional captures and 36,756 measured steps** since the prior
120-assessment checkpoint: **24 passes, 6 failures and 1 unknown**. It includes
the recovered v7 student once, with zero extra physics from reconciliation,
and excludes the active branch. All 120 previous hashes remain unchanged.

V6 (coordinator 2387363, child 2387430) **ended with exit 1** after completing all
three reference M3 boundaries. Session 69371 returned exit 1 and is closed.
At the first original M4 import, the frozen expansion driver passed a JSON
`inherited_model` string to the Path-only artifact reader, raising
`AttributeError: 'str' object has no attribute 'resolve'` before any new M4
physical intent. The [declared bookkeeping repair](evidence/research-progress-20260909/acquisition_path_repair/repair.json)
uses `scripts/research/motion2scene_resume_expanded_acquisition.py` to normalize
strings in that imported driver's artifact call. **No frozen file or plan was
edited.** The original collector, native source inventory, scorer, teacher,
ridge learner, candidate identities/order, budgets and histories are retained.
Existing fitting registrations still bind the exact original driver; the
successor sequence additionally binds the effective adapter and repair receipt.

Ten focused repair/original-continuation tests pass, including reproducing the
original error, importing the exact inherited model with no physics or fitting,
rejecting reordered prefixes and changed hashes, and restoring the original
reader on exceptions. Black and Ruff pass. The
[prelaunch validation](evidence/research-progress-20260909/acquisition_path_repair/validation.json)
checked all **60 inherited M0–M4 models**, preserved **93 existing assessments**,
and passed the unchanged native preflight with all **222 bound artifacts**.
A preliminary read-only check mistakenly resolved the virtualenv interpreter
symlink to base Python and was rejected by the unchanged inventory guard; the
invocation was corrected to the declared venv path. No environment was changed.
The predecessor processes were terminal and the GPU empty before v7 launched.
M16/M32 now also remain unlaunched if the M8 development stage exits unsuccessfully.

**The first eight physical executions after the path repair completed**, totaling
**9,536 measured steps**: seed 93201 reference M4's pre-update M3 student and all
seven teacher schedules fail. All eight are full 1,192-step captures; no new
unknown task outcome or retry was introduced. The
[independently verified M4 receipt](evidence/research-progress-20260909/acquisition_path_repair/first_reference_M4.json)
binds the exact prior model, scene and branch order, preserves the M3 teaching
prefix, and reproduces the new teacher group with all fourteen required prefix
matches and every legal teacher target available. The unchanged ridge fit is
complete. Its cumulative 39-capture prefix costs **46,488 steps**. All **93
pre-repair assessment hashes remain unchanged**. This entire M4 encounter is
bank-unsolvable and stays in all task/cost denominators; it is not evidence of
complementary passing responses or common-set policy improvement.

All three fourth reference encounters are now complete: every student and every
teacher fails. Seed 93202's two short branches are contact-established partial
failures with 996 steps each; its M4 encounter costs 9,144 steps, and cumulative
39-capture prefix costs 46,096. Seed 93203's eight M4 captures are complete,
costing 9,536 steps; its cumulative prefix also costs 46,096 because its two
partial failures occurred at M3. All known task failures are retained, and
unobserved tails are not imputed.

The [fifteen bound M4 completion receipts](evidence/research-progress-20260909/five_arm_M4/M4_boundaries_complete.json)
account for **585 captures and 696,536 measured physics steps**. Reference
contrast costs 138,680 steps across three corpora; each other arm costs 139,464.
These totals are now verified by the **completed independent raw-record audit**,
SHA `2a25c83e64a0bd34e3d4615a85aac90f52bf9acc4b1cfc0c08a4f09bb97c646e`.
It finished around 04:12 UTC. Watcher **2425228** and auditor **2438097** ended
normally; tool session **43325 returned 0 and is closed**. Its
[validation](evidence/research-progress-20260909/five_arm_M4/validation.json)
checks 175 pinned references and rejects a changed assignment. The
[launch](evidence/research-progress-20260909/five_arm_M4/launch.json)
invoked the unchanged `motion2scene_response_diversity.py` with the expanded
plan at budget 4. Completed output is
`m2s-five-arm-M4-response-diversity-20260910-v1/result.json`.
**Do not launch a duplicate or refit/re-execute historical prefixes.**
The audit adds no physical execution, fit, or protocol change. All twelve
original corpus reports and four original arm summaries match the earlier audit
exactly. [Five-arm M4 report](FIVE_ARM_M4_ACQUISITION.md), per-corpus CSV and the
measured M1/M4 acquisition figure are generated and checked. Eight relevant
response-diversity tests pass; the renderer passes Black and Ruff.

Across each arm's twelve assigned encounters, bank-solvable/adaptation-required
counts are uniform **12/2**, target-only **12/4**, reference **6/6**, executed
**12/12**, and replay **12/12**. Six reference bank-unsolvable tasks stay in the
denominator. Every individual corpus still admits one fixed response covering
all its solvable tasks. Executed construction improves useful-task yield here;
these measurements do not establish a downstream selection advantage. The
figure is a construction-yield curve, not a common-set policy learning curve.
All three executed/replay teacher tables and candidate orders match exactly.
Recorded geometric-search costs, including the failed reference-v1 search,
are separately visible in the report without invented per-arm amortization.

The [first M5 pre-update student](evidence/research-progress-20260909/five_arm_M4/first_M5_student.json)
passed in a full 1,192-step execution on assigned candidate
`primary_candidate_93201_0338`. Its committed policy and exact earlier collection
prefix are the original inherited M4 model, with no current-encounter teacher
data in the fit. This is an actual acquisition execution, not a newly trained
M5/M8 common-set policy result. Both seed-93201 executed/replay M5 encounters
are now complete, with identical candidate `primary_candidate_93201_0338`,
teacher outcomes and eight-capture cost of 9,536 steps each. Both pre-update
M4 students pass in 4.84 seconds; this does not evaluate the new M5 fits.
[Paired verification](evidence/research-progress-20260909/M5_first_pair/paired_M5_verified.json)
binds the exact generating models and unchanged prefixes. Actual replay uses
0.6 historical-gap + 0.2 uniform + 0.2 coverage weighting, with no fallback.
Its largest weight remains attached to M1 outcomes generated by M0, not a new
M5 physical gap; phase-70 M5 gap is unavailable after commitment.

The [preceding progress receipt](evidence/research-progress-20260909/five_arm_M4/M4_complete_and_M5_progress.json)
binds **19 new executions and 22,256 measured steps** since the previous
101-assessment checkpoint: two passes, seventeen failures, 0 unknown task outcomes.
It verifies all 101 earlier hashes unchanged and records the four live process
identities. This count excludes all eight first-reference M4 executions already
reported in the preceding progress turn, and includes only closed assessments;
active branches are not imputed.

The predecessor v5 coordinator (2332039), comparator runner (2332263), and
retained original acquisition worker (1955047) have **ended**. Session 35814
returned exit code 1 and is closed. Comparators themselves ended with exit 0
after 2,965.088 seconds. The original acquisition worker was SIGCONT-resumed,
then its unchanged preflight rejected a newly added **offline** selector file
inside the frozen editable `gear_sonic` package. That was the only inventory
difference: no existing runtime file changed. The first new teacher had no
intent or attempt at that rejection. V5 then failed its complete-M8 check, as
required; it did not falsely advance to learned-policy evaluation.

Recovery moved the byte-identical new selector to `scripts/research`, preserved
the original declaration/source snapshots, and updated only its offline import.
The original runtime preflight now passes with all **222 bound artifacts** and
the exact original editable/package identities. No check, frozen environment,
runtime, policy, scorer, acquisition queue or budget was amended. V6 was launched
only after authoritative process termination and this successful preflight.
Its [recovery](evidence/research-progress-20260909/comparators_complete/acquisition_recovery.json)
and [validation](evidence/research-progress-20260909/comparators_complete/recovery_validation.json)
retain the failure and source relocation. **Do not add even an unused new file
inside the frozen editable runtime package during this study.** Keep offline
development helpers under `scripts/research`.

All **48/48 comparator assignments** completed: 32 passes, sixteen failures,
**56,780 measured physics steps**, no unknown task outcome. This completes the
thirteen comparator episodes remaining at the 02:15 check (35 episodes and
41,284 steps), adding 15,496 measured steps. One contact-established failure
has a 756-step partial capture; its unobserved later fall status remains unknown.
The [generated measured table](M8_DEVELOPMENT_COMPARATORS.md) and
[corrected result](evidence/research-progress-20260909/comparators_complete/result.json)
show script **6/6**, finite-bank capability **6/6**, best fixed prior-splice
**5/6**, fixed sustained **4/6**, fixed short **3/6**, and neutral **2/6**.
Both prior-splice schedules average **−0.204 seconds versus script on five
mutually successful conditions**. These are actual executions at seed 8732 on
six reused development contexts, with all tasks retained. They establish a
matched comparator capability/time tradeoff; they are not learned M8 or
reserved-layout results.

The first summary reader marked 39 complete single-beam fall outcomes unknown
because it only recognized the multi-beam whole-episode field. The corrected
reader respects each existing scoring schema, without changing the scorer or
any physical data. V1 output and exact reader bytes are retained, and the
[correction receipt](evidence/research-progress-20260909/comparators_complete/analysis_correction.json)
verifies identical passage, timing, contact, capability and physical costs.
The final event count is no observed fall/upright-threshold failure in 47 complete
captures, with one unknown tail; fourteen captures have measured undesired
environment contact. The script's six successful executions show no measured
contact under the declared normal-force/counterpart rule, not a general safety
guarantee. Comparator runtime closure verification checked 110 distinct bound
artifacts and confirmed that the offline selector was never in that closure.

The reference-arm **M3 checkpoints for seeds 93201 and 93202 are complete**.
Their [first](evidence/research-progress-20260909/reference_M3/seed93201.json) and
[second](evidence/research-progress-20260909/reference_M3/seed93202.json) checks
bind the exact M2 pre-update policy, the assigned scene and seven-branch order,
the four-collection teaching prefix, ridge λ=10 and uniform phase weighting.
Each has 31 total acquisition captures and **36,952 measured steps**. Each new
M3 scene passes the two prior-splice schedules and fails the other five; each
pre-update student passes. This is acquisition evidence, not post-update
common-set performance. The third reference M3 checkpoint is now complete and
[independently re-audited](evidence/research-progress-20260909/reference_M3/seed93203.json):
its seven teacher branches and pre-update M2 student all fail. Five teacher
captures are complete; two short-schedule captures end after 996 measured steps
each with verified environment contact. Their known task failures remain
separate from unavailable later fall/recovery measurements. Teacher cost is
**7,952 steps**, and the complete 31-capture prefix costs **36,560 steps**.
All fourteen required prefix comparisons match exactly, and every legal
continuation label is available. The source teacher group recomputes exactly.
The entire bank-unsolvable encounter stays in task/cost denominators and cannot
be used to manufacture complementary passing responses.

The 02:57 [progress receipt](evidence/research-progress-20260909/reference_M3/progress_20260910T0258Z.json)
binds **eleven additional acquisition captures and 13,112 measured physics
steps** since the 02:43 initial check: three passes and eight failures. It
excludes the 77 already completed captures and leaves the active branch
unimputed. V6 `post_resume_verification.json` separately preserves the earlier
five-branch recovery snapshot and unchanged seventy pre-intermission hashes.
Do not count the scheduling pause, source relocation, or offline analysis as
physics.

**Recorded perception-to-decision examples are now extracted and plotted.**
`scripts/research/motion2scene_script_perception_traces.py` reads all six
completed script executions and recomputes sixteen actual neutral-phase
readouts from the delivered 114D features, legal masks, tick and frozen script.
All sixteen match their recorded commands. This is readout verification on
existing executions, not sixteen new policy evaluations. The
[trace artifact](evidence/research-progress-20260909/script_perception/result.json)
retains actual sensor/trajectory/result identities, packet capture times,
unknown fractions, switches and final physical outcomes.

The [inspected figure and decision table](figures/script_perception/README.md)
show the prior-only/sustained-only development pair, with PNG and PDF figures
generated by `scripts/research/motion2scene_plot_script_perception.py`. The
robot's net planar displacement from the first registered WAIT decision to
commitment is **0.127332 m at 1.0 s** for the early prior and **0.434379 m at
1.4 s** for the late sustained choice. Preaction packet capture times at those
commitments are 0.98 and 1.38 s. Both examples already have different upper-hit
bands at the first 0.30 s decision, so these traces **do not establish an
additional-information advantage for WAIT**. They show neutral execution
continuing while a later supported entry is retained. Unknown space remains
visible; no postcommit features are plotted, and surface visibility is not
equated with general alias resolution. The separate same-repertoire WAIT
control remains necessary.

The final neutral decision in the empty scene has no later entry and is
labeled `CONTINUE_NEUTRAL`, not a claim of retaining later choices. The original
extraction and source bytes are preserved; the
[label correction](evidence/research-progress-20260909/script_perception/trace_label_correction.json)
changes one offline label and adds explicit remaining entry ticks, without
changing a physical command. Six focused trace-reader tests pass, including
future packets, command/switch disagreement, missing ticks, commitment
boundaries and the final neutral choice. Black and Ruff pass the reader,
plotter and new test module. The figure was visually inspected. No new fit,
geometric query or physical execution was introduced by this trace work.

The v8 sequence reuses the 48 comparator captures when it finalizes/runs the
full M8 panel, then continues the same acquisition through M16/M32. It explicitly
defers the optional 204-episode motion-extension capability study, six related
fits and 24 extension-policy episodes to prioritize the central comparison.
Their declarations and data remain preserved. This change was declared before
new comparator outcomes and changes no primary acquisition queue or budget.
The v4 readiness proposal was validated but never launched; v5 completed the
comparators; v6 restored the runtime inventory and v7 repairs inherited-path loading. **Do not call the old M8 `prepare` command:** it rejects the 48
already prepared directories. The coordinator uses the new adapter's
`finalize` command and then the unchanged original full-panel runner.

All three reference M4 prefixes are complete; no M8 corpus is complete. The
single next executable dependency is completing the already resumed five-arm
M8 acquisition, then executing its ninety frozen learned-policy assignments.
The bounded response-
complementary pilot remains conditional on the M8 decision gate. Its first
bounded proposal selection is now implemented and completed, with the
geometric shortfall below; **do not launch v1 pilot physics or silently refill
its proposal budget**.
The earlier
[completed handoff receipt](evidence/research-progress-20260909/M4_completed_handoff_20260910T0048Z.json)
checks every new reference bootstrap model/policy/teacher/registration hash,
the first reference M1 model, M4 completion references and the live process.

**Bounded complementary selector v1: implemented, tested, no physical pilot.**
`scripts/research/motion2scene_complementary_acquisition.py`
selects the common passing schedule using only complete bank-solvable earlier
training tasks, then lowest mean real passage time and registry tie order.
All-failing and unknown tasks remain assigned but cannot falsely empty the
passing-set intersection. All three exact executed-contrast M2 prefixes select
`prior_splice_e015_r265`; their matched training passage means are 5.00, 4.14
and 4.38 seconds. Their 82,248 unique historical acquisition steps are reused
prefix cost, not new physics. Original uniform phase weighting, scene-wise
complete-continuation teacher, ridge λ=10 and measured-tie initialization were
verified against the saved M2 registrations.

The separate
[declaration](evidence/research-progress-20260909/complementary_pilot_v1/declaration.json)
precedes selection and freezes two paired M2-to-M4 encounters per corpus:
ordinary early-constraint, then ordinary versus targeted short-then-sustained.
The target is a distinct robust positive against predicted interference for
the earlier common schedule, preserving the original 81-offset, 0.01 m margins,
ranking, tracker, schedules, sensing, teacher, learner and physical branch cost.
The two arms share the same earlier prefix and strata. Original queues through
M32 and all six development geometries are excluded by geometry identity, without
reading their outcomes. Maximum physical assignments would be 96 acquisition
episodes (84 teacher, 12 pre-update student; 114,432 maximum steps), twelve fits,
and 36 matched policy evaluation episodes (42,912 maximum steps). None launched.

The [selection](evidence/research-progress-20260909/complementary_pilot_v1/selection.json)
examined the first 256 frozen draws in each stratum and seed: **1,536 proposals,
zero targeted candidates even before exclusions**. There were forty ordinary
contrasts before exclusions; only three early-constraint candidates remain
after excluding original assignments, and no short-then-sustained candidate.
No complete paired continuation exists. The
[pre-physics decision](evidence/research-progress-20260909/complementary_pilot_v1/prephysics_decision.json)
independently checks the same bounded array eligibility and explicitly prevents
launching this v1 pilot. It is a geometric support shortfall, not a physical
failure rate or a test of the learning hypothesis. No budget was expanded,
no original history changed, and no physical success was imputed. Any revision
needs its own bounded development declaration after the M8 decision.

The selector reused previously computed geometry: 347,056 historical clearance
queries lie within the considered draws; the full shared source pool cost was
967,424 queries and 457.371462 seconds for 3,840 proposals. New ranking time was
0.036435 seconds, with **zero new clearance queries, fits or physics steps**.
The original full pool cost remains separately visible. The first auxiliary
array check used the research venv, lacked sklearn and exited before writing;
the corrected IsaacLab-venv check completed. This did not rerun selection.

Reproduce with `.venv_isaaclab/bin/python
scripts/research/motion2scene_prepare_complementary_pilot.py declare
--expanded-plan <expanded-plan>/plan.json --m2-audit <M2-audit>/result.json
--development-study <M8-panel>/study.json --out <new-directory>`, then the
same script's `select --out <new-directory>`. Existing declarations/selections
are never overwritten. The original data root is
`/home/linjiw/research-data/groot-wbc/m2s-complementary-acquisition-pilot-20260910-v1`.
The original implementation bytes are retained in that directory's
`source_snapshot`, with explicit
[source relocation](evidence/research-progress-20260909/complementary_pilot_v1/offline_source_relocation.json)
and [validation](evidence/research-progress-20260909/complementary_pilot_v1/relocation_validation.json)
receipts. The selector bytes and preparation function ASTs are unchanged; only
the offline import/location changed to restore the runtime inventory. Do not
alter the completed selection or interpret relocation as another pilot run.

Validation: `pytest -q decoupled_wbc/tests/test_motion2scene_complementary_acquisition.py
decoupled_wbc/tests/test_motion2scene_prepare_complementary_pilot.py
decoupled_wbc/tests/test_motion2scene_acquisition_pool.py` with the IsaacLab venv:
**31 passed**. After relocation and the comparator-reader correction, the same
three modules plus `test_motion2scene_development_comparators.py` and
`test_motion2scene_development_panel_statistics.py` pass **51 tests**. Black and
Ruff pass the new implementation/test files; `git diff --check` passes.
This is implementation/offline evidence only.

The [M4 checkpoint report](M4_ACQUISITION_CHECKPOINT.md) records **468 acquisition
episodes and 557,856 measured physics steps**, including all bootstrap,
pre-update students and complete teacher branches. The independent raw audit
retains 117 teacher and ten student failures, with no unknown outcome in the
completed prefixes. At 139,464 steps per arm, adaptation-required yield is
2/12 uniform, 4/12 target-only and 12/12 each for executed contrast and replay.
Every corpus still admits one fixed schedule covering all four tasks. Across
constructors, 34 unique geometry/seed conditions have bank coverage 34/34 and
best-fixed coverage 32/34. Uniform, executed contrast and replay tie in the
recorded-input cross-corpus folds at 21/22, 21/23 and 22/23, versus target-only
16/22, 21/23 and 16/23. The tied arms select a constant prior within each corpus.
These are recorded branch selections, not new physical policy executions.
Higher task yield is supported; better perceptive selection remains unresolved.

The unchanged renderer generated the completed
[M1/M2/M4 evidence](evidence/research-progress-20260909/M1_M2_M4/result.json) in a
fresh directory, preserving M1/M2 artifacts. The working manuscript includes
the measured M4 table, response figure and readout; it compiles to 20 pages.
The figure's manuscript page was visually inspected. No new fit or physics
was added by the audit/readout/reporting work.
The [later portable-curve tools](PORTABLE_ACQUISITION_CURVE.md) now pass 61 tests,
reproduce all 36 M2 fits and three explicitly separate expanded-format CPU fits.
No M8/M16/M32 dataset export or new closed-loop policy result is implied. Resume
the existing expanded acquisition; the next decisive policy
experiment is the unchanged 138-episode M8 development panel.

Earlier handoff receipts are retained as dated history. The
[23:54 live handoff receipt](evidence/research-progress-20260909/acquisition_wait_20260909T2354Z.json)
verifies all eleven completed boundary/model/policy hashes and the watcher's
174 pinned references before the final M4 model completed.
The [collection completion snapshot](evidence/research-progress-20260909/M4_collection_completion_20260910T0006Z.json)
checks the original plan's complete collection inventory, per-row step sums,
distinct trajectory paths and every student's generating prior policy. It
retains 117 teacher and ten student failures, subsequently confirmed by the
completed raw-record audit. The original startup reservation remains separate.
The [M4-to-M8 handoff verification](evidence/research-progress-20260909/M4_to_M8_handoff_20260910T0014Z.json)
checks all twelve final model/policy boundaries, all sixty inherited M0–M4
model slots, unchanged scene/student/teacher paths, ridge λ=10, arm weighting,
paired queues through M32 and 173 staged source references. All three reference
corpora have independently assigned prefixes. No history was replaced.

All three seven-way M4 replay controls are complete. The
[verified M2/M4 control report](REPLAY_CONTROLS_M2_M4.md) rechecks 42 saved
policies, 252 recorded-context choices and 126 archived-signal weight recipes
with no new fits or physics. All controls score 5/6 on the same six development
contexts and select the same one schedule within each corpus. Gated/ungated
weights remain identical; all twelve nonempty M4 phase cues are eligible.
Replay's recorded-input null result extends through M4. Its independent
closed-loop benefit remains unproven, and the frozen M8 arms remain unchanged.
The [00:30 reference-bootstrap snapshot](evidence/research-progress-20260909/M4_M8_live_20260910T0030Z.json)
records the earlier two-bootstrap boundary. The completed M4 task-yield readout
agrees with the now-completed independent audit. Its stored cost excludes
bootstrap; the raw audit and manuscript include it. Preserve both receipts and
their stated accounting scopes.

| Claim/dependency | Accessible evidence at session entry | Interpretation |
| --- | --- | --- |
| Runtime | Isaac Lab environment, SONIC assets, RTX 5080, active original dispatcher | Physical simulation is available; preserve the active acquisition trajectory |
| Manuscript | `submission/traversal_method_v2.tex` and supplement read | Reports development script parity, cost regression, M1; no M8 or reserved results |
| Acquisition | Original plan/adoption V3, 12 M1 models and 11 M2 model receipts | One M2 corpus still collecting at entry; result-file existence is not a physical re-audit |
| Core comparison | Five-arm 32-encounter expansion already declared | Scene-wise complete-continuation teacher, common phase ridge λ=10; reference gets its own prefix |
| Replay | Seven fixed-data controls available at two M2 corpora, all 5/6 recorded branch proxies | No new policy rollouts or established replay gain |
| Held-out claim | Reserved protocol still unexecuted | No supported held-out superiority or acquisition-efficiency conclusion |

Entry hashes and counts: [initial-state.json](evidence/research-progress-20260909/initial-state.json).
Existing workspace modifications predate this session and must be preserved.

The completed original physical dispatcher was
`/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-dispatch-tie-proposed-v5/dispatch.py`,
using `m2s-primary-acquisition-plan-tie-proposed-v5/plan.json` and
`m2s-primary-acquisition-adoption-v3/adoption.json` under the same data root.
The earlier staged manifest was
`m2s-expanded-and-capability-stages-20260909-v3/sequence.json`. Its parent was
superseded by the recorded v5 handoff above while preserving its live acquisition
child. The optional extension stages are now explicitly deferred; the core
M16/M32 trajectory remains unchanged. Do not edit a live stage or bound source file.

The bounded prerequisite suite passed **54 tests** in 20.70 s: schedule teacher,
timed collector, sensor alignment, primary controller, expanded acquisition,
acquisition pool, timed outcome and observation timing. Its 33 warnings are an
existing invalid-escape deprecation in inspected source text. This is software
validation, not independent verification of all recorded physical outcomes.

New work: `scripts/research/motion2scene_response_diversity.py` re-audits complete
prefixes and measures passing-set patterns, all fixed-schedule coverages, minimum
schedule covers, and bank-minus-best-fixed coverage. It includes unsolvable
assigned tasks, keeps unknown bounds explicit, verifies paired replay queues,
and separates bootstrap, student, teacher physics from proposal computation.
It never edits queues or fits a learner. Eight focused tests pass.

The next decisive experiment remains the registered **M8 matched closed-loop
development panel**, after all original M4 and expanded M8 prefixes complete.
No current checkpoint or reserved outcome justifies changing the learner,
teacher, sensing interface, score, action bank, or frozen candidate ordering.

Session progress: all **12 M2 prefixes completed at 18:56 UTC**, and the original
dispatcher entered M3. Contrast/replay yield 6/6 adaptation-required tasks versus
target-only 3/6 and uniform 0/6 at identical 82,248 recorded steps per arm
including bootstrap. Late sustained still solves every acquired task. The
same-repertoire WAIT control is complete: one-time versus sequential limits are
6/6 versus 6/6 with initial information, 5/6 versus 6/6 with reveal at 1.00 s,
and 5/6 versus 5/6 with reveal at 1.40 s or never. These are recorded-table limits.
All three seven-way M2 replay controls tie at 5/6 recorded development contexts;
gated/ungated weights are identical because all recorded nonempty phase cues
are eligible. See [completed work and commands](RESEARCH_PROGRESS_20260909.md).

The independent M1/M2 physical prefix audits completed without discrepancies.
M2 contains 276 acquisition episodes including bootstrap, 328,992 recorded
physics steps, and no unknown outcome in those completed prefixes. The separate
historical reservation remains outside measured counts. M2 cross-corpus
readouts are now complete: uniform, executed contrast and replay tie at
11/11, 11/12, 10/11; target-only gives 10/11, 11/12, 10/11. Late sustained
covers all 17 unique validation scenes. These folds reuse scenes and contain
no new policy execution. Figures, corpus CSV, generated table, source snapshots
and validation receipts are saved under `evidence/research-progress-20260909/`.
The revised working manuscript compiles. All 82 tests across the disjoint
correctness and analysis suites pass; the five new source/test files pass
Black and Ruff. No active physical acquisition source was changed or new
simulation worker launched by this session.

The missing M16/M32 common-set native evaluation continuation is now implemented
and frozen; see [protocol, validation and commands](DEVELOPMENT_LEARNING_CURVE.md).
It preserves the existing M8 panel, adds 90 policy episodes at each later
checkpoint, and reuses its 48 script/fixed captures: 318 unique evaluation
episodes, 414 analysis rows. All M32 acquisition receipts and the completed M8
panel are required before either later evaluation can run. It does not alter
the active staged worker. Every later model binds its exact assigned teacher
prefix, ridge λ=10 and original arm weighting. No future model was substituted
with an earlier policy, and no new simulator was launched.

The continuation passes 61 focused tests (17 new), including no retry on unknown
outcomes, idempotent resume and counting shared captures once. Separate native
M1/M2 preparations verify identical runtime/sensor/scorer contracts with the
current executable assets. Its acquisition-cost reader reproduces the measured
M2 total of 328,992 steps. The final read-only snapshot at 19:37 UTC contains
12 M2, 3 M3 and zero M4 model receipts; all 15 final M32 gates are missing.
These receipt counts are progress indicators, not independent physical audits.
[Validation receipt](evidence/research-progress-20260909/learning_curve_validation.json)
binds source files, declaration and both smoke checks. Later execution and
statistics commands are documented but unexecuted. The next experiment remains
the unchanged 138-episode M8 development panel; the core held-out claim is still
unresolved.

The old reserved runner remains specific to the 972-episode M2/M4 proposal.
A separate [expanded reserved curve](RESERVED_EXPANDED_LEARNING_CURVE.md) is now
implemented, with an immutable proposal and assignment preview under
`m2s-reserved-expanded-curve-proposal-20260909-v1`. It assigns all 45 M8/M16/M32
models, the same strong script and all seven fixed schedules to the original
18 layouts and two execution seeds: 1,908 unique nominal episodes. Every stored
stress offset is preserved and remains disabled in the nominal comparison.
The proposal does not adopt or launch evaluation. The new runner includes
inventory freezing, reviewed-evidence adoption, preparation, serial execution
and a standalone statistics reader. It retains the existing physical validator,
collector, attempt accounting and all failure/unknown rules.

All 106 focused tests pass (21 new); four new files pass Black/Ruff. Full-matrix
synthetic tests exercise unknown attempts, shared references, unsolvable-bank
denominators, exact schedules and 20,000-draw paired corpus/layout statistics.
Actual negative preflights reject freezing/adoption at missing M32 data and
reject reserved preparation before adoption, creating no output directories.
[Validation receipt](evidence/research-progress-20260909/reserved_curve_validation.json)
records all retained geometry/sensor/scoring equalities and source hashes.
At 20:09 UTC, six original M3 model receipts exist; M4 remains incomplete.
The original acquisition process and expanded successor were confirmed live.
No new physical worker was started. The manuscript now states the proposed
318-development/1,908-reserved allocations explicitly as unexecuted. These are
execution/reporting dependencies completed, not new policy-performance results.
The next decisive experiment remains the original 138-episode M8 development
panel. After acquisition and development finish, assemble the nine gate-evidence
references from actual measurements and checks before the implemented adoption
command; the user has already requested that protected research workflow.

The [acquired response witness](ACQUIRED_RESPONSE_WITNESS_20260909.md) now verifies
the three executed-contrast M3 encounters from raw recorded teacher and student
captures: 24 episodes, 28,608 steps and no discrepancy with stored supervision.
The complete M2 pool plus these encounters contains 20 distinct geometry/seed
conditions. Bank coverage is 20/20 and best-fixed coverage 19/20, with one
same-seed pair requiring disjoint sustained versus prior-splice responses.
Its recorded upper-hit features differ at the first 0.30 s decision. This changes
the observed acquired-pool capability statement, not the completed M2 results or
the central performance conclusion. Every individual executed-contrast M3
corpus still admits one fixed prior covering all three tasks. The mixed-prefix
pool is an interim mechanism analysis, not a balanced M3 arm comparison.

The witness is source-bound at `m2s-acquired-response-witness-20260909-v1` and
mirrored under `evidence/research-progress-20260909/acquired-witness/`. The
recorded-state figure uses forward kinematics only and exports all first-phase
features and seven-schedule outcomes. The analysis runs no new physics, fits no
policy and changes no acquisition queue. Eleven focused tests and Black/Ruff
pass. The manuscript builds with the measured finding and a qualified supplement
figure. At 20:33 UTC, twelve M2, eight M3 and zero M4 model receipts exist.
The dispatcher and its `seed93203_observation_curriculum` M3 child were confirmed
live at 20:36 UTC; the expanded successor remains active. Receipt counts alone
are not independent physical audits. The next decisive experiment remains the
unchanged 138-episode M8 development panel; no reserved execution is reported.

The [portable M2 acquisition release](PORTABLE_ACQUISITION_M2_V1.md) is complete.
All twelve native export audits pass: 276 recorded episodes, 328,992 measured
steps, 82,248 sensor rows and 108 complete teacher phase tables. The 695 MiB
archive retains all 52 teacher and nine student failures. Both the original
export and a freshly extracted, relocated copy reconstruct all 36 frozen
M0/M1/M2 fits under a guard that blocks original data reads, simulator imports
and release writes. All 324 recorded teacher-packet action checks agree; the
maximum coefficient difference is 5.5512e-17. These are repeated recorded-input
checks, not newly executed policies or additional physical evidence.

The reader reproduces all M1/M2 task-yield and measured-cost rows. It preserves
student-generating-policy identities, exact chronological teacher prefixes and
the originally audited replay weights; it does not regenerate the full historical
physical-gap audit. The original unknown startup and failed initial fit are
packaged explicitly. The seven retained bootstrap trajectories belong to
`seed93203_analytic_contrast`, keep their source hashes and are counted once.
The zero measured startup steps and separate 1,192-step reservation stay distinct.

The archive's 17,233 member paths and bytes were verified before and after actual
GNU tar extraction. No raw pretrained weights or full motion bank is shipped.
The portable suite passes 22 tests, including the new measured-cost and unknown
bounds checks; Black/Ruff pass. Full receipts, corpus CSVs and source snapshots
are under `evidence/research-progress-20260909/portable-acquisition/completed/`.
The former partial export/smoke receipts remain as history. The exporter and all
reconstruction/package jobs completed; do not resume their old tool sessions.

At 21:46 UTC all twelve M3 model receipts exist and M4 has begun (one complete
M4 receipt at that snapshot). The original dispatcher and expanded M8 successor
remain live. Receipt counts are progress indicators, not independent physical
audits. No acquisition source, queue, learner, teacher, baseline or reserved
geometry was changed by this portability work. The core held-out claim remains
unresolved. The next decisive experiment is the unchanged 138-episode M8
common-set native development panel after its required acquisition prefixes.

The updated 20-page manuscript compiles with the completed portability result in
the supplement. Its existing underfull bibliography box is unchanged. The main
performance claim and abstract remain limited to completed development evidence.

At 23:07 UTC, seven M4 model receipts and dispatcher boundaries are complete:
`seed93203_uniform`, `seed93202_analytic_contrast`, `seed93202_target_only`,
`seed93201_analytic_contrast`, `seed93201_uniform`,
`seed93203_observation_curriculum`, and `seed93201_observation_curriculum`.
The original dispatcher and expanded M8 successor remain live. The last replay
corpus completed its student visit, seven teacher branches, historical replay
calculation and M4 fit, retaining eight encounter episodes and 9,536 measured
steps. [Completion and live-process evidence](evidence/research-progress-20260909/acquisition_wait_20260909T2307Z.json)
binds these receipts and dispatcher boundary 054. No restart or source change
was made. The dispatcher is now running `seed93202_uniform`, confirmed as
PID 2212922. This is verified acquisition progress, not a new common-set policy
comparison or raw-capture audit. The earlier
[handoff check](evidence/research-progress-20260909/acquisition_handoff_20260909T2152Z.json)
verified all 180 pinned references and the unchanged 138-assignment M8 manifest.

The existing full M4 response-diversity analysis now has a live CPU watcher at
`m2s-response-diversity-M4-watch-20260909-v1/watch.py`, PID 2174226, tool session
91946. Poll this exact process/session; do not launch a duplicate. It requires
all twelve original dispatcher completion markers, verifies 174 pinned source
references, and then invokes the unchanged `motion2scene_response_diversity.py`
at declared budget 4. It currently waits for five remaining M4 boundaries;
**the analysis itself has not run**. Its output will be
`m2s-response-diversity-M4-20260909-v1/result.json`. Existing output, source drift,
a stopped incomplete dispatcher or a failed audit causes a retained stop, not
an automatic replacement/retry. [Watcher code, validation and execution receipt](evidence/research-progress-20260909/M4-audit-watch/validation.json)
are saved. This attaches an existing recorded-prefix audit to acquisition
completion and changes no method, teacher, candidate ordering or stopping budget.

The attempted complete M3 response audit was correctly refused because M3 is not
a declared comparison checkpoint. No output directory/result was created and
the gate remains unchanged. Do not repeat that invocation. The next decisive
policy experiment remains the registered 138-episode M8 panel after acquisition.

The five-arm portable export/reconstruction implementation is now complete,
subject to actual later-checkpoint data and the documented full-recording schema
gate. [Source-bound validation](evidence/research-progress-20260909/portable-curve/validation.json)
records 61 passing tests (22 new), Black/Ruff, all 36 M2 reconstructions and
324/324 recorded action checks, plus three native/NumPy expanded-format fits
and 27/27 recorded action checks on an existing M2 corpus. Every M1/M2 physical
cost/yield row equals the original completed release. This work performs 78 CPU
fits across retained validation versions and zero new physics. Two serialization
defects were repaired in the new reader; earlier failed pre-fit checks remain
saved. The immutable M2 archive and active acquisition source files are unchanged.

The new exporter preserves all historical teacher/student/model prefixes,
checks paired queues, retains original startup receipts, and independently
re-audits later replay sidecars against their historical generating-policy
outcomes. It does not retroactively claim those sidecars were hash-bound by the
original fit result. Unsupported partial captures or unavailable neutral
histories are rejected before export; no assigned task is dropped. The actual
M2 audit is correctly rejected as a substitute for a five-arm later checkpoint.
Later export/relocation remains unexecuted. Commands and limits are in
[PORTABLE_ACQUISITION_CURVE.md](PORTABLE_ACQUISITION_CURVE.md). The revised
20-page manuscript compiles with this artifact scope in the supplement and the
existing underfull bibliography warning.
