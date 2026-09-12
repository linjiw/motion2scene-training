# Completed acquisition and mechanism work, September 9

**Latest completed checkpoint, September 10 UTC:** all twelve original M4
corpora, the independent raw-record audit, the primary cross-corpus readout and
the M4 seven-way replay controls are complete. The
[M4 checkpoint report](M4_ACQUISITION_CHECKPOINT.md) records 468 acquisition
episodes and 557,856 measured physics steps. At 139,464 steps per arm, uniform
acquires 2/12 adaptation-required tasks, target-only 4/12, and executed contrast
and replay 12/12 each. Every corpus still admits one fixed schedule covering
its full prefix. The pooled bank covers 34/34 unique geometry/seed conditions,
versus 32/34 for the best fixed schedule. Uniform, executed contrast and replay
tie in all three recorded-input cross-corpus folds; no new native policy
evaluation is implied. The updated manuscript and M1/M2/M4 figure compile.
All three reference bootstrap prefixes have finished under the existing M8
worker; the five-arm M8 acquisition is ongoing. The original dispatcher and
M4 audit finished normally and must not be restarted.

The following dated M2 and mechanism records are retained as history.

The original acquisition reached **M2 in all twelve corpora at 18:56 UTC** and
continued into M4 under its existing dispatcher. The central claim is still
unresolved: M2 improves one construction-yield measure, but its tasks remain
solvable by a single fixed schedule. M8 and reserved policy comparisons are not
complete.

| M2 constructor, three corpora | Solvable acquired tasks | Adaptation-required tasks | Best fixed schedule coverage | Recorded steps including bootstrap |
| --- | ---: | ---: | ---: | ---: |
| Screened uniform | 6/6 | 0/6 | 6/6 | 82,248 |
| Target-only | 6/6 | 3/6 | 6/6 | 82,248 |
| Executed contrast | 6/6 | 6/6 | 6/6 | 82,248 |
| Same executed scenes with replay | 6/6 | 6/6 | 6/6 | 82,248 |

The [M2 source receipt](evidence/research-progress-20260909/M2_yield.json)
contains all 192 encounter executions and 228,864 measured physics steps across
the four arms. Bootstrap adds 100,128 steps, making 328,992 recorded acquisition
steps across the twelve completed prefixes. Each corpus has 23 episodes and
27,416 steps. The separate historical 1,192-step unresolved reservation is not
recorded physics and is excluded from these capture totals. The three corpora
are the replication units; the six task assignments per arm are not six
independent held-out layouts. Contrast/replay scene identities and branch order
match through M32, checked against the frozen expanded plan.

Adaptation-required means measured neutral failure and at least one passing
adaptation. It does not imply the policy needs to distinguish different
adaptations. Late sustained passes every M2 acquired task across all four arms.
The new response-diversity analyzer reports all passing-set patterns and exact
minimum covers, keeps unsolvable tasks in the denominator, and withholds exact
cover claims for unknown branches. Bootstrap, student and teacher costs are
separate. This is a training-data measurement, not a common-set policy test.

Proposal costs remain separate: original shared pool preparation measured
480 candidates, 114,848 clearance queries and 55.0446 s of clearance search,
including rejected and unselected proposals. These shared computations are not
independent per-arm physical acquisitions. The expanded/reference pool receipts
retain their own query and time costs; neither geometric rejection nor the
169/645 reference-pair retention statistic is a physical collision rate.

The reference comparator has valid independent bootstrap/M1–M4 assignments for
each seed in the expanded plan. Those executions await the original M4 boundary.
Its pending prefix is not replaced by executed-constructor labels or a development
model.

The completed M2 cross-corpus readout reinforces this limitation. All arms for
one training seed use exactly the same other-seed validation scenes:

| Training corpus seed | Uniform | Target-only | Executed contrast | Replay |
| --- | ---: | ---: | ---: | ---: |
| 93201 | 11/11 | 10/11 | 11/11 | 11/11 |
| 93202 | 11/12 | 11/12 | 11/12 | 11/12 |
| 93203 | 10/11 | 10/11 | 10/11 | 10/11 |

These are recorded complete-branch selections, not new native policy episodes.
The 34 validation assignments per arm reuse 17 unique scenes. A fixed late
sustained schedule covers all 17. Executed contrast ties uniform at every seed;
the one-seed advantage over uniform observed at M1 is absent in this larger
readout. The validation pool changes between budgets, so this is not a
fixed-test-set learning curve. The
[source receipt](evidence/research-progress-20260909/M2_cross_corpus.json) retains
each assignment, exclusion, selected branch, paired time and model identity.

## WAIT with the same complete schedules

The new [finite control result](evidence/research-progress-20260909/WAIT_control.json)
compares sequential decisions with a one-time choice at 0.30 s among **all seven
complete schedules**, including late entries. An initially selected late schedule
continues neutral until its entry. Both controls use the same full physical
branch outcomes and the same declared corridor-feature reveal intervention.

| Corridor information available | One-time complete-schedule limit | Sequential WAIT limit |
| --- | ---: | ---: |
| 0.30 s | 6/6 | 6/6 |
| 1.00 s | 5/6 | 6/6 |
| 1.40 s | 5/6 | 5/6 |
| Never | 5/6 | 5/6 |

The suspected mechanism is information arriving after the first decision but
before the last useful commitment. The smallest change is whether subsequent
observations can change the selected full schedule. The matched control retains
the complete repertoire. The 1.00 s result supports this finite information
mechanism; the nominal tie prevents a claim that WAIT already improves ordinary
closed-loop passage. Exact nominal groups are singletons. No observation
tolerance was selected, no new model was fitted, and no new physics was run.
A physical policy benefit requires the eventual matched native control.

Generated directly from checked results: [acquisition response curves](evidence/research-progress-20260909/M1_M2/acquisition_responses.pdf),
[same-repertoire WAIT figure](evidence/research-progress-20260909/M1_M2/wait_same_repertoire.pdf),
[individual corpus data](evidence/research-progress-20260909/M1_M2/acquisition_by_corpus.csv),
and [generated LaTeX table](evidence/research-progress-20260909/M1_M2/acquisition_yield.tex).
The [generation receipt](evidence/research-progress-20260909/M1_M2/result.json)
binds inputs, outputs and rendering code. M1 and M2 curves reuse acquisition
prefixes; their costs must not be summed as separate experiments.

## Replay at M2

The existing checkpoint worker completed all three seven-way control comparisons.
Uniform, coverage-only, actual uniform/coverage fallback, historical gated gap,
historical ungated gap, refreshed supervised error, and historical-plus-error
priority each give 5/6 recorded development branch passages in every corpus.
The [source-bound summary](evidence/research-progress-20260909/M2_replay_summary.json)
preserves individual corpus results.

There are 21 final control models, produced by 66 full policy fits: one shared
initial fit plus seven controls with three refits each, per corpus. The recorded
control receipts do not measure fitting wall time, so none is imputed. These
additional CPU operations consume zero new physical branches.

Replay prioritization is active: each corpus has three positive historical phase
gaps. The visibility exclusion is inactive here: all six nonempty phase cue
records per corpus are eligible, and gated/ungated weights are identical at
every refit. Thus this checkpoint supports neither a visibility-gating benefit
nor a replay passage benefit. Historical gaps still belong to their original
pre-update models; supervised residuals are not refreshed physical gaps. Retain
the frozen comparison arm through its declared study, and retain replay as a
final headline component only if subsequent controlled evidence earns it.

## Implementation and reproduction

The method, common ridge learner, original scene-wise teacher, tracker, sensor
interface, scorer and live acquisition sources remain unchanged by this work.
The 54-test bounded correctness suite passed. The 28-test analysis/control suite
also passed; it covers the two new analyses and existing acquisition, replay and
information helpers. Black and Ruff pass on the new source and tests.

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_response_diversity.py \
  --plan /home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-plan-tie-proposed-v5/plan.json \
  --budget 2 --out /tmp/m2s-response-diversity-M2-reproduction

OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_wait_information_control.py \
  --source /home/linjiw/research-data/groot-wbc/m2s-six-context-development-policy-v1 \
  --out /tmp/m2s-wait-control-reproduction

OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python -m pytest -q \
  decoupled_wbc/tests/test_motion2scene_response_diversity.py \
  decoupled_wbc/tests/test_motion2scene_wait_information_control.py \
  decoupled_wbc/tests/test_motion2scene_acquisition_yield.py \
  decoupled_wbc/tests/test_motion2scene_checkpoint_controls.py \
  decoupled_wbc/tests/test_motion2scene_information_ablation.py \
  decoupled_wbc/tests/test_motion2scene_observation_teacher.py
```

Use a fresh output directory; the scripts preserve previous results. The response
analyzer independently re-audits complete training/teacher prefixes and can take
several CPU minutes. It also accepts the expanded plan at M8/M16/M32 once every
assigned corpus exists. Missing models cause an explicit error, not substitution.

The manuscript now includes the completed M2 and same-repertoire WAIT findings,
keeps replay nulls explicit, and moves detailed tree debugging to the supplement.
The [primary-source check](NOVELTY_SOURCE_CHECK_20260909.md) corrects the LfH
comparison. The [English and Chinese pitches](PITCH.md) describe the implemented
idea without promising unmeasured gains. The working abstract remains bounded
to completed development evidence; there is no final held-out-results abstract.

The single next decisive experiment is the already registered **138-episode M8
common-set native panel**, after its required acquisition prefixes finish. It
includes all 15 acquired policies, the unchanged strong script and all seven
fixed schedules. Inspect its task-response diversity before considering a
separate complementarity-aware acquisition intervention. M16/M32 and protected
reserved evaluation remain unfinished, so the project is not declared complete.

## Fixed continuation after M8

The [M16/M32 development evaluation continuation](DEVELOPMENT_LEARNING_CURVE.md)
is now implemented and declared before its models or evaluation outcomes exist.
It adds 180 policy episodes to the existing 138-episode M8 panel and shares the
original 48 script/fixed measurements, giving 318 unique episodes across three
checkpoints. Both later evaluations require completed M32 acquisition, so they
cannot inform its stopping or queues. The original M8 and live acquisition
workers remain unchanged. Sixty-one focused tests pass, and separate native
preparation and measured-cost checks pass without new physics. The commands
and [source-bound receipts](evidence/research-progress-20260909/learning_curve_validation.json)
are saved. No M16/M32 policy performance or held-out conclusion is added.

The [reserved expanded comparison](RESERVED_EXPANDED_LEARNING_CURVE.md) now has a
separate runner, standalone statistics reader and immutable assignment preview.
It replaces the old runner's hardcoded M2/M4 scope for the planned final study,
while preserving the old proposal and all reserved geometry/scoring. The new
allocation contains 1,908 nominal episodes: 45 acquired checkpoints and eight
shared fixed/script baselines on 18 layouts and two execution seeds. It is not
adopted or executed. All 106 focused tests pass, and real incomplete-data
preflights stop before reserved preparation. The manuscript records the new
allocations as pending work. No later-policy or held-out result is inferred.

## Acquired response complementarity

The [interim acquired-pool analysis](ACQUIRED_RESPONSE_WITNESS_20260909.md) adds
the three executed-contrast M3 encounters to the completed M2 pool. An independent
re-audit verifies their 21 complete teacher branches and three generating-M2
student executions: 28,608 recorded steps, with no new physics or training.
The bank covers all 20 unique geometry/seed conditions and the best fixed
schedule covers 19. One target-only/contrast pair has disjoint sustained/prior
passing sets and different recorded perception features at 0.30 s. Each
individual executed-contrast M3 corpus still admits a single fixed prior
covering 3/3 tasks. This is a post-hoc mechanism witness, not a balanced M3
performance comparison or robust information-separation result.

The measured [figure](evidence/research-progress-20260909/acquired-witness/acquired_response_witness.pdf)
uses recorded IsaacLab states and existing visual meshes with forward kinematics
only; all seven schedule outcomes and all 114 first-phase features are exported.
Small results, source snapshots, raw-source hashes and validation receipts are
preserved. Eleven focused tests and Black/Ruff pass. The manuscript compiles
with the finding and explicitly retains the unresolved M8 and held-out claims.

## Portable acquisition reconstruction

The [M2 portable release](PORTABLE_ACQUISITION_M2_V1.md) is complete: twelve
separate corpora, 252 full teacher branches and 24 actual pre-update student
visits, totaling 276 episodes and 328,992 measured steps. All 61 recorded failures
remain present. Native export audits verify 108 teacher phase tables and 82,248
sensor rows. The guarded NumPy reader reconstructs all 36 frozen M0/M1/M2 fits
and matches all 324 recorded teacher-packet choices, with maximum coefficient
difference 5.5512e-17. Both the original release and a freshly extracted,
relocated archive pass with original data reads and simulator imports blocked.
This is offline reproducibility evidence, with no new physical execution.

The public reader reproduces each corpus's M1/M2 physical task-yield and measured
cost rows, including bootstrap, complete teachers and actual student visits.
M2 costs 82,248 steps per arm and gives adaptation-required yields of 0/6, 3/6,
6/6 and 6/6 for uniform, target-only, executed contrast and replay. The original
teacher prefixes, ridge λ=10, measured-tie initialization and historical replay
weights are retained; no refreshed fitting residual is called a physical gap.

The byte-verified compact archive is 728,923,512 bytes (695 MiB). Actual extraction
verifies every one of its 15,162 file paths. Hardlinks preserve distinct episode
paths and all data bytes while avoiding duplicate storage. The original unknown
startup, conservative reservation and failed initial fit are included explicitly;
all seven reused bootstrap captures remain in `seed93203_analytic_contrast` and
are counted once. Existing license notices travel with the package, and raw
pretrained weights/full reference banks are excluded.

The reader's verified bytecode-write defect was repaired before final packaging;
the final reader keeps the release immutable. Twenty-two focused tests and
Black/Ruff pass. [Completed receipts, CSVs and exact commands](evidence/research-progress-20260909/portable-acquisition/completed/validation.json)
retain source hashes and validation history. All twelve M3 model receipts now
exist, and original M4 acquisition is underway. No M8 or reserved result is
claimed. The single next decisive experiment remains the registered 138-episode
M8 closed-loop panel; portability does not establish the central performance claim.

## Five-arm portable curve implementation

The new [curve exporter and NumPy reader](PORTABLE_ACQUISITION_CURVE.md) support
complete declared five-arm M8/M16/M32 prefixes while retaining the immutable M2
release and every original acquisition/model dependency. Actual later exports
remain data-gated and unexecuted. The reader preserves generating-policy
bindings, exact historical teacher prefixes and original versus expanded replay
formats. Expanded sidecars are checked against independently re-audited physical
history; their export-time hashes are not attributed retroactively to old fits.

The final guarded reader reproduces all 36 M2 fits, 324/324 recorded choices and
all 24 M1/M2 corpus cost/yield rows. Three native expanded-format fits on a
separate recorded-M2 CPU check reconstruct successfully with 27/27 recorded
choices. These are format/reproducibility checks, not online acquisitions or
policy executions. Total additional CPU fits across validation versions are 78;
new physical steps are zero. Byte-size metadata and nested JSON coverage-key
issues were caught by real-data validation, repaired, and retained as failed
pre-fit receipts. The 61-test focused suite (22 new tests), Black and Ruff pass.

[Completed evidence](evidence/research-progress-20260909/portable-curve/validation.json)
includes final source snapshots, guard/toolkit hashes, native/portable fit
receipts and the actual negative checkpoint preflight. The original M2 archive,
method implementation and acquisition queues remain unchanged. The supplement
describes the bounded artifact result; the 20-page manuscript builds with its
existing bibliography warning. At 23:52 UTC, ten M4 boundaries are complete and
the next corpus has six completed teacher captures. The M4 audit watcher still
awaits all twelve original completion markers. The next decisive experiment is
the already registered M8 common-set closed-loop panel.

At 23:54 UTC, `seed93201_target_only` completed M4. Eleven original dispatcher
boundaries now verify against their model/policy hashes; only
`seed93202_observation_curriculum` remains. All 174 watcher references still
match, and the original dispatcher, M8 successor and existing M4 CPU audit
watcher remain live. [Completion/handoff receipt](evidence/research-progress-20260909/acquisition_wait_20260909T2354Z.json)
preserves the snapshot. Full M4 physical re-audit and the M8 policy panel are
still pending; no new policy-performance claim follows from these receipts.

## Original M4 recording completion

At September 10, 00:06 UTC, all original M4 collections have completed receipts.
The [source inventory snapshot](evidence/research-progress-20260909/M4_collection_completion_20260910T0006Z.json)
follows the original plan slots, including the retained bootstrap path, and
checks per-row step sums, distinct trajectory paths, all seven teacher schedules
and each actual student's generating prior policy. It records 84 bootstrap
branches (100,128 steps), 336 encounter teacher branches (400,512 steps) and
48 pre-update student executions (57,216 steps): 468 distinct trajectories and
557,856 recorded physics steps. All 117 teacher and ten student failures remain
present. No completed-prefix receipt is labeled unknown; the separate original
unknown startup and conservative reservation are unchanged.

This snapshot reads stored completed assessments; it does not independently
re-audit raw trajectories. Eleven M4 fits are complete. The final replay worker
PID 2248207 was confirmed CPU-active at 00:06:48 UTC, and the dispatcher,
expanded M8 successor and existing M4 audit watcher remain live. The M4 raw
audit will require the final committed model/boundary. No M8 acquisition or
closed-loop comparison has completed. The next decisive policy experiment
remains the registered 138-episode M8 development panel.

At 00:12 UTC all twelve M4 model/boundary receipts completed, and the original
dispatcher exited normally. The existing CPU audit launched
`motion2scene_response_diversity.py --budget 4` as PID 2259482, under watcher
2174226 / tool session 91946. Expanded acquisition PID 1955047 began the
reference-envelope arm's independent bootstrap; seed 93201's seven-branch
bootstrap and M0 fit are now complete. The
[handoff verification](evidence/research-progress-20260909/M4_to_M8_handoff_20260910T0014Z.json)
checks all twelve final model/policy pairs, all sixty original inherited model
slots, exact teacher/student/scene prefix paths, common ridge and arm weighting,
paired candidate queues through M32 and 173 staged source references. The
original trajectory history and reserved evaluation remain unchanged.

The [M2/M4 replay-control report](REPLAY_CONTROLS_M2_M4.md) adds the completed
three-corpus M4 comparison. All seven controls remain at 5/6 on the same six
recorded development contexts, with one selected prior schedule per corpus.
Differences from uniform are zero for selected schedules, passing branches and
recorded time on the five mutually successful contexts. Three historical phase
gaps remain positive per corpus, but gated/ungated weights are identical and all
twelve nonempty M4 phase cues satisfy the visibility rule. The report retains
all individual corpus/method/checkpoint rows and the nested-prefix limitation.

The new saved-policy reader reproduces 252 recorded-context choices from 42
M2/M4 models and 126 weight recipes using archived signals. It performs no new
fits, raw-trajectory audit or physical execution. Black and Ruff pass, and the
supplement records the M4 null finding. The full M4 acquisition response audit
and primary cross-corpus readout remain active; the M8 closed-loop panel is
still pending its acquisition data.

At 00:30 UTC, the reference-envelope arm has two completed independent
bootstrap/M0 pairs (seeds 93201 and 93202), each retaining seven physical
teacher branches. [Current handoff evidence](evidence/research-progress-20260909/M4_M8_live_20260910T0030Z.json)
binds their model, policy and teacher receipts and the live M4 audit/M8 worker
processes. No M8 corpus is complete. The M4 stored-assessment task-yield readout
reports adaptation-required counts 2/12, 4/12, 12/12 and 12/12 for screened
uniform, target-only, executed contrast and replay. The independent raw-record
response audit and cross-corpus readout remain pending; acquisition response
figures and the manuscript's M4 construction claims must wait for those checks.
