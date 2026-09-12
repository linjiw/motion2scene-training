# Completed original M4 acquisition checkpoint

The independent raw-record audit and the saved-policy cross-corpus readout
completed on September 10, 2026 UTC. Executed contrast acquired more tasks that
require adaptation at the same measured physical cost. Every individual corpus
still admits one fixed schedule covering all its tasks, and the recorded-input
policy readout ties uniform. The core claim about held-out perceptive traversal
therefore remains unresolved.

This is the completed **original four-arm M4 checkpoint**, not the five-arm M8
comparison. The reference-envelope arm is acquiring its own independent prefix.
The original scene-wise complete-continuation teacher, seven complete schedules,
causal observation interface, scoring contract and phase ridge learner with
λ = 10 remain common. `analytic_contrast` names executed-envelope contrast in
the files; `observation_curriculum` is the same candidate sequence with the
implemented observation-aware replay. The information-consistent teacher is a
separate development extension and did not generate these acquisition targets.

## Physical acquisition and useful-task yield

| Constructor | Seed 93201 adaptation-required tasks | Seed 93202 | Seed 93203 | Total | Bank capability | Best fixed within arm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Screened uniform | 1/4 | 0/4 | 1/4 | 2/12 | 12/12 | 12/12 |
| Target-only | 1/4 | 1/4 | 2/4 | 4/12 | 12/12 | 12/12 |
| Executed contrast | 4/4 | 4/4 | 4/4 | 12/12 | 12/12 | 12/12 |
| Same executed scenes with replay | 4/4 | 4/4 | 4/4 | 12/12 | 12/12 | 12/12 |

Adaptation-required means a measured neutral failure and at least one passing
complete adaptation. The contrast versus target-only differences are 75, 75
and 50 percentage points across the three corpora, averaging 66.7 points.
This measures physical task yield on constructor-specific training scenes,
not perceptive policy performance on a common set. Three acquired corpora are
the replication units; the twelve assignments are not twelve independent
held-out layouts. No significance claim is made.

Every corpus consumes **46,488 measured physics steps**, including its original
bootstrap. This is **139,464 per arm and 557,856 across the four arms**.

| Acquisition role | Executed episodes | Measured physics steps |
| --- | ---: | ---: |
| Own bootstrap teachers | 84 | 100,128 |
| Encounter teachers | 336 | 400,512 |
| Pre-update students | 48 | 57,216 |
| Total | 468 | 557,856 |

The complete prefixes retain 303 passing and 117 failed teacher branches and
38 passing and ten failed students. No recorded branch outcome is unknown in
these completed prefixes. All 48 assigned encounters are bank-solvable; none
was removed on that basis. The original startup unknown and its separate
1,192-step reservation remain in the historical accounting, outside measured
capture totals. The reused seed-93203 executed-contrast bootstrap is counted
once, attached to its original corpus. No acquisition history was replaced.

Geometric proposal computation is separate: the original shared pool consumed
480 candidates, 114,848 clearance queries and 55.044625 seconds of measured
clearance search, including rejected and unselected proposals. These shared
search costs are counted once, not charged as four independently executed
searches. The new reference pool retains its separate accounting. Geometric
pair retention is not a physical collision rate.

## Response diversity and recorded policy selection

All twelve individual M4 corpora admit one fixed schedule passing 4/4 tasks.
Pooled within each arm, either prior schedule covers all uniform, executed and
replay tasks, and late sustained covers all target-only tasks. Contrast and
replay have identical candidate identities and ordering and no teacher-outcome
disagreements at any of the three seeds.

Across constructors, deduplication by geometry hash and execution seed gives
34 conditions, with no conflicting outcomes among repeated assignments.
The bank passes 34/34; either fixed prior passes 32/34. The minimum complete
schedule cover is two: one prior plus one sustained schedule. This establishes
complementary capability in the pooled acquired data, while explaining why an
individual training corpus can still encourage one fixed response. It does not
establish that available observations support a learned successful selector.

The saved M4 models were applied to recorded neutral-prefix observations from
the other acquisition seeds. All arms share the same validation set within
each training seed. The selected complete schedule was looked up in the
already executed teacher table; these are **not new closed-loop policy runs**.

| Training corpus | Uniform | Target-only | Executed contrast | Replay | Fixed prior | Bank capability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 93201 | 21/22 | 16/22 | 21/22 | 21/22 | 21/22 | 22/22 |
| 93202 | 21/23 | 21/23 | 21/23 | 21/23 | 21/23 | 23/23 |
| 93203 | 22/23 | 16/23 | 22/23 | 22/23 | 22/23 | 23/23 |

Uniform, executed contrast and replay choose only late prior for seeds 93201
and 93203 and only early prior for 93202. They tie the fixed prior's passage
counts. Their mean paired recorded passage-time differences versus early prior
are +0.01048 s, 0 s and +0.01091 s on 21, 21 and 22 mutually successful
conditions. Target-only gives +0.15250 s, 0 s and +0.27875 s on 16, 21 and 16
mutually successful conditions. These are recorded branch crossing-plus-
stabilization times, not newly measured policy times or full recovery duration.
The folds reuse the 34 conditions, and the pool grows between M1, M2 and M4;
this is not a fixed-test-set learning curve.

The separate [seven-way replay control report](REPLAY_CONTROLS_M2_M4.md) also
finds no recorded-input benefit through M4. Neither result establishes a replay
gain. The frozen M8 comparisons remain unchanged so that this null finding can
be checked with actual matched policy execution.

## Evidence and reproduction

- [Independent raw audit](evidence/research-progress-20260909/M1_M2_M4/M4_raw_audit.json):
  `sha256:e0340577761b4a26ac74204bcabbb8e9344740aa756b09157fe44abbc3bbad50`.
- [Saved-policy cross-corpus result](evidence/research-progress-20260909/M1_M2_M4/M4_cross_corpus.json)
  and [experiment manifest](evidence/research-progress-20260909/M1_M2_M4/M4_cross_corpus_experiment.json).
- [Derived summary with all 34 conditions](evidence/research-progress-20260909/M1_M2_M4/M4_summary.json).
- [M1/M2/M4 response curves](evidence/research-progress-20260909/M1_M2_M4/acquisition_responses.pdf),
  [individual corpus CSV](evidence/research-progress-20260909/M1_M2_M4/acquisition_by_corpus.csv),
  and [generation receipt](evidence/research-progress-20260909/M1_M2_M4/result.json).

The existing auditor completed this exact command from the repository root:

```bash
.venv_isaaclab/bin/python scripts/research/motion2scene_response_diversity.py \
  --plan /home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-plan-tie-proposed-v5/plan.json \
  --budget 4 \
  --out /home/linjiw/research-data/groot-wbc/m2s-response-diversity-M4-20260909-v1
```

The unchanged renderer completed:

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python scripts/research/motion2scene_render_acquisition_evidence.py \
  --acquisition \
  /home/linjiw/research-data/groot-wbc/m2s-response-diversity-M1-20260909-v1/result.json \
  /home/linjiw/research-data/groot-wbc/m2s-response-diversity-M2-20260909-v1/result.json \
  /home/linjiw/research-data/groot-wbc/m2s-response-diversity-M4-20260909-v1/result.json \
  --wait /home/linjiw/research-data/groot-wbc/m2s-wait-information-control-20260909-v1/result.json \
  --out /home/linjiw/research-data/groot-wbc/m2s-M1-M2-M4-acquisition-evidence-20260910-v1
```

Use a fresh output directory for reproduction. The raw audit requires the
recorded acquisition data and native project environment. The existing
cross-corpus completion receipt binds its saved policies and results. The
report assembly verified their model hashes, common validation assignments,
all twelve corpus costs, agreement between the stored-assessment yield and
the independent raw audit, and the 34-condition duplicate outcomes. The
renderer used completed measurements; its figure was visually inspected.
This analysis and rendering added no physics or model fits.

The single next decisive experiment is the prespecified **138-episode M8
common-set closed-loop development panel** after all fifteen M8 corpora finish.
Any complementarity-aware acquisition change must be a separately declared
development intervention after that checkpoint, preserving the frozen queues
and held-out benchmark. M8/M16/M32 and reserved policy results remain incomplete.
