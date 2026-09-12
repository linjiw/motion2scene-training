# First primary bootstrap model under the adopted tie amendment

At **04:34:26 UTC on September 9, 2026**, the adopted dispatcher completed M0 for
`seed93203_analytic_contrast`. The [completed boundary](/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-dispatch-tie-proposed-v5/complete_000.json)
and [training result](/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-execution-tie-v5/seed93203_analytic_contrast/models/model_000/result.json)
bind the actual controller-produced model. It is distinct from the earlier
external CPU validation fit. This note fixes the first-model milestone; it is
not a live count of later acquisition and reports no M2/M4 traversal result.

The measured development comparison remains the substantive performance result:
the updated learner and retuned strong script both pass 6/6 inspected contexts,
versus 5/6 for the original learner and constant prior. Both updated policies
add 0.50 s on each of the short and long passages. The learner matches the updated
script on obstacle passage times. See the [complete 36-attempt comparison](SIX_CONTEXT_POLICY_COMPARISON_V1.md).
Independent curriculum and held-out gains remain unestablished.

## What changed in fitting

The original seven empty-scene bootstrap executions produced five passes and two
finite-horizon hold failures. They supplied consequential teaching decisions at
0.30 and 1.40 s, but exactly equal successful action costs at 1.00 s. The original
consequential-only fitter stopped explicitly; its [failure and physical audit](/home/linjiw/research-data/groot-wbc/m2s-primary-tied-bootstrap-failure-v1/result.json)
remain unchanged.

[Adoption V3](/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-adoption-v3/adoption.json)
enables the same narrowly defined initialization rule for every future fit in
all four arms and all three corpus seeds:

- A phase must have **zero consequential supervised rows**. If any exist, the
  original fit, normalization and replay weighting apply, excluding tied rows.
- The fallback requires a complete source-bound matched table with at least two
  legal actions, all required future continuations admitted, and every legal
  action successful with exactly equal finite nonnegative cost. There is no
  approximate-equality tolerance. Replay weights must be positive and finite.
- Measured zero-regret targets admit zero coefficients and intercepts. Only
  supported legal columns are marked trained; normalization uses the tied rows
  and the existing 0.05 standard-deviation floor. The unchanged legal argmax
  breaks a tie by option order and can select WAIT.
- Missing, all-failed, mismatched, single-action or otherwise unsupported tables
  cannot initialize a head. WAIT continues the qualified neutral motion; it is
  not a protective stop or a feasibility guarantee in a new scene.

The ridge penalty remains 10, selected under the earlier consequential-only
rule, with no bootstrap-driven retuning. The amendment was informed by this
original empty bootstrap and its CPU validation. No primary contrast encounter,
primary student rollout or reserved outcome informed it. This is a prospective
comparison under a disclosed method revision, not a fresh confirmatory test of
an untouched learner. The immutable [amendment proposal](/home/linjiw/research-data/groot-wbc/m2s-measured-tie-method-amendment-proposed-v1/amendment.json)
retains its original proposal status; adoption V3 is the subsequent authority.

The actual M0 result records **two consequential decisions and one measured-tie
initializer**, at phase tick 50. Ticks 15 and 70 use the ordinary regression.
The initialized coefficients and intercepts are exactly zero. All three phases
select legal action 0 on their recorded bootstrap inputs. These are fitting
checks, not additional physical successes or a current-scene student evaluation.

## Original captures are reused once

The new controller reuses the [original seven-branch collection](/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-execution-recovery-v4/seed93203_analytic_contrast/bootstrap/teachers/result.json)
only in `seed93203_analytic_contrast`. A Python-isolated worker imports the
original collector snapshot, performs its exact nonphysical resolver checks,
and recomputes all seven original rows without writing their source artifacts.
The [original-interpreter audit](/home/linjiw/research-data/groot-wbc/m2s-bootstrap-reuse-staged-v1/physical_reaudit_isaac_python.json)
verifies their physical/source identities. Isolation guards restrict project
imports; this trusted local audit is not described as an operating-system sandbox.

The seven captures contribute **8,344 physics steps exactly once**. Re-auditing,
refitting and referencing them add no new physical samples or independent
replicates. No bootstrap is copied between corpora, and no development model
substitutes for a primary model. The earlier pre-simulator scene-resolution
failure retains its original unknown assessment, zero actual physics and a
separate 1,192-step reservation.

The [V8 ledger](/home/linjiw/research-data/groot-wbc/m2s-traversal-acquisition-ledger-v8.json)
adds those seven captures once to the previously audited development inventory:
231 complete study captures with 233,624 steps, plus one partial failure with
756 steps, total **234,380 study steps**. The separate 796-step instrumentation
failure gives **235,176 persisted steps** in this scope. Historical 144-branch
data, separately inferred unpersisted steps, the pre-simulator reservation and
primary acquisitions after this snapshot are outside that total. The
[V8 audit](/home/linjiw/research-data/groot-wbc/m2s-acquisition-ledger-v8-audit-v1/result.json)
records the identities and counting rule.

## Validation and remaining experiment

The [applied source checks](/home/linjiw/research-data/groot-wbc/m2s-measured-tie-source-application-v1/result.json)
pass 179 tests: 106 impacted checks and 73 tie/provenance checks. The
[common learner/runtime freeze V4](/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-common-freeze-v4/result.json)
and [native/environment freeze V6](/home/linjiw/research-data/groot-wbc/m2s-native-runtime-freeze-v6/result.json)
bind source code, installed assets and environment provenance. The native receipt
tracks 60 assets and 222 additional artifacts; renderer MDL dependencies remain
explicitly outside its PhysX-only scope. These checks do not reproduce a complete
operating-system binary image. All [12 actual configuration checks](/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-adoption-v3/configuration_check.json)
pass under adoption V3.

The primary assignment remains four arms × three corpora, with seven bootstrap
branches and four student-plus-seven-teacher encounters per corpus: 468 assigned
episodes, at most 557,856 recorded physics steps. The separate earlier 1,192-step
reservation is not a second bootstrap acquisition. M2/M4 performance and the
matched-cost curriculum comparison remain pending. The current
[V6 reserved proposal](TRAVERSAL_PROTOCOL_AMENDMENT_V6_PROPOSED.md) and
[reporting V3](TRAVERSAL_PROTOCOL_REPORTING_CLARIFICATION_V3_PROPOSED.md)
retain the 972 nominal assignments and previously fixed paired statistics;
reserved execution remains separately unadopted and unexecuted.

The [method overview](evidence/traversal-v2/method_overview.pdf) depicts implemented
observation-curriculum rounds after the bootstrap. Its 81-offset screen is
geometric; current student execution precedes seven teacher branches; replay uses
historical generating-model gaps on fixed geometry. With no positive eligible
gap, the mixture is 0.8 uniform plus 0.2 coverage. The figure makes no performance
claim for that replay mechanism.

## Pinned milestone artifacts

| Artifact | SHA-256 |
| --- | --- |
| Adoption V3 | `f5e69567fc3984b0a6afb0cc896e63d9281a5fba39d629710c1c5780e1ed9860` |
| All 12 configuration checks | `637318a5a926e79dd3ab27df0b111ed4d06b99d54163ccce874b07f66702cb7a` |
| Actual M0 training result | `08312d18351b325e94cce89237295a5355df6ee83282357f73d2927b09bdebce` |
| Actual M0 policy | `925a4e2e890c2a07203ef8658ae9d14bae14a729e19224a46635caf445fc77de` |
| Original-interpreter physical re-audit | `352b34fb291ccbbe2a9eaded1f103a4827b24836301eada3af97525cbeb00b4e` |
| Ledger V8 | `b67dad2dd088d1c9d42027fc705330925634fd6dec160808105bfd12238c25b9` |

The immutable [42-episode teacher release](PORTABLE_SIX_CONTEXT_DATASET_V1.md)
remains the NumPy learning quickstart: 12,516 aligned 114D packets, 18 available
phase tables (17 consequential), and 50,064 physics steps. The separate
[36-attempt policy release](PORTABLE_POLICY_PANEL_V1.md) preserves both original
scorer versions and the 756-step partial known failure; it adds no teacher labels.
Neither archive is rewritten by the primary learner amendment.
