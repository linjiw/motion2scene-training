# Multi-option policy development: completed first comparison

Eighteen actual episodes compare two student fitters and simple baselines on
three reused source-41002 development conditions under physics seed 8731. Every
episode has 198 exactly aligned sensor/physical rows, with 110D causal inputs.
All passing episodes execute the required return when adaptation is entered.
This is a development comparison. The first lower-scene acquisition and aggregated-student evaluation have now
completed; their separate result is recorded below.

| Policy | Pass | Empty time (s) | Nominal time (s) | Lower time (s) |
| --- | ---: | ---: | ---: | ---: |
| Softmax schedule imitation | 3/3 | 2.50 | 2.96 | 2.96 |
| Scripted, d070 preference | 3/3 | 2.44 | 2.94 | 2.94 |
| Constant d070 | 3/3 | 2.82 | 2.94 | 2.94 |
| Constant d085 | 3/3 | 2.84 | 2.96 | 2.96 |
| Always walk | 1/3 | 2.44 | — | — |
| Ridge value imitation, same 18 teacher branches | 3/3 | 2.44 | 2.94 | 2.96 |

Each time is one actual physical-simulation measurement on a passing episode;
failed times are null. Empty has no overhead constraint. Nominal/lower use the
known route-progress-0.55 beam at underside 1.249/1.239 m. Reused layouts,
previous lower-height qualification, and fitter selection after development
errors prevent a held-out or curriculum-superiority interpretation.

The softmax student enters d040 at reference phase 0.40 on empty, and d085 at
0.40 on nominal/lower. It passes but has measured cost errors relative to the
scripted controller. The value student reuses the **same 18 physical teacher
branches** and fits each option's measured finite-schedule regret by ridge
regression, with regularization 1e-6 and an unpenalized intercept. It selects
neutral on empty, d055 at 0.30 on nominal, and d085 at 0.40 on lower. Its lower
passage remains 0.02 s slower than the scripted d070 run. These are changes in
student fitting on fixed data, not evidence that scene generation improved.
Value outputs are unconstrained regression estimates, not success probabilities
or a global success guarantee.

The lower-scene teacher acquisition supplies complete finite-schedule targets
at the value student's own matched neutral visits. Waiting at a decision retains
its actual future adaptation continuation; forced walking is a different schedule.
The before/after aggregation comparison below uses the same value fitter.

The first five policies consume 11,880 physics steps; the value student consumes
2,376 more, for **14,256 physics steps across 18 episodes**. The teacher corpus
acquisition is separate and must also be charged when reporting training cost.
Sensor/entry/return times use reference phase; traversal time uses the physical
recording clock, one 50 Hz tick earlier at the same row.

The separate [portable dataset](../../../research-data/groot-wbc/m2s-multi-policy-development-dataset-v1-aligned/DATASET_CARD.md)
and [21.10 MiB archive](../../../research-data/groot-wbc/m2s-multi-policy-development-dataset-v1-aligned.tar.gz)
retain all 18 episodes (16 pass, two fail), all 3,564 raw packets and exact
alignment masks. There are 191 hashed files. Configured preference is separated
from executed option and switch logs: for example, preferred index 4 corresponds
to actual neutral/index 0 or d055/index 2 in two value-student episodes. Geometry,
labels and outcome/cost metadata are separate from causal student arrays.
The previous 49/19-episode archives remain unchanged.
Manifest SHA-256: `869aeb11c868086e9689a44bc4da2cac16d94437bbecbc94eea755727d5fe7d5`.

Source results: [five-policy run](../../../research-data/groot-wbc/m2s-multi-policy-development-v1/runtime_audit.json)
and [value policy on the old 18 branches](../../../research-data/groot-wbc/m2s-value-policy-old18-development-v1/result.json).
The proposed [runtime amendment](TRAVERSAL_IMPLEMENTATION_AMENDMENT_V2_PROPOSED.md)
separately assesses the finite horizon and source-specific qualification needed
before the original reserved layout recipe can be executed.


## First aggregation: completed negative result

Nine additional lower-scene teacher branches are complete at entry phases
0.20/0.30/0.40, costing 7,128 physics steps. All are admitted. Walking, d040 at
all three entries, and d055 at 0.30 fail; d070 at 0.30 and all three d085 entries
pass. The [completion receipt](../../../research-data/groot-wbc/m2s-value-policy-development-registration-v1/completed.json)
binds these teacher records to the old value student's exact neutral-state
prefixes. d070 takes 2.94 s versus the student's d085-at-0.40 passage in 2.96 s:
the gap is measured cost, with no passage failure to repair.

The [new 27-branch value student](../../../research-data/groot-wbc/m2s-value-policy-new27-development-v1/result.json)
then executes three actual episodes using the same value fitter and ridge
regularization. All three pass with zero recorded beam force, no fall/reset,
legal return and 594 exactly aligned packets, costing 2,376 physics steps.

| Fixed value fitter | Empty | Nominal | Lower | Sum across three conditions |
| --- | ---: | ---: | ---: | ---: |
| Original 18 teacher branches | 2.44 s, neutral | 2.94 s, d055 at 0.30 | 2.96 s, d085 at 0.40 | 8.34 s |
| After nine added teacher branches | 2.50 s, d040 at 0.40 | 2.94 s, d070 at 0.30 | 2.94 s, d070 at 0.30 | 8.38 s |

Aggregation improves lower passage by 0.02 s but regresses empty passage by
0.06 s; nominal time ties. The sum worsens by 0.04 s. This completed comparison
**does not establish an aggregation benefit**. Each condition has one episode;
these are local descriptive timing differences at 50 Hz. Both models and the
failed improvement remain preserved. The separately evaluated phase-specific
architecture below addresses this representation limit; it does not erase this
negative result.

The original 18-policy archive stays immutable. The incremental release below
retains the additional teacher and student records, including failures.
For reusable offline learning from the existing portable teacher package, see
[dataset baseline CLI](DATASET_BASELINE_V1.md).


## Phase-specific value model: measured paired data update

A separate [six-episode comparison](../../../research-data/groot-wbc/m2s-phase-policy-development-registration-v1/completion.json)
uses the same phase-specific value architecture and ridge regularization 1e-6
on the original 18 and aggregated 27 teacher branches. Inputs remain the same
110D sensor/state schema. Each model is frozen before its three actual episodes;
the architecture itself follows inspected development errors.

| Phase-specific model | Empty | Nominal | Lower | Sum |
| --- | ---: | ---: | ---: | ---: |
| Original 18 branches | 2.44 s, neutral | 2.94 s, d055 at 0.30 | 2.96 s, d085 at 0.40 | 8.34 s |
| Aggregated 27 branches | 2.44 s, neutral | 2.94 s, d070 at 0.30 | 2.94 s, d070 at 0.30 | 8.32 s |

Both pass 3/3 with zero recorded beam force, no fall/reset and actual legal
recovery. All 1,188 sensor packets are aligned; these six episodes cost 4,752
physics steps. The data update removes **one 50 Hz control tick (0.02 s)** on
the lower condition while retaining the other two times. It matches the strong
scripted controller. This is a bounded development data-update benefit with
a fixed phase-specific fitter, not a held-out result, replay-only ablation,
matched-budget constructor comparison or general performance estimate.

The [incremental dataset](../../../research-data/groot-wbc/m2s-aggregation-increment-development-dataset-v1-aligned/DATASET_CARD.md)
and [21.12 MiB archive](../../../research-data/groot-wbc/m2s-aggregation-increment-development-dataset-v1-aligned.tar.gz)
contain exactly nine lower teacher branches, three shared-model aggregated-policy
episodes and these six phase-model episodes: **18 episodes, 13 pass/five fail,
14,256 physics steps and 3,564 aligned 110D packets**. Three portable lower targets
bind the actual old student visit through the earlier policy package's immutable
manifest and episode ID; its rollout is not duplicated. All 195 file hashes pass.
Manifest SHA256: `915087f3fc84e273a7d0d6454fa1fc58ea689691911060aabacf7087753bb6b8`.

The [timing receipt](../../../research-data/groot-wbc/m2s-observation-timing-development-v1/result.json)
shows upper occupancy at reference phase 0.02 in both nominal/lower scenes,
but first ceiling-surface evidence at 0.22/0.34. Thus lower underside is not
observed in the ceiling layer before d070 entry at 0.30. The common verified
d070 response can use an early generic obstacle cue; this does not prove exact
height inference or elimination of partial observability. Empty-room upper
occupancy appears at 1.98 from other geometry and carries no semantic beam identity.
