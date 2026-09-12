# Learning complementary traversal decisions

The next method should preserve complementary capability while learning when a
faster response is appropriate. Recorded-data experiments and a completed
six-context native learner comparison now test that question using the existing
42-branch development corpus. The physical comparison retains ridge as the
common learner for the running acquisition experiment.

## What the experiments establish

The cost regression is present on the fitted teaching observations. The existing
six-context ridge chooses sustained adaptation on the short and long passages,
whose recorded branch times are 4.06 and 4.64 s. Consequently, observation shift
in a new execution is not needed to explain the existence of the regression.
It may still explain differences between recorded-branch and actual-policy costs.

A fixed depth-two learner predicts passage feasibility and successful passage
time separately. Each legal action has its own classifier and time regressor;
only successful measurements train time. Choose minimum predicted time among
actions with estimated feasibility at least 0.5; if none qualifies, maximize
estimated feasibility. Depth, threshold and seed were fixed for this comparison.
This changes the objective and representation together, so it is a practical
alternative, not an isolated test of nonlinearity.

| Learner | Training branch proxies | Training phase regret | Whole-context holdout proxies | Holdout phase regret |
| --- | ---: | ---: | ---: | ---: |
| Existing regret ridge, penalty 10 | 6/6 | 0.01916 | 4/6 | 0.19563 |
| Separate feasibility/time trees, depth 2 | 6/6 | 0.00713 | 4/6 | 0.24970 |

Regret averages the 17 consequential phase tables, including hypothetical neutral
phases after an earlier commitment. Branch proxies instead follow each model's
neutral decisions until its first commitment and read the corresponding complete
measured schedule. Neither quantity is a newly executed policy success rate.
All phases of the held-out context stay out of fitting. These contexts were
previously used for development; the six folds are not six independent corpora.

The tree recovers the 4.48 s prior schedule on the long training passage and
walking on the empty context, but worsens the two-beam choice from 4.12 to 4.20 s.
Both learners fail the early and complementary holdouts: each useful capability
has only one teaching context. We retain the original online learner for the
running comparison. The tree is a candidate for broader development comparison,
not a selected replacement based on training fit.

## Closed-loop learner comparison

The fixed tree was exported to a NumPy-only deployment model, reproducing all
18 training and all 18 whole-context-holdout decisions exactly. It then received
six new physical executions at seed 8732, paired with the existing six-context
ridge executions. Scenes, reference bank, sensor interface, physics seed and
physical success criteria are unchanged. The first neutral decision vectors
match the ridge recordings exactly in all six contexts.

| Development context | Ridge passage time (s) | Outcome-tree passage time (s) |
| --- | ---: | ---: |
| Empty | 5.38 | 5.90 |
| Short | 4.10 | 4.06 |
| Long | 4.68 | fail |
| Early | 2.72 | 2.72 |
| Two beams | 4.12 | 4.12 |
| Complementary | 4.54 | fail |
| Passage count | 6/6 | 4/6 |

All six tree measurements are complete, totaling 1,788 control rows and 7,152
physics steps. The long and complementary failures have measured environmental
contact; neither is a missing-data failure. On the four mutually successful
contexts, the tree is 0.12 s slower on average. This comparison uses one matched
physics seed and development contexts, not independent held-out corpora.

The failure is not an export mismatch. For example, at the complementary
context's first decision, the distant corridor unknown fraction moves from
0.875 in the teacher execution to 0.873333 in the policy execution. This crosses
the short-action feasibility tree's 0.874167 threshold; that head now predicts
success and its shorter estimated time causes immediate short commitment.
The resulting execution fails. In the empty context, a later unknown-fraction
threshold similarly changes the neutral feasibility prediction. These are exact
decision-rule explanations on recorded inputs, not counterfactual physical tests.

The candidate does not preserve useful capability despite its improved fitted
regret. We retain ridge for the controlled acquisition arms and prioritize
broader teaching data over further tuning on these six examples.

[Complete paired result](/home/linjiw/research-data/groot-wbc/m2s-outcome-tree-native-development-20260909-v1/comparison.json)
links every assigned outcome and the original paired ridge results.

## Observation-consistent teaching

The existing finite-tree prototype is now connected to the real teacher corpus.
It groups exactly equal current 114-dimensional student vectors, without scene
IDs or privileged physical-state hashes. Signed zero is canonicalized; no
measurement tolerance is introduced. Every phase has six distinct groups, so
the exact test finds zero information conflicts and no nontrivial cross-context
equivalence classes. This does not demonstrate task distinguishability under
sensor noise.

Controlled tests establish the intended mechanism: WAIT is rejected when two
identical later histories require incompatible actions, retained when a future
observation distinguishes the encounters, and assigned a set of acceptable
actions when multiple common responses pass. Incomplete outcomes remain unknown.
The solver requires refining information partitions. A finite-memory sensor
summary may forget earlier distinctions; the implementation rejects that case
rather than silently assuming a recurrent student.

## Replay at fixed physical-data cost

The implemented development controls are uniform, coverage-only, 0.8 uniform plus
0.2 coverage, historical gated priority, separately measured ungated priority,
current supervised-error priority, and historical priority multiplied by current
error. Current error is mean squared regret residual over legal measured actions.
It is recomputed on training inputs after each fit. It is not a current physical
failure probability. Historical controls require actual generating-policy gaps.

The completed experiment compares the four controls that need only the stored
teacher corpus. Each starts from a uniform ridge fit and receives three refits;
holdout data never enter priorities, normalization or training.

| Replay control | Whole-context holdout proxies | Mean phase regret |
| --- | ---: | ---: |
| Uniform encounters | 4/6 | 0.18952 |
| Coverage only | 4/6 | 0.18952 |
| 0.8 uniform + 0.2 coverage | 4/6 | 0.18952 |
| Refreshed supervised error | 4/6 | 0.19570 |

All controls share the same encounter-before-phase normalization. This differs
from the original development ridge's equal-context weighting separately within
each phase, explaining why the two uniform phase-regret values differ. Coverage
and current-error weighting show no passage benefit here. These results do not
compare the historical physical-gap algorithm: its own measured pre-update
student executions must finish before that comparison.

## Experiments that can resolve the remaining research question

The central hypothesis is that executed alternatives identify teachable
constraints more efficiently, allowing perceptive selection to generalize at
matched acquisition cost. Replay is a separate component hypothesis.

1. Finish the active M2/M4 runs. Extend development acquisition to 8, 16 and 32
   encounters, retaining independent corpus seeds. Keep every physically failed
   encounter in the acquisition cost and outcome accounting. Compare reference
   versus executed envelopes with the same bank, sampling domain, screen,
   physical labels and common learner. Compare feasibility-screened uniform,
   target-only and executed contrast separately from replay.
2. Fit each replay arm on controlled corpora using uniform, coverage-only,
   uniform/coverage, gated and ungated measured gaps, and refreshed error. An
   error refreshed on recorded inputs is inexpensive; an updated physical gap
   requires an actual rollout. Scores from another policy are not on-policy
   measurements. Select the common learner using development traversal outcomes
   across corpora, preserving passage first and mutually successful paired time.
3. Test action-relevant observation gating with controlled pairs whose known
   difference changes the acceptable continuation set. Remove an irrelevant
   distant constraint in a matched intervention before crediting that constraint
   as irrelevant; a lack of visibility alone is insufficient. Start with exact
   collisions, then declare noise quantization or tolerances before measurement.
4. Freeze the method, then evaluate the reserved tasks with strong scripts and
   constant schedules. Add all seven forced schedules on the same 18 layouts and
   two physics seeds: 252 nominal capability episodes in addition to the existing
   972 policy assignments. Keep capability failures in the denominator. Compare
   time only on mutually successful matched conditions and report their counts.
5. Preselect the final-budget checkpoint for placement stress and sensor
   sensitivity. Run the eight fixed stress offsets, missing-return rates
   0/10/30%, range noise standard deviations 0/1/3 cm, and delays 0/40/100 ms
   as separate development-chosen perturbation axes. Corrupt measurements before
   building the history, retain missing-space masks, and use paired perturbation
   seeds. The [sensor intervention runtime](SENSOR_SENSITIVITY_STUDY.md) and a
   four-condition short-passage physical check are complete; the multi-context,
   multi-seed evaluation remains to be executed. Neither implies camera transfer.

The nominal acquisition maxima for seven bootstrap branches and eight episodes
per encounter are:

| Encounters after bootstrap | Assigned episodes per corpus | Maximum physics steps |
| --- | ---: | ---: |
| 2 | 23 | 27,416 |
| 4 | 39 | 46,488 |
| 8 | 71 | 84,632 |
| 16 | 135 | 160,920 |
| 32 | 263 | 313,496 |

These are cumulative budgets, not independent datasets at each checkpoint. The
original controller supports four encounters and a 50,000-step ceiling. A
separate [expanded runner and fixed plan](EXPANDED_DEVELOPMENT_STUDY.md) now wait
for those M4 corpora, without changing their bound inputs. Larger acquisition
and fixed-task capability episodes have not yet been executed.

The [matched prior-extension study](OPTION_EXTENSION_STUDY.md) has now generated,
repaired and physically qualified four candidates from each procedure, with
identical observed qualification yield. Its independently fixed 12-task coverage
panel is prepared. This tests a repeatable capability procedure beyond removing
the one original successful prior derivative.

## Sources and runnable results

[Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html) motivates
the learning-potential comparison; our supervised residual control is not a
reproduction of its RL method. The official
[Kimodo project](https://research.nvidia.com/labs/sil/projects/kimodo/) already
demonstrates Kimodo-to-GEAR-SONIC tracking. Our proposed contribution is useful
execution-conditioned teaching, not connecting these tools.

Run from the repository root with NumPy and scikit-learn:

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python scripts/research/motion2scene_decision_study.py \
  --source /home/linjiw/research-data/groot-wbc/m2s-six-context-development-policy-v1 \
  --out /absolute/path/to/new-decision-study
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python scripts/research/motion2scene_replay_comparison.py \
  --source /home/linjiw/research-data/groot-wbc/m2s-six-context-development-policy-v1 \
  --out /absolute/path/to/new-replay-study
```

Completed [decision results](/home/linjiw/research-data/groot-wbc/m2s-decision-learning-study-20260909-v3/result.json)
and [replay results](/home/linjiw/research-data/groot-wbc/m2s-replay-learning-study-20260909-v1/result.json)
contain every fold, selected schedule and recorded cost; replay additionally
retains all three weight updates. The first decision-study invocation failed
while assembling output after fitting and produced no result; v2 completed, and
v3 reproduces its numbers after formatting and adding development-split checking.
No physical acquisition budget was consumed by these CPU studies.

Validation: 40 focused tests pass across decision learning, replay controls,
observation teaching, and the existing schedule learner/curriculum. Ruff and
Black pass on all five new Python files. Tectonic builds the 14-page working
manuscript with no undefined references or overflowing boxes; a bibliography
line produces one harmless underfull-box warning.

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python -m pytest -q \
  decoupled_wbc/tests/test_motion2scene_decision_study.py \
  decoupled_wbc/tests/test_motion2scene_replay_controls.py \
  decoupled_wbc/tests/test_motion2scene_observation_teacher.py \
  decoupled_wbc/tests/test_motion2scene_timed_schedule_learning.py \
  decoupled_wbc/tests/test_motion2scene_timed_schedule_curriculum.py
tectonic docs/motion2scene/submission/traversal_method_v2.tex --keep-logs
```
