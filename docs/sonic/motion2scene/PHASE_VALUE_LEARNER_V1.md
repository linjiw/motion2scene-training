# Value imitation at qualified decision phases

The new learner uses one value head at each actually qualified decision tick:
0.20, 0.30 and0.40s. It preserves the existing110 sensor/state/controller features,
measured physical-regret objective, original replay weights and fixed ridge
coefficient10⁻⁶. It changes the value function's capacity to distinguish decisions
at different phases. This is a learner intervention on known development data,
not evidence of a new curriculum or held-out traversal improvement.

The previous single linear value model fitted the aggregated table with rank8
for9 complete consequential decisions. Its recorded-data readout selected an
unnecessary d040 adaptation in the empty scene at0.40s. The original18-schedule
table had6 complete decisions and its smaller-regularization linear model selected
cost-optimal immediate actions at all6. The phase model is a fixed response to
that development finding; neither earlier models nor their physical outcomes
have been overwritten.

## Learning and legality contract

[The new module](../../gear_sonic/dataset_generation/hallucination/motion2scene_phase_value_imitation.py)
reuses the existing `physical_regret` and `fit_value_policy` implementations.
For each phase and supported option it minimizes weighted mean squared error of
measured continuation regret plus10⁻⁶ times the squared coefficient norm. The
intercept is unpenalized. Failure receives regret1; verified successful
continuations receive relative excess measured passage time. These values are
regression estimates, not predicted success probabilities.

A target requires a complete legal branch table, a verified passing continuation
and a consequential difference among legal actions. Completeness is checked on
the full table **before** any phase/option projection. Incomplete rows, tied rows
and rows with no verified solution do not change either normalization or fitted
coefficients. Each head uses the mean/std of its complete consequential rows,
with the unchanged0.05 standard-deviation floor. The original replay weights
remain attached to their actual recorded rows and are renormalized within each
option's weighted objective.

The qualification mask is:

| Decision phase | Available entry choices |
| --- | --- |
|0.20s|neutral/wait, d040, d085|
|0.30s|neutral/wait, d040, d055, d070, d085|
|0.40s|neutral/wait, d040, d085|

d055/d070 have no value targets at0.20/0.40s. Illegal or unobserved choices are
never relabeled as failures. The model stores an explicit trained mask in addition
to its qualification mask. A newly legal action with no trained physical value
raises an error. A consequential decision at a phase outside these three ticks
also raises an error. The phase comes from the existing controller clock and is
checked against the existing `phase_s` feature; it is not privileged scene input.
These heads are trained for neutral-state entry decisions only.

The NPZ schema is `motion2scene_history_phase_value_v1`: phase-specific mean/std
arrays have shape3×110, weights3×110×5 and bias3×5. Phase, option and feature names,
qualification/training masks and the fixed regularization are explicit. Entries
outside the trained mask are storage placeholders and cannot be queried as values.

## Registered finite-table results

Both fits were registered before either was run. Each registration binds the
teacher table, original value-learner registration, feature interface, physical
option registry, replay-weight file where applicable, and a copied source
dependency closure. The trainer has separate `register`/`fit` stages and refuses
a repeated fit under one registration.

| Fixed data | Previous linear readout | Phase-head readout | Per-phase augmented rank |
| --- | ---: | ---: | --- |
|18 physical schedules /6 complete decisions|6/6 optimal|6/6 optimal|2/2 at each phase|
|27 physical schedules /9 complete decisions|8/9 optimal|9/9 optimal|3/3 at each phase|

The selected measured continuation regret is zero for both phase models. The
aggregated linear baseline's mean selected regret was0.00234742 on its nine known
decisions. Both phase models select neutral/wait at all three empty-scene ticks.
At the nominal contrast's0.30s decision, the original-data phase model selects
d055 and the aggregated-data phase model selects d070; these have tied minimum
measured passage time in the nominal teacher table. Small ridge residuals can
therefore change the selected identity without changing measured teacher cost.

Artifacts:

```
/home/linjiw/research-data/groot-wbc/m2s-phase-value-old18-development-v1/
/home/linjiw/research-data/groot-wbc/m2s-phase-value-aggregated27-development-v1/
```

The original-data model SHA-256 is
`bf3e7a9f1f5e44a89e17b803d564a0d9ab137a7a6192c47059639b8f930ba5af`;
the aggregated-data model is
`d58b955962bdfff72c920ab44dcc4fa36baf39e34efd214e3bf2ff0f2d3d152f`.
`result.json` records every selected action, legal mask, measured continuation
value/branch, phase target membership and fitting error. There were no new
physical rollout steps in either fit. These six/nine table decisions are not
independent held-out test encounters.

## Minimal runtime integration

[The existing policy helper](../../gear_sonic/dataset_generation/hallucination/motion2scene_multi_option_policy.py)
now dispatches this explicit new schema in `load_multi_policy` and `choose_option`.
Its previous linear schema and readout calculation remain unchanged. Lazy imports
avoid a circular import through the existing value fitter's schema constant.

The runtime already computes qualified legality and reference-jump guards. For
the new schema, a single legal option is held without calling a value head.
Mandatory recovery still requests neutral when the existing return guard permits
it; a refused return continues the active option. Those cases record null policy
logits because they use no learned decision. There is no learned termination,
protective stop, direct adaptation-to-adaptation switch or new entry phase.
The simulator execution class was not edited.

The planned next comparison is actual closed-loop execution of both frozen phase
models on the three development layouts. Earlier choices can change later states,
so finite-table readouts cannot substitute for these rollouts. The old/new
comparison must use the same new learner and sensor/action interface; data and
architecture effects should remain separate. A clean prospective evaluation is
still necessary for any generalization or curriculum claim.

## Reproduction and verification

[Trainer](../../scripts/research/motion2scene_train_phase_value_policy.py):

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv_research/bin/python \
  scripts/research/motion2scene_train_phase_value_policy.py register \
  --out /absolute/path/to/fresh-phase-model \
  --source-run /home/linjiw/research-data/groot-wbc/m2s-value-aggregation-development-v1 \
  --option-registry /home/linjiw/research-data/groot-wbc/m2s-online-option-registry-v1/registry.json

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv_research/bin/python \
  scripts/research/motion2scene_train_phase_value_policy.py fit \
  --out /absolute/path/to/fresh-phase-model
```

Use `m2s-value-imitation-development-v2` as the source run for the original18
schedule comparison. The new trainer intentionally exposes no regularization
search argument; it inherits the exact registered teacher/replay configuration.

Seven new focused tests verify phase-dependent choices, independence of other
phase heads, exclusion of incomplete/illegal targets, original replay weighting,
unsupported phase/action errors, feature/hash binding and guarded runtime hold/
return dispatch. Together with the impacted value, multi-option and schedule
tests,17 tests pass. Scoped Ruff and Black checks pass. The separate artifact audit
checks source/model hashes and recomputes readouts through the actual runtime
helper without another fit or any physics.
