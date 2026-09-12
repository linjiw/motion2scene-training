# Reuse the portable traversal data

[`motion2scene_dataset_baseline.py`](../../scripts/research/motion2scene_dataset_baseline.py)
provides an offline value-imitation baseline using only portable episode arrays
and finite-schedule targets. It does not run physics, regenerate labels, query a
teacher, or read the original absolute artifact paths. The current 19-episode
teacher package supplies six matched decisions with the named 110D/five-option
schema. The policy-only archives support recorded-outcome analysis but do not
invent teacher targets for training.

Inspect an unpacked package from the repository root:

```bash
PYTHONPATH=. .venv_research/bin/python \
  scripts/research/motion2scene_dataset_baseline.py inspect \
  --dataset /home/linjiw/research-data/groot-wbc/m2s-multi-option-development-dataset-v1-aligned
```

To fit a development baseline, choose a new output directory:

```bash
PYTHONPATH=. .venv_research/bin/python \
  scripts/research/motion2scene_dataset_baseline.py fit \
  --dataset /home/linjiw/research-data/groot-wbc/m2s-multi-option-development-dataset-v1-aligned \
  --out /tmp/m2s-portable-value-baseline-example \
  --l2 0.000001
```

An equivalent data-only command has now executed in the separate
[portable reproduction](../../../research-data/groot-wbc/m2s-portable-dataset-baseline-reproduction-v1/reproduction_audit.json).
It recovers all six cost-optimal recorded decisions, with exact normalization/schema
and maximum coefficient difference 2.94e-14 from the original old18 value fit
(row-order numerical variation). It adds zero physics steps or evaluation episodes.
The output contains `registration.json`,
a pickle-free `policy.npz`, and `training_report.json`. Registration binds input
manifest hashes, exact episode/row references, feature names, option identities,
regularization and implementation hashes. The report is a training diagnostic;
it contains zero policy-evaluation episodes. Do not present fitting the six
teacher rows as held-out traversal performance.

The baseline regresses each option's physically measured finite-schedule regret,
using complete consequential legal targets and an unpenalized intercept. Feature
normalization uses supervised training rows only, with a standard-deviation floor
of 0.05. Decision weights are uniform in this reuse example; it does not reproduce
the research curriculum's encounter-level replay mixture. Negative predicted
regret supplies the runtime score, and legality masks remain required. Values
are unconstrained estimates, not success probabilities.

Action zero means **wait at the current phase**, with a recorded future
continuation. That continuation may enter adaptation later. A forced walking
branch is not a replacement label for waiting. Passing targets must retain an
admitted physical continuation, the recorded matching prefix, measured passage
cost, no reset/fall and completed recovery. Student inputs are loaded only from
`student_history.npz`; no scene geometry, source/layout ID, teacher outcome or
policy output is appended.

Every package hash is audited. Each teacher input must point to an eligible sensor
packet, match reference phase and legality, and belong to development/training.
Continuation episodes must share the same source, physics seed, condition and
beam and retain their exact option identities. The CLI rejects evaluation splits,
missing returns, incomplete continuation counts, schema mixing, duplicate packages
and duplicate physical decisions across releases. Multiple teacher packages may
be passed after `--dataset` only when their schemas agree and their physical
teacher decisions are distinct. Separate acquired episodes are retained even
when their deterministic trajectories happen to have identical bytes; duplicate
detection uses recorded artifact identity as well as its digest and row index.
Keep all descendants of a source/layout and its
perturbations in one externally designed split; this small development package
contains no untouched test split.

Available archives and their roles are linked from
[the method dataset section](TRAVERSAL_METHOD_V2.md#dataset-motion2scene-traversal-episodes)
and [current status](TRAVERSAL_V2_STATUS.md). The legacy 214D, binary 100D and
multi-option 110D schemas must remain separate. Apply recorded sensor eligibility
masks rather than silently aligning arrays by length or dropping a fixed tail.
The separate two-beam demonstration preserves both contact streams and offers
course-record loading/analysis, not a navigation benchmark or learned-course gain.


The [incremental teacher/policy package](../../../research-data/groot-wbc/m2s-aggregation-increment-development-dataset-v1-aligned/manifest.json)
adds three exact lower-scene teacher decisions. Passing it together with the
original 19-episode teacher package validates nine complete 110D targets:

```bash
PYTHONPATH=. .venv_research/bin/python \
  scripts/research/motion2scene_dataset_baseline.py inspect \
  --dataset /home/linjiw/research-data/groot-wbc/m2s-multi-option-development-dataset-v1-aligned \
  /home/linjiw/research-data/groot-wbc/m2s-aggregation-increment-development-dataset-v1-aligned
```

This combined read-only inspection has passed. The incremental targets retain
cross-package identities for the actual student visits; a matched teacher-prefix
array supplies the same causal input without duplicating the student's rollout.
A fit on this expanded package is an offline development baseline. The research
phase-specific policy result uses its separately registered fitter and physical
runs; it must not be attributed to this CLI by inference.
