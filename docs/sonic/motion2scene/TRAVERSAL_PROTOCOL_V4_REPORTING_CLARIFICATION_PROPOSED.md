# Proposed V4 reporting clarification

The [separate reporting specification](TRAVERSAL_PROTOCOL_V4_REPORTING_CLARIFICATION_PROPOSED.json)
defines the primary descriptive interval before primary acquisition or reserved
evaluation outcomes and before adoption. Its SHA-256 is
`c980f50e4ef1f79f1b56a5d8a798826f0968daa658b0467ff8e720e45e353c63`.
It provides no execution authority. Earlier development outcomes exist and
remain development evidence.

The original [V4 proposal](TRAVERSAL_PROTOCOL_AMENDMENT_V4_PROPOSED.json) remains
unchanged at SHA-256
`966d45ed93066a73a945a55f33c769459fe4c1736a455d4eec4ba581d40de2fd`.
Its wording specified a bootstrap of the 18 layout means and separate reporting
of variability across the three acquired corpora. This clarification adds
corpus resampling to the **primary descriptive interval**. The original
layout-only calculation remains a named sensitivity analysis. These are
different uncertainty procedures; the change is explicit and precedes the
primary experiments.

All 972 assignments, 18 nominal layouts, two evaluation seeds, policy/model
inventory, runtime and scoring requirements remain unchanged. Stress variants
and higher acquisition budgets remain proposed follow-ups. A future adoption
record must bind this clarification and the reviewed statistics implementation
by their full artifact hashes.

## Exact resampling procedure

Use 20,000 draws and seed `202609081822`. Corpus order is `93201, 93202, 93203`;
layout order is the lexically sorted 18 locked layout IDs. Evaluation seeds
`94301, 94302` stay together within each layout. Generate indices in this order:

```python
rng = np.random.default_rng(202609081822)
corpus_indices = rng.integers(3, size=(20000, 3))
layout_indices = rng.integers(18, size=(20000, 18))
```

Convert the index multiplicities to corpus weights summing to one and layout
weights summing to one for each draw. Their product gives the cell weight.
Reuse the **same two weight matrices for every comparison**, both checkpoints
and both completion and successful-time calculations. Use the 2.5th and 97.5th
percentiles with NumPy's `method="linear"`.

For completion, average the two paired evaluation-seed differences in each
corpus/layout cell and sum with the product weights. For successful time, use
the ratio of the weighted sum of mutually successful paired time differences
to the weighted number of those pairs. Every contrast is left minus right;
positive completion and negative time differences favor the left policy.

M2 and M4 share each corpus trajectory. They are paired checkpoints, not six
independent training replications. Constructor comparisons pair the same three
corpus-seed blocks. Each fixed baseline has only 36 physical episodes. Its
layout/seed outcomes can enter the algebra for all three corpora, using the
same layout draw, without becoming independently sampled baseline copies.

The layout-only sensitivity uses the same sampled layout weights and fixed
corpus weights of one third. Primary intervals cover the complete 18-layout
nominal panel; the declared scene-family strata are descriptive summaries.

## Fixed comparisons and missing outcomes

The specification enumerates all 34 comparisons in deterministic order:

| Comparison | Count |
| --- | ---: |
| Observation curriculum minus analytic contrast, at M2 and M4 | 2 |
| Analytic contrast minus uniform, at M2 and M4 | 2 |
| Analytic contrast minus target-only, at M2 and M4 | 2 |
| Each of four constructors at both checkpoints minus each of three shared baselines | 24 |
| M4 minus M2 within each constructor and corpus trajectory | 4 |

Unknown outcomes remain in every assigned denominator. A policy with `S`
verified passes and `U` unknown assignments among `N` assignments has completion
bounds `[S/N, (S+U)/N]`. Contrast bounds are the left lower bound minus the right
upper bound, and the left upper bound minus the right lower bound. A fully
measured task-criterion failure counts as a failure, including the explicitly
named invalid-initial-approach criterion; it is not an unknown or an invented
collision/fall.

If **any participating assigned outcome is unknown**, withhold both completion
intervals and both time intervals for that comparison. Retain completion bounds,
assigned and measured denominators, and the observed mutually successful time
mean and counts as descriptive information. Failure and unknown episodes have
no imputed successful passage time.

Time differences use only mutually successful matched episodes. Report the
matched-pair count and each side's unique physical-episode count: a baseline can
participate in 108 paired cells but has at most 36 unique recordings. If there
are no mutually successful pairs, report no time mean or interval. If any
bootstrap draw has zero mutual-success support, withhold that scheme's entire
time interval and report the zero-support count. Do not discard or redraw the
empty samples.

## Reporting and interpretation

Report each corpus's completion mean or bounds and paired time mean/counts
separately. Across the three corpus means, report the mean, sample standard
deviation, minimum and maximum. With unknowns, summarize lower and upper bounds
separately. Plot each M2/M4 point against its actual recorded acquisition steps;
also disclose the assigned maxima of 27,416 and 46,488 steps. Saving M2 adds no
acquisition physics.

Only three independent corpora are available per constructor. These crossed
bootstrap intervals are descriptive and can be unstable; they do not supply
a large-sample coverage guarantee. The 34 intervals have no multiplicity
adjustment and do not support simultaneous significance claims. Successful-time
differences are conditional on both policies succeeding and cannot alone
establish improved passage capability. The finite layouts, reused motion-source
ancestry and ideal simulator sensing retain their existing scope limits.

The pure [statistics CLI](../../scripts/research/motion2scene_evaluation_statistics.py)
accepts either the runner's complete `motion2scene_reserved_execution_v4`
result or a paused `evaluation_status` wrapper, preserving all 972 rows. It
produces `motion2scene_reserved_statistics_v1`; the primary fields are
`crossed_95_percentile` and `crossed_95_percentile_s`. The
`layout_only_95_percentile` counterparts are the sensitivity. The CLI validates
report identities and denominators; the execution runner remains responsible
for model, runtime and physical evidence admission.

The [creation receipt](/home/linjiw/research-data/groot-wbc/m2s-evaluation-v4-reporting-clarification-proposed-v1/result.json)
binds the specification, original V4 proposal, unchanged assignment preview
and exclusive-write builder. Creating this artifact issued no reserved geometry
query, simulator execution, or adoption.
