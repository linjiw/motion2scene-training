# Proposed measured-tie supervision amendment

2026-09-09 UTC. **Proposed only; no adoption or production application.** The
[immutable amendment](../../../research-data/groot-wbc/m2s-measured-tie-method-amendment-proposed-v1/amendment.json)
has SHA-256
`99bf0173eee4ca408e9ce80b9eae7077230a68ab60a1e69459557d9d7aec1f8b`.
It contains no future-plan reference; a subsequent plan can bind this artifact.

The original empty bootstrap completed seven physical branches and 8,344 steps.
Its first model fit failed because phase 50 had a complete, feasible equal-cost
table, which the consequential-only filter excluded. The original
[failure audit](../../../research-data/groot-wbc/m2s-primary-tied-bootstrap-failure-v1/result.json),
failed fit and captures remain unchanged. This amendment uses only that empty
bootstrap and its CPU validation; no primary contrast encounter, student rollout,
trained primary model or reserved evaluation outcome informed it.

The proposed opt-in, `allow_measured_tie_initialization`, operates **only when a
phase has no consequential supervised rows**. Complete tables with at least two
legal actions, every legal action successful, and exactly equal finite
nonnegative costs supply genuine zero-regret targets. Their ridge solution has
zero coefficients and intercepts, with normalization from those measured rows.
Missing, all-failed or incomplete-continuation tables remain excluded; zero or
invalid replay weights cannot authorize a head. Existing runtime validation and
legal WAIT tie-breaking remain unchanged.

Whenever consequential rows exist, fitting and normalization remain exactly as
before. Lambda stays at 10, selected under the earlier eligibility rule, without
retuning against this bootstrap. The same opt-in must apply to every future fit
across all arms and corpus seeds. The
[staged tests](../../../research-data/groot-wbc/m2s-measured-tie-initialization-staged-v2/result.json)
pass 19 cases; [actual CPU validation](../../../research-data/groot-wbc/m2s-measured-tie-bootstrap-validation-v1/result.json)
reproduces all seven teacher records, initializes only phase 50, and preserves
phase-15/70 arrays exactly. Its external validation model is not a primary model.

Under separate adoption and verified reuse, the same seven captures would count
once as the `seed93203_analytic_contrast` bootstrap in a new controller/model root.
Their 8,344 steps stay inside that corpus's original 46,488-step allocation,
leaving 38,144. No physics is repeated to break the tie. The earlier 1,192-step
presimulator unknown reservation remains separate from recorded physics:
`12 × 46,488 + 1,192 = 559,048` conservative maximum steps. Candidates, options,
seeds, episode order, reserved geometry and evaluation statistics are unchanged.
Prospective source freezes must be refreshed; historical source pins are retained.
