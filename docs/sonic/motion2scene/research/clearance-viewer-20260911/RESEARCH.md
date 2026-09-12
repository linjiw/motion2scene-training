# Critical proposals with an explicit clearance constraint

2026-09-11. Outcome-motivated exploratory amendment. This is a frozen-model generator analysis on previously inspected coordinates, not the original pilot, a new independent validation set, or a new neural training result. All earlier geometric results were accessible before this design.

## Relationship to LfH and LfLH

LfH constructs obstacle configurations around open-space motion and uses the resulting pairs for planner learning. LfLH learns an obstacle-distribution encoder, uses a fixed planner decoder for motion reconstruction, adds prior and collision terms, and then trains a downstream planner from generated scenes. Its paper explicitly mentions approximate-gradient methods such as REINFORCE for non-differentiable decoders. These motivate our inverse scene-construction problem; they do not validate this humanoid pipeline. Sources: [LfH project](https://www.cs.utexas.edu/~xiao/Research/LfH/LfH.html), [LfLH, equations 1–3 and section III](https://www.cs.utexas.edu/~xiao/papers/lflh.pdf).

Our critical-location proxy asks whether the target capsule trajectory clears a beam while another schedule has a capsule penetration witness. It does not establish motion optimality, necessity among all possible motions, or closed-loop feasibility. We have not implemented the original differentiable-planner reconstruction pipeline. The fixed geometric calculation anchors the meaning of an obstacle but cannot supply physical passage labels. Non-differentiable physics does not mathematically prohibit learning: a score-function estimator is possible, with variance and rollout cost to address. No such physical optimization was launched here. Earlier local prose claiming that non-differentiability makes learning impossible should be read with this correction.

## Implemented generator

Let q_theta(c | motion) be a frozen probability distribution over the existing 384 coordinate cells. For a target motion, define S = {c: lower_capsule_gap(c) >= 0.01 m}. Unknown cells are excluded. The lower bound incorporates the spatial capsule-axis covering correction, evaluated over the recorded frames. Define

    Z = sum_{c in S} q_theta(c)
    q_S(c) = q_theta(c) * 1[c in S] / Z, if Z > 0.

If Z=0 the sampler abstains. It does not silently fall back to unsafe or uniform placements. The implementation handles even subnormal positive retained probability without losing normalization.

For any distribution p supported in S, KL(p || q) = KL(p || q_S) - log Z. Therefore q_S is the KL-nearest feasible distribution to q. It preserves the relative probabilities of retained cells. This is a finite-support projection property, not a new empirical safety theorem. It uses only target-clearance fields, not the alternative-motion critical labels, at generation time.

The method conditions proposals on a geometry query result. It must be compared to a geometry-screened uniform generator with the same query access. Calling its gain an improvement in the neural network would be incorrect. At a new motion, this exhaustive implementation needs 384 target-placement geometry checks for this fixed domain. This analysis reuses 2,688 cached motion-placement pairs and performs zero new clearance queries. The original 32×48 field construction also queried training cells, which are outside this displayed 384-cell domain.

## Complete finite-domain comparison

Percentages below average all seven targets and both existing initialization seeds; uniform has one deterministic distribution per target. The original split's test coordinates have already been inspected, so they are now development evidence. No target or seed is selected for the aggregate.

| Frozen model | Raw critical mass | Projected critical mass | Raw penetration-witness mass | Raw mass retained |
|---|---:|---:|---:|---:|
| Uniform | 1.60% | 3.96% | 59.23% | 38.36% |
| Unconditional | 44.44% | 82.29% | 38.62% | 46.82% |
| Coverage | 46.18% | 80.45% | 33.79% | 50.07% |
| Coverage + masked completion | 45.93% | 78.50% | 31.74% | 51.94% |
| Coverage + ranking | 45.57% | 81.47% | 22.32% | 61.30% |
| Combined | 42.60% | 81.24% | 22.70% | 56.14% |

Every projected row has 100% mass on the checked clear support and zero penetration-witness mass by construction. No row abstained on this bank. Neutral still has zero critical mass, so the maximum all-target critical average is 6/7 = 85.71%, even for a critical oracle. It is not removed to inflate performance.

These are probabilities over candidate placements, not counts of physical passes. The denominator changes from q to q conditioned on S; retained mass is essential to interpretation. An 81.47% conditional critical probability does not mean that 81.47% of raw proposals are useful. Coverage plus ranking retains 61.30% on average; rejection sampling cost depends on each row's reciprocal Z, not the reciprocal of this aggregate mean. Exhaustive cached projection does not execute rejection sampling, and we report no measured rejection count or production latency.

The unconditional projected model reaches 82.29%, slightly above every motion-conditioned variant in this panel. The evidence does not establish motion-specific generalization. Much of the result could come from this bank's common critical band and the motion-specific geometry check. The shared data contain one source bank; two initialization seeds and thousands of grid cells are not independent motion populations. Effect intervals and p-values are NA: this exploratory deterministic conditional-domain comparison has no established independent sampling or exchangeability scheme. No equivalence or non-inferiority conclusion follows.

All 154 method/seed/mode/target rows are in `outcomes.csv`; all aggregate rows are in `summary.json`. Contact-qualified passage, new physical failures, technical episode outcomes, and physical unrun assignments are not created by this analysis. Their outcomes remain unavailable here; the interrupted pilot's ledger is unchanged.

## Implemented next loss, not yet trained

A generator can place almost all raw probability on rejected cells yet look good after renormalization. The next objective should jointly optimize conditional critical coverage and raw proposal yield:

    L = KL(U_critical || q_S) + lambda_accept * (-log Z) + beta * L_rank.

U_critical is uniform over declared training critical cells. q_S and Z are computed on training coordinates only. The first term rewards multiple critical placements; the second penalizes rejection. Ranking remains an optional matched ablation. Crucially, at lambda_accept=1 the first two terms equal ordinary uniform-critical cross entropy up to a label-dependent constant. Renaming them would not create a new learning method. lambda_accept>1 increases acceptance pressure relative to coverage; the implementation's candidate default is 2, an untrained engineering choice rather than an optimized value or physical decision threshold.

`coverage_acceptance_loss` implements the first two terms stably in log space. Targets with clear but no critical cells receive only acceptance supervision. Targets with no clear cells are excluded from this loss and must retain an abstention count; no fictitious positive is introduced. Tests check the lambda=1 identity, gradients for a no-critical target, and no-clear behavior. This loss was not used to retrain checkpoints in this session.

A discriminating follow-up should freeze a source-ancestry split before opening new outcomes; compare unconditional, coverage, and coverage-plus-acceptance under the same model and updates; and report raw and projected results together. On the current bank, any follow-up training is development, not fresh confirmation. Masked reconstruction remains an ablation because its previous result did not improve critical probability. Richer oriented obstacles, variable widths and full 3D distributions require a separately defined domain, geometry handling, and new design. The viewer does not imply they are implemented.

## Execution and decision

The export and finite cached-field analysis completed in 0.3656121849999181 s on CPU, measured by the exporter. This is not total session time, fresh geometry cost, or neural training time. There were zero new physics steps, zero new training updates, and no GPU work requested. All 298 recorded frames of all seven motions are retained. UI replay and screenshots are not dynamics experiments.

The first browser harness launch failed because the freshly installed Playwright expected a different browser revision. It never loaded the viewer. The harness was pointed to the existing Chromium executable and passed; no experiment or model was rerun. Source formatting and input-validation hardening after the initial export did not change the reported probabilities; the final audit recomputes the projection and verifies them. Original prior manifests remain untouched.

- SUPPORTED FOR THE DECLARED FINITE ESTIMAND: projected proposals concentrate mass on critical checked-clear placements in this domain; compared with the raw generator, the change includes generation-time geometry access.
- INCONCLUSIVE: motion-conditioned learning adds value beyond an unconditional generator with the same geometry check.
- BLOCKED/UNTESTED: the newly implemented acceptance loss improves learning; no new fit was performed.
- BLOCKED/UNTESTED: physical utility over contemporaneous Screened Uniform, source-ancestry transfer, Phase 3, or hardware. Existing gates remain in force.

The next permitted research action is a dated, budgeted offline design for independent-source evaluation and acceptance-loss training once eligible ancestry-separated inputs are verified. Physical expansion still requires the original pilot and adoption gates; this viewer and analysis do not waive them.
