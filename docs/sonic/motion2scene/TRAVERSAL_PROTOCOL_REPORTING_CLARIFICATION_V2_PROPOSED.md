# Proposed reporting clarification V2: recovery references

[Clarification V2](TRAVERSAL_PROTOCOL_REPORTING_CLARIFICATION_V2_PROPOSED.json) binds the [V5 infrastructure proposal](TRAVERSAL_PROTOCOL_AMENDMENT_V5_PROPOSED.json) and recovered primary acquisition paths. It changes **no reporting algorithm, estimand, comparison, policy identity, geometry or assigned episode**. It is unadopted and grants no execution authority.

The authoritative parent is `parent_protocol_proposal`, pointing to V5. The retained `parent_v4_proposal` identifies the original scientific-scope proposal; `previous_reporting_clarification` identifies immutable clarification V1. The wire schema remains `motion2scene_v4_reporting_clarification_v1`, with a separate `clarification_revision=2`, so the existing statistics interface remains compatible.

The `amendment_scope` field and its timing statement retain the history of clarification V1: it changed the original V4 layout-only primary bootstrap to paired crossed corpus/layout resampling before original primary adoption. V2 is a later infrastructure-reference amendment, issued after that primary adoption and a retained pre-simulator software failure. No primary physical outcome or reserved evaluation outcome was available or used to choose this recovery. The old adoption is not transferred to the recovery proposal or reserved evaluation.

All calculation-defining fields match V1 exactly:

- 20,000 draws, NumPy `default_rng(202609081822)`, corpus-index draws before layout-index draws, and the same matrices reused for all 34 comparisons.
- Three acquired-corpus blocks crossed with 18 layout blocks, with both evaluation seeds kept inside each layout. M2/M4 remain paired checkpoints. Baselines have 36 unique episodes each and are not independent corpus replications.
- Completion means or unknown identification bounds; paired successful-time ratios with explicit unique-episode and joint-success counts. Unknown outcomes withhold intervals. Any zero-support bootstrap draw withholds that time interval instead of silently conditioning on nonempty draws.
- Layout-only intervals remain a named sensitivity. Per-corpus variation and only-three-corpora limitations remain explicit. No simultaneous significance or population-coverage guarantee is added.

[The original explanation](TRAVERSAL_PROTOCOL_V4_REPORTING_CLARIFICATION_PROPOSED.md) provides the unchanged equations, draw order and 34-comparison inventory. The learning-curve x-axis remains actual recorded study physics steps; the retained predecessor's 1,192-step conservative reservation is disclosed separately for the affected corpus and is not plotted as measured physics.

V2 SHA-256 is `5322bd83cf1c7ebbbffab87566087435a73f787492b3b88706467d0d9cd321d8`. The canonical serialized unchanged statistical fields have SHA-256 `2c993ab6420590d3b43a608e8e7c3d8d8ae8f4f0e11188ece819e02845c7519c`. The [creation receipt](/home/linjiw/research-data/groot-wbc/m2s-evaluation-infrastructure-recovery-proposed-v5/result.json) verifies exact statistical equality and all 972 assignment identities/order. Final reserved adoption must bind this artifact, V5, the actual model inventory, and reviewed execution/statistics implementations separately.
