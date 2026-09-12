# Standalone evaluation statistics and checkpoint curves

`scripts/research/motion2scene_evaluation_statistics.py` reads the final V4 evaluation receipt or a paused `evaluation_status` event containing `receipt.report.rows`. It requires the complete 972-row assignment matrix, including explicit `not_run` and `technical_missing` rows. It uses Python and NumPy only; it does not import the robot package, simulator, tracker or training stack and does not open model weights.

```sh
python scripts/research/motion2scene_evaluation_statistics.py \
  --input /path/to/evaluation/result.json \
  --out /path/to/new-statistics-directory \
  --reporting-clarification docs/motion2scene/TRAVERSAL_PROTOCOL_V4_REPORTING_CLARIFICATION_PROPOSED.json
```

Use an output directory outside the input's immutable directory; it must not already exist. `--expected-sha256` optionally binds the receipt digest. The output records source/input hashes, NumPy version and the optional reporting clarification hash before analysis. Binding a proposed clarification does not adopt the evaluation protocol.

The CLI writes `result.json`, `learning_curves.csv` and `paired_comparisons.csv`. It preserves every input row, validates the 24 learned checkpoints and three shared baselines, rejects duplicate pairings or inconsistent model/acquisition metadata, and reports each policy's assigned outcomes. M2/M4 curves contain both actual cumulative acquisition physics steps and their fixed assigned maxima of 27,416 and 46,488; evaluation repeats do not multiply acquisition cost. Three per-corpus curves and their mean, sample standard deviation, minimum and maximum are retained. The declared layout families and single/course strata are summarized separately.

There are 34 comparisons: observation minus analytic, analytic minus uniform, and analytic minus target-only at M2 and M4; each arm's M4 minus M2; and each arm/checkpoint minus the shared walk, constant and script baselines. Directions, corpus identities, layout ordering and model artifacts are explicit.

The clarified primary descriptive interval uses 20,000 paired crossed bootstrap draws with seed 202609081822. NumPy's default PCG64 generator first draws a `(20000,3)` acquired-corpus index matrix, then a `(20000,18)` layout index matrix. Normalized frequency weights form their product. The same matrices are reused across every comparison and checkpoint. Both evaluation seeds remain together within each layout. Their generated weight matrices are hashed in the output. A shared baseline has 36 unique recordings, with the same layout draw across all corpora; it is never independently resampled as three baseline copies. The original layout-only interval is reported as a named sensitivity using the same layout weights and fixed corpus weights.

Completion means use every assigned pair. Unknown outcomes retain explicit lower/upper completion-difference bounds and withhold intervals for that comparison. Time differences use only mutually successful matched episodes, with their counts and per-corpus values. The time bootstrap reweights the paired difference sum and joint-success count together. If any draw has zero joint-success support, that interval is withheld with the zero-support count; draws are not silently dropped or replaced. No failure time, crouch penalty or mechanical-energy value is imputed.

These are descriptive intervals with only three independently acquired corpora. M2/M4 share each acquisition trajectory; they are paired checkpoints rather than independent training replicates. The two evaluation seeds are repeated measurements rather than independent layouts. No multiplicity correction or population-coverage claim is made. The statistics reader validates metadata and arithmetic, while the evaluation runner and archived native audits remain responsible for physical provenance.

Validation includes 21 focused tests for complete/paused matrices, metadata consistency, cumulative acquisition counts, exact paired block resampling, shared-baseline identities, unknown bounds, disjoint success subsets, zero-support cost draws, portable output/hash handling and the immutable 34-comparison reporting contract. Separate full-configuration runs on synthetic 972-row complete and paused inputs succeeded with original workspace/assets, model reads and simulator/project imports blocked. Those synthetic values are software validation only; no reserved evaluation outcome was consumed.
