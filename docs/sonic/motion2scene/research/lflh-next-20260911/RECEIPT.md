# LfLH next-method research receipt

**Boundary:** exploratory research and CPU engineering training completed on September 11, 2026. The original physical pilot remains blocked; its registration and outcomes are unchanged. No new Isaac Lab/MuJoCo dynamics, GPU training, hardware action, reserved-layout access, or pilot-validation tuning occurred.

## Work completed

- Inspected original LfLH, Dyna-LfLH, LfH-CP and adjacent scene generation/environment design literature. The linked research report distinguishes sourced facts from proposed mathematics and methods.
- Inspected existing local LfLH code. Reproduced optimistic clearance in the actual legacy `SdfChoiceDecoder`: a weighted average can hide a negative minimum. Added an isolated lower-bound geometry implementation and tests, including missed capsule-axis intersections.
- Verified and reused seven previously qualified recorded trajectories through their SHA-256 references. No physical acquisition/evaluation contexts were used. Derived training and dense-frame capsule arrays without stepping any simulator.
- Wrote a dated engineering design before the new fits, then completed exactly eight fits: four architecture families × two initialization seeds × 120 updates. No checkpoint selection, optimization extension, or fit retry was performed.
- Retained all eight final checkpoints, update histories, raw sample rows, original metric definitions, and corrected geometric-witness analysis.
- Produced the method/design report, architecture CSV, scientific figure, source snapshot and input/output bindings.

## Actual execution and accounting

| Quantity | Recorded value | Meaning |
|---|---:|---|
| Completed fits | 8 | Unconditional, MLP, CNN, Transformer; seeds 101 and 102 |
| Optimizer updates | 960 | Sum of completed per-fit update receipts |
| Conditioned proposals scored | 448 | 56 per fit; correlated in-bank samples |
| Input-rotated proposals scored | 448 | Matched latent noise; diagnostic, not independent replication |
| Training fit wall time sum | 5.2473 s | Measured per-fit training loops |
| Geometry evaluation wall time sum | 6.9437 s | Measured per-fit scoring loops |
| Whole benchmark loop wall time | 12.7373 s | Includes loop overhead/checkpoint work, excludes preparation and later analysis |
| Subsequent metric analysis | 3.5828 s | Recomputed geometric witnesses, no new training or physics |
| Physics steps in this work | 0 | No dynamics launch |
| GPU training in this work | 0 | CPU tensors, two torch threads |
| CPU utilization / energy | NA | Not measured |
| Data preparation wall time | NA | Not independently instrumented |
| New physical outcome labels | 0 | Reconstruction targets are schedule identities |

The declared ceiling was eight fits, 120 updates per fit, and 600 seconds for the benchmark. The ceiling is not execution cost. The runner checks its wall budget before updates and between fits; its evaluation calls are not preemptible, so the wall threshold is a cooperative stop rather than a strict process timeout. The completed loop was well below it. Runtime versions and normalization are in `benchmark/started.json`; the broader installed-package snapshot is in `runtime-packages.json`.

The process completed with exit code 0. All fits have `state=complete`; there were no failed training attempts or unrun fit cells. The later correction is an analysis amendment, not a rerun of a physical outcome. No model was retrained following its results.

## Interpretation and correction

This is a single existing bank, not a source-ancestry-held-out sample. The decoder ranks geometric clearance with equal base costs; it does not implement the original LfLH differentiable dynamics planner or the frozen pilot teacher. The architecture parameter counts differ. Two initialization seeds and a short shared update count cannot establish an architecture ranking, convergence, data scaling law, or humanoid utility.

The original `contrast` field in `receipt.json` and `results.json` uses a negative lower bound as an interference proxy. That is not a valid penetration witness. It remains present for auditability and must not be quoted as certified collision. `posthoc-analysis.json` and `draws.csv` add `capsule_model_contrast`, requiring a sampled penetration witness for an alternative. The conditioned counts coincide in this batch. Neither metric establishes native-geometry collision or closed-loop behavior.

A positive spatially corrected capsule bound concerns the supplied frames and capsule model only. No temporal or tracking-error bound was established. A sparse-versus-dense disagreement means the coarse certificate failed at dense frames, not automatically that a measured collision occurred.

## Missing artifacts and blocked expansions

- The 24 historical reference CSV paths in `docs/hallucination/lflh_candidates.json` are unavailable under `/data/robotixx/groot-wbc-kimodo-m0/sweepcf_release/motions/clips/`; the local search did not recover the first referenced clip. The legacy SDF training experiment was not silently rerun with substitutes.
- No independent source-ancestry dataset and split has been established for this new method. Larger learning curves and transfer are proposed, not completed.
- A faithful LfLH decoder reproduction, mixture/flow/diffusion training, native-geometry/time-bound qualification and a calibrated physical critic remain future work.
- The original pilot’s episode 171 terminal receipt is still the existing execution blocker. Nothing here resolves it or authorizes bypassing the frozen runner.
- No new physical allocation, utility threshold, non-inferiority margin or adoption gate was introduced. The separately proposed future comparisons remain unadopted.

## Verification

Executed:

```bash
.venv_isaaclab/bin/python scripts/research/lflh_next/benchmark.py prepare --out docs/motion2scene/research/lflh-next-20260911/benchmark
.venv_isaaclab/bin/python scripts/research/lflh_next/benchmark.py run --out docs/motion2scene/research/lflh-next-20260911/benchmark
.venv_isaaclab/bin/python scripts/research/lflh_next/analyze.py docs/motion2scene/research/lflh-next-20260911/benchmark
.venv_isaaclab/bin/python -m pytest decoupled_wbc/tests/test_motion2scene_lflh_geometry.py decoupled_wbc/tests/test_motion2scene_support_pool.py decoupled_wbc/tests/test_motion2scene_support_validation_panel.py decoupled_wbc/tests/test_motion2scene_passage.py -q
```

**74 tests passed**, including seven new geometry tests and 67 existing support/passage checks. The figure was visually inspected. Black and Ruff were unavailable on PATH and in the frozen interpreter; an offline `uvx black` lookup also failed because Black was not cached. No lint/format pass is claimed and the hash-bound training source has not been reformatted after execution.

To reproduce training, choose a **new output directory** for both `prepare` and `run`. The original directory is intentionally non-overwriting. This would be a new engineering run, not a continuation of the pilot. The 24 pre-fit bindings are verified before loading and training; source closure snapshots and output hashes bind the current implementation and produced artifacts. Hashes created now do not prove any earlier registration date.

Read [the research report](RESEARCH_REPORT.md), [per-fit results](benchmark/architecture-results.csv), [all draw outcomes](benchmark/draws.csv), and [the declared design](benchmark/design.json). No publication, remote upload, commit, or external message was sent.
