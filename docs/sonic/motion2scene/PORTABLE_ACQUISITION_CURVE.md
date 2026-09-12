# Portable five-arm acquisition curves

The later-checkpoint exporter and NumPy reader are implemented and validated
against recorded data. **No M8, M16 or M32 release has been exported:** those
physical acquisition prefixes are still incomplete. The existing completed
[M2 release](PORTABLE_ACQUISITION_M2_V1.md) remains unchanged.

The new exporter accepts a complete independently audited five-arm checkpoint
and preserves each original corpus separately. It retains every inherited and
new M0-through-checkpoint model, the exact chronological teacher prefix, all
seven branches per teacher scene, and each actual pre-update student execution.
Student encounter `i` remains attached to generating model `i−1`. The paired
executed-contrast/replay candidates, scene definitions and branch ordering must
match the frozen plan. The reference-envelope arm must have its own complete
prefix. The common scene-wise teacher, ridge penalty 10 and adopted measured-tie
initializer are preserved.

The physical release includes failed complete recordings. Its current inherited
schema requires a recorded neutral history at every supervised phase and full
captures. Unsupported partial captures or unavailable neutral histories cause
an explicit preflight failure; assigned tasks are never removed to complete an
export. If such data occur in later acquisition, a physical-schema extension is
required before release. Unknown startup outcomes remain in the retained
historical receipts, with zero measured startup steps and the separate original
reservation. The already reused bootstrap stays charged to its original corpus.

The reader reconstructs declared M8/M16/M32 curve points from actual episode
measurements. Bootstrap, teacher branches and student visits all contribute to
physical cost; bootstrap is excluded from useful-task yield. Nested prefix costs
are not additive. Geometric proposal computation remains separately visible in
the source audit. Reconstruction does not execute a policy or refresh a physical
gap.

Original fits retain their originally bound replay-weight artifacts. The
expanded native fitter uses a different schema: its `replay.json` sidecar is not
hash-bound by the original fit result. The exporter therefore hashes that
sidecar at export, independently regenerates its signals from the original
teacher/student measurements, and verifies the adopted replay rule. The reader
checks that proof, generating-policy identities and the exact historical weight
recipe. A new residual cannot replace a historical measured outcome.

## Completed validation

The [validation receipt](evidence/research-progress-20260909/portable-curve/validation.json)
binds the three new source/test files, isolated toolkit, real-data checks and
negative preflight. The final reader was run under Python 3.11.16 / NumPy 2.4.6
with original experiment/repository reads, simulator imports and release writes
blocked by a Python-level trusted-reader guard. This is not an OS sandbox.

| Check | Measured result |
|---|---:|
| Final reader reconstructing the immutable M2 release | 36 / 36 fits |
| Matching M2 recorded teacher-packet action checks | 324 / 324 |
| Maximum M2 coefficient difference | 5.5512 × 10⁻¹⁷ |
| Native expanded-format fits on recorded M2 replay data | 3 |
| Matching NumPy reconstructions of those format checks | 3 / 3 |
| Matching expanded-format recorded action checks | 27 / 27 |
| Maximum expanded-format coefficient difference | 3.1226 × 10⁻¹⁷ |
| Focused tests, including 22 new tests | 61 passed |
| New physical executions from this validation | 0 |

All reconstructed M1/M2 yield and measured-cost rows equal the completed M2
reader's results. The expanded-format check uses one recorded M2 corpus in an
explicitly separate CPU validation directory; it is neither an M8 prefix nor a
replacement acquisition history. Historical outcomes remain bound to their
actual generating models even when the file format is being tested.

Real-data validation exposed and repaired two reader issues: byte-size metadata
must not alter original path/hash identities, and JSON-serialized nested replay
coverage keys must be restored as tuples before the registered weighting
function is evaluated. Failed checks are retained. There were 78 CPU fits in
total across native format checks and two completed reader versions. These are
repeated fitting operations, not 78 physical policy trials. Black and Ruff pass.

Validation data and scripts are at
`/home/linjiw/research-data/groot-wbc/m2s-acquisition-curve-portability-validation-20260909-v1`.
The final toolkit is `toolkit_readonly_v3`; the final M2 reconstruction is
`M2_regression_v3/result.json`. The original M2 archive and active acquisition
sources were not modified.

## Commands

The following focused validation command was executed successfully:

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python -m pytest -q \
  decoupled_wbc/tests/test_motion2scene_acquisition_curve_dataset.py \
  decoupled_wbc/tests/test_motion2scene_acquisition_dataset_baseline.py \
  decoupled_wbc/tests/test_motion2scene_schedule_dataset_baseline.py \
  decoupled_wbc/tests/test_motion2scene_timed_schedule_dataset.py \
  decoupled_wbc/tests/test_motion2scene_expanded_acquisition.py \
  decoupled_wbc/tests/test_motion2scene_checkpoint_controls.py \
  decoupled_wbc/tests/test_motion2scene_response_diversity.py
```

The following M8 export commands are **unexecuted and data-gated**. Run only
after all assigned M8 checkpoints exist; use a fresh output directory. For later
declared checkpoints, use their actual complete audit and corresponding budget.
The exporter does not acquire data or substitute earlier checkpoints.

```bash
TASK_DATA=/home/linjiw/research-data/groot-wbc
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_response_diversity.py \
  --plan "$TASK_DATA/m2s-expanded-acquisition-plan-20260909-v1/plan.json" \
  --budget 8 --out "$TASK_DATA/m2s-response-diversity-M8-portable-v1"

OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_export_acquisition_curve_dataset.py \
  --audit "$TASK_DATA/m2s-response-diversity-M8-portable-v1/result.json" \
  --adoption "$TASK_DATA/m2s-primary-acquisition-adoption-v3/adoption.json" \
  --check

OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_export_acquisition_curve_dataset.py \
  --audit "$TASK_DATA/m2s-response-diversity-M8-portable-v1/result.json" \
  --adoption "$TASK_DATA/m2s-primary-acquisition-adoption-v3/adoption.json" \
  --history "$TASK_DATA/m2s-acquisition-M2-portable-20260909-v1/historical_startup/history.json" \
  --out "$TASK_DATA/m2s-acquisition-M8-portable-v1"

TASK_RELEASE="$TASK_DATA/m2s-acquisition-M8-portable-v1"
OPENBLAS_NUM_THREADS=1 .venv_research/bin/python \
  "$TASK_RELEASE/tools/portable_baseline/scripts/research/motion2scene_acquisition_curve_dataset_baseline.py" \
  --release "$TASK_RELEASE" --out "$TASK_DATA/m2s-acquisition-M8-reconstruction-v1"
```

The exported toolkit requires Python 3.10+ and NumPy, with repository/upstream
licensing retained. Raw pretrained weights and the full motion bank are not
redistributed. A release archive and relocation verification remain separate
steps after a real later-checkpoint export exists.
