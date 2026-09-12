# Held-out route validation and shared-clock reference construction

Measured 2026-09-05. The route policy and fresh seeds were frozen at `b0a81c6`; the execution
manifest was committed at `ed0d9d8` before launch. These are obstacle-absent Isaac/SONIC simulation
results. The independent unit is the generation seed, with eight new carriers.

## Held-out result

All eight registered references generated and passed Q0/Q1 and the frozen straight-route
predicate. All eight completed their registered execution. Seven passed the tracker gates and
five of those seven passed the reference-relative route gate: **5/7 = 71.4%**, below the
preregistered 80% criterion. Relative retention is therefore not validated under that criterion.

| seed | tracker | relative route | endpoint error (m) | heading error (rad) | cross-track RMSE (m) | failure |
|---:|---|---|---:|---:|---:|---|
| 42001 | pass | pass | 0.071 | 0.047 | 0.026 | none |
| 42002 | pass | pass | 0.158 | -0.148 | 0.030 | none |
| 42003 | pass | fail | 0.186 | 0.359 | 0.052 | heading tolerance 0.349 rad |
| 42004 | pass | pass | 0.167 | 0.019 | 0.093 | none |
| 42005 | pass | pass | 0.090 | -0.033 | 0.055 | none |
| 42006 | pass | pass | 0.167 | -0.077 | 0.090 | none |
| 42007 | fail | fail | 0.378 | -0.126 | 0.106 | disallowed contacts; endpoint and cross-track tolerances |
| 42008 | pass | fail | 0.141 | -0.133 | 0.118 | cross-track tolerance 0.100 m |

Predictions 1, 2 and 3 (generation, reference admission and tracker-survival yield) were met.
Prediction 4 (at least 80% of tracker survivors retain the route) was falsified. Prediction 5
(separate complete denominators) is satisfied by the final result. One interim four-run report
incorrectly adjudicated prediction 4 before batch completion; it remains archived with an explicit
invalid-adjudication note. Only the complete v2 report is authoritative.

The historical absolute route classifier marks all seven tracker survivors `over_turn`, while
the contact-rejected motion is its only `valid_straight` achieved path. This is disagreement
between different measurements, not proof that every tracker survivor has a correct route.
The measured cross-track deviations remain relevant to scene placement. In particular, the
0.118 m error cannot be assumed harmless when constructing critical scenes with centimetre margins.

The batch used **0.068 contended GPU-hours**. It paused once at 6551 MiB free VRAM and resumed
under the unchanged 7500 MiB startup guard. No other GPU workload was stopped.

## Shared-clock reference construction

The earlier duck batch's 12 scientific rejections all included endpoint tracking error as their
only rejection reason. To test timing as a possible cause, the shared-clock design was frozen
before constructing its outputs. It uses the existing retiming implementation and applies the
same time map to neutral, 55 mm and 85 mm reference levels within each of the eight original
carrier groups. The minimum local pace is 70% of the original pace.

All **8/8 groups, 24/24 references**, preserve identical horizontal root routes, root orientations
and timing across levels. They pass Q0/Q1; the 16 adapted references pass paired S3. Frames increase
from 120 to 143--144, approximately 19--20% longer duration. All eight central height profiles are
ordered; the minimum adjacent separation per group ranges from 0.0124 to 0.0327 m. These are
reference envelope differences, not exact beam-clearance intervals or execution evidence.

The proposed nine-cell Q3 pilot uses the first three registered carrier IDs. Its physics launch
requires the completed held-out route validation to meet the 80% criterion. Since that criterion
failed, **zero shared-clock physics cells launched**. The preparation code refuses this state.
No Q4 ladder or dataset-eligible critical scene is admitted.

## Evidence and reproducibility

- Held-out data root: `/home/linjiw/research-data/groot-wbc/cg-wbc-route-retention-heldout-v1`.
- Authoritative result: `q3/relative_retention_result_v2.json`, SHA-256
  `d2de7ac0b634fbe7a35fc4335ceacf3bab66beab8b7f6b28eb9f682abaa13e6c`.
- Complete run record: `q3/run_record.json`, SHA-256
  `c0f0c99e2bc4dbab42515aaa369fe647233b91143a1fe2870a27deaf6c9bff01`.
- All eight route overlays: `q3/route_comparison.png` and `q3/route_comparison.svg`.
  Both paths are translated to the origin and rotated by the same reference chord; the lateral
  plotting scale is exaggerated and explicitly labelled.
- Shared-clock candidates:
  `/home/linjiw/research-data/groot-wbc/cg-wbc-v2-shared-seed-confirmatory/e1_shared_clock_duck_v1/candidates.json`,
  SHA-256 `85f26079c7540e4e17e9bb97586690b9aafd34d350c48451e22a56a19e372a0c`.
- Verification: 72 standalone tests and 55 source retiming/conversion/trajectory tests passed;
  focused Ruff and whitespace checks passed.

The next uncertainty is whether the route misses represent estimator sensitivity, controller
drift, or both. Any new analysis must retain the measured failure values and separate estimator
calibration from motion admission. A failed yield prediction alone does not justify relaxing
the physical tolerances. Learned hallucination remains gated by qualified ladders, exact critical
geometry and obstacle-present preference reversal.
