# E1 Q3 neutral-control route-retention calibration v1 result

The preregistration and analysis implementation were committed at `f07b817448a86b93f2f8a2188db67014509a4adb`
before the analysis ran. The complete machine-readable result is
`/home/linjiw/research-data/groot-wbc/cg-wbc-v2-shared-seed-confirmatory/e1_controlled_duck_ladders_v1/q3_v1/route_retention_calibration_v1.json`
(SHA-256 `3af56d62ecce530de0324752389165632c2c2f75a7960af9ebd8412ee42e317b`).

## Result

All 8/8 frozen neutral carrier groups were measured. Direct reference-to-achieved route metrics
were:

| metric | median | observed range |
|---|---:|---:|
| achieved/reference path length | 0.975 | 0.958--0.992 |
| achieved/reference net displacement | 0.969 | 0.952--0.988 |
| endpoint error | 0.148 m | 0.067--0.216 m |
| route-shape RMSE | 0.097 m | 0.039--0.141 m |
| cross-track RMSE | 0.046 m | 0.029--0.063 m |
| absolute signed-heading error | 0.111 rad | 0.015--0.145 rad |
| cumulative-curvature excess | 0.675 rad | 0.533--0.837 rad |

The absolute straight-route classification is not stable over the registered measurement grid.
Depending on centerline smoothing and spatial station spacing, achieved validity ranges from 0/8
to 8/8. Reference validity itself ranges from 4/8 to 8/8. At the original 0.50 s/0.50 m setting,
1/8 achieved paths pass; changing only station spacing to 0.75 m makes 8/8 pass. No setting is
selected from this result.

## Interpretation and gate decision

The existing cumulative-absolute-curvature threshold is too scale-sensitive to serve as the sole
controller-retention gate. This does **not** prove that every neutral route was retained: endpoint
and shape errors remain nonzero. It supports replacing the absolute second classification with a
reference-relative retention test while keeping the reference's independently frozen route-validity
requirement.

The v2 S4 result remains 0/16 and Q4 admission remains zero. A relative rule must be frozen and
tested on new held-out neutral executions before it can govern a new qualification lineage.
