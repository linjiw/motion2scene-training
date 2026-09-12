# E1 Q3 neutral-control route-retention calibration v1

Status: frozen before running the new reference-versus-achieved analysis.

## Scope

This is descriptive instrument calibration on all eight neutral controls registered in
`E1_CONTROLLED_DUCK_Q3_V1_MANIFEST.json`. The independent unit is the base carrier/ladder group.
The full 8/8 denominator is retained, including any unmeasured execution.

It may distinguish absolute route-predicate sensitivity from controller route loss. It may not:

- relabel the frozen v2 route or S4 results;
- change the registered Q3 acceptance policy;
- admit a motion to Q4;
- select a new route threshold from these same controls;
- authorize an analytic dataset or learned hallucinator.

## Reference-to-achieved comparison

Reference and achieved root paths are translated independently to start at the origin and
resampled at 101 normalized-arclength stations. No rotation, scale, endpoint alignment, or
nonlinear shape warping is allowed. The audit reports:

- reference and achieved path length and net displacement;
- achieved/reference ratios for both quantities;
- signed-heading error and excess cumulative absolute curvature;
- endpoint Euclidean, along-chord, and cross-chord error;
- route-shape, local along-track, and local cross-track RMSE and maximum error;
- both frozen absolute route classifications.

These measurements have no pass/fail threshold in this calibration.

## Frozen sensitivity grid

The absolute classifier is rerun for every Cartesian combination of:

- centerline smoothing: 0.25, 0.50, 0.75, and 1.00 seconds;
- station spacing: 0.25, 0.50, and 0.75 metres.

Every combination is reported. No combination is selected after seeing the results. If a revised
retention rule is justified, it must be frozen separately and evaluated on newly executed,
held-out neutral carriers before it can affect S4 or Q4.
