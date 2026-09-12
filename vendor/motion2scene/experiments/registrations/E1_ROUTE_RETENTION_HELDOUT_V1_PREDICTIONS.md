# E1 held-out relative route-retention predictions

Frozen before generation seed 42001 or any later registered seed is sampled.

1. All 8/8 registered neutral references will generate without an infrastructure failure.
2. At least 6/8 generated references will pass Q0, Q1, and the independently frozen
   `valid_straight` reference predicate.
3. At least 5/8 admitted references will pass the existing obstacle-absent Q3 tracker-survival
   gate at runtime seed 7700.
4. At least 80% of Q3 tracker survivors will pass `relative_route_retention_v2_heldout`.
5. The held-out report will preserve generated, reference-admitted, Q3-survived, and
   route-retained denominators separately; a skipped or missing cell is not a failure or a pass.

The relative gate's thresholds were chosen after inspecting the earlier eight-control calibration,
so performance on those controls is not validation evidence. Only the new seed-42001--42008
executions adjudicate prediction 4.
