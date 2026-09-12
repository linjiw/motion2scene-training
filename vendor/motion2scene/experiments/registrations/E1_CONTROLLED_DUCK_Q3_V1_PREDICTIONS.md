# E1 controlled-duck Q3 v1 predictions

Frozen before any controlled-ladder Q3 rollout. The independent unit is one of eight base
carrier seeds; the neutral, 40 mm and 55 mm levels are repeated measurements within it.

1. At least 5/8 neutral carriers will pass the single-seed obstacle-absent trackability gate.
2. At least 3/8 carrier groups will have both neutral and 40 mm levels pass Q3 and retain a
   measurable controller-level crouch event. S4 itself requires the separately frozen
   magnitude/IoU analysis and is not inferred from tracker survival.
3. The 55 mm level will not have a higher carrier-level Q3 pass rate than the 40 mm level.
4. Reference-level ordered geometry will not be called an admitted ladder unless at least
   three levels in a carrier group retain the event under Q3 and later Q4.

All scientific rejections remain in the denominator. A rejected neutral makes its adapted
levels dependency-skipped; those skips remain reported and are not treated as passes or
failures of the crouch operator.
