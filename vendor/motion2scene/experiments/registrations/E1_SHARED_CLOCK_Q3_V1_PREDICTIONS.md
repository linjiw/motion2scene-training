# Shared-clock duck Q3 pilot

Question: can a slower clock shared by every ladder level support controller-retained crouches?
The existing neutral/d055/d085 references are resampled together with minimum pace ratio 0.7.
The first three source carrier IDs (41001, 41002, 41003) form a bounded acquisition pilot, selected
by ID before any shared-clock execution. All eight reference construction attempts remain reported.
Within each group, neutral/d055/d085 share the horizontal route, root orientation, frame count,
time map and approximate contact timing. The historical batch differs in timing and its upper
intensity, so it provides context rather than an isolated causal effect of retiming.

Execution: at most nine serial obstacle-absent trajectory-only runs, seed 7800, 7500 MiB free-VRAM
startup guard. Adaptations depend on their own neutral's tracker survival. Scientific rejections
are not retried. Infrastructure errors stop the batch. The source controller and acceptance gates
remain those pinned by the earlier manifests.

Predictions frozen before launch:

1. At least two of three shared-clock neutrals pass tracker survival.
2. At least two of three d055 motions pass tracker survival.
3. At least one of three complete neutral/d055/d085 groups passes tracker survival and the
   reference-relative route gate, with both adapted levels retaining S4 paired semantics.

Q3 route retention uses the held-out `relative_route_retention_v2_heldout` rule. This pilot can
launch only after the complete fresh neutral-validation batch meets its 80% retained-survivor
criterion. Paired semantics retain the frozen 50 mm whole-body effect, 60% magnitude retention,
and 30% event IoU thresholds. Both neutral and target must retain their own reference routes.

Every generated level remains in the denominator. Three-level Q3 success makes a group a Q4
candidate only. Q4, exact critical geometry, obstacle-present reversal and hallucination training
remain later experiments with their own evidence requirements.
