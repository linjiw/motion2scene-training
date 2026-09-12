# Claim–Evidence Map

This ledger separates proposed mechanisms from measured evidence. It should be updated before any
headline prose.

| ID | Proposed claim or question | Required evidence | Current status | Current evidence / limitation |
| --- | --- | --- | --- | --- |
| C0 | Fresh Kimodo prompts yield kinematically distinct G1 candidates. | Full attempted denominator; taxonomy coverage; shared-seed paired functional effects and semantic predicates. | Shared-seed confirmatory acquisition measured; usable S3 yield is poor. | `cg-wbc-v2-shared-seed-confirmatory`: 144/144 generated across eight independent seeds; 132/144 pass Q0 and 144/144 pass Q1. Duck paired peak height reduction has median 0.242 m, but only 1/24 reaches S3 and it fails Q0. Arm tuck reaches 0/24 S3. Complete shoulder-turn, step-over and carry predicates remain `not_measured`; they are not counted as failures or successes. |
| C1 | Qualified motion banks are materially better substrate than raw generator output. | Q3/Q4 yields and downstream comparisons for filtered vs unfiltered banks. | Prerequisite measured; comparison untested. | Q3 yield is 5/6 among CPU-selected cells, but there is no equal-budget rollout of unfiltered motions and Q4 is unmeasured. Do not attribute the yield improvement to qualification yet. |
| C2 | Matched motion ladders expose nonempty critical geometry intervals. | Same-carrier neutral/weaker/target/stronger ladders; exact interval width and jitter tests. | Reference mechanism repeats; executable premise still fails. | Across eight controlled same-carrier duck groups, 7/8 are ordered reference candidates and 10/32 adjacent intervals remain nonempty after 10 mm target/strike margins. Q3 accepts 8/8 neutral, 3/8 40 mm and 1/8 55 mm levels, but 0/16 adapted executions reaches frozen S4. Thus 0 ladders and 0 intervals are dataset eligible. |
| C3 | Critical scenes make a target preferred, not merely feasible. | Target feasible; weaker rejected; stronger feasible but costlier; obstacle-removal reversal. | Untested. | Current generated clutter clears its routes but three carriers later failed the semantic gate. It is a preflight artifact, not an eligible compatible-only dataset arm. |
| C4 | Critical-token factorization preserves criticality while increasing conditional scene diversity. | Matched direct-scene and token-decoder arms; exact yield, diversity, realism, and removal tests. | Untested. | No learned hallucinator exists. |
| C5 | Motion2Scene data improves the forward scene-to-skill decision. | Equal-learner/equal-data comparison with named OOD splits, calibration and false-safe metrics. | Untested. | This is the primary downstream utility claim, not an inverse reconstruction metric. |
| C6 | Geometry-qualified scenes remain valid under SONIC execution. | Obstacle-absent and obstacle-present physics, target vs alternatives, multiple seeds. | Obstacle-absent Q3 measured; no retained ladder. | The v2 controlled batch completed 24/24 cells: 12 accepted and 12 rejected. All eight neutrals survive, versus 3/8 shallow and 1/8 medium crouches. Achieved paired envelope retention is measurable for some accepted edits, but 0/16 reaches frozen S4. A preregistered 8/8 neutral-control audit measures median 0.975 path-length retention, 0.046 m cross-track RMSE and 0.111 rad heading error; absolute achieved validity changes from 0/8 to 8/8 over the frozen smoothing/station grid. This establishes scale sensitivity of cumulative curvature, not S4. No Q4, obstacle-present or preference-reversal evidence exists. |

## Smallest contribution to test first

Latest instrument check (2026-09-05): on eight fresh neutral carriers, 7/8 pass tracker survival
and 5/7 survivors pass the preregistered relative route gate. The 80% validation criterion is not
met. Seed 42003 misses heading tolerance and seed 42008 misses cross-track RMSE tolerance; seed
42007 is rejected for contacts and also fails route tolerances. A shared-clock construction has
8/8 matched, reference-qualified three-level groups, but zero executed shared-clock groups.
See `experiments/registrations/E1_ROUTE_RETENTION_HELDOUT_V1_RESULT.md`. These results do not
change C2/C6 to execution-qualified or authorize C3--C5.

1. A versioned, failure-accounted schema for executable humanoid motion and matched alternatives.
2. An analytic critical-interval generator for one overhead-beam family.
3. A counterfactual exact verifier that can print a losing cell and obstacle-removal result.
4. A controlled utility experiment comparing random, compatible-only, and analytic-critical data.

The first paper-quality mechanism claim is unavailable until Q4-strict same-carrier ladders make
item 2 produce nonempty intervals and item 3 demonstrates target/alternative reversal. Until then,
outputs are research substrate and feasibility evidence; learned LfLH remains out of scope.
