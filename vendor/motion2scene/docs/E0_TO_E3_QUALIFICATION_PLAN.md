# Motion2Scene qualification-to-critical-scene decision record

Status: adopted research guidance, operationalized 2026-09-04. Literature links and summaries
in the supplied guidance are background context; this record governs local experiments and does
not independently revalidate those citations.

## Decision

Do not train a learned hallucinator yet. The immediate bottleneck is a bank of same-carrier,
semantically valid, controller-retained motions with ordered functional envelopes. The required
chain is:

```text
motion acquisition
  -> paired semantic and route qualification
  -> Q3 controller retention
  -> Q4 robustness
  -> same-carrier ordinal ladder
  -> analytic critical interval
  -> obstacle-present physical preference reversal
  -> learned critical-token LfLH
```

The first method claim uses ordinal minimal feasibility. For ladder level `k`, a scene label is
the lowest level that clears it; no arbitrary whole-body effort weights are needed.

## What E0 established

The 18-reference `cg-wbc-v1-pilot` is permanently `pilot_characterization_v1`. It established
that the prompt pipeline runs and that behavior yield is uneven: ducking is much more reliable
than step-over or lateral narrowing. It did not estimate Kimodo's conditional distribution,
isolate a causal prompt effect, establish Q4, create matched ladders, or support critical-scene
learning.

The current single-seed Q3 evidence is also narrower after the retained-state audit:

- 6/6 frozen V8 cells were executed and 5/6 passed the original tracker gate;
- paired S0--S4 is 0/6 measured because v1 uses different seeds for every prompt;
- under the new route-centerline gate, 1/6 retained a valid requested route, 2/6 had an invalid
  reference route, and 3/6 lost route validity during tracking;
- duck and shoulder-turn retain the older self-baseline predicate, but those labels are not
  upgraded to S4.

This pilot may develop prompts and predicates. It may not enter a final test split.

## Frozen v2 motion-acquisition design

`cg-wbc-v2-shared-seed-confirmatory` crosses six body modes with three routes at each of eight
new generation seeds: 144 registered references. Generation seed is the independent unit;
body-mode and route cells are repeated measures. Each non-neutral cell is compared with the
same-seed, same-route walk.

The prompt grammar receives one revision to ask for finite onset and recovery and to discourage
route reversal. Prompts, thresholds, generation settings, and the analysis pairing are frozen in
[`configs/e0_kimodo_shared_seed_factorial_v2.json`](../configs/e0_kimodo_shared_seed_factorial_v2.json).
No prompt changes are allowed after the first v2 output is viewed.

## Semantic and route evidence

Reference semantics use S0--S3:

| Level | Required evidence |
| --- | --- |
| S0 absent | no paired functional envelope change |
| S1 elicited | paired effect exceeds its frozen functional threshold |
| S2 localized | finite event with onset and recovery |
| S3 route-aligned | localized event occurs on a valid progressing route |
| S4 controller-retained | achieved state retains event magnitude and location |

Duck uses whole-body top height, not root height. Lateral families use G1 body/capsule width along
the local route normal, not world `y`. Shoulder turn additionally needs shoulder-axis yaw. Step
needs unilateral apex gain, a supporting foot, landing, recovery, and no bilateral flight.

Route validity uses a 0.5-second-smoothed centerline sampled every 0.5 m so gait sway is not counted
as route curvature. It records net/path ratio, signed heading, integrated absolute turn,
monotonicity, reversal, self-intersection, and lateral departure.

The subsequent neutral-control calibration showed that cumulative absolute curvature remains
measurement-scale sensitive: achieved validity ranges from 0/8 to 8/8 over the frozen smoothing
and station-spacing grid. Therefore, this absolute predicate continues to qualify references, but
it is not by itself evidence that a controller lost a valid route. A new execution lineage must
use a preregistered reference-relative retention gate validated on held-out neutral controls. The
original v2 classifications and S4 counts remain unchanged.

## Q3 and Q4

Q3 is decomposed into tracker survival, route retention, semantic retention, and functional
envelope retention. Completing a clip is insufficient if its obstacle-negotiation event vanishes.

Every S4 Q3 motion entering Q4 runs three frozen physics seeds. Q4-robust means exactly two or at
least two of three (reported with the exact count); Q4-strict means three of three. The main motion
bank uses Q4-strict. Q4-robust-but-not-strict is a stress-test tier. The frozen protocol is
[`E0_Q4_PROTOCOL_V1.json`](../experiments/registrations/E0_Q4_PROTOCOL_V1.json).

## Matched ladders

The first family is duck, followed by separate arm-tuck and shoulder-turn families. Step-over is
parked until a Q4-qualified source exists. A ladder is admitted only when it has at least three
levels sharing carrier, start, route, timing, approximate speed, contact phase, and duration; only
adaptation intensity may vary. Every level is requalified. Attempted and admitted ladder counts
must both be reported.

## Analytic critical scenes

For target level `k` and weaker level `k-1`, the robust critical set requires target clearance and
weaker penetration after preregistered margins. A scene record must include a positive criticality
margin, target pass, weaker failure, stronger pass when available, obstacle-removal reversal,
registered geometry-jitter reversals, and exact-verifier agreement.

The first physical result must execute both alternatives in the same scene. Both must work in
open space; the weaker must fail because of the obstacle while the target completes the local
crossing. Carrier/ladder group is the independent unit. Nuisance scenes and physics seeds are
repeated measurements.

## Promotion gates

| Gate | Promotion evidence |
| --- | --- |
| A: motion substrate | at least 12 admitted Q4-strict ladders with three ordered levels |
| B: analytic identifiability | nonempty critical-set yield, widths, margins, removal and jitter checks |
| C: physical reversal | target succeeds and matched weaker fails in obstacle-present SONIC |
| D: diversity need | a named limitation of analytic sampling that learning can test |

Only after A--C pass may a small one-token MDN or conditional flow be trained. The forward learner
then predicts skill, minimal intensity, event position, and confidence/refusal; it does not begin
as a 29-DoF trajectory generator.

## Resource-contract note

The supplied guidance referred to a still-pending 9,000 MiB Q3 gate, but the frozen V8 pilot had
already completed under an amended 6,000 MiB registration. Empirical trajectory-only processes
fit below about 4.9 GiB. The pending eligible-census retry and new Q4 protocol use 7,500 MiB as a
shared-machine startup guard. This is not a GPU reservation, and no unrelated process is killed.
The threshold is not lowered during a registered run.

## Current next result

The next scientific result is not a hallucinator. It is an execution-qualified, same-carrier duck
ladder with a nonempty exact beam interval and obstacle-present target-versus-weaker reversal. If
v2 duck clips track but lose crouch, achieved rather than reference envelopes define the beam. If
prompt acquisition fails, controlled carrier transformations or retargeted motions replace it;
the Motion2Scene inverse hypothesis remains unchanged.
