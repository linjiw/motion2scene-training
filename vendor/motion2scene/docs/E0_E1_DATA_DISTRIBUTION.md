# Fresh Kimodo corpus and first Motion2Scene distribution

## What was prompted

The E0 corpus uses one fixed grammar:

```text
A person walks at a steady pace {route clause} {body-mode clause}
```

The design is an exactly balanced Cartesian grid:

- body mode: walk, duck-under, arm-tuck, shoulder-turn, step-over, carry-walk;
- route: straight, gentle-left, gentle-right;
- speed language: steady for every cell;
- sampling: one deterministic Kimodo sample per cell.

This produces 18 prompts and 18 motions. Every motion has 120 frames at 30 Hz, uses 100 denoising
steps, and stores its exact prompt and seed. The prompt cache is 18 × 4,096 bfloat16 embeddings.
Seeds follow `1000 + 1000 × prompt_index`.

The exact prompts and source-side results are frozen under:

```text
/home/linjiw/research-data/groot-wbc/cg-wbc-v1-pilot/taxonomy/
/home/linjiw/research-data/groot-wbc/cg-wbc-v1-pilot/motions/
```

One sample per prompt measures grid coverage, not Kimodo's conditional probability distribution.
Estimating prompt-level yield requires multiple independent seeds per grid cell.

## What Kimodo actually produced

Across all 18 generated references:

| Metric | Minimum | Median | Maximum | CV |
| --- | ---: | ---: | ---: | ---: |
| Mean speed | 0.480 m/s | 0.823 m/s | 1.316 m/s | 0.248 |
| Path length | 1.903 m | 3.263 m | 5.220 m | 0.248 |
| Absolute heading change | 0.017 rad | 1.730 rad | 6.589 rad | 0.903 |
| Minimum root height | 0.589 m | 0.728 m | 0.750 m | 0.064 |
| Root-height range | 0.033 m | 0.049 m | 0.198 m | 0.645 |
| Upper-body joint RMS excursion | 0.179 rad | 0.445 rad | 1.226 rad | 0.537 |

Requested route direction separates at the median: gentle-left has +2.008 rad signed heading,
gentle-right −1.640 rad, and straight −0.180 rad. Duck-under prompts have a 0.613 m median minimum
root height versus 0.731 m for walk. These are reference-kinematic observations only.

## Qualification distribution

The full denominator is retained:

| Gate | Passed / attempted | Interpretation |
| --- | ---: | --- |
| Generation | 18/18 | valid Kimodo qpos artifacts |
| Q0 reachability | 17/18 | conservative joint-limit screen |
| Q1 severe self-intersection | 18/18 | G1 capsule screen |
| Semantic predicate | 5/12 measured | requested event visible in reference |
| Semantic unavailable | 6/18 | walk/carry are `not_measured`, never inferred as passes |
| Worth obstacle-absent rollout | 10/18 | no CPU reason to refuse physics |
| Q3 measured so far | 6 accepted, 1 rejected, 3 unmeasured | one seed, `screen_empty`, zero external force for gradeable cells |

Step-over has 0/3 semantic yield. This is why no step motion was sent to physics. The six-cell V8
pilot accepted 5/6; arm-tuck failed endpoint tracking. The fixed eligible-census expansion added
one accepted walk-right motion before shared-GPU contention left three cells unmeasured.

## First critical distribution

For an overhead target and a taller weaker alternative, beam underside height is constrained by:

```text
target_reach + safety_margin <= beam_height
beam_height <= weaker_reach - strike_margin
```

The index-000 nominal→40 mm reference pair has:

- target reach: 1.242300 m;
- weaker reach: 1.298725 m;
- raw gap: 56.425 mm;
- safety and strike margins: 10 mm each;
- usable critical interval: 1.252300–1.288725 m, width 36.425 mm.

Twenty heights are sampled with one uniform draw in each of 20 equal-probability strata. This
prevents random clumping near one boundary. Each height therefore preserves both target clearance
and weaker-motion collision depth.

Rendering nuisance is sampled independently:

| Nuisance | Distribution |
| --- | --- |
| Thickness | uniform 0.04–0.16 m |
| Along-route span | uniform 0.20–0.50 m |
| Cross-route span | uniform 1.00–2.00 m |
| Yaw jitter | uniform −0.08–0.08 rad |
| Material | categorical: painted wood, steel, concrete |

Changing nuisance support cannot change the seeded critical-height stream. This is the concrete
factorization:

```text
motion ladder -> critical height interval -> stratified height
                                      \-> independent rendering nuisance
```

The next 40→55 mm reference gap is only 19.394 mm, so 10 mm margins on both sides make it empty.
Both controlled targets also failed Q3 endpoint tracking. Consequently the current analytic
artifact contains 20 diagnostic scene parameters but **zero dataset-eligible scene–motion pairs**.
The schema independently requires passed Q3/Q4 target tracking and passed exact geometry before a
record may carry a critical token.

## GPU budget

Camera-free trajectory rollouts do not require 9,000 MiB free VRAM. Completed cells take roughly
25–37 seconds and fit the measured sub-4.9 GiB process envelope. The original 6,000 MiB launch gate
works under stable availability. Because this machine runs uncoordinated jobs, the pending retry
uses 7,500 MiB as a shared-machine guard after one mid-startup cuBLAS allocation failure. This does
not reserve the GPU; serial execution and stop-on-contention remain mandatory.
