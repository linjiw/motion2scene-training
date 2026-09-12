# E0-Q3 Pilot Predictions

Registered: 2026-09-04T08:22:54-04:00, before any rollout in this lineage.

## Question

Do six fresh Kimodo G1 motions that passed the Q0 reachability screen complete one obstacle-absent
SONIC tracking pass, and does the executed subset retain enough behavior variation to justify
building matched motion ladders?

This is an E0 motion-bank qualification experiment. It is not a Motion2Scene scene-generation
test: the scene is an infinite plane and the motions are not matched alternatives.

## Fixed sample

Six motions, one per prompted body mode, with two straight, two gentle-left, and two gentle-right
routes. Selection was fixed before physics and is not conditioned on a rollout outcome. Every
motion has one 4.0 s / 120-frame Kimodo sample and one simulator seed.

## Predictions

1. At least 4 of 6 evaluated motions pass the existing `reference_trackability` acceptance policy.
   Each individual cell is provisionally predicted `accepted`; any rejection is retained and the
   aggregate prediction is adjudicated over all six attempted cells.
2. The two prompted left-turn executions have positive signed heading change and the two prompted
   right-turn executions have negative signed heading change.
3. The selected duck execution has a minimum root height at least 0.05 m below the selected plain
   walk execution.
4. If at least four episodes are evaluable, their action-token diversity has pooled effective rank
   at least 5.0 and between-episode effective rank at least 3.0.
5. Infrastructure success requires the explicit SONIC marker, one trajectory, a runtime success
   manifest, and exactly 50 Hz capture. Missing any item is infrastructure failure, not scientific
   rejection.

## Interpretation

- Passing Q3 makes these six candidates eligible for matched-ladder construction, not for the
  learned hallucinator by itself; Q4 still requires multiple seeds.
- A body-mode or route prediction miss is evidence that text-conditioned reference variation did
  not survive control, even if the tracker accepts the episode.
- Fewer than four acceptances redirects work toward motion generation/qualification before any
  inverse-scene learning.
- The compatible clutter package already generated from these motions is retained only as E1's
  `compatible-only` baseline.

## Unit, denominator, and scope

The unit is one independently generated motion clip at one fixed simulator seed. The denominator is
six attempted clips. With one sample per prompt and one physics seed, this pilot supports an
existence/feasibility decision only; it cannot support reliability, prompt-distribution, or
generalization claims.
