# E0-Q3 Pilot V2 Predictions

Registered: 2026-09-04T08:28:08-04:00, before any rollout in this lineage.

## Question

Do six fresh Kimodo G1 references that passed the composed reachability,
self-intersection, and available behavior-semantic gates complete one obstacle-absent SONIC tracking
pass, and does execution preserve the prompted event/route distinctions?

This is E0 motion-bank qualification on an infinite plane, not a critical-scene experiment.

## Fixed sample and denominator

Six clips are selected from the ten references worth a rollout: two walks plus one each of
duck-under, arm-tuck, shoulder-turn, and carry-walk. The routes are balanced at two straight, two
gentle-left, and two gentle-right. Step-over is absent because all 3/3 generated references failed
the pre-registered semantic eligibility rule, not because of a physics result. The experimental
unit is one independently generated clip at one fixed simulator seed; the denominator is all six
attempted clips.

## Predictions

1. At least 4 of 6 evaluated motions pass `reference_trackability`. Each individual cell is
   provisionally predicted accepted; completed rejections stay in the denominator.
2. The two left-turn executions (indices 001 and 016) have positive signed heading change, and the
   two right-turn executions (005 and 011) have negative signed heading change.
3. The duck execution retains a torso drop of at least 0.08 m and a minimum root height at least
   0.05 m below the straight-walk execution.
4. The arm-tuck and shoulder-turn executions each retain at least 0.06 m within-episode half-width
   reduction and reopen after the event, matching the existing semantic predicate.
5. If at least four episodes are evaluable, action-token diversity has pooled effective rank at
   least 5.0 and between-episode effective rank at least 3.0.
6. Infrastructure success requires the SONIC marker, exactly one trajectory, a runtime success
   manifest, and 50 Hz capture. A missing item is infrastructure failure, never a scientific
   rejection.

## Interpretation boundary

One seed supports feasibility only. Q4 requires repeats. Passing this pilot makes a motion eligible
for matched-ladder construction but does not establish criticality, reliability, prompt-level
generation rates, or transfer. Fewer than four acceptances redirects work toward motion generation
and qualification before inverse-scene learning.

