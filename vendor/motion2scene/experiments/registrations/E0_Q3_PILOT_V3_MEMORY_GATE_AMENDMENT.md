# E0-Q3 Pilot V3 Memory-Gate Amendment

Registered: 2026-09-04T13:13:08-04:00, before any rollout cell in V2 or V3 was executed.

## Decision

The V2 scientific question, sample, predictions, seeds, controller, motions, and acceptance policy
remain frozen. V2 yielded before spending because its conservative 9,000 MiB launch gate was not
met. V3 changes only that operational gate to 6,000 MiB and writes to a new output lineage.

## Evidence

The repository's measured GPU budget reports approximately 4.8 GB for one single-environment SONIC
rollout and explicitly sets 6 GB, rather than 4.8 GB, as the safe launch threshold. The extra margin
covers PhysX's initial allocation plus the scene, robot, and renderer. The corresponding capacity
utility defines `DEFAULT_REQUIRED_MIB = 6000`. Both files are hash-pinned in the V3 manifest.

The 9,000 MiB V2 setting was a conservative pilot policy, not an observed minimum. No V2 rollout
had begun when this amendment was made, so this change cannot be conditioned on experimental
outcomes.

## Resource rule

Run exactly one rollout at a time. Do not attempt two concurrent rollouts: with a measured 4.8 GB
per rollout and roughly 7.9 GB currently free, that would exceed available memory. If the 6,000 MiB
gate still produces a CUDA, PhysX-allocation, or renderer-memory failure, stop the batch, classify
it as infrastructure, and do not reduce the gate again without new measured evidence.

The maximum registered spend remains six rollouts and 0.625 contended GPU-hours.
