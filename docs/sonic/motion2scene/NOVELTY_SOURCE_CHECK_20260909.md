# Primary-source positioning check

Checked 2026-09-09. This is a focused check of the nearest three research ideas,
not an exhaustive priority claim.

| Source | Established precedent | Motion2Scene distinction that still needs evidence |
| --- | --- | --- |
| [LfH project and linked papers](https://www.cs.utexas.edu/~xiao/Research/LfH/LfH.html) | Collect executed motions in open space and construct obstacle/perception configurations around those motions | Contrast complete humanoid schedules with executed whole-body geometry, then physically verify transitions, traversal and recovery in proposed scenes |
| [Prioritized Level Replay, ICML 2021](https://proceedings.mlr.press/v139/jiang21b.html) | Prioritize training environments by estimated future learning potential, including TD error | Historical measured student/teacher gaps and supervised residual controls for a fixed finite continuation learner; no claim to originate replay or reproduce PLR's RL algorithm |
| [Robust Asymmetric Learning in POMDPs, ICML 2021](https://proceedings.mlr.press/v139/warrington21a.html) | Privileged teachers can demand actions that partial-information students cannot reproduce; A2D adapts expert and trainee jointly | An explicit finite information-group continuation calculation, separately evaluated from the original scene-wise acquisition teacher |

The manuscript previously described an “assumed executable plan” as the inverse
construction predecessor. That contrast is too broad: LfH explicitly uses motions
executed in open space. The related-work paragraph now acknowledges this.
Neither execution conditioning, environment replay, nor information mismatch
alone supports a novelty or performance claim. The central unmeasured question
remains downstream traversal at matched actual acquisition cost.
