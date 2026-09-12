# Teaching decisions that the student can actually select

The completed information-ablation study directly tests the difference between
a passing schedule in each scene and a passing policy that can distinguish the
scenes before commitment. It reuses the six-context, seven-schedule physical
teacher table and changes only access to its recorded scene-summary features.

All 28 corridor features are set to one common constant before a specified
reveal decision. Proprioception, phase, active-option indicators and legal-action
masks are unchanged. These remaining features are exactly equal across the six
contexts at all three recorded neutral decisions. At the reveal decision, the
original recorded scene summary becomes available. This is a controlled
information-access ablation, not a model of camera latency or missing ray returns.

| First decision with scene features | Observation groups at 0.30 / 1.00 / 1.40 s | Scene-wise bank capability | Best observation-consistent passage count |
| --- | --- | ---: | ---: |
| 0.30 s | 6 / 6 / 6 | 6/6 | 6/6 |
| 1.00 s | 1 / 6 / 6 | 6/6 | 6/6 |
| 1.40 s | 1 / 1 / 6 | 6/6 | 5/6 |
| Never | 1 / 1 / 1 | 6/6 | 5/6 |

When information arrives at 1.00 s, the universal-passage teacher assigns the
single initial group WAIT; the subsequent groups admit appropriate prior or
sustained continuations. When information arrives only at 1.40 s, the early
context has already lost its final prior commitment opportunity. Earlier
commitment cannot solve all contexts either: the prior and sustained responses
are complementary. The teacher therefore rejects the claim that a common
all-passing continuation exists. That rejection does **not** mean zero possible
passages: the exact passage-count dynamic program finds a five-context optimum.

The finite solver chooses one action per observation group. Because the declared
groups retain earlier distinctions, dynamic programming sums optimal successor
group counts and compares them with each immediate complete schedule. Unknown
branch outcomes prevent an exact capability result. Exhaustive enumeration on
20 small randomized finite trees matches the dynamic program. These checks and
the real recorded-table experiment establish the finite information constraint;
they do not establish generalization of a fitted policy or robustness to a
physical sensor model.

[Completed information experiment](/home/linjiw/research-data/groot-wbc/m2s-information-ablation-20260909-v1/result.json)
contains every information-group teacher table and source binding.

## From an information bound to trainable continuation targets

The teacher now constructs a best attainable common policy even when no policy
passes every member of an information group. For each legal action it computes
the number of successful encounters and the sum of their measured passage times.
A commitment uses its measured complete schedule; WAIT combines the selected
policies of its successor information groups. Backward induction maximizes the
passage count first and minimizes successful-time sum only among equal-count
alternatives. Encounters have uniform empirical mass. All passage-optimal actions
and all exact time minimizers are retained; deterministic tie breaking selects a
policy for evaluation. Unknown branches withhold targets, while a completely
measured all-failing group remains explicitly unsolvable.

This construction matters even when the universal-success set is empty. In a
controlled three-scene example, every scene has a fast late option, but the
observations never distinguish which one to use. A scene-wise teacher therefore
prefers WAIT in all three scenes, although any common late action passes only
one. The new teacher chooses a shared early option that passes two. Revealing
the scene before the late decision instead makes WAIT pass all three. Exhaustive
enumeration of every common policy on 50 randomized small trees matches both
the dynamic program's passage count and its secondary time objective. This is
a finite-method test, not an additional humanoid rollout.

The targets can use the existing ridge learner without fabricating physical
outcomes. Let $N^*$ be the maximal group passage count and $C_*,C^*$ the minimum
and maximum successful-time sums among actions attaining it. An action with
fewer passages receives regret $(N^*-N_a)/N^*$. A maximal-passage action receives
$(C_a-C_*)/(N^*C^*)$. With positive passage times, even its largest time regret is
strictly smaller than losing one passage. Singleton targets reduce exactly to
the original physical regret. The explicit-target fitter preserves the original
phase normalization, uniform encounter weights, unpenalized intercepts and ridge
penalty of 10. This target ordering is not a guarantee on fitted runtime scores.

On the six-context recorded study, the constructed policies attain 6/6, 6/6,
5/6 and 5/6 under the four reveal conditions. The paired same-ridge experiment
has the following result:

| First scene information | Scene-wise targets: passing branch proxies | Information-consistent targets: passing branch proxies | Paired proxy-time change |
| --- | ---: | ---: | ---: |
| 0.30 s | 6/6 | 6/6 | 0 s, 6 mutual successes |
| 1.00 s | 6/6 | 6/6 | 0 s, 6 mutual successes |
| 1.40 s | 5/6 | 5/6 | −0.004 s, 5 mutual successes |
| Never | 5/6 | 5/6 | −0.004 s, 5 mutual successes |

With unmasked observations, the fitted coefficients and saved models are
identical. Under late or absent information, the new targets change the shared
choice from the 1.00 s prior to the 0.30 s prior. Both miss the complementary
context. Thus the formulation corrects an attainable-continuation problem and
can train the same learner, but these six contexts show no passage gain over
the scene-wise baseline. The four-millisecond mean difference is a recorded
branch-cost difference, not an established closed-loop efficiency gain.

[Finite policy experiment](/home/linjiw/research-data/groot-wbc/m2s-information-policy-study-20260909-v1/result.json)
and [same-ridge target comparison](/home/linjiw/research-data/groot-wbc/m2s-information-learning-study-20260909-v1/result.json)
contain the policies, action values and per-context selections. Both reuse the
existing 42 physical teacher branches and add zero physics steps. The original
acquisition arms remain unchanged; broader teaching and physical evaluation of
this extension are separate experiments.

## Action-relevant continuation gate

The [new finite gate](../../scripts/research/motion2scene_action_relevant_gate.py)
tests whether the current information group admits a common successful action,
instead of requiring a surface receipt from every enabled beam. This is an
alternative supervision-eligibility rule; it does not identify the physical
effect of removing an obstacle. The available recordings contain complete
schedule branches, not matched beam-removal interventions.

For an information group at phase $k$, retain an immediate commitment if its
measured complete schedule passes in every group member. Retain WAIT only if
every successor information group admits a successful continuation. At the final
phase, WAIT is the neutral complete schedule. Among compatible actions, select
the one minimizing group-mean measured passage time while retaining the full
acceptable action set. Missing remaining branches withhold this calculation.
This backward recursion uses the existing universal-passage teacher; when its
set is empty, the passage-count teacher above can still provide useful training
targets. An empty priority gate must not be interpreted as zero achievable
passages or as a reason to erase those examples from uniform/coverage training.

The gate and teacher must change together. A scene-wise WAIT label may use a
different future action in each indistinguishable scene. Accepting WAIT as a
common immediate action does not authorize reusing those future choices or
their cost targets. The implementation distinguishes immediate-action
compatibility from agreement with the selected common-policy continuation.
It remains a development comparison, not a replacement in the frozen primary
acquisition or evidence of improved replay fitting.

The [completed calculation](/home/linjiw/research-data/groot-wbc/m2s-action-relevant-gate-development-20260909-v2/result.json)
reuses the six contexts and all 42 measured
teacher branches. Its eligibility counts are over all 18 recorded neutral
decisions, including decisions not reached after an earlier commitment:

| Scene features first available | All-beam cue gate | Common-success action gate | Newly admitted decisions | Admitted decisions requiring a different immediate action |
| --- | ---: | ---: | ---: | ---: |
| 0.30 s | 18/18 | 17/18 | 0 | 0 |
| 1.00 s | 13/18 | 17/18 | 5 | 1 |
| 1.40 s | 8/18 | 5/18 | 0 | 0 |
| Never | 3/18 | 0/18 | 0 | 0 |

For the reveal intervention, a raw recorded surface receipt is not counted as
usable while its scene-summary features are masked. The empty task has no beam
cue requirement. At a 1.00 s reveal, the common initial WAIT leads to appropriate
later choices and admits five nonempty-context decisions before scene-specific
cues are available. The long passage's scene-wise early commitment must
change to the shared WAIT. With a 1.40 s reveal, the shared early group has
no all-passing policy. Under nominal sensing, the original cue rule already
admits every phase; the new rule only removes the early scene's final-phase
decision, after its last passing prior commitment is no longer legal.
There is therefore no nominal admission gain in this development set.

These counts measure finite target eligibility, not learned-policy passage,
independent scene coverage, sensor robustness, or a physical replay advantage.
Exact vector groups can be singletons and do not establish that a noisy student
can distinguish the same tasks. Fourteen focused gate tests cover shared early
commitment, causally usable WAIT, late information conflicts, missing outcomes,
set-valued targets, masked receipts and action/continuation distinctions.
The combined gate, observation-teacher and passage-first information tests pass
32/32. This study rechecks the raw teacher collections and adds zero physics
steps. The first result and its exact executed source are preserved separately;
the second result clarifies that immediate-action agreement is weaker than
agreement about the continuation, with identical eligibility counts.

The information issue is established in prior work: Arora, Choudhury and Scherer
analyze cases where MDP-based POMDP approximations fail to choose useful
information-gathering actions. Our connection is methodological, not a claim
that their theorem directly establishes this traversal gate: the present
calculation enforces shared choices in a measured finite schedule tree.
See [Hindsight is Only 50/50](https://arxiv.org/abs/1804.02573), 2018.

## Controlled replay fitting

The checkpoint comparison now fits seven controls on each completed corpus:
uniform, coverage-only, 0.8 uniform plus 0.2 coverage, historical gated gaps,
historical ungated gaps, refreshed supervised error, and historical gaps reduced
by current supervised error. Every control uses the same ridge learner and three
refits. The ungated comparison removes only the cue deadline; measured outcomes,
matched histories, runtime identities and causal generating-model prefixes remain
required. Current error also covers complete measured zero-regret ties; the
learner's existing tie-initialization rule is unchanged.

On the first acquired observation-curriculum checkpoint (one nonempty encounter
plus bootstrap), all seven controls select the same five passing recorded
development branches, missing the complementary passage. The historical-gated
model reproduces the acquired model within 6e-18 in its coefficients. This
establishes a working comparison with actual historical measurements, not a
replay performance advantage.

[Checkpoint control result](/home/linjiw/research-data/groot-wbc/m2s-checkpoint-replay-controls-20260909-v1/seed93201_observation_curriculum_M1/result.json)
includes seven runtime-compatible policies. The
[learning-curve worker](/home/linjiw/research-data/groot-wbc/m2s-checkpoint-replay-learning-curves-20260909-v1/experiment.json)
waits for all three observation-curriculum corpora at M2, M4, M8, M16 and M32,
then fits every control. It does not launch physical evaluation or use reserved
outcomes. These fixed-data controls retain the original generating-policy
histories; their scores are not presented as on-policy measurements of each new
control.

The first completed M2 replay comparison (seed 93201) again gives all seven
controls 5/6 recorded development passages. Every control chooses the prior at
1.00 s on all six contexts and misses the complementary passage. This is one
completed corpus, not the three-seed M2 result and not new physical evaluation.
The [M2 result](/home/linjiw/research-data/groot-wbc/m2s-checkpoint-replay-learning-curves-20260909-v1/seed93201_observation_curriculum_M2/result.json)
retains all seven fitted policies and their weighting histories.
The [second completed M2 corpus, seed 93203](/home/linjiw/research-data/groot-wbc/m2s-checkpoint-replay-learning-curves-20260909-v1/seed93203_observation_curriculum_M2/result.json)
has the same seven-way 5/6 tie and prior-at-1.00-s choices. The
[third M2 corpus, seed 93202](/home/linjiw/research-data/groot-wbc/m2s-checkpoint-replay-learning-curves-20260909-v1/seed93202_observation_curriculum_M2/result.json)
is now complete as well: all seven controls again give 5/6, but select the prior
at 0.30 s in every context. Thus all three corpora show a seven-way passage tie
and no context-dependent schedule selection on the six-context recorded panel.
All miss the complementary passage. This is the completed three-corpus M2
recorded-data comparison, not new closed-loop or reserved evaluation.
