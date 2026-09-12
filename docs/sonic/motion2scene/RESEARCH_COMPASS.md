# Motion2Scene Research Compass

> **Provenance.** This is the principal investigator's governing reference, supplied 2026-09-10 and
> reproduced here verbatim. It is deliberately stable: it is not an experiment log and must not be
> edited to track results. Its own rule applies to itself — *"Later execution status must be taken
> from later records, not inferred from this document."* For current status see
> [RESEARCH_STATE.md](RESEARCH_STATE.md). Its stated basis is the September 2026 manuscript and the
> persistent research state as of 2026-09-10 13:28 UTC, which is **before** the M8 development panel
> completed; the M8 result is recorded in RESEARCH_STATE.md, not here.

**Working title:** Motion2Scene: Execution-Conditioned Teaching for Perceptive Humanoid Traversal

**Purpose:** A stable research and writing reference, separate from the evolving experiment log. This
document proposes no change to an active experiment.

## The research question

Given a frozen low-level tracker and a qualified motion repertoire, can scenes constructed from
controller-executed alternatives, together with complete-continuation supervision, train a
sensor-based policy to traverse new overhead constraints more successfully or efficiently at a
matched measured acquisition cost?

The goal is improved use of available capability. A completed checkpoint, an increased contrast
count, or an exported dataset is not the goal itself.

## The story

Knowing how to perform a motion does not specify where or when to use it. Motion2Scene uses what the
robot can actually execute to construct candidate lessons in which motion choices have different
consequences. Complete simulator executions verify those alternatives and teach a policy which legal
commitment to make, or whether to continue neutral motion while retaining later choices. The
scientific test is whether this teaching procedure improves actual traversal beyond development
examples.

One sentence: **Use executable motions to design the experiences that teach perceptive decisions.**

## What the current method owns

**Execution-conditioned construction.** Whole-body execution envelopes, including transitions and
return, screen scene proposals. Positive clearance over finite placement offsets and nominal
interference of an alternative concentrate physical labeling on candidate contrasts. The screen
proposes; scene-present physical execution determines outcomes.

**Complete-continuation teaching.** Matched branch histories link each immediate legal action to its
measured complete schedules. `WAIT` is valued through available later continuations, not equated with
walking through the entire obstacle. Passage takes priority over measured successful passage time.

**Causal policy learning.** The common phase-specific learner receives the declared sensor/history,
robot-state and legality features. Privileged scene parameters and future outcomes are teaching
information, not runtime policy inputs. Commitment selects a complete supported schedule.

**Supporting mechanisms.** Historical observation-gated replay changes fitting weights, not scene
geometry. The finite information-consistent teacher is a separately implemented development
extension; it is not the teacher used by the ongoing primary acquisition comparison. Neither
mechanism receives credit for a performance benefit until isolated evidence supports it.

## The unresolved link

A scene where adaptation is necessary is not automatically a lesson in selecting among adaptations. A
corpus can contain many positive/negative contrasts while one fixed schedule still solves all of it.
Useful decision coverage may concern either passage or real passage-time differences. Merely
balancing action labels is not the objective.

Likewise, a scene-wise successful schedule is not automatically selectable from available
information. Causal inputs prevent future leakage but do not create missing sensor evidence. The
target is physically feasible, decision-relevant, informationally usable supervision — not maximal
difficulty or maximal collision frequency.

## The evidence that completes the story

The core result must connect construction to actual policy performance on a common evaluation and
then on protected held-out layouts, with comparable acquisition costs and the same relevant execution
interface. Report individual corpus variation, bank capability, actual policy passage, and matched
successful passage-time comparisons. Teacher-table lookups remain distinct from newly executed
policies.

Construction yield explains a mechanism; it is not a substitute for the downstream result. A common
teacher across construction arms does not isolate the teacher's incremental benefit. Comparisons
should only support the components they actually vary.

A strong script is a legitimate comparator. Do not weaken it or redesign evaluation layouts to
manufacture its failure. If replay adds no value, simplify the method. If the central collection
hypothesis is unsupported, test a focused, separately versioned intervention rather than rewrite the
null result as success.

## Scope boundaries

The current study concerns simulated overhead traversal along a fixed approach, seven qualified
schedules, three legal decision phases, and at most one adaptation followed by recovery. It does not
establish general navigation, steering, protective stopping, repeated skill chaining, hardware
transfer, or energy efficiency. Passage time is crossing plus stabilization, with success
additionally conditional on the specified remaining-horizon requirements; it is not full
maneuver-completion time.

The motion prior and tracker supply inherited capabilities. Motion2Scene supplies a
collection-and-teaching procedure for using those capabilities. Reusable branch-linked data is a
companion asset; new-learner or new-robot utility requires additional evidence.

## Rule for staying on track

Keep the research question stable, permit evidence-driven method revisions, and keep performance
claims conditional on completed measurements. Judge each proposed activity by the link it tests:
executable alternatives, useful training decisions, learned behavior, or held-out benefit. Do not let
a checkpoint name, a replay module, or dataset size replace the central question.

---

*Basis: the supplied September 2026 manuscript and the persistent research state updated
September 10, 2026, 13:28 UTC. Later execution status must be taken from later records, not inferred
from this document.*
