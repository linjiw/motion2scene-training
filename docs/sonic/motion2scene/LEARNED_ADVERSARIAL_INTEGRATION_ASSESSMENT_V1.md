# Learned and adversarial acquisition: integration decision

Assessment date: 2026-09-08. **Keep analytic construction in the fixed primary
comparison. The smallest justified neural follow-up is a bounded test of the
existing execution-conditioned beam proposal as an initializer for geometric
search.** Its inclusion must depend on acquisition cost, coverage, or downstream
benefit against a strong analytic comparator. The present artifacts establish
neither a neural acquisition advantage for controller-executed motions nor a
learned/adversarial generator that improves humanoid traversal.

This is a source review and proposed follow-up, not a new registered experiment
or an amendment to ongoing acquisition. No primary acquisition or reserved
evaluation outcomes were read for this assessment. No inference, fitting,
physics, or runtime changes were performed. Historical development results below
remain separate from the current seven-schedule comparison. The working
[manuscript](submission/traversal_method_v2.tex) remains authoritative; older
design documents describe proposals as well as implementations.

## What exists, and what it actually optimizes

The repository contains several different learned scene mechanisms. They should
not be presented as one trained LfLH system.

| Implementation | Actual input and output | Objective and status |
| --- | --- | --- |
| [First Gaussian hallucinator](../../gear_sonic/dataset_generation/hallucination/learned_hallucinator.py) | Motion-envelope input; two obstacle parameters | Retained reproducibility artifact. The [retraction](../hallucination/REPORT_LFLH_COMPARISON.md) documents answer-dependent parameterization, ineffective conditioning, a variance-floor artifact, and an inconsistent occupancy metric. Its headline comparison is withdrawn. |
| [Multi-obstacle hallucinator](../../gear_sonic/dataset_generation/hallucination/lflh.py) | A pair of nominal and target up/left/right extent profiles, shape `(B,2,3,S)`; means and diagonal log-scales for `K` boxes, each with station, lateral position, vertical center and three half-extents | Shared temporal convolutions, mean/max pooling, pair/difference features and an MLP. A fixed soft motion-choice decoder supplies target cross-entropy; additional terms cover relaxed-scene preference, minimality, KL, shape and obstacle repulsion. This is an engineered finite-library choice objective, not reconstruction through SONIC. The report's later claimed conditioning/plausibility gains were also [withdrawn](../hallucination/REPORT_LFLH_MULTI_OBSTACLE.md). |
| [SDF replacement and trainer](../../scripts/research/hallucination/train_lflh_sdf.py) | Same paired extent encoder; rebuilt **reference** candidate clouds and route transform in the decoder; multiple boxes | Target-choice cross-entropy plus a 0.05 m cloud-clearance hinge, Gaussian KL and annealed shape penalty. It replaces the defective directional decoder, but does not reproduce the full original LfLH loss. This trainer omits the earlier relaxed-scene and repulsion terms. |
| [Motion-only event proposal](../../gear_sonic/dataset_generation/hallucination/motion2scene_events.py) | Target capsule midpoint/axis/radius, route-relative XY, absolute height, normalized progress and time; 64 encoded samples, eight local route anchors; station/underside proposals | The actual width-32 model uses a normalized residual temporal block and eight correlated Gaussian components. No alternative enters its encoder; target/neutral references enter the geometric training loss. The [source-phase study](SOURCE_PHASE_V1_RESULT.md) and [search-distillation study](DISTILLATION_V1_RESULT.md) are reference-only development experiments. |
| [Execution-conditioned proposal](../../gear_sonic/dataset_generation/hallucination/motion2scene_execution_proposal.py) | Achieved positive and negative envelope summaries, actual approach history and sensor configuration; a four-component density over station and underside | A 64-unit tanh MLP fits accepted analytic witnesses by equal-context negative log-likelihood. This is analytic distillation, not self-supervised reconstruction through a planner and not a student-regret objective. The completed v3 pilot has no new physical labels. |
| [Current construction queues](../../gear_sonic/dataset_generation/hallucination/motion2scene_acquisition_pool.py) and [replay curriculum](../../gear_sonic/dataset_generation/hallucination/motion2scene_timed_schedule_curriculum.py) | Fixed geometric candidates and qualified executed alternatives; later, actual physical teacher/student branches and sensor-history receipts | Geometry queues are frozen before outcomes. Analytic and observation-curriculum arms share their candidate order. The intervention weights replay of acquired encounters; it does not learn obstacle coordinates, mutate layouts, or predict visibility from ground-truth scene parameters. |

There are two important implementation boundaries in the older LfLH code. The
multi-obstacle training function contains an obsolete comment saying it sees only
the observed motion; the executable `profiles` construction stacks **both** nominal
and target profiles. Also, [the SDF decoder](../../gear_sonic/dataset_generation/hallucination/sdf_decoder.py)
samples capsule axes and time and uses a soft weighted minimum. That is a useful
geometric surrogate, not exact continuous capsule clearance. Subsampling can
overestimate clearance, so its positive distances cannot certify separation.
The world conversion rounds station to an integer index, preventing a useful
gradient through station selection. These are reasons to retain the current
independent native-geometry screen instead of transplanting that old trainer.

The reference event model's implemented loss is

\[
L=\sum_{a=1}^{8}\pi_a\left[
L_{\mathrm{choice}}+5\left(\frac{[m-c^+]_+}{m}\right)^2
+5\left(\frac{[m+c^-]_+}{m}\right)^2\right],\qquad m=0.01\,\mathrm m.
\]

Here the hardest target and easiest upright reference across supplied offsets
define the margins. The [training driver](../../scripts/research/motion2scene_event_scaling.py)
samples one scene per anchor and checks nominal plus four sampled corners per
update. The fixed upright/crouch preference costs are geometric modeling choices;
they are not measured traversal time, work, or energy. Later [distillation](DISTILLATION_V1_RESULT.md)
adds a weighted set-matching term from searched reference witnesses.

For the executed proposal, the exact conditioning contract is

\[
z=[e^+_{16\times6},e^-_{16\times6},\operatorname{vec}(h_{1:15}),\nu],
\quad q\sim g_\phi(q\mid z),\quad q=(s,h_{\mathrm{under}}).
\]

Each envelope bin holds whole-body XYZ minima and maxima from achieved capsule
trajectories, covering entry, maintenance and recovery. In the
[v3 driver](../../scripts/research/motion2scene_solution_curriculum.py), history
contains joint positions/velocities, root position/quaternion and applied joint
actions. Sensor configuration contains mounted origin, ray directions and range;
it is **not an observed obstacle history**. The saved conditioning vector has
1,642 entries. The four-component correlated Gaussian lives in logit coordinates,
mapped by sigmoid to `s in (0.1,0.9)` and underside in `(1.1,1.45)` metres.
Its loss is

\[
L_{\mathrm{NLL}}=-\frac1N\sum_i\frac1{|A_i|}\sum_{q\in A_i}
\log p_\phi\!\left(\operatorname{logit}
\frac{q-q_{\min}}{q_{\max}-q_{\min}}\mid z_i\right).
\]

The omitted physical-coordinate Jacobian is independent of the fitted parameters
for these fixed targets/bounds; the logged value is a latent-coordinate fitting
diagnostic. There is no student input, sensor-timing loss, physics gradient or
adversarial update. Three contexts supply 29 geometric witnesses, with equal
context weight despite witness counts of 10, 1 and 18. Conditioning fields being
present does not establish useful generalization across those fields.

## What the measurements support

| Development evidence | Result and denominator | Permitted interpretation |
| --- | --- | --- |
| [Corrected SDF archive](../hallucination/lflh_sdf.json), 11 training and five excluded clips | On 80 excluded-clip draws, target selection is 23/80 and positive cloud clearance is 15/80; random-prior values are 12/80 and 0/80 | Some surrogate improvement over a weak prior. Selection and clearance are separate marginals; the archive does not provide their conjunction. The clearance rate tests `>0`, not the training margin of 0.05 m. No strong analytic or downstream comparison. |
| [Reference search distillation](DISTILLATION_V1_RESULT.md), eight previously observed evaluation source groups | Hybrid raw 334/384; original raw 292/384; matched-query extra-training control 310/384. Hybrid five-evaluation search 377/384 versus original seventeen-evaluation search 384/384 | Learned witness distillation can improve pooled reference proposal yield. It loses on one source versus the query control, and reduced search loses seven outputs. Neither source-wise preservation nor coverage criterion passes. |
| [Global analytic search](evidence/analytic-global.json), 16 reference cases | 128/128 accepted requests, but only 84 distinct within-case placements; mean reference-map station coverage 98.96% versus 37.33% for learned pattern search | A strong analytic baseline exists; repeats must not be treated as new data. Previously inspected reference maps make this development evidence. |
| [Distinct analytic selection](evidence/analytic-distinct.json), same 16 cases | 127/128 accepted distinct outputs at the same 6,884 queries per job; the registered 128/128 prediction fails | Distinctness removes padding but retains one actual rejection. Do not round this into perfect acceptance. |
| [Executed-context proposal v3](../../../research-data/groot-wbc/m2s-solution-curriculum-v3/result.json), three fitting contexts | Analytic local resampling 123/192; learned 41/192; uniform 0/192 | The strongest directly matched comparator beats the neural sampler. No obstacle-present execution, observation-timing check, new-context transfer or downstream training occurs in this pilot. |

The last comparison uses a revised **113-offset** screen: the positive alternative
must retain 10 mm clearance at all offsets, while the negative need only interfere
by 10 mm nominally. Of 4,140 grid proposals, 29 pass this contract; four also pass
the earlier all-offset contrast contract. The current acquisition pool instead
uses its separately fixed **81-offset** native outer-positive/inner-negative
screen. Results under these contracts cannot be merged into a common robustness
rate.

For the current seven-schedule family, the implemented analytic comparator
searches all distinct positive/negative option pairs within each predeclared
candidate. It maximizes the smaller of robust-positive and nominal-negative
margin slack, with deterministic option/index ties; the acquisition plan then
applies its shared coverage rule. Thus it is stronger than a target-only fit or
random rejection sampler. This describes its algorithm, without inspecting or
asserting any current primary result.

The v3 nonlearned sampler chooses accepted analytic witnesses uniformly and adds
jitter of ±0.02 normalized route progress and ±0.0025 m underside. Accepted-bin
counts by context are 11/2/20 for analytic resampling and 6/5/14 for learned
sampling. The learned arm covers more coarse bins on the context with only one
analytic witness, but fewer in the other two. Bins are **0.02 normalized progress
by 0.01 m height**; a legacy result key incorrectly calls the first dimension
`2cm_station`. These are occupied bins, not a measured fraction of the feasible
scene population.

Cost also favors retaining the analytic baseline. The v3 witness search costs
28,620 clearance queries and 8.727 s; fitting adds 0.919 s. Each sampling arm then
uses 43,392 audit queries. Proposal computation is about 0.000264 s for analytic
resampling and 0.001974 s for the neural sampler; auditing costs 12.553 and
12.698 s, respectively. These tiny proposal timings are local measurements, not
deployment latency claims. A learned method must charge its teacher search and
fitting as well as inference. In the older binary nominal experiment, recorded
analytic versus learned construction loops likewise cost 9.420 versus 15.984 s
and 13,872 versus 27,744 search queries; that [historical cost table](submission/paper.tex)
belongs to a different apparatus and is not a complete end-to-end comparison.

Thus, **there is limited positive evidence for learning reference proposal
distributions, but no current evidence for neural advantage over the strongest
matched executed-motion constructor, or for improved humanoid traversal caused
by a learned generator**. The retracted results provide no support in either
direction about LfLH itself.

## What to borrow from the literature

LfH uses motion collected in open space to synthesize constrained perceptual
training examples. This is already prior art for motion-derived training scenes;
its runtime includes separate classical checks.
[LfH, §§III–IV](https://arxiv.org/html/2007.14479v4).

LfLH learns an inverse scene distribution through a fixed differentiable planner,
combining plan reconstruction, obstacle priors and collision regularization.
Its generated scenes are filtered before training a sensor-driven local planner.
For this project, the useful lesson is the separation of proposal learning,
verification and downstream training. Replacing Ego-Planner with a finite
whole-body option scorer changes both the feasible set and the optimality claim;
it is an adaptation of the idea, not a reproduction of their system.
[LfLH, §§III-B/C and IV-A.1](https://arxiv.org/html/2108.09793v1).

LfH-CP separates learned critical obstacle configurations from procedural dynamic
trajectories that pass through them while avoiding the plan. Its temporal
presence mechanism is useful inspiration for identifying relevant decision
events. A static overhead passage cannot disappear outside one critical frame:
all objects must be checked jointly through the complete humanoid execution.
[LfH-CP, §§III-D/E](https://arxiv.org/html/2509.26513v1).

PAIRED motivates environments with the return gap between protagonist and
antagonist; ACCEL combines regret with editing existing levels. Neither supports
a claim that raw classification loss identifies useful humanoid scenes or that
our fixed queues implement adaptive environment generation.
[PAIRED](https://arxiv.org/abs/2012.02096),
[ACCEL](https://proceedings.mlr.press/v162/parker-holder22a.html).

These primary sources were rechecked for this assessment. This is a focused
mechanism comparison, not an exhaustive novelty survey or an independent
reproduction of their experiments.

## Smallest next intervention: conditional proposals for bounded search

**Proposed only, after the fixed primary comparison.** Reuse
`ExecutionConditionedProposal` as a two-dimensional initializer for one static
beam. Keep the controller, four reference files, seven qualified schedules,
native geometry, sensor renderer/history, legality and outcome definitions fixed.
The deployed student remains the existing 114-feature sensor/state policy; the
generator's achieved-motion summaries and scene-design metadata never enter it.

The change is confined to how station/underside candidates are proposed. Other
beam dimensions and pose components are fixed by a newly declared development
context and appended to generator conditioning alongside schedule entry/return
metadata. Positive/negative schedules are assigned before search, from the
qualified bank. Use the same common early approach history and the schedule's
recorded complete executed envelope; late entry is not a license to invent an
unrecorded approach state. The existing head already accepts an explicit
conditioning dimension, so this requires a new offline driver and model artifact,
not a new controller or generative architecture. Two-beam density learning should
wait: independent single-beam samples need joint validation and do not constitute
a learned course distribution.

| Proposed experiment component | Required controlled design |
| --- | --- |
| Fitting | Reuse the four-component, 64-unit model and 400-step recipe initially. Fit only independently checked analytic witnesses; retain rejected proposals and actual teacher-search cost. Freeze three fitting seeds, source/context splits, normalization and witness deduplication before fitting. |
| Generalization unit | Exclude complete declared beam-dimension/pose contexts from fitting, not random draws from the same witness set. The present bank has development source-41002 ancestry; context transfer alone is not new motion-source transfer. If only the three old fitted contexts are used, call the test in-context amortization. |
| Strong comparisons | Domain-uniform initialization; event-aware analytic station/height search with distinct outputs; local resampling of the same analytic witnesses; learned initialization. Give corrected arms the same bounded refinement, margins, offsets and requested distinct-output count. No weak uniform-only headline. |
| Query accounting | Freeze candidate budgets and all random streams before queries. Count preprocessing, every candidate/offset/option clearance, failed refinements, independent audits, fitting and teacher acquisition. Report curves against both total query count and measured CPU time; an evaluation of all seven alternatives costs more than one pair. |
| Geometry endpoint | Distinct accepted candidates per requested slot and per total acquisition budget, plus per-context coverage and both margin violations. Constant/shuffled-input controls must keep the verifier's original motion pair unchanged. A fit-loss decrease is not a result. |
| Physical promotion | Only after the geometric comparison is useful, register a fresh matched teacher/student acquisition at equal physical-step budgets with actual 65-ray observations. Geometric admission is a proposal screen; physically failed and technically unknown attempts retain their respective accounting. |

Do not remove independent rejection even if likelihood improves. Do not select
only easy contexts after seeing outcomes. If analytic resampling/search remains
as good or better after complete cost and coverage accounting, keep the neural
component out of the paper's central method. A geometry-only win can justify an
acquisition-acceleration component; a traversal claim additionally needs the same
downstream learner trained and executed on the resulting corpora. Any later
confirmatory comparison needs a separately registered unseen panel; the current
reserved panel must not become a tuning set for this follow-up.

## Smallest later adversarial extension, if the proposer earns its place

The current replay score already supplies a conservative target for a later
extension. It assigns gap 1 to an admitted student failure when a complete legal
teacher succeeds, or

\[
g=\max\left(0,\frac{T_{\rm student}-T_{\rm teacher}}
{\max(T_{\rm student},T_{\rm teacher},10^{-12})}\right)
\]

when both pass. Source, outcome, exact history, feature and legality identities
must match; current-phase delivered sensor cues must meet the declared deadline.
This cue test measures availability, not semantic distinguishability or a newly
measured transition deadline. Missing evidence yields an unavailable gap, not a
hard-example score. The implemented replay mixture is 0.2 uniform, 0.2 coverage,
0.6 verified gap; when no positive gap exists, the last component becomes uniform,
giving **0.8 uniform plus 0.2 coverage**. Historical gaps retain their generating
model identity and age; they are not all errors of the latest fitted student.

A minimal **new** adversarial acquisition mechanism would fit the same proposal
to previously measured witnesses with these declared weights, then sample new
nearby station/underside candidates, preserve a robust geometric positive option,
and physically verify the selected scenes. Compare it with **identically weighted
analytic witness resampling and the same refinement**. This isolates the learned
density from the usefulness of the gap weights. A nonlearned local mutation arm
can establish whether adaptive geometry helps before introducing neural fitting.

Every new scene must receive its own pre-update student run and matched legal
teacher branches. New sensor acquisition and teacher failures count in the
budget. The current group's labels cannot train the model used for its student
run; stale parent-scene gaps cannot be reported as the new scene's measured gap.
Optimizing the existing differentiable geometric score alone is not adversarial
student training. Weighting already acquired scenes without generating new
layouts remains replay. Neither variant expands the controller's available
capability, enables repeated adaptations, or adds route planning.

The inclusion test is therefore specific: **does this small additional acquisition
component improve held-out closed-loop completion or measured successful traversal
time at the same total acquisition budget, beyond strong analytic construction
and the same replay rule?** Until that evidence exists, use “execution-aware
analytic construction with verified replay” for the implemented method and keep
“learned/adversarial proposal” in the separately labeled follow-up.

## Evidence identities checked for this review

These hashes identify historical result files read directly; the surrounding
study registrations and original failures remain unchanged.

| Artifact | SHA-256 |
| --- | --- |
| `m2s-solution-curriculum-v3/result.json` | `132a0e370e05ae132fb812df55793fd1881cd45ac9582b484baed602f62761b3` |
| `docs/hallucination/lflh_sdf.json` | `b67491b2292d5ecee93605955cce04b8531ae388299ea318eaed7bdc8f34fb67` |
| `docs/motion2scene/evidence/analytic-global.json` | `3491a79bbe3f48725cd4a46b5e893ff43d88070fd7436ed33f1f25ba1c98e10f` |
| `docs/motion2scene/evidence/analytic-distinct.json` | `f497292055c1a69bc18f5ac86d5d7038147e15975506efb861716abf96061b7b` |
| `docs/motion2scene/evidence/distillation.json` | `d8e53ace9908277513a4eaf4a641952e8052fe80c21afa7600683e63c35020cd` |
