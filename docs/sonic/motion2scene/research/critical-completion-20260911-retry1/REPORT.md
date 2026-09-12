# Critical-location learning with masked completion and contrastive objectives

The implemented improvement is a distribution over multiple obstacle locations, trained from geometric response sets rather than a single preferred schedule. It substantially concentrates probability on critical locations in this one-bank interpolation test. However, most of that benefit is also available to an unconditional model. Contrastive ranking improves the collision-related tradeoff; masked completion does not improve the primary metric in this experiment. The full combined model is therefore not promoted as the best method.

This is a new exploratory offline study following the first LfLH engineering benchmark. It is not a modification of the registered physical pilot, an original preregistration, a source-transfer result, or evidence of improved physical passage. All previous pilot and protected-layout gates remain outside this work.

## What changed

The previous generator mapped a motion profile to one diagonal Gaussian over beam position and height. It could make a target motion clear without generating a meaningful distinction between schedules. Its simplified reconstruction decoder also asked for one winning schedule even when several motions could be valid.

The new `CriticalNet` maps an archived motion profile and spatial coordinates to an energy field over **32 × 48 beam placements**. A categorical distribution derived from that field can represent several disconnected placement regions. Two auxiliary output channels predict normalized geometric distance fields for masked completion. All five trained variants use the same 27,715-parameter architecture and the same training geometry; only conditioning or loss terms differ.

For this test, a “critical location” means a beam placement where the target capsule trajectory has a conservative positive clearance bound and at least one alternative capsule trajectory has a sampled penetration witness. It is a finite-bank, geometry-specific definition. The model does not learn critical times, dynamic obstacles, realistic sensor inputs, or a whole navigation policy yet.

The data are the seven previously qualified executed motions, with capsule geometry evaluated at all 298 recorded frames. The beam’s position and underside vary; its orientation and dimensions remain fixed. A predetermined pattern of 4 × 4 coordinate blocks holds out 384 of 1,536 locations. These are withheld **coordinates within the same bank**, not independent held-out motion sources.

## How the image-learning ideas were adapted

Masked autoencoders learn representations by reconstructing hidden portions of an input. That motivates hiding spatial geometric fields and recovering them from motion and visible context. It does not imply that image reconstruction losses provide robot collision guarantees. [He et al., *Masked Autoencoders Are Scalable Vision Learners*](https://arxiv.org/abs/2111.06377).

RePaint conditions generation on the known image region during diffusion sampling. Here the useful analogy is to explicitly distinguish observed and missing parts of a scene representation. The implementation is a small masked-completion network, **not** a diffusion model or a reproduction of RePaint. The main evaluation supplies no observed field values, so completion cannot obtain credit for a partially supplied answer. [Lugmayr et al., *RePaint*](https://arxiv.org/abs/2201.09865).

Contrastive learning motivates distinguishing compatible from incompatible motion–scene pairs. Multiple compatible schedules must remain positives, and uncertain pairs must not become negatives just because they differ from the selected schedule. The implemented term is a geometry-supervised pairwise ranking loss, rather than the original supervised-contrastive embedding objective. [Khosla et al., *Supervised Contrastive Learning*](https://arxiv.org/abs/2004.11362).

Separating consequential locations from the rest of scene generation is also informed by LfH-CP. Its critical-point factorization motivates this research direction; its dynamic-navigation claims are not inherited by this fixed-beam humanoid model. [Ghani et al., *Learning from Hallucinating Critical Points*](https://arxiv.org/html/2509.26513v1).

## The learning objectives

Let \(a\) index an archived motion and \(e\) a beam placement. Let \(\ell_a(e)\) be the spatially corrected lower bound on capsule clearance at recorded frames and \(u_a(e)\) the minimum sampled distance without that correction. With the fixed geometric margin \(m=0.01\) m, define

\[
P(e)=\{a:\ell_a(e)\ge m\},\qquad
N(e)=\{a:u_a(e)\le-m\}.
\]

The remaining pairs are uncertain. A negative lower bound alone never establishes penetration. Even a capsule penetration witness concerns the represented capsule model, not necessarily the native robot collision mesh or an obstacle-present rollout.

For each target, define its geometric critical set

\[
C_a=\{e:a\in P(e),\ N(e)\ne\varnothing\}.
\]

The network produces scores \(s_\theta(a,e)\), normalized over a declared coordinate domain:

\[
q_\theta(e\mid a)=\operatorname{softmax}_{e}[s_\theta(a,e)].
\]

### Coverage loss

On training locations only, the base objective covers all available critical cells:

\[
\mathcal L_{\rm cover}
=-\frac{1}{|A_+|}\sum_{a\in A_+}
\frac{1}{|C_a^{\rm train}|}
\sum_{e\in C_a^{\rm train}}\log q_\theta(e\mid a),
\]

where \(A_+\) contains targets with at least one critical training cell. This is cross-entropy against a uniform distribution on the available positive set. It discourages concentrating all mass on just one member of that set. It is a chosen distributional preference, not a theorem that uniform coverage maximizes learning utility.

A target with no positive cells contributes no fabricated positive label. It remains present in the evaluation and denominator. In this bank, neutral walking has **zero critical cells in both training and withheld coordinates**. Thus a contrast-only objective cannot supply a walking-specific critical example here. A future curriculum needs separately defined all-clear/background contexts and an appropriate action-cost or tie policy to teach when adaptation is unnecessary.

### Contrastive ranking loss

For each training placement with clear and penetrating motions, use

\[
\mathcal L_{\rm rank}
=\operatorname{mean}_{e,\ a\in P(e),\ b\in N(e)}
\log\left(1+\exp(s_\theta(b,e)-s_\theta(a,e))\right).
\]

The term raises the score of geometrically compatible motions relative to witnessed incompatible motions at the same location. Multiple clear motions are not forced to compete. Unknown pairs receive no negative label. The source contains tests for these distinctions and for gradient direction.

This term can reduce probability on dangerous parts of the geometric field, but a predicted score is not a certified lower bound. Its physical significance still needs obstacle-present labels. It also cannot resolve a student’s perceptual ambiguity because the current model uses an archived motion profile, not a deployed sensor history.

### Masked-field completion loss

Training creates a block mask revealing 25% of training blocks on average. The input contains only allowed training values, with a separate mask channel; withheld coordinate values are explicitly blanked. The network reconstructs the missing training values in two distance channels, clipped to ±0.2 m and normalized by 0.2 m. The auxiliary loss is Smooth L1 on the missing training entries.

The base distribution loss always uses **zero observed field input**. The auxiliary completion pass is separate. This matches the primary inference condition and prevents a completion model from relying exclusively on supplied geometric labels. The total objective is

\[
\mathcal L=\mathcal L_{\rm cover}
+0.3\,\mathbf1_{\rm contrast}\mathcal L_{\rm rank}
+0.3\,\mathbf1_{\rm masked}\mathcal L_{\rm completion}.
\]

Weights, masks, coordinate resolution, split, seeds and update counts were fixed before the fits. The only technical amendment was a dtype correction before any successful update. No weight sweep or favorable-checkpoint selection followed the results.

## The completed experiment

Two initialization seeds, 301 and 302, trained each of five variants for 200 updates: unconditional coverage, conditioned coverage, coverage plus masked completion, coverage plus ranking, and both auxiliary losses. All ten final checkpoints were locked before scoring. The evaluator also rotates motion inputs as a diagnostic and includes uniform, geometry-screened uniform, a critical-set oracle, and the previous Gaussian CNN.

All learned-model primary scores use **no revealed map values**. The score is the exact probability mass on critical cells among the 384 withheld coordinates, averaged over all seven targets. It is not an empirical passage rate. Probability is normalized on that fixed coordinate-domain restriction, chosen before its labels; this evaluates the model on a requested region of the map. The old Gaussian is density-projected onto the same finite grid and normalized there, rather than rerun as continuous samples.

| Method | Critical-location mass | Certified-clear capsule mass | Sampled penetration-witness mass |
|---|---:|---:|---:|
| Uniform grid | 1.60% | 38.36% | 59.23% |
| Geometry-screened uniform* | 3.96% | 100.00% | 0.00% |
| Previous Gaussian CNN† | <0.01% | 100.00% | <0.01% |
| Unconditional coverage model | 44.44% | 46.82% | 38.62% |
| Conditioned coverage model | **46.18%** | 50.07% | 33.79% |
| Coverage + masked completion | 45.93% | 51.94% | 31.74% |
| Coverage + contrastive ranking | 45.57% | **61.30%** | **22.32%** |
| Both auxiliary losses | 42.60% | 56.14% | 22.70% |
| Critical-set oracle* | 85.71% | 85.71% | 0.00% |

Learned-method entries are descriptive means of two fits. The remaining probability for a non-abstaining method lies in uncertain geometry. No confidence intervals or p-values are claimed. Seven schedules, shared coordinates and two initializations do not provide independent source replications.

\* The geometry-screened and critical-set oracles inspect withheld geometric labels when constructing their distributions. They have additional inference-time query access. They are geometric benchmarks, **not** the missing contemporaneous physical Screened Uniform comparator. The critical oracle abstains on neutral walking, so its all-target mass is 6/7 rather than 1.0; that target is not omitted.

† The previous CNN had different architecture, data supervision, objective and training duration. This is a package comparison, not evidence that a single loss term caused the difference. Every new trained arm has the same geometry supervision and architecture, making the within-new-model loss comparison more informative.

![Probability tradeoff across the five trained variants](comparison.png)

Dots show both initialization seeds; bars are descriptive means, not uncertainty intervals. [All summary rows](summary.csv) and [all per-target results](per-target.csv) retain both seeds, rotated inputs, unsupported targets and all baselines.

## What the ablations actually say

**Coverage is learnable in this finite bank.** The conditional map concentrates much more probability on critical cells than uniform sampling. However, the unconditional map is only 1.74 percentage points behind on average and outperforms it at seed 301. A large improvement over random placement therefore does not establish a strong motion-conditioning contribution. Much of this bank’s critical region may be common across targets.

**Contrastive ranking offers a useful geometric tradeoff.** Relative to coverage alone, witnessed-penetration mass falls by 11.47 percentage points while primary critical mass decreases by 0.62 points. This is consistent with the loss suppressing incompatible motion–scene pairs. The comparison does not establish equal critical yield or non-inferiority; no margin or suitable uncertainty bound was registered. A 22.32% penetration-witness mass is still substantial and cannot justify executing unfiltered proposals or making safety claims.

**Masked completion does not improve the primary score here.** Its average is 0.25 points below coverage alone. The combined method is 3.58 points lower and varies considerably across seeds: 38.34% versus 46.87% critical mass. This experiment supplies no basis to claim that adding inpainting helps the generator. It also does not prove inpainting has no value with more independent data or a different representation.

**Input information has a detectable but limited effect.** Rotating motion profiles reduces conditioned scores for both seeds of the conditional models. That is an input-dependence diagnostic, not a transfer result. The competitive retrained unconditional baseline remains the stronger warning against interpreting that dependence as a large useful contribution.

**The neutral case exposes a curriculum limitation.** Its lack of critical cells must not be “fixed” by replacing the target or redefining success after observation. The next objective design needs to value useful non-contrast backgrounds alongside adaptations. Binary geometric contrast alone cannot provide every decision a traversal policy needs.

## Critical-location maps

[View all seven targets](critical-maps.png). The map artifact shows ground-truth capsule critical cells, coverage probabilities and contrastive probabilities. White blocks are training coordinates, omitted from this visualization of the withheld domain. Color bars are labeled separately; no favorable target subset is selected. The full variant comparison remains in the tables even though the map illustration focuses on coverage and ranking.

The learned locations are beam station and underside. Their relation to a critical *time* is implicit in where each recorded motion occupies space. An explicit event-time model, multi-obstacle inpainting, and sequential decision policy are not implemented by these maps.

## Next design, without assuming an inpainting benefit

The most defensible candidate for further investigation is **coverage plus contrastive ranking**, retained alongside coverage-only and unconditional controls. That preference is exploratory and based on the measured tradeoff, not a passed adoption gate. Keep masked completion as an ablation until it earns its extra compute.

A useful next distinction is between recovering a smooth distance image and identifying a decision boundary. Smooth L1 can reduce average field error without preserving a narrow critical band. A proposed follow-up would compare masked regression against a boundary-weighted or ternary compatibility reconstruction objective, with fixed weights chosen before a new experiment. This explanation is not yet demonstrated, and the current experiment did not isolate reconstruction accuracy as its cause.

For richer scene inpainting, represent obstacles already fixed by a scene prefix as immutable context and generate only the missing obstacle parameters. Preserve known geometry exactly; do not let a reconstruction network alter a supplied obstacle to make the example easier. Separate free-space constraints, forbidden body contact, and allowed support contact. A diffusion or flow model should only be introduced once independent motion sources and genuinely multimodal scene structure justify it.

Before another scaling claim, build an ancestry-qualified motion split. Increasing coordinates around this one bank does not provide new motion ancestry. Test frozen models on new executed source families, retain every failure, and compare against analytic search given the same information and counted geometry budget. Evaluate corpus response complementarity and causal observability separately from location-map entropy.

Finally, a new physical utility comparison still requires the earlier pilot to be reconciled and the applicable gates resolved, followed by a separately registered budget and implementation lock. None of the geometry labels here replaces contact-qualified passage, continuation teaching, or the contemporaneous Screened Uniform comparison. The frozen controller, physical scorer, reserved layouts, and existing acquisition assignments were not changed.

## Evidence and implementation

- [Original design](../critical-completion-20260911/design.json) and [recorded technical failure](../critical-completion-20260911/technical-failure.json).
- [Technical amendment and unchanged scientific configuration](design.json): float64 geometry targets converted to float32 network inputs after the first forward failed with zero updates. Geometry bounds and table bytes remain unchanged.
- [Geometry receipt](geometry-receipt.json): 1,536 placements × seven motions, 10,752 motion-placement pairs; 10.7803 seconds measured preparation-loop wall time. Geometry was reused, not queried again for the retry.
- [Training receipts](training-results.json): ten completed fits, 2,000 updates, 16.5362 seconds measured successful training-loop wall time. The original failed initialization has unmeasured terminal wall time and is retained separately.
- [Evaluation lock](evaluation-lock.json), [results](evaluation-results.json), [probabilities](probabilities.pt) and [per-target table](per-target.csv): no inference-time geometry supplied to learned models.
- `scripts/research/lflh_next/critical.py` and `critical_evaluate.py`; seven new regression tests in `decoupled_wbc/tests/test_motion2scene_critical_learning.py`.
- **81 tests pass** across the new objective/dtype/leakage tests, earlier geometry tests, and impacted support/passage tests. Black passes; Ruff passes with only E402 excluded for the standalone scripts’ path bootstrap. Tools were installed through isolated `uvx`, without changing the simulation interpreter’s packages.

All measurements are local CPU computation. There were **zero new simulator steps, zero physical labels, and zero GPU training runs**. The artifact manifest binds this report, data, code, checkpoints, failed-attempt records and outputs. Hashes establish current integrity, not an earlier preregistration date.
