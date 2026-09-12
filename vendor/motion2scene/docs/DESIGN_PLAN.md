# Motion2Scene-LfLH: Design Plan

Status: governing research guidance, normalized from the user-provided plan on 2026-09-04.
The six linked literature references are user-supplied leads and must be checked against the
primary papers before they support novelty or quantitative claims.

## Working identity

Preferred title:

**Motion2Scene: Learning Critical 3D Traversal Scenes from Executable Humanoid Motion**

Method-oriented alternative:

**Learning Humanoid Traversal from Hallucinated Critical Scenes**

The project is not a graphics problem about producing visually realistic rooms. Its robotics
question is:

> Given a physically executable humanoid motion, can we synthesize a diverse distribution of
> realistic scene constraints under which this motion is feasible and meaningfully preferred over
> simpler alternatives, then use those synthesized pairs to learn the forward scene-to-motion
> decision?

The research program is

\[
\text{executable motion}
\rightarrow \text{critical scene distribution}
\rightarrow \text{synthetic scene-motion dataset}
\rightarrow \text{scene-conditioned skill/event policy}
\rightarrow \text{physical execution}.
\]

## 1. LfH and LfLH foundation

A conventional planner computes

\[
p^*=f^*(S,c_0,g),
\]

but paired examples \((S,p^*)\) are expensive in constrained environments. Learning from
Hallucination reverses the arrow: begin with a safe open-space plan and construct a scene for which
that plan is appropriate,

\[
S^*=g(p,c_0,g).
\]

The motion supplies the supervision. LfLH replaces a hand-designed inverse with a learned
distribution,

\[
S\sim g_\psi(S\mid p,c_0,g), \qquad \hat p=d(S,c_0,g),
\]

and learns the hallucinator through reconstruction under a frozen decoder. A second, ordinary
supervised stage learns \(S\rightarrow f_\theta\rightarrow p\) from synthesized pairs.

Here, “self-supervised” means no ground-truth scene or expert scene/action label is required for
each motion. It does not mean the method has no motion bank, physical model, decoder, or prior.
Training a conditional scene generator on existing paired scenes alone is not LfLH.

## 2. Why a direct humanoid port fails

1. **Generated is not executable.** Visually plausible humanoid motion may skate, penetrate the
   floor, violate joint/dynamic limits, use unsupported contacts, or lie outside the tracker’s
   distribution. Input must be an execution-qualified bank.
2. **The inverse is underdetermined.** Ducking may imply a beam, shelf, branch, or nothing. The
   output is \(p(S\mid M)\), not one reconstructed room.
3. **Exact trajectory optimality is too strong.** Several 29-DoF trajectories solve one scene.
   Recover an equivalence class—skill, event location/intensity, phase, route, and speed—not every
   joint angle.
4. **A full differentiable humanoid planner is impractical initially.** Use a fixed,
   differentiable motion-library decoder for inverse learning, exact geometry for verification,
   and MuJoCo/Isaac/SONIC for selected execution tests.
5. **Direct room generation invites mode collapse and steganography.** Predict low-dimensional
   critical constraint tokens, then procedurally sample nuisance scene details.

## 3. Core hypothesis and falsifier

> A successful humanoid motion does not identify a scene, but it may identify a low-dimensional
> family of critical geometric constraints. Learning that distribution and rendering realistic
> nuisance variation can yield useful supervision for scene-conditioned traversal.

A compatible scene only satisfies \(F(M,S)=1\); an empty room often does. A critical scene also
makes a weaker alternative fail:

\[
F(M_i,S)=1, \qquad F(M_{weak},S)=0.
\]

Preferably the target is the least-cost feasible member of a matched family:

\[
M_i=\arg\min_{M\in\mathcal A_i} E(M,S).
\]

The premise is weakened if matched, qualified motions rarely produce nonempty critical geometry
intervals or if target preference disappears under small geometry perturbations.

## 4. Representations

Represent a qualified motion as root/body poses, joints, contacts, velocities, and accelerations.
Canonicalize into route-relative progress \(s\), lateral offset \(n\), height \(z\), and phase
\(\phi\). Derived features include swept body height, left/right width, foot clearance, support
structure, root speed, and body-region envelopes.

A critical token is

\[
k_j=(\tau_j,b_j,s_j,\phi_j,n_j,[\ell_j,u_j],\rho_j),
\]

where the fields encode constraint type, affected body region, route position, phase, normal,
valid geometry interval, and confidence/presence. Examples include an overhead beam binding the
head/torso, a floor obstacle binding a swing foot/shin, or one wall binding a shoulder/arm.

Render scenes as

\[
S=R(K,\nu),
\]

with nuisance variables for thickness, width, material, irrelevant furniture, room form,
lighting, and sensor noise. Keep the critical layer causally controlled and the background layer
realistic but nonbinding.

## 5. System modules

### A. Execution-qualified motion bank

Unify MTC, existing SONIC-tracked motion, successful Scene2Motion repairs, and qualified
ARDY/Kimodo samples in one G1 schema. Track evidence separately:

| Tier | Requirement |
| --- | --- |
| Q0 | finite kinematics and valid joint limits |
| Q1 | no severe self/floor penetration |
| Q2 | support/contact and dynamics gates |
| Q3 | obstacle-absent local tracking completion |
| Q4 | robust tracking over multiple physics seeds |

Only Q3/Q4 records enter the main inverse-training bank. Always report carrier yield
\(N_{qualified}/N_{generated}\) at every tier.

### B. Matched motion ladders

Construct neutral, weaker, target, stronger, and alternative-skill motions sharing start, goal,
root route, duration, speed, phase, and base identity as closely as possible. Controlled edits are
preferable to unrelated prompts. Define “preferred” only within this disclosed library.

### C. Analytic Motion2Scene-LfH baseline

Prove geometric identifiability before learning:

- Duck: choose beam height between target and weaker swept-height profiles with margin.
- Squeeze: choose opening width between target and weaker swept-width profiles with margin.
- Step: choose obstacle height between target and weaker swing-clearance profiles with margin.

These intervals provide target clearance, alternative deficit, interval width, and perturbation
robustness. If most intervals are empty, improve the motion ladders rather than train a network.

### D. Learned critical-token hallucinator

Approximate \(q_\psi(K\mid M,\mathcal A)\). Start with a TCN or small temporal Transformer, one
obstacle family, and a mixture-density output for one or two tokens. Later consider body-region
graphs, set decoding, discrete type/presence variables, and conditional flows.

### E. Frozen differentiable decoder

Use an energy over collision, margin, support, execution risk, and effort:

\[
E(M,S)=\lambda_c\Phi_{collision}+\lambda_m\Phi_{margin}
+\lambda_s\Phi_{support}+\lambda_x\Phi_{execution}+\lambda_e C_{effort}(M).
\]

The fixed decoder selects among candidates with a softmax over negative energy. It may use
capsule/box signed-distance approximations and fixed support/dynamics penalties; it is never
silently retrained to favor the target. Exact geometry rechecks all accepted scenes and physics
labels a smaller execution subset.

## 6. Inverse objective

Use a sum of:

- target-selection reconstruction, \(-\log P_D(M_i\mid S,\mathcal A_i)\);
- target feasibility and safety margin;
- preference gap against the best alternative;
- obstacle minimality, measured by removing each critical obstacle;
- scene realism or grammar support;
- conditional diversity in realistic nuisance dimensions;
- robustness under motion and scene perturbation.

Do not reward raw parameter distance as diversity; report conditional bin coverage, entropy,
pairwise distances, multi-obstacle joint coverage, and distance to real-scene parameter support.

## 7. Canonical self-learning example

For a neutral/shallow/medium/deep crouch ladder with increasing effort, the hallucinator samples a
beam position and height. If the beam is too high, neutral walking wins; if too low, the target
collides; inside the critical interval, weaker motions fail and the target beats the needlessly
stronger motion on fixed effort. The target motion and decoder teach the interval without a paired
real beam.

## 8. Curriculum

1. Scalar overhead adaptation: walk plus crouch-depth ladder; one beam family.
2. Lateral adaptation: neutral, arm tuck, shoulder rotation, and eventually sidestep/sidle;
   doorway and asymmetric gap families.
3. Step-over: controlled height and lead-side ladders, only after reliable phase/contact evidence.
4. Sequential events: beam/gap/box combinations at separate route positions.
5. Real-scene insertion: controlled critical primitives inside diverse scene backgrounds.
6. Dynamic constraints: only after static criticality is stable.

## 9. Forward learner

The first downstream model predicts an interpretable skill/event tuple rather than 29-DoF motion:

\[
(O_S,g,c)\rightarrow(\text{skill},\text{intensity},\text{event position},
\text{lead side},\text{confidence/refuse}).
\]

Retrieval/generation and Scene2Motion then compile that decision into motion for SONIC. A full
scene-conditioned diffusion model is deferred until inverse-generated data improves this simpler
decision under a matched learner and data contract.

## 10. Optional one-round self-expansion

Train \(\pi_0\), deploy it on held-out scenes, verify/repair with Scene2Motion and SONIC, add only
robust successes to the bank, and retrain once. Evaluate bank/scene coverage change, traversal
change, and collapse toward easy skills. One round is enough for the first test.

## 11. Hard problems and safeguards

- **Meaningful optimality:** lowest-cost feasible member of a matched, disclosed family—not global
  optimality.
- **Matched alternatives:** root route, speed, duration, phase, and identity must not confound the
  critical geometry.
- **Scene steganography:** constrain token vocabulary, use a frozen physical decoder, apply removal
  tests and priors, and separate nuisance rendering.
- **Surrogate disagreement:** report
  \(\Pr[F_{surrogate}(M,S)\ne F_{exact}(M,S)]\); filtering is not a substitute for reporting
  pre/post-validation yield.
- **Realism:** combine a learned critical layer with a grammar or unpaired background prior.
- **Geometry versus execution:** compare kinematic, obstacle-absent qualified, and robust
  multi-seed banks.

## 12. Experimental program

### E0 — Motion bank qualification

Measure tier-by-tier yield, tracking completion, support/contact, penetration, joint dynamics,
test-retest behavior, and family coverage. Start with duck, lateral tuck/squeeze, and step-over.
Proceed only when matched, trackable alternatives support meaningful critical intervals.

### E1 — Analytic Motion2Scene-LfH

Compare random placement, compatible-only scenes, a most-constrained swept corridor, and critical
interval sampling. Measure target feasibility, weaker failure, target preference, nonempty
interval rate, minimality, perturbation robustness, and exact-validation yield.

### E2 — Learned Motion2Scene-LfLH

Compare analytic sampling, direct CVAE scene parameters, original-style Gaussian LfLH,
critical-token LfLH, and a conditional-flow token model. Ablate alternatives, preference gap,
minimality, realism, diversity, execution qualification, and token factorization.

### E3 — Forward scene-to-skill learning

With one learner and matched data budgets, compare random procedural, compatible-only, analytic
critical, learned LfLH, learned critical-token, limited real, and real-plus-hallucinated data.
Use motion-, geometry-, composition-, and background-OOD splits. Report skill/intensity/event
accuracy, calibration, false-safe rate, risk–coverage, closed-loop completion/contact, and calls
to generation/repair.

### E4 — Physics verification

Sample by skill, difficulty, criticality margin, scene novelty, and confidence. Run target and
matched alternatives across multiple physics seeds. Keep geometry, obstacle-absent tracking, and
obstacle-present execution as separate evidence tiers.

### E5 — One-round self-expansion

Measure whether one deploy/verify/add/retrain round expands the qualified bank and scene-family
coverage without collapsing to easy skills.

## 13. Dataset record and splits

Group records by base motion, alternative set, and critical token. Store motion provenance and
hashes; Q0–Q4 evidence; route/skill/intensity/phase/envelopes; alternative identities; token
interval; critical and nuisance scene primitives; signed clearance/deficit labels;
removal/jitter counterfactuals; execution outcomes; and code/model/simulator/controller hashes.

Evidence tiers are M (motion), Q (qualification), G (exact geometry), T (obstacle-absent tracking),
and P (obstacle-present physics). Missing tiers are `not_measured`.

Split complete base-motion and critical-scene groups. Never split intensity variants, raw/repaired
versions, nuisance renderings of one token, near-copies of real scenes, or physics seeds across
train/test. Maintain development, motion-OOD, geometry-OOD, composition-OOD,
background/appearance-OOD, and cross-prior splits.

## 14. Relationship to adjacent work

Motion-conditioned scene synthesis work supports the premise that bodies reveal free space,
contact, and object-placement information. Motion2Scene’s intended delta is executable robot
motion, counterfactual traversal constraints rather than aesthetic room reconstruction, and
downstream traversal utility rather than visual plausibility alone. Novelty and source statements
remain hypotheses until the cited primary papers and current literature are checked.

## 15. Repository boundary

Keep this project separate. Pin the Scene2Motion/SONIC repository at a specific commit or extract a
small shared `humanoid_scene_core`; never depend on a moving branch. Reuse robot envelopes,
foot/contact features, exact geometry probes, scene grammar, hashing, and validation contracts.
Do not copy the whole parent project or its research history.

Planned package boundaries:

```text
motion/      canonicalization, qualification, contacts, alternatives
scene/       tokens, grammar, renderer, priors
inverse/     analytic LfH, hallucinator, critical decoder, losses
decoder/     motion energy, soft selector, SDF surrogate
verify/      exact geometry, counterfactuals, execution
forward/     scene encoder, skill policy, retrieval
dataset/     schema, builder, splits, validator
experiments/ E0 through E5
```

## 16. Ten-week sequence

- Weeks 1–2: schemas, canonicalization, matched duck/squeeze ladders, analytic intervals, exact
  target/alternative checks, and the first synthetic pairs. Gate on nonempty robust intervals.
- Weeks 3–4: differentiable capsule/box surrogate, fixed energy/selector, and exact-disagreement
  analysis. Gate on held-out ranking agreement.
- Weeks 5–6: one-obstacle mixture model, direct-CVAE baseline, critical-token model, diversity and
  minimality. Gate on diversity/realism without material criticality loss.
- Weeks 7–8: scene-to-skill/intensity learner with random/LfH/LfLH comparisons and named OOD axes.
  Gate on downstream utility and false-safe behavior.
- Weeks 9–10: balanced obstacle-present SONIC verification, target-versus-alternative execution,
  final dataset, figures, and claim/evidence map.

## 17. Internal go/no-go gates

- **A — Analytic identifiability:** approximately 90–95% target exact-geometry feasibility,
  weaker rejection above random, mostly nonempty robust intervals, and nonredundant obstacles.
- **B — Learned hallucination:** high exact-validation yield, more conditional modes than direct
  baselines, near-analytic criticality, and realistic parameter support.
- **C — Dataset utility:** better named-axis OOD selection/traversal or fewer false-safe launches
  under equal-data/equal-model comparisons; reduced real-pair demand.
- **D — Robotics closure:** obstacle-present traversal, target beating matched weaker alternatives,
  qualified banks beating raw generator output, and benefit on independent scenes.

These are management thresholds, not results or promised paper claims.

## 18. Required ablations

1. Compatible versus critical scenes.
2. Direct scene output versus critical tokens.
3. Kinematic versus execution-qualified motion.
4. Exact trajectory versus skill/intensity reconstruction.
5. Critical-only versus critical plus realistic nuisance.
6. One-shot versus one self-expansion round.

The candidate headline—still a hypothesis—is that execution-qualified, counterfactual-critical
scenes teach a forward traversal policy more effectively than collision-compatible scenes.

## 19. Recommended first decisive experiment

Use 16–32 base walk carriers with 3–5 matched crouch levels. Derive exact G1 swept-height
intervals, sample 20 beam scenes per target level, and hold out carrier IDs and beam-height ranges.
Compare analytic critical LfH against compatible-only, random procedural, and learned LfLH.
Measure target feasibility, weaker failure, fixed-decoder selection, interval width, jitter
robustness, scene diversity, forward skill/intensity prediction, and a small obstacle-present
SONIC subset. This stage should be predominantly CPU/MuJoCo. Add lateral gaps second and
step-over third.

## 20. Claim boundary

Do not claim scene reconstruction from arbitrary motion. The bounded claim to test is:

> Given an executable humanoid motion and matched alternatives, Motion2Scene learns a distribution
> of minimal critical constraints under which that motion is the preferred traversal behavior,
> and uses those constraints as supervision for the forward scene-to-motion problem.

Long-term composition:

\[
\begin{aligned}
&\text{Motion2Scene: } M\rightarrow S_{critical}\\
&\text{Forward learner: } S\rightarrow\text{skill/event}\\
&\text{Scene2Motion: }(S,M)\rightarrow M_{repaired}\\
&\text{SONIC: }M_{repaired}\rightarrow\text{execution}.
\end{aligned}
\]

## User-supplied literature leads

1. Learning from Hallucination: <https://arxiv.org/abs/2007.14479>
2. Learning from Learned Hallucination: <https://arxiv.org/abs/2108.09793>
3. LfH-CP: <https://arxiv.org/html/2509.26513v1>
4. MTC: <https://arxiv.org/abs/2603.05993>
5. MIME: <https://research.adobe.com/publication/mime-human-aware-3d-scene-generation/>
6. SUMMON/POSA-related lead supplied as: <https://arxiv.org/abs/2301.01424>

