# Distillation, flow matching and scene-conditioned humanoid control

The central hypothesis is that reliable task-conditioned behavior requires three components to work together: an expert that can execute the task, student training on states encountered during its own execution, and a public conditioning interface containing enough information to select the required behavior. A more expressive action distribution addresses only part of that problem. This report separates those components and connects them to the implemented experiments.

The bounded experiments are now complete: see the [measured results and next decisions](FLOW_DISTILLATION_RESULTS_20260913.md). They include 64,000 main optimizer updates, 192 smoke updates, 3,052 reference-tracking episodes and 20 independently scored task episodes. Flow did not outperform the current motor foundation at this budget; none of the scene-conditioned students passed a navigation task.

## Relevant research

**BFM.** Behavior Foundation Model uses a CVAE and online DAgger: the current student visits simulator states, the privileged proxy supplies same-state action targets, and action reconstruction is combined with KL regularization. Its low-level interface covers root, body positions and joints. Mask availability moves from full toward Bernoulli(0.5); reference-state initialization, hard-motion sampling and termination remain part of the training recipe. The reported setup uses 8,192 environments and a G1 with frozen wrists. This is substantially different from a small offline token-predictor experiment. [1](https://arxiv.org/html/2509.13780v1)

**OmniXtreme.** The closest action-flow comparison distills motion-specific teachers into a flow policy using same-state queries during student rollout. Its algorithm clears the buffer for each iteration and fits a noised-action velocity field with Beta-distributed flow times; inference uses Euler integration. Subsequent residual PPO freezes the base and learns corrections with more demanding actuation modeling. The base and residual retain different previous-action histories. Our uniform-time, single-teacher, normalized-action pilot is an adaptation, not a reproduction of that system. [2](https://arxiv.org/html/2602.23843v1)

**BeyondMimic.** This work is especially relevant to navigation: its diffusion stage models future state-action trajectories from observation history. Test-time costs can therefore operate on predicted positions, velocities and body clearance, including slowing near a waypoint and avoiding obstacles. This differs from generating only the next 29 action values. Directly adding an obstacle cost to an action-only denoiser does not provide the future robot geometry needed to evaluate that cost. [3](https://arxiv.org/html/2508.08241v1)

**Flow Policy Gradients for Robot Control.** FPO++ studies reward-based training and fine-tuning of expressive flow policies, including humanoid motion tracking. It uses a flow-matching surrogate rather than requiring ordinary action likelihoods. It is an alternative to residual PPO, not evidence that a Gaussian PPO implementation can simply accept a deterministic flow endpoint as if it were a Gaussian mean with an unchanged likelihood model. [4](https://arxiv.org/html/2602.02481v1)

**PhysiFlow.** This paper places flow matching in a motion-generation component coupled to a tracking component. Its relevance is the separation between generated motion and physical tracking, rather than evidence for replacing this project's same-state action predictor. Architectural similarity at the word “flow” is insufficient to establish equivalent supervision or control interfaces. [5](https://arxiv.org/html/2603.05410v1)

**ResMimic.** This work motivates learning task-specific corrections on top of a general tracking policy, particularly for loco-manipulation. It supports investigating a residual adaptation path; it does not establish that a small residual can recover a student that already fails most full-command tracking attempts. [6](https://arxiv.org/abs/2510.05070)

The underlying conditional flow-matching objective learns a continuous transport field using sampled paths between noise and data. That is a training objective for a generative model, not a guarantee that the supplied data cover successful task behavior. [7](https://arxiv.org/abs/2210.02747)

The official OmniXtreme repository currently advertises pretrained checkpoints and sim-to-sim evaluation, with training components described as planned releases. Our experiment uses locally implemented methods and the existing SONIC assets; it does not silently substitute an incompatible released policy or claim to execute the authors' full training recipe. [8](https://github.com/Perkins729/OmniXtreme)

## What the existing experiment actually trained

The completed baseline is a public-prior transformer feeding 64 quantized SONIC tokens into a frozen decoder. It has approximately 5.2 million trainable parameters and was fitted for 40,000 updates, starting with 21 whole-qualified teacher episodes and adding five rounds of student-state queries. A separate MLP control received 10,000 updates. This is a BFM-style interface adaptation with a fixed decoder, rather than the paper's action-decoder training problem.

The original-seed final result was 5/89 train completions in full mode, 5/89 with the navigation mask, and 0/20 development completions in either mode. Repeated evaluation showed that the navigation-mask train count varies substantially with evaluation seed. Mean train duration progress improved to approximately one third of the motion, while development stayed near one tenth. Those observations support continuing research, not calling the motor foundation reliable.

The old navigation mask exposes reference-derived body-frame horizontal velocity, yaw rate and target height. It has no destination or obstacle inputs. It therefore cannot demonstrate destination selection, obstacle avoidance or stopping at a goal. Calling this interface navigation is convenient shorthand for a reduced command profile, but it is not a measurement of the intended scene-navigation task.

On recorded nominal rows, shuffling commands measurably increased action error, so complete command blindness is not supported. The frozen decoder also reconstructs recorded teacher actions within approximately 1.5e-5 maximum absolute error. That checks checkpoint/order/decoder wiring. It does not demonstrate that the learned token predictor is expressive enough, that quantization is harmless in rollout, or that its predictions remain accurate after drift.

The actual data scale is more informative than the optimizer-step count. The five DAgger collections simulated 216,181 transitions but retained 40,967 locally supported query rows. Some simulator transitions occurred after first failure or reference reset and were not useful supervised examples. The final fit repeatedly sampled 47,400 nominal-plus-query rows. Uniform sampling over episodes balances long and short recordings, but cannot create missing late-motion states or successful recovery trajectories.

## Competing explanations and tests

### Coverage and compounding error

A student can have low average one-step error while repeatedly making a small error in a balance-critical direction. The next state then differs from the training state, and the error compounds. Labeling that visited state is useful only when the expert's action is a meaningful correction for the same requested task. A strict support filter protects against bad labels but can also remove most of the very states needed to learn recovery.

The new pilot preserves the strict whole-motion metric and separately adds locally supported teacher prefixes. It constructs exact contiguous windows starting at 25%, 50% and 75% of all 89 training references, then executes the teacher from those window starts. This broadens temporal coverage without changing joint order or inventing interpolated dynamics. It approximates the coverage benefit of intermediate-state starts; it is not a full implementation of randomized RSI, hard-negative mining or history burn-in.

Each window is separately bound to its original motion, source frame and serialized native motion. A successful window never becomes evidence that the original full motion completed. The endpoint remains a paired navigation request, while the current robot pose is measured and recorded. Reference position is never substituted for the measured pose after drift.

### Action representation and optimization

The pilot compares direct action regression and action flow using the same inherited public encoder, teacher dataset, action normalization, command mixture and update count. The regression model tests whether a direct action head is a useful alternative; the flow model tests a richer conditional transport objective. Both continue to consume the same 930D history feature layout. That layout is partitioned into feature blocks, not reinterpreted as ten chronological frames.

This is a cleaner head comparison than comparing a newly trained flow model directly with a historical token student trained under a different schedule. Even so, parameter counts differ slightly and the flow objective is not numerically identical to normalized action regression. The historical token model remains a practical reference point, not a fully matched architecture ablation. In particular, a negative flow result at this budget does not prove that diffusion or flow policies are unsuitable for humanoids.

The new direct heads normalize targets using training-only action statistics. The first fit estimates those statistics; continuation preserves them so the meaning of an output coordinate does not change when new queries are added. The DAgger follow-up carries AdamW optimizer state across stages with identical architecture and learning rate. Each stage reinitializes its sampler with the recorded training seed, so this is optimizer continuation, not an exact interrupted-run resume.

### Multimodality and temporal consistency

A coarse task request can permit several routes or gaits. A deterministic squared-error predictor may average incompatible actions when the observed history and request cannot disambiguate them. A flow model can represent multiple conditional outputs, but a single paired continuation per context gives weak evidence about that distribution. More importantly, choosing unrelated samples at every control tick may destroy a behavior that would be coherent under a persistent choice.

The implemented sampler accepts explicit noise and caches the condition encoder while integrating the action field. The initial evaluation compares zero initial noise with a Gaussian sample held over the episode, and compares one versus eight Euler steps. Zero noise is a deterministic diagnostic, not a claim to compute the mean of the learned distribution. Episode-held Gaussian noise is a temporal-consistency hypothesis, not a reproduction of independent sampling at each action.

The first 10,000-update fits produce no complete motion in either new head. Eight-step flow performs better than one-step flow, but it trails the old token baseline; sampled episode noise also performs poorly. The appropriate response is to investigate fitting, useful labels and rollout coverage, rather than proclaim a flow advantage based on declining denoising loss.

### Residual corrections

Two residual mechanisms must be distinguished. A posterior residual changes a latent distribution during representation learning. An action residual changes the executed actuator command. Neither should be described as residual reinforcement learning unless rewards and an RL update actually train it.

The implemented diagnostic freezes the existing BFM and SONIC decoder and learns a zero-initialized action correction bounded to ±0.05. It receives measured history, explicitly masked commands and the base action. Its objective is same-state teacher-action imitation plus a small residual-magnitude penalty. This directly tests whether small corrections help the strongest current student without allowing wholesale changes to its representation.

The pilot does not implement actuation-aware residual PPO or FPO++. Those are subsequent research branches. A small correction limit can make a stable baseline easier to preserve, but it also imposes an upper bound on how much tracking error can be repaired. If errors require larger corrections, increasing the bound without measuring falls and motor behavior may simply turn the residual into a replacement policy.

## Direct scene/navigation conditioning

The primary architecture directly conditions the public policy on measured history, an explicit navigation request and known-map geometry. It does not require a separate high-level navigation director. The new context encoder attends to one navigation token and a masked set of up to five exact obstacle primitives. The navigation token contains body-relative start and goal, arrival tolerance, terminal speed bound, required hold duration and an explicit complete-map bit.

The geometry features retain obstacle center, full dimensions, orientation and primitive type. Beam height and vertical extent are preserved; a heightmap-only encoding would be insufficient to express free space beneath an overhang. Padding is distinct from valid geometry. The complete-map bit prevents this version from silently accepting unknown camera space as free space. A future camera encoder needs an explicit partial-observation schema, visibility, timestamps and memory.

A zero-initialized context output preserves the inherited encoder at migration. The new context contribution becomes trainable during fitting. Tests check that migration preserves outputs, that hidden detailed commands cannot leak even when replaced by NaNs, and that available context can affect the policy after its new path is trained. The actor accepts neither motion IDs nor reference phase nor privileged future trajectories.

The first three context arms are direct regression trained always with full commands, direct regression trained with masking, and flow trained with masking. A fourth arm directly extends the existing BFM token prior with scene/goal context while preserving its pretrained prior, posterior, token adapter and frozen SONIC decoder. Its migration test verifies exact original full-command tokens before adaptation; this is the primary comparison for retaining the current motor foundation. The masked mixture is 30% full, 20% partial root and 50% context-only. The fitting implementation selects one mode for a minibatch of sampled episode rows; it does not train recurrent episode sequences. Native collection and evaluation keep their declared control mode fixed through a rollout. This distinction matters when later adding memory or mode-switching experiments.

Paired plane tasks and four exact scene tasks provide the first context pilot. Their labels are explicitly exploratory. The original scene/task qualification failed, and the mere existence of teacher-action labels is not proof that the task succeeds. A changed-goal request with an unchanged incompatible reference is rejected; that failed collection attempt is preserved separately and contributes no training labels.

## Independent task evaluation

Reference tracking and task success need separate denominators and stopping conditions. A robot taking a valid alternate route can fail a reference-pose threshold while successfully navigating. Conversely, closely reproducing a reference does not guarantee that the reference reaches the requested destination safely and stops there.

The new native scene evaluator measures goal distance and speed from the current simulator state, captures pair-resolved contact forces at 200 Hz, and scores the required hold. It does not stop solely for reference-pose mismatch. It stops at goal hold, undesired contact, a declared absolute pelvis-height fall guard, or the bounded task horizon. Its initial fall/contact profile is deliberately limited to these upright traversal probes; it is not a universal scorer for ground-contact skills.

There is also a subtle simulator issue: reference exhaustion can trigger command resampling that resets the robot independently of ordinary termination. The evaluator clamps reference bookkeeping below that boundary and captures physical measurements before any native reset. An unexpected reset is treated as an evaluation failure. This prevents a teleport or a reset from being counted as goal arrival.

The current four-scene pilot includes training-related tasks and does not establish held-out scene-family generalization. The 20 development motions remain separately evaluated under the reference-tracking benchmark. Subsequent work needs multiple independent scene/motion families, different goals from the same initial state, and obstacle changes that require different successful continuations. A shuffled-context action difference is only a sensitivity diagnostic; the desired evidence is improved physical task success under the correct context.

## Executed pilot and research decisions

The first pilot sequence totals 59,000 new optimizer updates excluding small smoke fits: 20,000 for the first two action heads, 10,000 after expanded teacher coverage, 12,000 across three DAgger rounds per head, 15,000 for the three context arms, and 2,000 for the frozen-BFM residual. The additional context-BFM arm adds 5,000 updates, bringing the bounded total to 64,000, plus 192 smoke updates. It uses one training seed to validate the pipeline before the previously proposed larger three-seed study. Native evaluations follow DAgger fits and context fits; task-scored scene rollouts are reported separately.

The complete metrics, failures and stage receipts are stored in `/home/linjiw/research-data/m2s-flow-context-20260913`. The results report accompanying this document is the authority on which stages completed and their measured outcomes. A configured stage is not a completed experiment, and a finished offline fit is not evidence of physical control success.

The next research decision should follow the observed failure mode. If flow denoising improves but integrated actions remain poor, examine sampler calibration and the conditional action model. If both direct heads improve only after new rollouts, prioritize expert coverage and collection frequency. If the bounded residual helps while base changes hurt, preserve the motor representation and investigate corrections. If context models ignore scene changes or cannot stop, obtain matching successful continuations and test the conditioning interface before scaling architecture.

A joint state-action flow model remains a worthwhile subsequent comparison for test-time geometric guidance. It requires synchronized, contiguous trajectories with correct masks across resets; independently sampled action rows cannot be repackaged as such trajectories. A path or goal auxiliary prediction head may help learn context, but its output must not quietly become an externally supplied reference at deployment. These extensions should be compared against the direct context policy rather than replacing the research question mid-evaluation.

## Sources

1. Zeng et al. *Behavior Foundation Model for Humanoid Robots*. 2025, v1. [Paper](https://arxiv.org/html/2509.13780v1), especially Sections III–IV.
2. Wang et al. *OmniXtreme: Breaking the Generality Barrier in High-Dynamic Humanoid Control*. 2026, v1. [Paper](https://arxiv.org/html/2602.23843v1), Algorithm 1 and Section III.
3. Truong et al. *BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion*. 2025, v1. [Paper](https://arxiv.org/html/2508.08241v1), Section IV. This review names the inspected version; later revisions exist.
4. *Flow Policy Gradients for Robot Control*. 2026, v1. [Paper](https://arxiv.org/html/2602.02481v1).
5. *PhysiFlow: Physics-Aware Humanoid Whole-Body VLA via Multi-Brain Latent Flow Matching and Robust Tracking*. 2026, v1. [Paper](https://arxiv.org/html/2603.05410v1).
6. *ResMimic: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation via Residual Learning*. 2025. [Paper record](https://arxiv.org/abs/2510.05070).
7. Lipman et al. *Flow Matching for Generative Modeling*. 2022. [Paper](https://arxiv.org/abs/2210.02747).
8. OmniXtreme authors. [Official implementation repository](https://github.com/Perkins729/OmniXtreme), inspected September 13, 2026.
