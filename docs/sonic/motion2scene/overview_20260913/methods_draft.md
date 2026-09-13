# Motion2Scene: from execution-aware scene construction to context-conditioned humanoid behavior

Methods overview and working introduction · 13 September 2026

This draft synthesizes the repository's Motion2Scene, teacher training, BFM-inspired distillation, motor recovery, and navigation studies. It describes a connected research program with implemented components and explicitly identified extensions. It does not imply that every component has already been trained together as one system. Results are a dated evidence snapshot; the separate navigation recovery study remains governed by its own final receipts.

## Research overview

![Research overview](01_research_overview.png)

**Figure 1. Research overview: from motion evidence to scene-conditioned humanoid behavior.** **(A)** Generated, recorded, and authored motions are converted into G1 references, checked for embodiment and dynamics consistency, and retained with their ancestry and split assignments. Motion2Scene uses candidate behaviors and their controller-executed body envelopes to propose environments in which motion selection matters. Learned inverse proposals and analytic search are alternative construction mechanisms. A geometric screen seeks a feasible target continuation, contrast with competing responses, and information available before the decision deadline. Matched physical executions then establish passage, contact, timing, and failure labels. Measured outcomes inform subsequent curriculum construction; geometric compatibility alone does not certify a task demonstration. **(B)** A SONIC-initialized tracking teacher is optimized with privileged simulator and reference information and then fixed for student supervision. The strongest current motor student predicts missing desired-reference frames from measured history and a current detailed command, preserving the teacher's reference encoder, finite scalar quantizer, and history-conditioned action decoder. Student-state teacher queries and nominal replay refine the predictor through bounded data aggregation. **(C)** A navigation branch receives robot history, a localized task request, and known-map geometry, and predicts the detailed command internally through the unchanged motor specialist. Task demonstrations and supported recovery experience train this branch; autonomous goal completion, terminal hold, contacts, and falls evaluate it. Solid connections denote implemented interfaces, while dashed connections denote proposed broader integration or extensions whose completed outcome is not established in this snapshot. Colors identify scene construction, motor learning, and task conditioning. Robot drawings are conceptual. The diagram describes information and training dependencies rather than a claim of end-to-end navigation success.

## Introduction draft

Humanoid motion control and scene-conditioned navigation impose different demands on a learned policy. A motion tracker receives a detailed specification of how the body should move. A navigation policy instead receives a destination and environmental constraints, and must decide which body motion will satisfy them. In a constrained passage, that decision can involve when to lower the body, how long to maintain an adaptation, and when to recover. Near a destination, approaching the goal is only part of the task: the robot must also reduce its speed and remain inside the requested region. Reliable tracking provides essential motor capability, but task-conditioned behavior additionally requires appropriate supervision about which continuation to execute and information sufficient to choose it.

Motion2Scene addresses the data side of this problem by asking which environments make available motion alternatives useful. Rather than attaching arbitrary geometry to a reference, it constructs scenes around the spatial and temporal structure of candidate behaviors. This direction builds on Learning from Learned Hallucination, which learns obstacle configurations associated with motion plans obtained in open space. Our humanoid study makes controller execution, legal transitions, and decision-time information explicit because a kinematically clear reference can still produce contact or fail during entry and recovery. The central object is therefore a scene paired with a physically evaluated continuation, including its failures and measurement conditions. [Learning from Learned Hallucination](https://arxiv.org/abs/2108.09793).

A second challenge arises when transferring motor expertise into a student that has less reference information. The Behavior Foundation Model framework motivates treating detailed and sparse commands as different specifications of behavior and learning from a privileged expert through masked online distillation. Our implementation investigates that principle within the SONIC motor representation. The early masked token students and direct-action comparisons show that low imitation loss alone is insufficient to establish reliable execution. The later motor-recovery study instead preserves the pretrained reference encoder, quantizer, and decoder, and learns the missing reference context needed to use them. This separates learning how to specify a behavior from relearning all of its motor realization. [Behavior Foundation Model](https://arxiv.org/html/2509.13780v1), [SONIC](https://arxiv.org/html/2511.07820v1).

We organize the method around three coupled requirements: **executable teaching targets, sufficient public conditioning, and training coverage of learner-visited states**. Motion2Scene supplies candidate task conditions and measured comparisons between continuations. A privileged tracking teacher supplies action and motor-representation targets. A public student predicts the information required by the retained motor model, first from a current detailed command and then from scene and goal context. Online aggregation exposes the student to states caused by its own actions. For navigation, a queried action receives stronger recovery status only when a compatible continuation actually completes the task from the encountered state. This distinction prevents nominal reference tracking or unexecuted teacher advice from being counted as demonstrated task recovery.

The current implementation supports a concrete account of the remaining uncertainty. The recovered full-command specialist completes 88 of 89 training motions and 10 of 20 development motions on the original evaluation seed. Its repeated training counts are 85 and 86 of 89, with 10 of 20 development completions in both repeats. These results use a richer current-command interface and a different preserved representation than the historical masked student, so they support the combined motor adaptation rather than an isolated claim about model capacity. The same motor passes eight stopping and corridor tasks when supplied with detailed commands, whereas the original goal/map adapter passes none. A later causal-localization comparison produces one success in the eight-task panel, but the successful task fails on two additional seeds. These observations identify task conditioning and closed-loop support as continuing research problems. [Motor evidence](../BFM_MOTOR_RECOVERY_RESULTS_20260913.md), [navigation pilot](../NAVIGATION_MOTOR_PILOT_20260913.md), [terminal-control evidence](../NAVIGATION_TERMINAL_CONTROL_20260913.md).

The resulting research program connects informative environment construction to preserved motor distillation and scene-conditioned execution. Its methodological value lies in making the interfaces between these stages explicit: what motion a scene admits, what an expert can execute, what the student observes, and which visited states its labels support. The next decisive tests concern whether complementary, task-qualified experience improves autonomous scene-dependent behavior at matched acquisition and learning budgets while retaining the established motor specialist. Broader route selection, obstacle avoidance, perception transfer, and generalization remain hypotheses to be evaluated under those contracts.

## Method introduction

Our objective is to learn humanoid behavior from motion-derived environments and privileged execution supervision while progressively reducing the information required at inference. The method has three stages. First, we construct candidate scenes from alternative motion envelopes and use physical execution to determine which continuations remain useful. Second, we train a tracking teacher and distill its motor behavior into a student that predicts missing desired-reference context through a retained SONIC representation. Third, we learn scene-and-goal conditioning through the recovered motor specialist, using independently scored task demonstrations and supported learner-state recovery data. The architecture in Figure 2 distinguishes training-only teacher information from each public actor interface. Figure 3 illustrates why scene contrast, response complementarity, and terminal control produce different teaching requirements.

Let $x_t$ denote the physical robot state, $h_t$ its measured proprioceptive history, $r_t$ the teacher's desired-reference window, and $a_t\in\mathbb{R}^{29}$ the native action vector. A detailed command $c_t$ specifies present-time behavior quantities, with a mask $m_t$ identifying available entries. A navigation request $g_t$ contains the localized start/goal and terminal requirements, and $S_t$ contains the available obstacle geometry. Detailed-command execution uses $h_t,c_t,m_t$; navigation execution uses $h_t,g_t,S_t$, with optional causal localization features. The training system may additionally access privileged simulator quantities, true future references, and task-bound outcome records. These objects have distinct roles and are not interchangeable inputs.

## 1. Motion assets, repaired references, and executable options

Motion sources include generated, recorded, and authored references. Each source must pass through a consistent embodiment interface: joint order, coordinate convention, body definitions, timing, and native serialization must agree with the G1 controller. Repair and offline screening address joint-limit excess, penetration, discontinuities, and implausible dynamics. They determine which references are eligible for an experiment; they do not establish that a policy can execute them. The repaired teacher study uses 89 screened training clips and preserves all 20 assigned development clips for evaluation. These belong to the newer motor study and should not be confused with the earlier finite traversal-command banks.

For scene construction, an option is a complete supported continuation, including its motion identity, entry phase, duration, return request, and transition guards. This matters because a low posture that is feasible in isolation can become infeasible during the transition into or out of it. We therefore represent each qualified option by its achieved body trajectory over the full relevant interval. Body capsules provide an inexpensive approximation for geometric queries. Physical execution retains the authoritative record of tracking, contact, termination, and continuation legality. [Traversal specification](../TRAVERSAL_METHOD_V2.md).

## 2. Motion2Scene construction and decision-relevant supervision

Given an achieved trajectory $\hat\tau^a$ for each option $a$, an inverse constructor proposes a scene $S$ in which a target continuation can execute and a competing response is inadequate or more costly. One implemented learned beam model encodes capsule geometry with body and temporal features, then predicts a mixture distribution over bounded beam station and height. The geometric objective rewards target feasibility and preference over competing motion. Analytic and uniform constructors provide separate comparisons. The learned model is a proposal mechanism; it is neither the contact oracle nor evidence that learned generation outperforms analytic construction. [Inverse-model implementation](../../../gear_sonic/dataset_generation/hallucination/motion2scene_inverse.py).

An interpretable *screening target* for adaptation-required scenes is

$$
d(S,\hat\tau^{a_+})\geq\delta_+,
\qquad d(S,\hat\tau^{a_-})\leq-\delta_-,
$$

where $d$ is a signed capsule–obstacle clearance, $a_+$ is a proposed useful option, and $a_-$ is a competing option. The exact implemented constructors use their declared objectives, perturbations, and margins; this expression summarizes the contrast principle rather than replacing those specifications. A geometric violation predicts interference, but measured force and physical task scoring determine whether execution fails. Conversely, a positive geometric clearance can miss tracking-induced contact.

Each candidate is evaluated through named, matched continuations. Their outcomes are bound to the scene, initial or pre-decision history, policy, motion assets, and scorer. Teacher selection first respects physical task feasibility, then uses the available task-specific cost, such as measured passage time. Missing or unadmitted branches remain masked. Waiting is meaningful only when a legal later continuation still exists, and a scene-dependent target must be distinguishable from the observations available at its decision point. [Information-consistent teaching](../INFORMATION_CONSISTENT_TEACHING.md).

The curriculum must also distinguish **adaptation-required yield** from **response complementarity**. A corpus can contain many scenes that defeat walking while still allowing one fixed crouching schedule to solve every task. Such a corpus may teach a constant policy. A complementary corpus contains tasks whose successful response sets require different choices. This is why the measured response matrix, the strongest constant response, and observable decision timing are useful curriculum diagnostics. The completed M8 construction study found no downstream advantage of executed contrast over screened uniform in its matched comparison, despite earlier acquisition-yield differences. The current framework therefore treats curriculum superiority as an empirical question, not as an automatic consequence of generating harder scenes. [M8 evidence and interpretation](../RESEARCH_STATE.md).

## 3. Privileged tracking teacher training

The tracking teacher is initialized from SONIC and optimized in the native simulation training stack using PPO with the configured motion-tracking rewards and auxiliary terms. Its information includes privileged simulator state and the desired-reference window. At the level of the reinforcement-learning objective,

$$
\max_\psi\;\mathbb{E}_{\pi_\psi}
\left[\sum_t\gamma^t R_{\mathrm{track}}(x_t,r_t,a_t)\right].
$$

Here $R_{\mathrm{track}}$ denotes the actual configured reward rather than a new reward claimed in this draft. The repaired-data protocol begins from released actor/critic weights with a fresh optimizer and simulator. It trains on plane tracking without obstacles or cameras, so scene feasibility does not enter this teacher's training merely because scene annotations exist beside the motion data. The selected teacher checkpoint is held fixed during subsequent distillation. [Teacher protocol](../REPAIRED_TEACHER_NAVIGATION_PLAN_20260912.md), [native trainer](../../../gear_sonic/research/hindsight_training/tracker.py).

Teacher quality is established by whole-motion evaluation and its failure accounting. A scene expert requires additional evidence: a continuation must be compatible with the requested goal and map, and physical execution must satisfy that task's contact and terminal conditions. We use task-specific successful continuations as supervision where available. A general obstacle-aware planner is not implied by the tracking teacher.

## 4. BFM-inspired distillation and preserved motor recovery

![Teacher and student architecture](02_teacher_student_architecture.png)

**Figure 2. Teacher–student architecture and information boundaries.** **(A)** In full-command mode, the public motor specialist receives measured 930-dimensional history and a 114-dimensional present-time command. It reconstructs the current 64-dimensional target frame and predicts residuals for nine additional target frames; native packing produces the encoder's 640-dimensional input. The preserved SONIC encoder and finite scalar quantizer produce 64 token values, and the frozen decoder combines those tokens with measured history to produce 29 native action values. In navigation mode, a separate branch predicts the detailed command internally from a 10-dimensional task request, up to five masked 15-dimensional obstacle primitives, measured history, and optional causal localization features. **(B)** Motor training uses same-state teacher action and token targets, with an optional desired-reference auxiliary loss. Navigation training instead matches the fixed motor's action at the same history under the expert command, alongside structured command supervision. Freezing motor weights preserves the specialist while allowing input gradients to train the navigation branch. **(C)** Data aggregation records the states visited by the learner and distinguishes candidate teacher advice from physically supported recovery. **(D)** The earlier BFM-style comparison uses a masked command-conditioned Gaussian prior, a privileged training posterior, and a token adapter into a fixed SONIC decoder. Its posterior is absent at inference. It is architecturally distinct from the deterministic anticipatory motor. **(E)** Direct reference prediction, coherent reference-flow generation, task reinforcement learning, and camera-based belief are subsequent comparisons. The native history layout is not a sequence of ten uniform feature frames; reference packing and physical temporal order are handled explicitly.

### 4.1 Earlier masked variational student

The earlier implementation uses a 79-dimensional root/body/joint interface with explicit availability masks. A public encoder conditions a Gaussian behavior prior on measured history and masked commands. A training posterior additionally receives privileged simulator state and the future reference. A learned adapter maps a latent sample to native quantized SONIC tokens, and a frozen decoder produces the action:

$$
p_\theta(z_t\mid h_t,m_t\odot c_t,m_t),\qquad
q_\phi(z_t\mid h_t,m_t\odot c_t,m_t,s_t^{\rm priv},r_t).
$$

The implemented variational objective combines teacher-action reconstruction with $\mathrm{KL}(q_\phi\|p_\theta)$, optional token reconstruction, and configured public-prior action supervision. At execution only the prior is used. This is a BFM-inspired adaptation to SONIC's fixed decoder; it is not a reproduction of BFM's complete architecture or reported training scale. Later context-token and direct-action regression/flow models remain documented comparison branches. [Masked distillation implementation](../BFM_DISTILLATION_IMPLEMENTATION_20260912.md), [flow/context results](../FLOW_DISTILLATION_RESULTS_20260913.md).

### 4.2 Current anticipatory motor specialist

The motor-recovery study preserves more of the pretrained representation. Let $E$ be the teacher's reference encoder, $Q$ its finite scalar quantizer, and $D$ its history-conditioned decoder. The student learns a deterministic reference forecaster $F_\theta$:

$$
\hat r_t=F_\theta(h_t,c_t),\qquad
z_t=Q(E(\operatorname{pack}(\hat r_t))),\qquad
a_t=D(z_t,h_t).
$$

The current command expands the earlier 79-dimensional interface by adding 29 current target joint velocities and six relative-orientation values. The arrival coordinate remains unavailable. A 64-dimensional current target frame consists of current target joint positions, joint velocities, and orientation. It is fixed by the supplied command; the forecaster predicts residuals for the remaining nine frames. Its input is normalized measured history plus the current command, and its zero-initialized output begins by repeating the current target. Crucially, native packing is not a naive reshape of the 640-dimensional reference into chronological frames. [Anticipatory motor](../../../gear_sonic/research/scene_distillation/anticipatory_motor.py), [reference layout](../../../gear_sonic/research/scene_distillation/reference_layout.py).

The implemented motor objective can be written as

$$
\mathcal{L}_{\rm motor}=
\operatorname{MSE}(a_t,a_t^*)+
\lambda_z\operatorname{MSE}(z_t,z_t^*)+
\lambda_r\operatorname{SmoothL1}(\hat r_t/\sigma_r,r_t/\sigma_r).
$$

The reference term is optional and its coefficient is configuration-specific. Teacher tokens and actions provide aligned targets at the same measured state. Only the forecaster is updated; the teacher encoder and decoder remain unchanged. This model anticipates **desired reference motion**. It does not predict the probability of a collision or the robot's future executed physical state. [Motor loss implementation](../../../gear_sonic/research/scene_distillation/motor_training.py).

### 4.3 Online aggregation and self-learning structure

The motor student is rolled out in simulation and the fixed teacher is queried at the student's actual pre-action state. Fresh examples are mixed with nominal teacher replay. Randomized reference starts, nominal-start coverage, explicit burn-in, and recorded takeovers broaden the visited distribution. This follows the data-aggregation principle of training on states induced by the learner. [DAgger](https://proceedings.mlr.press/v15/ross11a.html).

In this project, “self-learning” refers to measured iterative acquisition and refitting. It is not a claim of a single end-to-end self-supervised optimizer updating the generator, teacher, and navigation policy simultaneously. The three relevant feedback loops are: measured option responses guiding curriculum construction; same-state teacher queries refining the motor predictor; and task-supported navigation recovery informing adapter updates. Teacher-assisted trajectories remain separate from unassisted policy evaluation. All failed, censored, and rejected attempts retain their interaction cost. A teacher query can be a valid numerical target while still lacking evidence of physical recovery from that state.

## 5. Scene-and-goal conditioning through the recovered motor

The public navigation actor receives measured history, a body-relative start/goal and terminal request, and a complete known map represented by at most five masked obstacle primitives. Each primitive retains center, dimensions, orientation, and shape information. Three-dimensional geometry matters for overhead constraints: free space beneath an overhang cannot be represented faithfully by a single top-surface height alone. Missing or padded geometry is explicitly distinguished from valid map entries.

The current navigation comparator uses a shared obstacle MLP and masked mean pooling, followed by a command predictor. Its public API has no detailed reference commands, reference clock, motion identifier, or future reference. The predictor produces a detailed command internally:

$$
\hat c_t=I_\eta(h_t,g_t,\operatorname{SetEnc}_\eta$S_t$,\ell_t),
\qquad a_t=M(h_t,\hat c_t),
$$

where $M$ is the complete frozen anticipatory motor and decoder, and $\ell_t$ is an optional causal localization feature. The current localized profile estimates body-frame velocity from measured world-position differences over five control intervals and appends a validity bit. The task context has ten values: body-relative start and goal, goal tolerance, terminal speed limit, required hold duration, and a complete-map indicator. The task deadline remains part of the execution/scoring contract rather than an input in this ten-value profile. Terminal heading is a possible later extension, not an implemented scored input here. [Navigation actor](../../../gear_sonic/research/scene_distillation/navigation_motor.py), [localization contract](../NAVIGATION_TERMINAL_CONTROL_20260913.md).

The structured navigation objective is

$$
\mathcal{L}_{\rm nav}=\operatorname{MSE}
\left(M(h_t,\hat c_t),\operatorname{stopgrad}M(h_t,c_t^*)\right)
+0.1\,\mathcal{L}_{\rm command}.
$$

The command term averages normalized error over heading, velocity, height, yaw rate, body keypoints, joint position, joint velocity, and orientation groups; the unavailable arrival coordinate is excluded. The motor retains input gradients so that its executed-action sensitivity trains the adapter, while all motor parameters and normalization buffers remain fixed. The original teacher action is retained as a diagnostic. Computing the primary target through the same motor avoids requiring the context branch to compensate for an arbitrary difference between its fixed backend and the original teacher.

This internal command interface is a tested comparator, not a requirement that the user supply an external navigation director. A direct scene/goal-to-reference branch through the same frozen encoder and decoder is a natural subsequent comparison. Direct context-token models have already been evaluated on an earlier, weaker motor foundation; they should not be conflated with that proposed comparison on the recovered specialist. A reference-flow model would additionally require task-qualified temporal targets and a coherent sampling/replanning scheme.

## 6. Task-qualified demonstrations and recovery

Nominal demonstrations must execute a compatible continuation in the requested scene and satisfy an independent task score. The current stopping collection contains four clips and eight clear/corridor executions, but only one recorded motion ancestry group. Scene replicas therefore do not constitute independent source families. In the recovery extension, demonstrations are also collected through the exact frozen motor backend used by navigation, addressing a possible mismatch between original-teacher states and deployed-motor states. [Supported recovery protocol](../NAVIGATION_SUPPORTED_RECOVERY_20260913.md).

For learner-state recovery, the navigation policy first drives the robot to a declared switch point. The fixed motor then executes the named continuation from the actual physical state without a phase jump or synthetic reset. Only a suffix that independently completes the required hold before the original deadline, with the prescribed contact and fall checks, is eligible as successful recovery. The first suffix row is a learner-state intervention; later rows are expert-executed recovery observations. They are not counted as separate learner queries. Unsuccessful attempts remain recorded but do not become positive task targets.

The bounded comparison preserves the backend and contrasts additional demonstration replay with a mixture of qualified recovery and replay at equal additional optimizer budget. Recovery acquisition consumes extra simulation and must be reported separately. Its final outcome is not supplied by this overview: use the study's final report and receipts when available. Geometric scene proposals, exploratory action queries, complete nominal demonstrations, and executed recovery suffixes are separate evidence classes.

## 7. Evaluation and the scene-context research agenda

![Scene and learning illustration](03_scene_and_learning_illustration.png)

**Figure 3. Conceptual illustration of scene construction and task learning.** **(A)** Alternative achieved motions provide different body envelopes. A proposed beam can preserve clearance for an adaptation while intersecting an upright response; physics must still verify the complete transition and passage. **(B)** Obstacle extent changes the duration requirement, while transition-sensitive encounters can favor different schedules. The schematic response table illustrates why a corpus needs complementary successful choices to identify a scene-dependent policy. It is an explanatory table, not experimental data. **(C)** A navigation request requires choosing a continuation compatible with both the destination and obstacle geometry, followed by braking and terminal hold. The illustrated alternate routes express research targets rather than demonstrated navigation. **(D)** Construction, physical label qualification, and learner-state refitting form a chain of supervision whose integrity is checked separately at each stage. The figure motivates tests that change the obstacle arrangement at a fixed start/goal, change goals in a fixed scene, and distinguish goal entry from stable completion. All poses and paths are schematic; contact and task-success claims come from independently scored executions.

We evaluate motor realization and task selection separately. Whole-reference tracking measures completion and tracking error under the native benchmark. Sparse command execution requires command-specific velocity, yaw-rate, and posture metrics alongside physical survival. Goal/map navigation requires the robot to reach the requested region, slow down, and remain there under contact, fall, and deadline constraints. An alternate valid route should not fail only because it deviates from the original joint trajectory, and short surviving prefixes should not be counted as successful navigation.

The current eight-task stopping scorer requires 3D pelvis-to-goal distance at most 0.25 m and 3D speed at most 0.10 m/s for 50 consecutive control decisions at 50 Hz. It retains prohibited-contact failures above the declared 1 N threshold, a 0.25 m absolute pelvis-height fall guard, and the original deadline. Contact is recorded at 200 Hz. Measurements are captured before simulator resets; reference exhaustion is prevented from teleporting the robot into an apparent success. This is a bounded upright-task scorer, not a general evaluator for crawling or arbitrary ground-contact behavior.

Scene reasoning is tested by successful behavioral changes under valid context interventions. Useful families include the same start/goal with different blocked passages, the same scene with different destinations, different overhead clearances requiring distinct supported postures, and varied arrival conditions requiring different deceleration. Irrelevant obstacles provide an additional control. All changed conditions need compatible executed continuations before their actions can serve as positive targets. Context shuffling can reveal sensitivity, but it cannot by itself establish that the policy uses geometry correctly.

The known-map study currently assumes an ideal localized pose stream. Camera and partial-map execution require an observation interface with visibility, timestamps, unknown-space semantics, and memory. They are a separate transfer stage. Held-out evaluation should respect motion ancestry, scene family, and repair lineage; a random row split or multiple replicas from one source cannot support broad generalization. Existing development motions have been inspected repeatedly, so an untouched evaluation set is still needed after further tuning.

## Evidence snapshot and claim boundaries

| Component | What the cited records establish | Claim still to test |
| --- | --- | --- |
| Motion2Scene | Implemented inverse/analytic construction and physically measured option labels; M8 executed contrast and uniform tie in downstream passage | Superior scene-conditioned policy learning from the constructor |
| BFM-style CVAE and action-flow comparisons | Executed training and evaluation; the early flow/context study did not produce successful scene navigation | A general advantage from changing the action distribution |
| Recovered full-command motor | 88/89 train and 10/20 development on the original seed; repeats 85/89 and 86/89 train, both 10/20 development | Arbitrary-mask foundation behavior or broad transfer |
| Stopping/corridor motor controls | Eight successful full-command tasks across four clips, one recorded ancestry group | Obstacle-induced route selection and independent-family generalization |
| Goal/map adapter | Original 0/8; localized comparison 1/8, with failure on two repeats of that successful task | Reliable autonomous task improvement |
| Navigation-supported recovery | Implemented collection/admission protocol and bounded comparison | Final task outcome must follow the study's own completed receipts |
| Cameras, reference flow, task RL | Explicit subsequent branches; earlier action-flow and supervised residual experiments remain separate | Qualified perception transfer, coherent route generation, and task-RL benefit |

The central contribution being developed is a method for connecting physically grounded task construction to preserved motor distillation and task-conditioned learning. Strong motor recovery is supported within the reported simulation study. General scene-conditioned navigation remains the research objective.

## Source and implementation map

The dated reports below define the scope of this draft; earlier snapshots are not silently combined into a single experiment.

| Topic | Primary local source |
| --- | --- |
| Motion2Scene design and option semantics | [Traversal method](../TRAVERSAL_METHOD_V2.md), [project guide](../PROJECT_GUIDE.md) |
| Acquisition results and M8 null | [Persistent research state](../RESEARCH_STATE.md), [post-M8 decision](../POST_M8_DECISION_V1.md) |
| Observation-consistent teacher | [Information-consistent teaching](../INFORMATION_CONSISTENT_TEACHING.md) |
| Teacher protocol | [Repaired teacher plan](../REPAIRED_TEACHER_NAVIGATION_PLAN_20260912.md) |
| Early BFM implementation | [BFM distillation](../BFM_DISTILLATION_IMPLEMENTATION_20260912.md) |
| Direct masked context and flow | [Direct context design](../DIRECT_MASKED_SCENE_NAVIGATION_20260913.md), [completed flow results](../FLOW_DISTILLATION_RESULTS_20260913.md) |
| Selected motor architecture and results | [Motor recovery](../BFM_MOTOR_RECOVERY_RESULTS_20260913.md), [compact evidence](../evidence/motor-recovery-20260913/README.md) |
| Navigation architecture and task data | [Contribution report](../MOTOR_TO_NAVIGATION_CONTRIBUTION_20260913.md), [navigation pilot](../NAVIGATION_MOTOR_PILOT_20260913.md) |
| Causal observations and task objective | [Terminal-control study](../NAVIGATION_TERMINAL_CONTROL_20260913.md) |
| Recovery support and ongoing comparison | [Supported-recovery protocol](../NAVIGATION_SUPPORTED_RECOVERY_20260913.md) |

External conceptual references verified for this draft: [LfLH](https://arxiv.org/abs/2108.09793), [BFM](https://arxiv.org/html/2509.13780v1), [SONIC](https://arxiv.org/html/2511.07820v1), and [DAgger](https://proceedings.mlr.press/v15/ross11a.html). They motivate the relevant ideas; their results are not evidence for this implementation.

## Figure files and reproduction

Each figure is supplied in editable SVG, vector PDF, and high-resolution PNG. The SVG keeps text as text. The diagrams use schematic vector geometry drawn locally with Matplotlib and contain no generated simulator evidence. The companion HTML is a local reading copy with figure links and the complete draft; the PDF document is a portable reading version.

- [Figure 1 SVG](01_research_overview.svg) · [PDF](01_research_overview.pdf) · [PNG](01_research_overview.png)
- [Figure 2 SVG](02_teacher_student_architecture.svg) · [PDF](02_teacher_student_architecture.pdf) · [PNG](02_teacher_student_architecture.png)
- [Figure 3 SVG](03_scene_and_learning_illustration.svg) · [PDF](03_scene_and_learning_illustration.pdf) · [PNG](03_scene_and_learning_illustration.png)
- [Figure source](draw_figures.py)

Re-render the figures from the repository root:

```bash
.venv_research/bin/python docs/motion2scene/overview_20260913/draw_figures.py
```

This documentation work does not launch, pause, or modify research training.
