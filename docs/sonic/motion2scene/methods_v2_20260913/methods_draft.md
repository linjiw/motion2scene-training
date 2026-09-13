# From Motion–Scene Pairs to Goal-Conditioned Humanoid Control

**A learned hindsight scene proposer, a full-command motor specialist, and a scene-context navigation student**

Research manuscript revision · 13 September 2026 · Evidence snapshot, not a claim of completed downstream generalization.

## Abstract

A motion library specifies how a humanoid can move but usually does not specify the environment that makes a particular movement useful. We study a staged approach that recovers candidate scene context from motion, preserves a pretrained whole-body motor representation, and learns to invoke that representation from navigation goals and geometry. Motion2Scene first constructs obstacle candidates around motion events and learns a conditional categorical distribution using reference clearance, proximity, and interference with geometric counterfactuals as self-supervision. A fixed geometric filter and sequential sampler produce reproducible motion–scene pairs. Separately, a reference-tracking teacher supplies motor supervision. Our recovered full-command specialist predicts missing desired-reference context from current commands and measured history, then uses a frozen reference encoder, finite scalar quantizer, and action decoder. A downstream navigation student predicts the full command from proprioception, a goal request, and a known obstacle map while preserving the complete motor backend. We qualify recovery supervision through executed continuations from actual learner states. Current evidence establishes these interfaces and exposes remaining limitations: the generator has sparse counterfactual support, the motor remains weaker on development motions, and an initial navigation improvement from 3/8 to 4/8 tasks falls to 1/8 on an evaluation-seed confirmation. The resulting framework supports controlled studies of when motion-derived scene context becomes useful physical supervision; broad navigation gains remain an open research question.

## 1. Introduction

Whole-body motion data and task-directed control provide different kinds of information. A recorded trajectory can show a humanoid turning, lowering its torso, or moving its arms, yet omit the obstacles and goals that motivated those actions. Conversely, a navigation request identifies where a robot should go without specifying the joint trajectories needed to arrive, stop, and remain stable. Connecting these two descriptions requires more than placing a scene around a motion or fitting a policy to teacher actions. The scene must be geometrically meaningful, the behavior must remain executable, and the downstream controller must select a suitable continuation using only its permitted observations.

We investigate this connection through three coupled learning problems. The first is **motion-conditioned scene inference**: given an entire reference motion, what obstacle configurations preserve that motion while making simpler geometric alternatives less compatible? The second is **motor distillation under reduced reference information**: how can a student retain a teacher's movement capability when it receives current commands rather than the teacher's future reference window? The third is **scene-and-goal command generation**: how can a navigation policy provide the rich command needed by the motor without receiving motion identity, reference phase, or an external reference trajectory?

Our Motion2Scene component treats obstacle placement as a hindsight inference problem. The motion is already known when the scene is generated. Candidate obstacles are anchored to events in that motion, and a fixed geometry model asks whether each candidate clears the original motion, lies close enough to be relevant, and interferes with a root-path shortcut or a held-articulation probe. These tests define a distributional training target for a learned scorer. At generation time, the scorer proposes among the fixed candidates; geometric conditioning and overlap-aware sequential sampling produce the final scene. This procedure supplies its own labels from motion and geometry. It neither reconstructs a unique original environment nor converts a geometric counterfactual into a verified control policy.

For motor learning, we start from a SONIC reference-tracking teacher and explicitly account for the information lost during distillation. Our full-command interface contains current root-related quantities, body keypoints, joint positions and velocities, and relative orientation. A deterministic forecaster uses this command and measured history to predict the missing desired-reference frames. The forecast passes through the teacher's retained reference encoder and quantized motor interface to its action decoder. This design makes the prediction target interpretable while preserving a previously trained motor representation. The forecaster predicts desired reference context, not the future physical state produced by the robot's actions.

The navigation student sits above that recovered motor. It encodes a known obstacle map, combines it with a body-relative navigation request and proprioceptive history, and predicts the current full command internally. We train the navigation actor through the frozen motor using targets generated by that same motor at the same measured history. This isolates command-generation error from an avoidable mismatch between two action backends. Because low supervised error does not ensure successful arrival and stopping, we further test recovery from actual navigation-induced states and admit positive recovery labels only when the executed continuation completes the original task on time.

The current contribution is a specified, auditable learning pipeline and a set of discriminating experiments. We provide an explicit learned hindsight proposal mechanism; a full-command distillation architecture that preserves the quantized motor path; and a navigation training and recovery protocol that distinguishes compatible geometry, teacher labels, executed support, and autonomous task success. The available results motivate this decomposition but do not yet establish that learned scene generation improves downstream data efficiency or that the navigation student generalizes beyond its small training task family. Those claims require the matched comparisons described in Section 8.

![Integrated research overview](fig0_research_overview.png)

**Figure 1. Integrated research overview.** Motion2Scene proposes static geometry in hindsight from complete reference motions; the schematic contrasts a curved target path with a straight geometric probe intersecting an obstacle. A separately trained tracking teacher supplies supervision to a current-command reference forecaster, which retains the pretrained encoder, quantizer, and decoder. The navigation actor then generates the rich motor command internally from measured history, a navigation request, and a known map. The lower bridge identifies the additional work required to turn geometry and motor capability into positive action supervision: execute a continuation, score the physical task, and admit only supported data. The recovery experiment supplies a bounded instance of this bridge; it does not yet establish downstream utility of the learned scene distribution. Training and sampling detail is expanded in Figures 2–4.

## 2. Relationship to BFM, BFM-Zero, and learned hallucination

BFM motivates treating different control modes as specifications of behavior. Its masked control interface, privileged proxy teacher, conditional variational model, and online distillation provide the closest architectural reference for our earlier student. Our revised figures use the same useful visual separation of information sources, learned modules, and inference paths. Our recovered full-command specialist, however, uses deterministic desired-reference anticipation and a retained SONIC motor path. It is not a reproduction of BFM's CVAE or evidence of arbitrary-mask control. [Behavior Foundation Model for Humanoid Robots](https://arxiv.org/html/2509.13780v1).

BFM-Zero offers a complementary example of a clearly separated training system and downstream interface. Its forward–backward representation learning and behavior embedding support its own reward, goal, and tracking inference procedures. Our navigation adapter does not implement those critics or that latent task-inference rule. The connection is the research goal of reusing motor competence through a compact downstream interface, rather than an equivalence between the learning algorithms. [BFM-Zero](https://arxiv.org/html/2511.04131v1).

Learned hallucination motivates inferring environments around known behavior to obtain useful navigation supervision. In our dataset-generating branch, this idea is implemented with a fixed candidate dictionary and geometry-derived categorical targets. The separately implemented continuous inverse-beam diagnostic instead differentiates a fixed geometric motion-preference objective through a sampled scene distribution. Appendix A describes that branch so its objective is not confused with the generator that produced the current dataset. [Learning from Learned Hallucination](https://arxiv.org/html/2108.09793v1).

## 3. Problem formulation and information boundaries

Let $M$ denote a complete retargeted reference motion, $S$ a static obstacle scene, $h_t$ the measured proprioceptive history, $c_t$ a current motor command, and $g_t$ the public navigation request. We distinguish three functions:

$$q_\psi(j\mid M),\qquad a_t=\mathcal{M}_\theta(h_t,c_t),\qquad \hat c_t=G_\eta(h_t,g_t,S_t).$$

The first scores a discrete obstacle recipe $j$. The second is the recovered motor specialist, including its forecaster, encoder, quantizer, and decoder. The third generates a command from task context. Navigation executes $a_t=\mathcal{M}_\theta(h_t,\hat c_t)$ with the complete motor frozen.

| Stage | Permitted information | Learned component | Output and qualification |
| --- | --- | --- | --- |
| Offline scene generation | Entire reference, native geometry, geometric probes | Recipe-scoring network | Sampled static geometry; no action labels or dynamic guarantee |
| Tracking teacher training | Reference future and simulator training information | SONIC actor and critic | Executed tracking behavior; obstacle/task support needs separate evaluation |
| Full-command motor distillation | Measured history and current full command; true future only for training targets | Desired-reference forecaster | Quantized motor tokens and native 29D actions |
| Navigation training/inference | History, goal request, known map; optional backward-looking localization | Obstacle encoder and command predictor | Internally generated current command, executed through fixed motor |
| Recovery collection | Actual learner prefix, explicit full-command intervention | Collection supplies labels; refitting updates navigation only | A suffix is positive supervision only after an independent task score |

Complete motion is legitimate offline input to the generator. It is excluded from the navigation actor. Likewise, true future reference can supervise motor training without being a student inference input. These distinctions should remain visible in every architectural figure and data receipt.

## 4. Motion2Scene: learning a hindsight obstacle proposer

![Learned hindsight scene proposer](fig1_hindsight_generator.png)

**Figure 2. Learned hindsight obstacle generation.** **(a)** Native forward kinematics converts the complete reference into motion features. A fixed event detector selects five anchors; lateral offsets, heights, and shape templates create 225 candidate recipes. **(b)** A motion MLP encodes a 128D local summary and a recipe MLP encodes six candidate attributes. A shared head outputs one score per candidate, yielding the categorical distribution $q_\psi(j\mid M)$. **(c)** A fixed geometric teacher tests all-frame reference clearance, proximity, and interference with two local counterfactual probes. These labels define $p^*$; conditional cross-entropy and a clear-mass penalty update only the scorer. **(d)** The frozen model scores candidates, which are filtered for reference clearance and sampled sequentially after excluding overlap with already selected objects. The heatmaps show recorded probabilities for clip 00916, arranged as 225 recipe cells rather than a spatial map. Each panel is independently color-scaled to expose relative probabilities; color intensity cannot be compared quantitatively between raw and conditioned distributions. **(e)** Hindsight uses an observed complete motion to propose compatible scenes. Physical continuation qualification is a subsequent experiment: these geometric labels do not establish successful obstacle-present robot behavior. Robot images are reference-pose mesh renders, not executed rollouts.

### 4.1 Native motion features and fixed event anchors

The dataset-generating implementation is `hindsight_study.py`, with kinematics and events in `motion_events.py`. We extract seven tracked locations: center of mass (COM), hip, head, two hands, and two feet. COM uses model masses and inertial centers; the hip uses its body origin, the head and hands use visual-mesh centers, and the feet use ankle-roll body origins. These definitions matter because a label tied to the visual surface should not silently be substituted with a joint origin.

Each reference yields 64 time-varying channels: seven heights, seven speed norms, seven acceleration norms, 21 hip-relative position coordinates, six body angular-speed norms, three COM velocity components, three COM acceleration components, six pelvis-orientation coordinates, pelvis yaw rate, route turn rate, local chord deviation, and hip horizontal speed. Position traces are smoothed with a seven-frame cubic Savitzky–Golay filter and differentiated at the stored sampling frequency. This is an offline, future-aware calculation; the navigation actor does not receive these event features.

An explicit activity score combines turn activity, head-relative vertical motion, and COM acceleration with weights 0.4, 0.3, and 0.3. Each term is normalized using a training-set 95th-percentile scale and clipped at three. Turn activity combines the magnitudes of route turning and pelvis yaw rate. The detector retains both endpoints and three activity peaks separated by at least 12% of the motion duration, with a fixed fallback filling the five-anchor set. The anchors are motion-dependent but are not learned parameters.

### 4.2 Candidate geometry and recipe encoding

For each anchor, we enumerate five lateral offsets, three center heights, and three primitive templates. Lateral offsets are −0.8, −0.4, 0, 0.4, and 0.8 m; heights are 0.4, 1.0, and 1.6 m. The templates are a beam of full dimensions 0.3 × 1.6 × 0.2 m, a box of 0.6 × 0.6 × 0.8 m, and a sphere of diameter 0.6 m. Thus each motion supplies 5 × 5 × 3 × 3 = 225 candidates.

An obstacle center is positioned relative to the anchor's root location, using the lateral direction perpendicular to the local horizontal path tangent. When speed is below 0.05 m/s, root yaw supplies the orientation fallback. After placement, the object remains static in world coordinates. The enumeration defines the support of the distribution: the learned model does not invent arbitrary object shapes or continuously regress their dimensions. The study also does not qualify physical mounting structures for elevated obstacles.

For each anchor, the mean and standard deviation of the 64 channels over a ±0.5 s window form a 128D motion summary. Training-only statistics normalize this vector with a scale floor of 0.1. Each recipe contributes six inputs: normalized time fraction, lateral offset, and height, each divided by three in the implemented encoding, followed by a three-way shape indicator.

The scorer has 21,185 trainable parameters. The motion branch is an MLP with widths 128–64–64; the recipe branch is 6–64. Their 128D concatenation enters a shared 128–64–1 scoring head. Hidden layers use SiLU. Softmax over the 225 resulting logits gives $q_\psi(j\mid M)$. This architecture learns which members of a motion-conditioned candidate set deserve probability mass; it is a categorical proposer, not an object-coordinate regressor.

### 4.3 Geometry-derived hindsight targets

The geometric teacher is fixed. For each obstacle candidate, it evaluates the original reference over all frames using a conservative outer-capsule approximation. Axis sampling subtracts both capsule radius and a sampling-cover term from obstacle signed distance, providing the implemented lower clearance bound. A separate native inner-primitive calculation can witness penetration: a negative inner value establishes interference under that model, whereas an outer overlap alone does not prove a collision.

We define three binary labels. $C_j$ indicates that the all-frame outer clearance lower bound is at least 0.02 m. $N_j$ indicates that the candidate is clear and the reference inner-distance upper bound is at most 0.12 m. $K_j$ indicates that the candidate is clear for the original reference but yields a negative inner-distance value for at least one local geometric counterfactual.

The two probes operate within ±0.5 s of the anchor. A **root-chord probe** replaces the local horizontal root path by the straight chord between its endpoints and shifts the body geometry accordingly. An **entry-articulation probe** holds the body geometry at its entry articulation in body coordinates while following the actual pelvis translation and orientation. These probes ask whether the observed path or articulation changes matter geometrically. They do not preserve all contact or dynamic constraints; for example, held articulation can produce an invalid foot-contact evolution. Accordingly, $K_j$ is a geometric contrast label, not proof that a physically feasible alternative policy would fail.

The target weights and normalized target are

$$w_j=0.1C_j+N_j+3K_j,\qquad p_j^*=\frac{w_j}{\sum_k w_k}.$$

The baseline term retains clear support, the near term favors relevant geometry, and the contrast term emphasizes candidates that distinguish the reference from the probes. There are no human-provided target obstacle coordinates. The self-supervision comes from known motion, fixed geometry, and declared probes; the implementation does not learn the geometric teacher jointly with the proposer.

### 4.4 Learning objective and sampling law

Let $Z_C=\sum_j q_\psi(j\mid M)C_j$ be the raw model mass on clear candidates. Conditioning on that support gives

$$q_j^C=\frac{q_\psi(j\mid M)C_j}{Z_C},\qquad \mathcal{L}_{\rm gen}=\operatorname{CE}(p^*,q^C)-2\log Z_C.$$

Cross-entropy shapes relative probability within the clear set, while the second term penalizes placing raw mass outside it. On the nonempty target branch, the expression is equivalently $\operatorname{CE}(p^*,q)-\log Z_C$. The implementation retains a clear-mass-only branch for an empty target with valid clear support; an empty clear set supplies no ordinary fitting gradient and must remain an explicit abstention. Neither simulator dynamics nor a learned planner is differentiated during this fit.

The declared exploratory comparison uses geometry, hindsight, and hindsight-without-motion-feature targets/inputs, each with two seeds, 200 Adam updates, learning rate 0.001, and batch size 16. The no-motion control zeros the local motion summary, but candidate anchors and recipes remain motion-derived; it is therefore not a control that removes every source of motion information.

After fitting, the selected proposer is frozen. The sampler computes raw scores, reapplies the full-reference clearance mask, draws one uniform number, and uses the categorical inverse CDF to select an object. Before the next draw, it removes candidates overlapping previously selected objects and renormalizes. Empty support produces a recorded abstention rather than repeated draws until a preferred result appears. This is a learned single-recipe distribution combined with a fixed sequential interaction rule, not a jointly learned distribution over complete rooms.

The current build contains 120 references, split into 100 training and 20 development motions, and 240 scene records with either three or five obstacles. With the same seed, the three-object scene is the prefix of the five-object scene. Each record preserves geometry, source lineage, frozen checkpoint identity, and sampling information. These are geometry-only records: the initial build marks them ineligible as executed teacher data. Repairing a reference later also requires reevaluating its clearance against the scene before reusing a previous geometric label. [Generator report](sources/REPORT.md); [build receipt](sources/build-receipt.json).

## 5. Teacher training and the full-command motor student

![Teacher and full-command motor architecture](fig2_teacher_full_command.png)

**Figure 3. Privileged teacher training and full-command motor distillation.** **(a)** The SONIC tracking actor and critic are initialized from the release model and trained on repaired reference motions using simulator rollouts and tracking rewards. Privileged simulator quantities belong to the training system; they are not all direct actor inputs. **(b)** The current full-command schema contains 114 coordinates, of which 113 are available because arrival remains masked. The student additionally receives a 930D measured history. **(c)** A 1044–512–512–576 forecaster predicts nine residual 64D desired-reference frames relative to the current frame. Native packing produces the 640D reference input to the frozen encoder; its latent is quantized and decoded with history into 29 native action values. Only the forecaster is updated during this motor stage. **(d)** Online collection obtains same-state teacher actions and tokens, mixes fresh queries with replay, and trains by action/token imitation with an optional reference loss. Red dashed paths indicate parameter updates; green paths indicate action inference. Teacher-assisted collection is distinguished from unassisted evaluation. The earlier masked CVAE is documented in Appendix B; it is not the recovered specialist drawn here.

### 5.1 Tracking teacher and the limits of its expertise

The repaired teacher protocol uses 89 training motions and 20 development motions. It initializes the SONIC actor and critic from a release checkpoint with a fresh optimizer, uses 128 parallel environments and 24 rollout steps, and allocates 32,000 PPO iterations, corresponding to a budget of 98.304 million transitions. This teacher recovery is performed on a plane. It is not evidence that the actor has learned to navigate the generated obstacle scenes. Checkpoints, reference generations, and subsequent physical evaluations must therefore remain separately identified. [Teacher/motor recovery results](../BFM_MOTOR_RECOVERY_RESULTS_20260913.md).

The selected teacher's saved actor configuration uses gravity direction, angular velocity, joint positions, joint velocities, and previous actions in its 930D history, together with an encoder of future joint and relative-orientation reference information. Root-position error and base linear velocity appear in the privileged critic, not directly in this actor path. In particular, an ideal horizontal translation preserving the actor's joint, orientation, angular-velocity, and history inputs need not change its action. A successful nominal tracker is consequently not automatically an expert for correcting arbitrary global goal-position errors. The 114D student receives additional relative keypoint and desired-velocity information, so this exact input-invariance argument does not apply unchanged to it. [Teacher information boundary](../NAVIGATION_SUPPORTED_RECOVERY_20260913.md).

### 5.2 Full-command interface

The current command has the following layout. Index intervals are zero-based and half-open.

| Group | Coordinates | Dimension | Meaning |
| --- | --- | ---: | --- |
| Heading | 0:2 | 2 | Current heading representation |
| Velocity | 2:5 | 3 | Desired velocity |
| Height | 5:6 | 1 | Desired root height |
| Yaw rate | 6:7 | 1 | Desired yaw rate |
| Arrival | 7:8 | 1 | Unavailable; masked and excluded |
| Keypoints | 8:50 | 42 | Fourteen 3D body targets |
| Joint position | 50:79 | 29 | Current desired joint pose |
| Joint velocity | 79:108 | 29 | Current desired joint velocity |
| Relative orientation | 108:114 | 6 | Two rotation-matrix columns |

The earlier 79D interface ends after joint position. The full-command experiment adds 29 joint velocities and six orientation values. It requires all supported current coordinates; the forecaster rejects incomplete commands other than the declared unavailable arrival entry. Arbitrary-mask behavior should not be inferred from the general BFM motivation or from the existence of earlier masked student classes.

### 5.3 Desired-reference anticipation through a fixed motor representation

Let $r_t^0$ concatenate current desired joint positions, joint velocities, and relative orientation into 64 values. The forecaster receives the 930D history and 114D command, normalized with training statistics. The input scale floor is 0.05 and normalized inputs are clipped to ±20. An MLP with widths 1044–512–512–576 and SiLU hidden activations outputs nine residual frames, rescaled by a fixed per-coordinate reference standard deviation:

$$\hat r_t^k=r_t^0+\sigma_r\odot F_\theta(h_t,c_t)_k,\qquad k=1,\ldots,9.$$

A zero-initialized final layer starts by repeating the current target. Concatenating the current frame and nine predictions yields ten 64D frames. Crucially, the native 640D encoder layout is not a naive chronological flattening. Joint-position frames are flattened into 290 values, joint-velocity frames into another 290, and their concatenation is reshaped into ten 58D blocks before appending six orientation values per block. `reference_layout.py` centralizes this packing.

The frozen encoder has widths 640–2048–1024–512–512–64 with SiLU hidden activations. Its 64 outputs are reshaped as two 32D vectors and passed through the native finite scalar quantizer with 32 levels per dimension. The resulting 64 token values and measured history enter the retained decoder, with widths 994–2048–2048–1024–1024–512–512–29, to produce 29 native action values. These values use the controller's action convention and should not be described as raw joint torques. Encoder, quantizer, and decoder are fixed during forecaster fitting, while the implementation's quantizer gradient path permits learning through the retained motor computation.

### 5.4 Distillation, replay, and evaluation

At a student pre-action state, the fixed reference-conditioned teacher supplies an action target and token target. The motor objective is

$$\mathcal{L}_{\rm motor}=\operatorname{MSE}(a_t,a_t^*)+\lambda_z\operatorname{MSE}(z_t,z_t^*)+\lambda_r\mathcal{L}_{\rm reference}.$$

The optional reference term is a scale-normalized Smooth L1 loss on the predicted frames; its activation and weight belong to the experiment configuration. The target future is permitted for supervision but excluded from the student's inference input. Fresh same-state queries and nominal replay preserve their source identity and timing. Randomized starts and explicit teacher takeovers can supply useful training states, but their intervention fraction must be reported separately from autonomous success.

The retained specialist reaches 88/89 training motions and 10/20 development motions in the selected motor report; two repeat evaluations reach 85/89 and 86/89 on training motions and 10/20 on development motions. Earlier 79D and 114D transformer runs and the offline anticipator provide useful diagnostics, but their changed interfaces, architecture, and optimization prevent attributing all improvement to a single factor. The result supports using this checkpoint as a fixed backend for a bounded navigation study, not claiming a general behavior foundation model has been established. [Motor recovery results](../BFM_MOTOR_RECOVERY_RESULTS_20260913.md).

## 6. Navigation and scene-context distillation

![Navigation context architecture and qualified recovery loop](fig3_navigation_context.png)

**Figure 4. Scene-and-goal control through an unchanged motor specialist.** **(a)** The public actor receives measured history, a ten-value navigation request, and a known map of up to five primitive obstacles; an optional four-value localization feature uses only backward-looking pose differences and a validity bit. **(b)** A shared 15–64–64 obstacle encoder and masked mean pool produce a 64D map representation. A 1004D, or localized 1008D, input passes through a 512–512–114 command predictor. It internally completes the full motor command; external full-command requests can use the preserved motor directly. **(c)** The entire motor, including forecaster, encoder, quantizer, decoder, and inherited normalizers, remains fixed. Actions are judged by physical arrival, speed, sustained hold, contact, fall, and deadline criteria. The robot poses and map path are schematic, not an asserted successful navigation trace. **(d)** The structured loss compares actions against the same frozen motor evaluated at the target command and the same measured history, and balances eight normalized command groups. Recovery data are admitted only after a full-command continuation from an actual learner prefix earns its own on-time goal hold. Gradients update the navigation encoder and command predictor, not the motor. Direct reference prediction, temporally coherent command generation, task reinforcement learning, and camera-based context remain downstream research comparisons.

### 6.1 Public observations and known-map encoding

The navigation input type contains history $h_t\in\mathbb{R}^{930}$, a 10D navigation request, five 15D obstacle rows, and five validity bits. Each obstacle row contains body-frame center (three), size (three), the first two rotation columns (six), and a three-way shape indicator. A shared 15–64–64 SiLU MLP encodes each row. Masked mean pooling forms an order-invariant 64D representation, using a clamped valid-object count to handle a known empty scene. Unsupported geometry, more than five obstacles, and an unknown map are not silently represented as free space.

The ten request values comprise body-relative start and goal positions, a goal tolerance, terminal-speed threshold, required hold, and a map-known indicator. The actor requires the map-known entry to be valid. This mode does not include a remaining-time signal or a terminal-heading request. The geometry interface is a structured known map, not a camera, depth encoder, occupancy completion model, or learned perception system.

The optional localization input estimates body-frame velocity from the current pose and the pose five control steps earlier, divided by the actual elapsed time, and adds a validity bit. It invalidates the first five steps after reset. No simulator velocity is passed as this feature, but the study assumes an ideal pose stream; robustness to estimated localization remains untested.

### 6.2 Command generation and same-backend supervision

The command predictor concatenates normalized history, request, pooled map, and optional localization. Its widths are 1004–512–512–114 without localization and 1008–512–512–114 with it. Hidden activations are SiLU; the last layer is zero-initialized. Its output is denormalized into a current command. The unavailable arrival coordinate remains masked when the command enters the fixed motor.

For a target command $c_t^*$ at recorded history $h_t$, we compute

$$a_t^*=\operatorname{stopgrad}\mathcal{M}_\theta(h_t,c_t^*).$$

The structured objective is

$$\mathcal{L}_{\rm nav}=\operatorname{MSE}\left(\mathcal{M}_\theta(h_t,\hat c_t),a_t^*\right)+0.1\mathcal{L}_{\rm group}(\hat c_t,c_t^*).$$

Here $\mathcal{L}_{\rm group}$ averages the normalized coordinate MSE within each of the eight available command groups and then averages those group losses. This avoids allowing the 42D keypoint group to dominate a one-dimensional height or yaw-rate target solely because it has more coordinates. Arrival is excluded. Original-teacher action error remains a separate diagnostic; it is not substituted for the representable same-backend target.

Freezing parameters does not mean detaching the motor's input computation. Gradients pass through the motor to the navigation prediction while all inherited motor tensors and normalizers remain unchanged. Tensor and full-command anchor checks verify preservation. Inference requires only the public navigation input and the internally generated command; no motion ID, phase, target future, or outside full command is supplied.

### 6.3 From teacher-state imitation to supported recovery

State distribution is a second issue beyond target representability. Even a same-backend target evaluated on original-teacher histories may inadequately cover histories the deployed motor or navigation actor visits. We therefore first collect exact-motor demonstrations by executing the frozen full-command specialist and recording its actual pre-action histories, current commands, and actions. This provides a direct state-distribution control before assigning all residual error to the navigation architecture.

We then run the current navigation actor to a declared switch time and intervene with the frozen full-command motor's nominal continuation. State, history, reference timing, and the original deadline are retained. There is no phase jump, reference translation, or robot teleport. A continuation is admitted only if its own physical suffix earns a fresh 50-tick goal hold before the deadline and the entire attempt has no prohibited contact. Hold accumulated by the navigation prefix cannot qualify a shorter suffix. Failed attempts remain in coverage and cost accounting and provide no positive imitation rows.

Only the first admitted suffix row is a verified learner-state query. Later rows are motor-executed recovery states. This distinction prevents counting a long successful continuation as hundreds of independently supported learner queries. The procedure is a bounded intervention-based aggregation experiment motivated by learner-induced distribution shift, not dense expert labeling or a test of DAgger's full guarantees. [DAgger](https://proceedings.mlr.press/v15/ross11a.html).

### 6.4 Physical task contract

The current navigation panel contains eight tasks from four clips and one ancestry group. Success requires the root to remain within 0.25 m of the goal in three dimensions while 3D speed is at most 0.10 m/s for 50 consecutive control ticks at 50 Hz. Prohibited contact above 1 N, pelvis height below 0.25 m, and the original task deadline constrain the attempt; contacts are checked at the 200 Hz simulation step. There is no additional posture-success criterion in this scorer.

Entering the goal sphere, producing scene-sensitive actions, or obtaining low action MSE is insufficient. The robot must approach, brake, and maintain the hold on time. Recoveries that succeed only after extending the deadline remain failures under the original task. This contract is central to interpreting both training data and downstream results.

## 7. Current evidence and what it establishes

### 7.1 Generator diagnostics

The hindsight study has 22,500 training and 4,500 development candidates. Only 74 training and eight development candidates satisfy the geometric contrast label; 62/100 training motions and 17/20 development motions contain no such candidate. Thus the candidate dictionary and geometric probes provide sparse contrast supervision.

The following values are raw, pre-filter model probability masses averaged over two fitted seeds and all 20 development motions. They are not sampled-scene collision rates or robot success rates.

| Proposer | Contrast mass | Near mass | Clear mass | Penetration mass | Unknown mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Geometry target | 0.312% | 24.437% | 76.565% | 20.645% | 2.790% |
| Hindsight target | 0.317% | 19.606% | 71.691% | 25.479% | 2.830% |
| Hindsight, zero motion summary | 0.292% | 16.111% | 68.924% | 28.321% | 2.755% |
| Event heuristic | 0.233% | 7.313% | 38.372% | 58.729% | 2.899% |

The hindsight fit does not demonstrate the desired combined improvement in contrast, proximity, and clear probability. Geometric filtering is therefore an explicit component of generation, not evidence that the network alone has learned collision safety. Label generation used 25,335 native kinematic frames, 27,000 target geometry queries, and 54,000 counterfactual queries, with zero physics rollouts. The recorded local CPU study took approximately 20.5 s in total; this is a machine-specific receipt, not a throughput benchmark. [Recorded study](sources/REPORT.md).

### 7.2 Motor recovery and the navigation gap

| Diagnostic | Training motions | Development motions | Interpretation |
| --- | ---: | ---: | --- |
| Fresh full-reference teacher | 88/89 | 15/20 | Privileged reference-tracking control |
| Teacher with current frame repeated | 16/89 | 1/20 | Missing reference future is consequential |
| Selected full-command anticipator | 88/89 | 10/20 | Recovered specialist; development gap remains |
| Anticipator repeat evaluations | 85/89 and 86/89 | 10/20 and 10/20 | Selected-checkpoint repeatability, not independent training seeds |

An earlier navigation terminal-control study illustrates why fitting metrics cannot replace physical outcomes. Its archived model has measured action MSE 0.004055 on the expanded 3,900-row view and completes 0/8 tasks. Structured fitting reduces MSE to 0.000296 but still completes 0/8; adding causal localization gives MSE 0.000299 and 1/8. That isolated completion fails on two further evaluation seeds. These are training-family diagnostics, and different training views prevent treating every row as an equal-data ablation. [Terminal-control audit](../NAVIGATION_TERMINAL_CONTROL_20260913.md).

### 7.3 Completed exact-motor and recovery experiment

The latest study executes the full-command motor successfully on all eight tasks, producing 3,076 demonstration rows. A 6,000-update localized navigation fit completes 2/8 tasks. Two continuations then receive the same 3,000-update optimization budget: replay alone reaches 3/8, while a 50% qualified-recovery / 50% demonstration-replay mixture reaches 4/8 on evaluation seed 91260. Both use batch 256, AdamW at 0.0001, training seed 91370, and identical optimizer restarts and inherited normalization.

Six of eight learner-prefix interventions qualify, producing 1,250 motor-executed recovery rows and six learner-state queries. Two failed interventions contribute no training rows. The admitted pool therefore contains 4,326 rows, while every attempted intervention remains recorded. The recovery branch gains two tasks and loses one relative to replay, for one net additional completion; all 24 main navigation evaluations remain free of prohibited contact and above the fall threshold.

A predeclared full-panel confirmation of the selected recovery checkpoint on evaluation seed 91261 yields only 1/8. All four main-seed successes fail, while a previously failed task succeeds. Four adaptively selected confirmation failures subsequently receive full-command controls, of which three pass and one still fails the hold requirement. Those four controls are not a complete oracle panel. The evidence identifies both navigation command-generation instability and local motor-continuation limits. It does not support promoting this checkpoint as a robust navigation improvement. [Completed recovery study](../NAVIGATION_SUPPORTED_RECOVERY_20260913.md).

## 8. Next experiments and downstream research

The proposed end-to-end contribution should be evaluated by whether generated scenes increase useful, physically qualified training coverage at a controlled cost. The current geometry, motor, and navigation results come from linked but distinct studies. Combining their favorable numbers does not establish a single successful scene-generation-to-navigation experiment.

A first matched comparison should separate **scene proposal** from **supervision collection**. Compare declared analytic or event proposals with the learned hindsight proposer, and nominal demonstrations with qualified learner-state recovery. Match source ancestry, candidate support where possible, reference repair, navigation architecture, optimizer budget, and task evaluation. Report every proposal, abstention, clearance rejection, executed teacher attempt, supported query, recovery row, and physics transition. Evaluate on held-out source families and layouts rather than counting multiple scenes from one motion as independent generalization cases.

Generator improvements should first address the sparse contrast support. Candidate shape, station, or pose families can be expanded in a separately registered experiment; probe design should then be assessed for physical relevance. Learning a broader distribution without a stronger supervision signal may merely spread probability across equally uninformative recipes. A generator utility study must compare downstream completion at a matched simulation budget, not only geometric contrast mass or acceptance rate.

For downstream control, the command bottleneck offers several bounded comparisons. A direct goal/map-to-desired-reference head can test whether explicitly predicting 114 intermediate coordinates helps or hinders task control. Temporally coherent reference chunks can test whether independently generated current commands undermine braking and stable hold. Task-feedback optimization in command space, constrained by imitation replay and the unchanged motor, can test whether a nominal reference teacher lacks the goal-correcting actions needed at learner arrival states. Each route needs its own executed results; none is implemented merely by drawing it as a possible downstream branch.

The public context interface can later expand from known primitive geometry to camera-derived or uncertain scene representations. That would require a perception or belief model, observation-history design, missing-map behavior, and new evaluation conditions. Current known-map results do not establish visual navigation, arbitrary-scene generalization, language control, or BFM-Zero-style zero-shot reward inference. A credible downstream program should add these interfaces one at a time while retaining the same task and motor-preservation checks.

## 9. Discussion and conclusion

The central methodological distinction is between explanatory geometry and actionable supervision. A candidate scene may clear a reference and obstruct a geometric shortcut without admitting a useful physical teaching continuation. A nominal tracker may execute a motion well without correcting a navigation-induced endpoint error. A student may reproduce teacher-state actions accurately without reaching and holding a goal under its own closed-loop state distribution.

Our framework makes these transitions explicit. Motion2Scene learns a distribution over declared obstacle recipes from hindsight geometric labels. Full-command distillation recovers missing desired-reference context through a retained motor representation. Navigation then generates that command from scene and goal information, with recovery labels admitted only after physical qualification. The current results support this experimental decomposition and reveal where its assumptions fail. Demonstrating reliable downstream transfer from learned scene generation remains the next empirical objective.

## Appendix A. Continuous inverse-beam learning is a separate branch

![Continuous inverse beam diagnostic](figA_inverse_beam.png)

**Figure A1. Differentiable inverse-beam diagnostic.** A body/temporal encoder maps a target capsule sequence to a four-component correlated Gaussian mixture over two unconstrained variables. Reparameterized samples are transformed into bounded beam station and height. A fixed geometric selector compares target and competing motion energies, producing a preference loss and target-clearance barrier; a KL term regularizes the mixture toward a standard-normal prior in the unconstrained coordinates. Gradients update the encoder and mixture head. Unlike Figure 2, this diagnostic learns continuous placement parameters through a fixed motion-choice objective and uses no distribution of target obstacle recipes. It did not generate the hindsight-801 dataset.

Each capsule frame has seven values per body: midpoint coordinates, axis coordinates, and radius. A shared 7–16 Tanh body encoder is followed by temporal convolutions with kernels 5, 5, and 3, width 64, Tanh activations, and temporal average pooling. A conditioned head produces four sets of mixture logits, two means, two scales, and a correlation. Scales use softplus plus 0.03; correlation is bounded by 0.95 times Tanh.

For component $k$, sample $u=\mu_k+L_k\epsilon$, with standard-normal $\epsilon$, and transform by a bounded sigmoid to beam station 0.35–0.65 and center height 1.10–1.45 m. Beam dimensions are fixed at 0.1 × 1.2 × 0.1 m, with fixed yaw. During training, all four components contribute two reparameterized samples each, weighted by their mixture probabilities; the KL estimate uses the full marginal mixture density.

The fixed candidate energy for motion $a$ is

$$E_a=b_a+200(0.005)\operatorname{softplus}\left(\frac{0.01-d_a}{0.005}\right),$$

where $d_a$ is geometric clearance and the base costs are zero for upright walking and one for crouching. The preference and feasibility objectives are

$$\mathcal{L}_{\rm pref}=\operatorname{softplus}\left(\frac{\max_{a\in A_{\rm target}}E_a-\min_{b\in A_{\rm other}}E_b}{0.25}\right),$$

$$\mathcal{L}_{\rm clear}=\left(\frac{\max(0,0.01-\min_{a\in A_{\rm target}}d_a)}{0.01}\right)^2.$$

The fit minimizes their weighted expectation plus distribution regularization:

$$\mathcal{L}_{\rm inv}=\mathbb{E}[\mathcal{L}_{\rm pref}+5\mathcal{L}_{\rm clear}]+0.02\operatorname{KL}\left(q_\psi(u\mid M)\Vert\mathcal{N}(0,I)\right).$$

No target obstacle coordinates enter this objective. The desired motion choice and fixed geometry provide the learning signal. The specified diagnostic uses 300 Adam updates at 0.002; its seeds and single carrier are separate from the two-seed categorical study. As in the categorical generator, favorable geometric objectives do not by themselves establish obstacle-present robot execution. [Implementation](../../../gear_sonic/dataset_generation/hallucination/motion2scene_inverse.py); [inverse-learning specification](../INVERSE_LEARNING_V1.md).

## Appendix B. Earlier BFM-inspired masked variational student

The earlier `TransformerMotionFoundation` accepts a 930D history and 79 command values with availability masks. It splits history into ten 93D feature blocks; these blocks must not be described as ten chronological observation frames. The command is padded to 80 values and divided into eight ten-value blocks, each concatenated with its mask. History and command blocks are projected to width 256 and processed as 18 tokens by four transformer layers with eight attention heads, GELU activations, feed-forward width 1024, learned positional embeddings, and no dropout. Mean-pooled history and command features are concatenated and mapped to a 192D condition.

A 192–256–256–128 prior predicts a 64D mean and log variance. A posterior using the condition, a 1645D privileged training vector, and a 640D future reference passes through 512–256–128 layers and predicts a residual posterior mean and log variance. A 64–512–512–64 latent adapter feeds the retained quantized token interface and frozen decoder. Training combines action reconstruction, posterior-to-prior KL, and prior action imitation. A later full-command transformer adds a zero-initialized 70–256–192 branch for the 35 added coordinates and their masks.

This lineage explains the project's BFM terminology and its mask-based experiments. The recovered motor in Figure 3 instead uses a deterministic 576-output forecaster and requires full current commands. The manuscript should not merge the posterior/prior arrows of this older architecture with the current forecaster or attach BFM-Zero's forward–backward critics to either model. [Variational student](../../../gear_sonic/research/scene_distillation/transformer.py); [full-command motor](../../../gear_sonic/research/scene_distillation/anticipatory_motor.py).

## Appendix C. Provenance, reproducibility, and revision notes

This revision completes the supplied teacher–student draft by adding the actual dataset-generating Motion2Scene mechanism, specifying the recovered motor and navigation interfaces, and replacing the formerly proposed recovery experiment with its completed results. It does not edit the attachment or claim that future experiments have been run. The attachment text and input hashes are retained in `sources/`.

| Evidence | Local source |
| --- | --- |
| Categorical generator, targets, training | [hindsight_study.py](../../../scripts/research/lflh_next/navigation/hindsight_study.py) |
| Native motion features and event detection | [motion_events.py](../../../scripts/research/lflh_next/navigation/motion_events.py) |
| Clearance-conditioned probabilities | [constrained.py](../../../scripts/research/lflh_next/constrained.py) |
| Sequential sampling | [sampling.js](../../../scripts/research/lflh_next/navigation/sampling.js) |
| Paired dataset assembly | [build.py](../../../scripts/research/lflh_next/dataset/build.py) |
| Forecaster and native reference packing | [anticipatory_motor.py](../../../gear_sonic/research/scene_distillation/anticipatory_motor.py), [reference_layout.py](../../../gear_sonic/research/scene_distillation/reference_layout.py) |
| Navigation actor and structured objective | [navigation_motor.py](../../../gear_sonic/research/scene_distillation/navigation_motor.py) |
| Motor evidence | [recovery report](../BFM_MOTOR_RECOVERY_RESULTS_20260913.md) |
| Terminal-control evidence | [terminal report](../NAVIGATION_TERMINAL_CONTROL_20260913.md) |
| Latest supported-recovery evidence | [recovery study](../NAVIGATION_SUPPORTED_RECOVERY_20260913.md) |
| Frozen generator fits and dataset receipt | [model lock](sources/model-lock.json), [build receipt](sources/build-receipt.json) |
| Attachment and evidence identity | [input manifest](sources/input_manifest.json) |
| Rendered pose sources | [pose provenance](assets/pose_provenance.json) |

Generator and reference-repair generations must remain hash-bound. The motor's normalization and all retained encoder/quantizer/decoder tensors must remain fixed in navigation comparisons. Split at motion/source ancestry before scene expansion; the three- and five-object variants are correlated. Record failed and unsupported attempts with their costs. Evaluate stable physical completion separately from geometric acceptance, action error, and assisted data collection. These requirements make a future claim about learned scene utility testable without inflating the current evidence.
