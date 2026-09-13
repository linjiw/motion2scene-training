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

