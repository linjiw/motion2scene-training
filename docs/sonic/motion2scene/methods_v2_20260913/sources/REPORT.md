# Motion observations and hindsight obstacle proposals

New exploratory amendment written 2026-09-11T20:23:37Z; all earlier route/fixed-grid development outcomes accessible. Same 100 training and 20 previously inspected development motions. No original pilot change, protected-layout access, Isaac Lab episode or hardware execution. The current user requested body-motion-driven proposals. This record reports both the implementation and its adverse result.

## Research question and prior work

A turn is evidence of a motion change, not evidence that an obstacle caused it. For a Z route, an informative static obstacle would block a shorter corner-cutting alternative while keeping the demonstrated route clear. An overhead obstacle might similarly distinguish a lowered pose from an upright alternative. Random turning, style and animation artifacts are alternative explanations.

[LfLH](https://arxiv.org/html/2108.09793v1) learns obstacle distributions through a motion-planning decoder, with the observed motion supplying the learning signal. It motivates learning an inverse relationship between motion and obstacles. Our small reference-geometry experiment has no demonstrated optimal-planner reconstruction guarantee or downstream humanoid utility.

[Learning from Hallucinating Critical Points](https://arxiv.org/html/2509.26513v1) already separates critical obstacle configurations from subsequent obstacle-trajectory generation. This is directly relevant prior work, so identifying useful locations is not by itself a novelty claim. Its ground-robot dynamic setting and planner formulation do not establish the current humanoid pipeline. The possible contribution here remains body-aware, physically qualified decision data, which is untested.

## What was implemented

Native-model COM is the mass-weighted sum of body inertial centers transformed by recorded body poses. Model mass is 33.341142 kg; this is not a hardware measurement. Observations include hip body origin, head and hand visual-mesh centers, and foot ankle-roll body origins. Head orientation refers to the attached torso body, not an independently articulated head joint. COM has no orientation. The viewer labels these distinctions.

Each full-record frame has 64 channels: seven point heights, seven speed norms, seven acceleration norms, 21 hip-relative coordinates, six body angular-speed norms, COM velocity and acceleration xyz, six pelvis-orientation matrix entries, pelvis yaw rate, route turn rate, local chord deviation and hip xy speed. Positions are smoothed with a seven-frame cubic Savitzky-Golay filter before finite differences at the stored fps. Angular speeds come from relative rotation matrices. This is offline hindsight with future frames; these features must not enter the frozen causal 114-dimensional policy inputs. Derivatives are kinematic estimates, not forces or contact labels. The inherited root/height normalization and unqualified generated references remain limitations.

The event detector is explicitly specified, not learned: weighted turn activity, head-relative vertical speed and COM acceleration, each normalized by its training-frame 95th percentile and clipped. Endpoints plus three separated peaks choose five anchor times. This observation-based proposal rule is shared across every arm; 225 candidates per motion keep the same lateral offsets, heights and three shapes. Continuous learned placement and learned anchor discovery were not implemented in this step.

A 21,185-parameter network scores each candidate from its recipe and a 128-dimensional local summary: mean/std of the 64 observations over a +/-0.5 s window. The geometry arm learns sufficient-near labels. The hindsight arm learns a target proportional to `0.1*clear + 1*near + 3*contrast`. The no-motion arm receives zero motion features with identical architecture. All have seeds 801/802, Adam .001, batch 16, 200 updates and the same geometry/labels. The loss is weighted conditional coverage plus twice negative log retained-clear probability; a zero-target row retains the full clearance penalty. Six final checkpoints were locked before scoring, with no selection or additional fitting.

The two artificial alternatives are a local root-position chord and a pose held at the window-entry articulation, translated and rotated with the actual pelvis. Their endpoints/pose-transition behavior has not been physically qualified. In particular, held feet can be incompatible with stance. They are geometric probes, not executable teacher continuations. Observed-reference clearance uses all recorded frames and conservative native outer bounds; a negative alternative witness uses only native inner primitives within its window. No outer-mesh overlap is treated as penetration proof. All sizes, rotations and body geometry are included.

## Results: supervision is sparse and the new objective did not deliver the hoped-for gain

Only 74/22,500 training cells and 8/4,500 development cells show reference-clear/alternative-penetrating contrast. 62/100 training and 17/20 development motions have no such cell. Zero rows remain. The observed development contrasts occur in only three motions, and those are not established independent experimental replicates.

| Raw model | Kinematic contrast mass | Sufficient-near mass | Clear mass | Penetration witness | Unknown |
|---|---:|---:|---:|---:|---:|
| geometry | 0.312% | 24.437% | 76.565% | 20.645% | 2.790% |
| hindsight | 0.317% | 19.606% | 71.691% | 25.479% | 2.830% |
| hindsight_no_motion | 0.292% | 16.111% | 68.924% | 28.321% | 2.755% |
| explicit_event | 0.233% | 7.313% | 38.372% | 58.729% | 2.899% |
| analytic_target | 4.074% | 69.397% | 100.000% | 0.000% | 0.000% |

Learned rows average both seeds over all 20 motions. Explicit-event probabilities depend only on the manually specified event score. Analytic-target probabilities use the complete geometric labels directly and are a privileged label-oracle diagnostic, not a deployable learned baseline. The no-motion network still shares motion-derived anchors and recipe timing, so its construction is not motion-independent.

Hindsight contrast mass is essentially unchanged from geometry (about 0.317% versus 0.312%), while its penetration mass is worse (25.48% versus 20.64%). Motion features improve this particular hindsight model over its no-motion counterpart, but that does not beat the geometric objective or establish physical utility. New and prior studies use different candidate supports, so their percentages are not controlled before/after policy effects. Action diversity, event concentration and geometric contrast do not substitute for contact-qualified passage.

All per-seed and per-motion raw/projected rows are in outcomes.csv. All 40,500 candidate probability rows include reference and both alternative bounds. Paired descriptive counts and differences are retained separately. p and intervals are NA: no registered exchangeability/independent-unit scheme, shared generated source/corpora/contexts, and no confirmatory physical estimand. No equivalence or non-inferiority claim is made. Clearance projection uses the bounds at scoring time; zero projected penetration follows rejection and is not learned safety.

## Resource and audit receipt

Six completed fits, 1,200 updates, one study attempt, no technical retry. 25,335 kinematic pose calls; 27,000 reference and 54,000 counterfactual candidate queries. Measured labelling 15.7198 s; training/checkpoint loop 3.6421 s; total study 20.5033 s. CPU two torch threads. GPU usage, energy and peak memory NA; zero physics steps. The viewer additionally evaluates 1,200 kinematic poses for mesh replay, not a dynamics experiment. Source hashes, actual executed code, models, labels, outcomes and display assets are archived.

27 focused tests pass, including time-correct derivatives, Z-turn detection, shortcut endpoints, zero-target clearance loss, route coordinate handling and conservative geometry/loss checks. Browser checks cover all selectable motion signals, COM's undefined orientation, body markers, both seeds, distribution colors, projection, mesh playback, mobile layout and abstention. Exact rendered coordinates and probabilities match the stored labels/checkpoints.

## Decision and next permitted action

Meaningful teacher-data utility: INCONCLUSIVE / UNTESTED. The learned hindsight objective did not improve the desired combined geometric outcome in this development comparison. The intended body-aware observations and their inspection tool are implemented; they are not proof of criticality.

The next discriminating change should address candidate/alternative coverage, rather than adding epochs or enlarging the network after this result. The current .4 m lateral grid and rigid obstacle templates may miss thin regions that separate a route from a short local chord; this is an untested explanation, not a diagnosed sole cause. A separately dated study could optimize bounded obstacle position/shape using conservative reference-clearance and alternative-interference losses, with rejection/abstention on failure and an analytic optimization baseline. Do not choose its margin or interval after examining outcomes.

Ultimately substitute matched, SONIC-qualified continuations for artificial pose holds and chords: same start, task, goal and observation contract, complete continuation labels, and the original contact/stabilization rules under an authorized physical budget. For teacher learning, preserve causal sensing inputs, source-ancestry partitions and all whole-chain outcomes. No new physical run, dataset utility claim or 1,000/10,000 scaling is permitted by these proxy results alone.
