# Foundation training and scene/navigation handoff — 2026-09-12

## Executed work

The authorized 3000-update foundation fit completed on 21 repaired, whole-qualified teacher episodes (6433 labels). A further bounded 1000-update exploratory DAgger fit also completed. No teacher optimizer training restarted. All five standalone foundation evaluations exited successfully; zero completion is a measured policy result, not a simulator crash.

The initial fit's last logged prior action MSE was 0.003083. Low offline loss did not translate into closed-loop control:

| Student | Command profile | Completed / 109 | Mean progress on 21 fitted motions | Development mean progress |
|---|---|---:|---:|---:|
| Initial 3000 updates | Full | 0 | 14.6% | 4.5% |
| Initial 3000 updates | Root | 0 | 15.0% | 4.5% |
| Initial 3000 updates | Navigation | 0 | 14.7% | 4.5% |
| Plus 1000 DAgger updates | Full | 0 | 19.9% | 5.2% |
| Plus 1000 DAgger updates | Navigation | 0 | 17.0% | 5.3% |

The evaluation uses 89 repaired train and 20 development clips, seed 91260, FP32 public-prior inference and native reference-tracking termination. No posterior drives the robot. Navigation here means the four-command mask, not scene/goal navigation. Reference-pose termination may reject a different valid gait, so these results do not replace a reference-independent velocity-following evaluator. Nevertheless, failure under full commands as well means we cannot attribute everything to that evaluation limitation.

The DAgger collection used 80% teacher interventions and 21 train motions: 10479 environment transitions, 8351 interventions, 3576 numerically supported query rows. It completed 21/21 assisted episodes; this is **not student-only success**. Those query labels are explicitly exploratory, not physically qualified recoveries. The fit balanced episodes and used a fresh optimizer initialized from the initial foundation weights. Its mask mixture and sampling also changed, so it is not an isolated estimate of DAgger's causal benefit.

## Why the next motor experiment must change

On 1295 recorded training rows, swapping command vectors while retaining history barely changes initial-student imitation MSE: full-command MSE 0.003222 versus shuffled 0.003280. DAgger increases sensitivity somewhat (0.003727 versus 0.004161), without producing complete rollouts. This supports a history-dominated shortcut / limited command dependence as a hypothesis; it is not proof of the single failure mechanism. Perturbed requests may be outside demonstrated combinations.

The [BFM paper](https://arxiv.org/abs/2509.13780) combines masked online distillation with a conditional VAE. Our small, initially offline SONIC-token student is an adaptation, not a reproduction or an established full behavior foundation. [DAgger](https://arxiv.org/abs/1011.0686) specifically addresses the policy-dependent observation distribution in imitation learning; its motivation supports collecting student-state labels, but does not guarantee that one assisted round fixes our system.

Next discriminating motor experiments: first verify exact teacher-action replay and same-state student/teacher action errors at initialization and over the first failure window; then compare a full-command-only baseline against masked training with explicit command-response/teacher-token supervision. Preserve history and vary reachable requests under the teacher to construct meaningful conditional labels. Increase online coverage progressively with explicit support/failure accounting. Add reference-independent command rollouts with fall/contact limits before claiming gait control; keep native full-reference evaluation alongside them. Do not silently loosen tracking qualification to make more labels eligible.

## Scene export bug fixed and validated

The earlier task preparation incorrectly permuted serialized SONIC DOFs as though they were live Isaac Lab state. Serialized DOFs already use MuJoCo order. The incorrect conversion changed some joint angles by as much as 2.36 rad. The first four scene-probe results are invalidated for teacher-performance interpretation; original evidence remains under scene-tasks/.

Added `sonic_motion_entry_to_qpos` to the data adapter, a nonuniform-joint round-trip regression test and a shape-rejection test. Added the reusable `prepare_repaired_tasks` CLI with episode/motion hash checks, train-only checks and source-prefix preservation assertions. Rebuilt all 20 proposals under **scene-tasks-v2/**, verified all original motion fields before the added hold, and reran four native probes. The plane collection and foundation training never used the corrupted scene export and are unaffected.

Corrected results:

| Scene teacher probe | Goal reached | Contact check | Strict full qualification |
|---|---|---|---|
| 00463 clear | Yes, final distance 0.104 m | Pass | Fail: tracking/hold |
| 00463 corridor | Yes, 0.109 m | Pass | Fail: tracking/hold |
| 00134 clear | Yes, 0.097 m | Pass | Fail: tracking/hold |
| 00463 low beam | No, 4.401 m | Fail, peak undesired normal force 1806.7 N | Fail |

The clear/corridor mean body errors are about 0.113–0.121 m, exceeding the fixed 0.10 m limit; the 50-tick terminal-hold check also fails. Appending a static final pose does not establish physically stable stopping. A proper deceleration/arrival continuation must be executed and qualified. The low beam needs a matching traversable posture/continuation, not merely an obstacle placed above a walking reference.

## Navigation and residual implementation status

Implemented components already present: public measured-history/goal/known-map encoder; masked prior/posterior foundation; frozen SONIC decoder; recurrent four-command director; bounded action residual; same-state query/scene registry contracts; recurrent imitation; bounded residual PPO with task rewards. This turn executed the foundation and exploratory query paths, corrected scene conversion, and exercised the scene/contact qualifier. The downstream scene navigation and residual PPO paths are **not yet physically validated**.

The current candidate is bound by hash in the navigation templates. Its actual failed command qualification is recorded; navigation preflight correctly rejects it. There are zero qualified scene continuations in the four executed probes. Therefore no navigation imitation or residual PPO was launched. This is an evidence dependency, not a request for additional user permission.

Execution order remains: reliable public motor control -> stable stop/turn/posture command checks -> executed scene/goal/terminal-hold qualification -> nominal navigation imitation -> supported DAgger -> matched command-only versus bounded action-residual comparison. Residual starts at zero, limit 0.05 per native action coordinate, with magnitude/smoothness penalties; these are experiment bounds, not physical safety certificates. Keep the base foundation/decoder frozen downstream. Use future trajectory and motion ID only for expert/posterior supervision. Low-beam traversal requires qualifying richer posture/skill commands beyond the current four-value navigator.

The design and contract implementation are reviewable, but a usable full foundation/navigation policy has not been achieved. The priority is control responsiveness and arrival behavior, rather than adding downstream optimization on an unqualified motor model.

## Evidence and validation

Packet: /home/linjiw/research-data/m2s-bfm-repaired-navigation-prep-20260912/.

- foundation/receipt.json; dagger-round-1/fit/receipt.json; student-results.json; all five native evaluation metrics/commands/receipts.
- command-sensitivity.json; dagger-round-1/collection/collection.json with interventions/query support retained.
- SCENE-EXPORT-CORRECTION.json; scene-results-v2.json; scene-tasks-v2/ native contact and tracking evidence.
- foundation-command-qualification.json; navigation-preflight-after-training.json.
- 51 focused tests passed, including the two new conversion regressions. Black/Ruff checks passed for changed code. Reusable CLI generated and validated another 20-task packet with motion-prefix preservation.

The bounded fits and evaluations are complete. No long-running student/navigation job remains active from this experiment.
