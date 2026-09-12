> 2026-09-12 update: [100-motion experiment, scene implementation, and trajectory-role correction](../../../../docs/motion2scene/BFM_SCENE_NAVIGATION_COVERAGE_20260912.md). The final student is not navigation-qualified; use the corrected video/scene receipts linked there.

# BFM foundation, scene navigation, and action-residual training

September 12, 2026. This revision implements the user's request to enable student residual learning. It supersedes the earlier command-only primary proposal in `FOUNDATION_NAVIGATION.md`; command-only remains the necessary comparison. This is our BFM-style adaptation, not a reproduction of the authors' implementation.

## Architecture and objective

1. The selected SONIC teacher remains frozen and consumes its native target/reference inputs. Its decoder is bound to the same checkpoint as its action labels.
2. `MaskedMotionFoundation` learns a scene-independent Gaussian prior and privileged posterior, plus a quantized SONIC token adapter. Its public inputs are 930D measured history and explicitly masked motion commands. Only its posterior receives the 1645D critic observation and 640D future target input.
3. A recurrent navigation head consumes measured history, body-frame start/goal, and the known obstacle map. It emits forward/lateral velocity, yaw rate, and anchor height into the frozen foundation. Future reference, phase, route and motion ID do not enter this actor.
4. A bounded action residual modifies the frozen motor output: `action = frozen_decoder(tokens, history) + limit * tanh(residual_head(public_hidden))`. The new head starts at zero. The proposed pilot limit is 0.05 per native action coordinate, before native action processing; this is an engineering parameter, not a hardware safety limit. The original prior/decoder weights remain unchanged even when residual actions differ.
5. After imitation, a tanh-Gaussian PPO residual can optimize measured task reward. The command navigator and motor foundation remain frozen during this stage. The PPO actor can start from the learned imitation residual; it replaces that residual rather than stacking an additional correction. Deterministic initial actions then preserve the imitation policy. Stochastic exploration remains within the declared bounds.

The existing standardized latent residual remains optional, with default scale zero. Do not enable both adaptations in the primary experiment: action residual is the requested main candidate; command-only and latent residual are separate comparisons. More adaptation capacity does not guarantee improvement.

The [BFM paper](https://arxiv.org/html/2509.13780v1) motivates masked control, a conditional VAE and online distillation; its downstream residual is in action space. Our fixed SONIC decoder, 79-component command interface, navigation director and bounded residual recipe are specific engineering choices.

## Command and data contracts

`commands.py` defines 79 values plus 79 availability bits: eight root/control slots, 14×3 body-point targets in the measured anchor frame, and 29 target joint angles in native order. The root slots retain the earlier field names; `base_height` is implemented using the configured motion anchor's height above the environment origin. The initial collector does not invent an arrival-speed command and masks that slot out. Masks are an equal mixture of full, root, keypoint, joint and four-command navigation profiles. They never hide scene obstacles.

`collect.py` attaches read-only hooks to the actual G1 encoder and action decoder, retaining the inputs and actions from the same control decision. Physics is stepped only after capture. The teacher-rollout path keeps every motion attempt and admits complete episodes only when the explicitly locked root/body error criteria pass. These are tracking screens, not scene or contact qualifications.

`DaggerFoundationCollectionCallback` drives the simulator with the student's public prior and queries the teacher on the exact current observation. It censors reset/termination tails and records a per-state numerical support mask. These predictions are explicitly **not physically qualified recovery labels**. Fitting them requires `allow_exploratory_queries=true`; they cannot silently enter a qualified scene-teacher dataset.

`aggregate.py` binds teacher and student-state collection manifests and preserves their distinct sources, failures and hashes. `train.py` balances episodes rather than frames, so long episodes do not automatically dominate. Mixing two teacher episodes and two queried episodes yields equal episode-level weight to the two sources, despite unequal frame counts.

The foundation objective is posterior action reconstruction + `0.001 KL(q||p)` + `0.1 public-prior action MSE`. The direct prior term is our explicit adaptation. Teacher token/action parity is checked before fitting. The actual TF32 teacher differed from full-precision replay by up to 0.003684 native action units across the inspected rows. The pilot uses an audited absolute tolerance of 0.0045 plus relative tolerance 0.0002; the previous default remains 0.0002, and the API rejects absolute tolerances above 0.005. This numerical allowance is not an action-quality or safety tolerance.

Checkpoints contain weights, optimizer, stage, selected teacher hash, configuration, CPU/CUDA RNG and NumPy RNG. Exact offline optimizer continuation and weights-only initialization are distinct configuration options. The initial smoke checkpoint predates CUDA RNG persistence and therefore cannot be called an exact CUDA resume; round 1 used its weights with a fresh optimizer. No simulator-state resume is implemented.

## Scene expert and navigation collection

`QualifiedSceneRegistry` selects a fixed, executed continuation for an exactly hashed task. A task binds the training motion, collision USD hash, start/goal, obstacle map, split and reward profile. Changed geometry or changed goal requires its own task and qualification. This finite registry is not an arbitrary route planner; an unqualified task produces no labels.

A scene receipt must bind the teacher/task, actual evidence files, 50 Hz control, 200 Hz contact evidence, whole motion completion, environment-contact checks, goal reach and terminal hold. Hashes enforce consistency with the supplied receipt; they do not establish the truth of a third-party receipt. The implementation does not automatically turn a reference clearance calculation into a physical qualification.

`NativeNavigationRuntime` supports one articulated environment with the existing scene-USD loader. It checks the loaded scene/reference and uses native measured history. `collect_navigation_episode` implements expert-only warm start and public-prior DAgger with recorded intervention probability, recurrent reset, state hash, observation hash, query time, task and checkpoint binding. Nominal teacher-driven queries may use the exact scene qualification; after any student action, labels require a separately qualified recovery envelope. Missing labels stay NaN with a false mask. The trainer verifies the bound query ledger against the stored public observations and never substitutes zeros for missing actions.

Navigation fitting reconstructs recurrent state from the complete public prefix, then applies truncated backpropagation to the selected sequence. Its loss is same-state teacher action MSE + action-residual magnitude penalty + residual temporal smoothness penalty. Smoothness does not cross episode resets. The proposed coefficients are 0.1 for each residual penalty. A public-prior command qualification is required before navigation fitting or student-driven scene collection. Expert-only collection may precede that qualification because the student does not act.

`NavigationStageCallback` connects these stages to the existing `eval_agent_trl.py` entry point: retain its scene-loading overrides, set `eval_callbacks=[im_eval]`, replace `callbacks.im_eval._target_` with `gear_sonic.research.scene_distillation.navigation_runtime.NavigationStageCallback`, and supply `callbacks.im_eval.stage_config=/absolute/config.json`. The stage configuration binds the task registry, checkpoints/hashes, command/residual bounds and a new output directory. It dispatches either `collect_navigation` or `residual_ppo`. Use the same external wall-time monitor as other native evaluation runs.

## Residual improvement and evaluation

`residual_rl.py` contains the bounded tanh-Gaussian actor/critic, transformed-action log likelihood, clipped PPO update, KL early stopping and live rollout loop. It freezes the navigation/foundation modules. GAE bootstraps artificial truncations only from proper terminal observations and stops recursion across resets. The live adapter refuses unavailable terminal observations; a task may explicitly designate native timeouts as genuine task deadlines, in which case they are terminal failures with no bootstrap.

Two reward profiles are available: unchanged native tracking reward, and `known_goal_progress_v1`. The latter measures the configured robot anchor **before autoreset**, rewards reduction in goal distance, penalizes elapsed time and nonfoot net contact force, penalizes native termination, and rewards reaching within 0.25 m at ≤0.1 m/s for 50 control ticks. Its numerical formula is in `online.py`. These are proposed simulation task/reward settings. The 50 Hz net-force reward includes self-contact and is not the required 200 Hz environment-contact qualification metric.

Compare command-only, bounded action residual, and optionally latent residual under separately bound equal budgets. Report goal completion with terminal hold, contacts, falls, motion completion/drift, action/residual magnitudes and smoothness, query failures, interventions and actual simulation/optimizer cost. PPO reward gain and imitation loss are not adoption criteria. Restore the original ancestry-held-out and physical-utility gates before research generalization or dataset-utility claims. Reserved layouts and original acquisition arms remain untouched.

## Current execution and next training gates

See [the implementation/pilot receipt](../../../docs/motion2scene/BFM_DISTILLATION_IMPLEMENTATION_20260912.md). Native teacher collection, offline fitting, prior-only physical evaluation and student-state collection have executed. Scene navigation collection/fitting and residual PPO have not executed against real scene tasks because the foundation and scene expert are not qualified. Their algorithms have contract/integration tests; this is not a claim that the scene adapter has completed a physical rollout.

Use a new immutable config and output directory for each training round:

```bash
.venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.preflight /absolute/training-config.json
.venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.train --config /absolute/training-config.json --output /absolute/new-run
.venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.aggregate --inputs /absolute/teacher/collection.json /absolute/dagger/collection.json --output /absolute/new-aggregate.json
```

The prepared navigation config deliberately reports missing scene labels and public-prior qualification. The next research gate is a larger, explicitly budgeted foundation/DAgger experiment plus expansion of the teacher's reliable motion support. Two short motions cannot establish a reusable behavior foundation. Then qualify command profiles, construct matched changed-scene/changed-goal tasks, execute and qualify their scene continuations, and run navigation imitation followed by residual PPO. The earlier proposed million-query/two-million-transition/two-hour envelope is a future-stage ceiling, not authorization to silently extend the smoke-test budget.
