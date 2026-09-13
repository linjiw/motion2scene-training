# Action-flow and direct-context research experiments

This extension adds experimental action regression, conditional flow matching, direct goal/scene conditioning, and a bounded action residual around the existing BFM. It retains the original token-student implementation and teacher checkpoints.

Read the [research review](../../../docs/motion2scene/FLOW_DISTILLATION_RESEARCH_20260913.md) for the distinctions between action flow, joint state-action diffusion, masked BFM distillation and residual RL. The current experiment packet is `/home/linjiw/research-data/m2s-flow-context-20260913`; its stage receipts and results determine actual completion and performance.

The pilot has completed. Read the [results report](../../../docs/motion2scene/FLOW_DISTILLATION_RESULTS_20260913.md) before choosing a checkpoint: all development completions and independent student task successes remain zero. These checkpoints are research artifacts, not improved production controllers. `final-artifacts.json` in the packet lists the final model paths and hashes.

## Entry points

From the repository root, with the existing Isaac Lab environment installed:

```bash
PYTHONPATH=. .venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.train_action_flow \
  --config /path/to/experiment-config.json --output /path/to/new-fit-directory

PYTHONPATH=. .venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.token_residual \
  --config /path/to/residual-config.json --output /path/to/new-residual-directory

PYTHONPATH=. .venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.prepare_context_windows \
  --catalog /path/to/context-train.json --output /path/to/new-window-directory
```

The output directories must be new. Copy and adapt a recorded config from the packet; `regression-config.json`, `flow-config.json`, and the later context/residual configs contain all required paths, hashes and budgets. Paths above are placeholders, not bundled datasets or runnable downloads. The existing repository's Isaac Lab and model/data setup remains required.

`ActionStudent` supports `kind=regression|flow`, explicit architecture dimensions, `context=true|false`, and 1–64 Euler steps. A flow sampler requires an explicit noise tensor. The known-map context profile supports at most five primitives and rejects an incomplete map. It does not implement camera navigation. Action statistics are checkpoint buffers computed from training labels and preserved on continuation.

The fitting config binds `teacher_checkpoint`, `teacher_sha256`, `dataset_manifest`, `dataset_manifest_sha256`, and `train_ids`. It declares optimizer settings, a finite update/wall budget, a seed and `profile_weights`. Profiles are `full`, `root`, `navigation` and `context`, with `context` hiding all 79 detailed command components. The `navigation` profile is still the old velocity/yaw/height mask; it is not an alias for context-only inference.

Set `allow_exploratory_queries` or `allow_teacher_prefixes` explicitly when using those data tiers. These flags do not grant physical qualification. Context datasets additionally require `context_schema=complete_known_map_goal_v1`, plus synchronized `navigation_context`, `obstacles_body` and `obstacle_mask` arrays. Original foundation shards have no such context fields and cannot be used as context datasets without verified recollection or reconstruction.

Initialization can copy the existing transformer's encoder (`encoder_only=true`) or migrate a direct-action checkpoint to add new context/residual parameters. Migration rejects incompatible shared tensors. `continue_optimizer=true` preserves AdamW state only for identical architecture and learning rate; this is not exact interrupted-run resumption with restored sampling RNG.

## Native callbacks

Use the existing `gear_sonic/eval_agent_trl.py` entry point and the recorded native argv from the packet as the source of environment, motion and teacher overrides:

- `action_native.ActionEvaluationCallback`: full/reference-navigation/context-mask tracking evaluation of direct-action checkpoints. Context models require `context_catalog`. `noise_mode=zero|episode` and optional `flow_steps` define the sampler explicitly.
- `action_native.ContextPrefixCollectionCallback`: teacher collection with measured pose and paired known-map task labels, using `collection_lock` with catalog path/hash and `minimum_prefix_rows`.
- `action_native.ActionDaggerCollectionCallback`: same-state teacher queries during direct-action student rollouts. Lock records student hash, control mode, teacher probability and seed. Support is a contiguous tracking-error prefix, not a qualified recovery guarantee.
- `token_residual.TokenResidualEvaluationCallback`: native tracking evaluation of the frozen-BFM action-residual checkpoint.
- `direct_scene_runtime.DirectSceneTaskCallback`: task evaluation in one exact collision scene; consumes a JSON `stage_config`. It uses the `SceneQualificationEnvCfg` contact sensors but independently scores goal/hold/contact/fall/deadline, without reference-pose termination.

A context catalog binds each task to the actually loaded native motion and collision scene. A plane catalog cannot claim obstacle execution. A changed-goal catalog that still uses an incompatible old continuation is rejected for collection. The catalog and action shards retain hashes. A successful crop/prefix must not be reported as original whole-motion success.

## Preserving the BFM motor foundation

`context_token` provides a fourth context arm: the original transformer prior/posterior/token adapter plus a zero-initialized public context path, followed by the frozen SONIC decoder. Fit it with `python -m gear_sonic.research.scene_distillation.context_token --config ... --output ...`; the config binds the original foundation and paired-context dataset. `ActionEvaluationCallback` and `DirectSceneTaskCallback` also load its `context_token` checkpoint stage. This preserves the pretrained motor path rather than replacing it with a randomly initialized action head.

## Residual limits

`token_residual` freezes the existing BFM and decoder, zero-initializes a small action correction, and fits same-state action error with a residual-magnitude penalty. The pilot limit is ±0.05 in native action units. This is supervised residual fitting. It is not actuation-aware PPO, FPO++, learned dynamics correction or a hardware-ready controller.

The generic direct-action model also supports a bounded residual head for follow-up studies. Keep its action statistics, base checkpoint, inference noise and residual-enable setting bound to each evaluation. Do not compare changing both sampler noise and residual state as if it isolated one effect.

## Validation

```bash
.venv_isaaclab/bin/python -m pytest -q \
  decoupled_wbc/tests/test_action_flow_distillation.py \
  decoupled_wbc/tests/test_scene_distillation.py \
  decoupled_wbc/tests/test_foundation_navigation.py \
  decoupled_wbc/tests/test_bfm_coverage_navigation.py \
  decoupled_wbc/tests/test_transformer_foundation.py
```

Tests cover flow time direction, integration endpoints, explicit sampling noise, context frame transforms, masked-value isolation, zero-initialized migration, frozen-base residual behavior, bounds and task success logic. Native runs are still required to measure physical behavior. The public scorer permits alternate trajectories; its upright traversal fall/contact profile is not appropriate unchanged for ground-contact skills.
