# Larger transformer foundation and extended DAgger experiment

The longer experiment completed all 40,000 transformer updates and five DAgger rounds. The final original-seed result is 5/89 train completions in both full and navigation-mask modes, and 0/20 development completions. Eight additional evaluations confirm weak development performance and seed-sensitive train completion. See the [completed evaluation report](BFM_NAVIGATION_READOUT_20260913.md) and [next direct-context experiment protocol](SCENE_NAVIGATION_NEXT_EXPERIMENT_20260913.md). This is a motion-command control experiment, not a qualified scene/navigation policy. The packet retains progress.json, evaluation-summary.json and monitor.log for the completed stages.

## Implemented candidate

TransformerMotionFoundation has **5,202,432 parameters**, versus **1,037,184** for the existing MLP. Four encoder layers use width 256, eight attention heads, dropout zero, and independently initialized layer weights. Ten tokens partition the 930D native measured-history storage into 93-feature blocks; these are explicitly feature blocks, not a claim about chronological frames. Eight additional command tokens carry the 79 command values and availability bits (padded to 80). Learned position embeddings distinguish the blocks. Hidden command values are cleaned before attention.

A 64D Gaussian prior/posterior feeds the SONIC token adapter. The prior consumes public history and masked commands; only the training posterior receives the 1645D critic state and 640D future reference. The transformer token adapter has no separate history bypass around the conditioned behavior latent. Explicit latent noise and deterministic default inference remain supported. Actor inference does not receive motion ID, privileged state or future reference.

The unchanged decoder stays bound to teacher checkpoint 1400. Added optional full-command teacher-token MSE (weight 0.1) alongside action reconstruction, prior action MSE (weight 1) and posterior KL (weight .001). Partial-command examples do not receive this direct token target. This is an engineering ablation, not a proven cure for command insensitivity. Historical MLP checkpoints remain loadable via a default architecture factory; evaluation, DAgger collection, navigation construction and fitting now reconstruct the architecture from checkpoint configuration.

## Locked budget and collection changes

- Transformer initial fit: 10000 updates, batch 128, AdamW learning rate 1e-4. Command mixture full/root/navigation = .7/.2/.1.
- Longer MLP control: 10000 updates, same optimizer/batch/loss/mask recipe, evaluated separately. This tests the initial recipe/capacity comparison; it does not match the transformer’s full later online budget.
- Five transformer DAgger rounds: **6000 updates each**, yielding **40000 total transformer updates**, plus the separate 10000-update MLP control. The completed 100-update transformer smoke is additional validation cost.
- Each DAgger round attempts all 89 screened training motions in one native batch, at most 600 control ticks / 53400 transitions. Five-round collection ceiling: **267000 transitions**, excluding evaluations. Twenty development motions enter evaluation only.
- Teacher intervention probabilities: .8, .6, .4, .2, 0. Command profiles alternate full/navigation/full/navigation/full. Actual interventions and support masks are retained; assisted completions never count as standalone student success.
- Cumulative aggregation retains all collection attempts and masks unsupported/post-failure rows. Query labels remain explicitly exploratory rather than physically qualified recovery labels.
- Source-balanced sampling gives half the batch in expectation to nominal teacher episodes and half to accumulated query episodes whenever both exist, then samples episodes/rows uniformly within each source. This prevents growing query collections from erasing nominal demonstrations. It does not certify the queried teacher’s recovery capability.
- Every fit is followed by standalone full-command and navigation-command evaluation on the same 89 train +20 development motion IDs, seed 91260, with command telemetry and native termination. Initial and every round checkpoint are retained. Changed datasets use weights-only initialization with a fresh optimizer; these rounds are not claimed as exact optimizer continuation.
- Total external wall cap: **6 hours**. Per-fit cap 2 hours; collection cap 15 minutes; evaluation cap 10 minutes. No automatic resume or retry; launch is locked and rejects an existing progress record.

## Research interpretation and navigation path

The [BFM paper](https://arxiv.org/html/2509.13780v1) uses masked online distillation and a CVAE. Our transformer and frozen SONIC-token adapter are an adaptation. Capacity, removing the bypass, token supervision, and expanded collection are a bundled candidate recipe; any gain cannot be attributed to attention alone. The MLP control helps assess initial fit differences but further ablations are needed to separate these effects.

[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) is a relevant alternative generative framework. It is **not implemented or launched in this experiment**. Our immediate failure is poor closed-loop command-conditioned control despite low offline loss. A one-pass transformer is a more direct first comparison; iterative generative inference and its sampling/latency tradeoffs should be evaluated as a separate model after the data and control interfaces work. That choice is an engineering inference, not a claim that flow matching is inferior.

Completion, failure-censored tracking error, command response, and unseen-motion progress determine whether this candidate helps. Training loss, teacher-assisted survival and a larger parameter count do not establish success. The native scorer still includes reference-pose termination; add an independently specified fall/contact-limited command evaluator before claiming general velocity-control quality. Keep the full-reference comparison intact.

Navigation remains a downstream stage: physically reliable public motor control -> stable stop/turn/posture commands -> qualified scene/goal continuations with terminal hold and contacts -> nominal navigation imitation -> supported on-policy queries -> matched command-only versus bounded action residual. The corrected scene exporter from the prior turn is retained. Low-beam traversal and terminal stopping still lack qualified demonstrations. No residual PPO or scene-navigation fit is silently started by this runner, even if some tracking scores improve.

## Verification and monitoring

The new model completed 100 GPU smoke updates. All **51 focused tests passed**, including masking hidden values, gradient flow from public command inputs, deterministic checkpoint reconstruction, posterior-only privilege boundaries, legacy checkpoint compatibility and existing navigation/distillation contracts. Black/Ruff checks passed on changed code. The actual large fit has started and passed hundreds of updates without an out-of-memory failure.

Packet: /home/linjiw/research-data/m2s-bfm-transformer-dagger-20260913/.

- plan.json: fixed stage, update, collection and time budgets.
- run-experiment.py and monitor-process.json: detached runner and PID.
- source-manifest.json and source-at-launch.tar.gz: source snapshot checked at each stage.
- progress.json: stage PID/state/timing; per-stage process.log and receipt.json.
- transformer-initial-model/metrics.jsonl: live initial-fit loss; later stage model folders contain their own logs/checkpoints.
- evaluation-summary.json: emitted as physical evaluation stages finish.
- complete.json or failure.json: terminal experiment outcome.

No improved tracking/navigation result is claimed before these evaluations finish.
