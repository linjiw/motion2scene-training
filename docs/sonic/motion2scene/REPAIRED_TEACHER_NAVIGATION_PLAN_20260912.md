# Repaired teacher training and the navigation improvement plan

The new repaired-data teacher is **running**, not yet qualified. The best existing student remains the earlier 12,000-update checkpoint: 1/100 exact full-command completions and 0/100 exact navigation-command tracking completions. The subsequent assisted student did not improve those results.

## Best matched student replay

Open `/home/linjiw/research-data/m2s-bfm-best-student-replays-v2-20260912/index.html`, or its `comparison.mp4`, `full-command.mp4`, and `navigation-command.mp4`.

Both panels use **the same checkpoint and motion 00490**. This motion is the only full-command completion and also the maximum-progress navigation example among the 100 motions. Full commands retain 298 measured steps (5.96 s); navigation retains 235 pre-failure steps (4.70 s), corresponding to native progress 235/299 = 78.6%. Reference and measured-body roles are corrected. The completed panel freezes its final pose; the failed panel excludes all subsequent states. These are measured skeleton replays, not RGB camera videos. The displayed coordinates are relative to the reference's starting XY position.

The v2 renderer uses the completed native trace to recover the actual 299-step denominator. Original 30 Hz metadata gave a 300-step approximation that could include the terminating sample. The first render directory is superseded by v2. No simulator rerun was needed.

## New teacher run

Packet: `/home/linjiw/research-data/m2s-sonic-repaired-teacher-128env-20260912`.
Inputs: `/home/linjiw/research-data/m2s-sonic-repaired-v1_1-20260912`.

- Train on exactly **89** `screened/train` clips. Preserve the exclusion reasons for the other 11 training candidates.
- Evaluate on **all 20** `matched/repaired/development` clips, including the two that remain unresolved by offline screening.
- Verify the entire repaired manifest and each selected export hash. The actual native CPU loader passed for both splits, with 435,897 finite values checked at beginning/middle/end.
- Initialize a new policy and critic from the **SONIC release**, with a fresh optimizer and simulator. This avoids treating optimizer history from the old motion distribution as a repaired-data resume.
- **128 environments**, 24 control steps per rollout, **32,000 PPO iterations**, five PPO optimization epochs per rollout, eight minibatches. Actor LR starts at 2e-5 under the existing adaptive schedule.
- Maximum 98,304,000 rollout environment transitions / 393,216,000 environment physics steps. This matches the transition ceiling of 8,000 iterations at 512 environments. Increasing iterations compensates for smaller rollouts; PPO data-reuse epochs remain five.
- 48-hour wall cap; periodic checkpoints every 500 iterations and latest checkpoint every 100. No automatic restart or doubled retry budget. Initialization/reset cost is distinguished from measured training transitions.
- Headless plane training, no obstacles or cameras. Local progress, reward/loss metrics, memory and motion-exposure records are enabled. The detached monitor persists beyond this conversation.

Other GPU processes initially left about 3.8 GiB available, so 128 was chosen instead of 256. Early PyTorch peak allocation was approximately 1 GiB; this is not total simulator/device memory. The first training iterations ran successfully and exercised all 89 resident motions. The longer run still needs monitoring and post-training evaluation.

`evaluation-protocol.json` predeclares a matched 20-clip development comparison of the SONIC release and final repaired teacher, seed 91231, with the same environment configuration. The supervisor runs that comparison after successful training completion. It does not select checkpoints using development scores during training. `progress.jsonl`, `metrics.jsonl`, and eventual failure/completion receipts retain the actual work performed.

Repaired references are offline-screened inputs, not already qualified teacher rollouts. The dataset's reported reduction in bad-dynamics frames, zero exported joint-limit excess and near-zero ground penetration are reasons to test this new distribution; they do not prove policy improvement.

## Why abstract commands can work—and what the present result means

The user's abstraction intuition is consistent with BFM: a behavior can be specified at different levels, from detailed joint/keypoint targets to root velocity commands. However, abstract commands leave many behaviors valid. Walking with different step phases and arm poses can satisfy the same velocity request. Removing full commands is therefore a conditional behavior-generation problem, rather than simply compressing a unique full-pose target. BFM uses masked control interfaces and a conditional variational model; it evaluates locomotion with root linear/angular velocity errors rather than requiring the original exact joint sequence. [BFM, control interface and locomotion evaluation](https://arxiv.org/html/2509.13780v1).

Our two current profiles are:

| Profile | Supplied information | Appropriate interpretation |
|---|---|---|
| Full | Available root controls, 14 body position targets and 29 joint targets | Detailed reference tracking |
| Navigation commands | Body-frame vx, vy, yaw rate and height | Sparse low-level motion commands |
| Scene/goal navigation | Known obstacle map, measured history and requested start/goal | A downstream task, not evaluated by those two tracking runs |

The 0/100 result was obtained under **native reference-tracking terminations**, including foot-position mismatch against the exact reference. It remains a failed tracking result and cannot be relabeled as navigation success. But it also does not establish that zero motions could satisfy a separately defined velocity/goal task. A different valid stepping pattern can be penalized by that scorer. Conversely, good velocity RMSE over a short surviving prefix is insufficient evidence of stable navigation.

I added `command_quality.py` and `CommandEvaluationCallback` to capture synchronized commanded/measured vx, vy, yaw rate and height, with native failure censoring and no invented success label. Pure metric tests pass; the native telemetry callback is prepared for the next student evaluations and has not been physically exercised in this turn. A subsequent command-only qualification must explicitly replace reference-style termination terms with declared physical stability/timeout checks while retaining a separate exact-tracking evaluation.

## Recommended framework and discriminating experiments

1. **Repair and qualify the tracking teacher first.** Use the new 89-clip fit, then test all 20 development clips. Inspect support, drift, full duration and termination reasons. Keep teacher failures in the denominator.
2. **Train a dependable sparse-command motor skill.** Preserve whole-qualified demonstrations, balance motion IDs *and reference-time quarters*, and collect teacher interventions/recoveries throughout the motion. Use a curriculum from full controls to root controls (including heading), then sparse navigation commands. Begin with walking, turning and stopping; interaction/return-to-start clips do not have a unique point-goal interpretation. Compare curriculum and flat masking with the same examples and update budget.
3. **Match the objective to command abstraction.** Report command velocity/yaw/height errors and physical survival separately from exact pose error. Add task RL plus an imitation anchor after the sparse motor skill can act stably. Do not force an unspecified arm pose or precise gait phase through a four-command interface. Test a coherent episode/chunk latent or recurrent prior against the deterministic prior; do not resample unrelated behavior modes every control tick.
4. **Distill a composite navigation teacher.** A planner or qualified continuation selector uses scene/start/goal to generate feasible motion guidance; the tracking teacher supplies same-state motor actions. The student learns that combined response from public observations. GuideWalk provides a relevant precedent: navigation guidance plus locomotion teacher, DAgger, then PPO with an auxiliary cloning objective. This is a proposed adaptation to SONIC, not a reproduced result. [GuideWalk methodology](https://arxiv.org/html/2606.10449v1).
5. **Fine-tune bounded residuals on the actual task.** Freeze the qualified foundation/decoder initially, zero-initialize a per-joint bounded action residual, and optimize progress, stable stopping and measured collision costs with a teacher anchor. Compare residual-disabled and enabled variants. Change obstacles at fixed start/goal, and change goals in fixed scenes, so scene conditioning must affect the selected behavior. Goal success/contact/hold metrics—not pose-copy accuracy—decide this stage.

The strongest current diagnosis is missing temporal supervision: the first 100-motion aggregate had only two rows after halfway and none in the last quarter. Assistance improved that coverage but the follow-up still failed standalone evaluation. Fixing data coverage, control semantics and evaluation comes before simply enlarging the network or adding more offline epochs. New teacher training is underway; further student/RL training is not silently queued behind an unqualified teacher.

## Validation and provenance

The repaired motion libraries passed native CPU loading; ten existing teacher-trainer/accounting tests and two new command-metric tests passed. Scoped Black and Ruff checks passed. Training source/configs were archived before launch. Later renderer and command-evaluation additions do not modify the running teacher's training path. See the packet's timestamped status snapshot for observed iterations, memory and checkpoint state; it is a snapshot, not the final training receipt.

## Subsequent implementation and smoke result

The command curriculum and reference-phase sampler are implemented. A matched CPU experiment completed 240 updates per arm, but uniform masking slightly outperformed the curriculum; neither result qualifies physical navigation. See [the experiment report](BFM_CURRICULUM_SMOKE_20260912.md) for data recovery, metrics, command-sensitivity findings and next gates.
