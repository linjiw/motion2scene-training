# Flow, DAgger, residual and direct-context pilot: completed results

September 13, 2026. **Completed:** 64,000 new optimizer updates plus 192 smoke updates, 28 native reference evaluations (3,052 motion episodes), and 20 independent task episodes. All 21 bounded fits completed. The frozen teacher remains unchanged. None of these research students is promoted as a reliable navigation controller.

The implementation and negative results answer a useful question: replacing the pretrained motor decoder with a newly learned action-flow head does not improve this setup at the tested budget. Retaining the motor foundation helps, but adding context labels alone has not produced successful navigation. The main evidence gaps are successful task demonstrations, retained obstacle/stop supervision and useful recovery states.

## Matched reference evaluation

Each cell reports **completed motions / denominator (mean fraction of reference duration reached)**. This duration metric is not spatial distance to the goal. The train/development split is 89/20 repaired motions. Evaluations use the existing native tracking termination and seed 91260. The development set has been inspected repeatedly in this project and is not an untouched final test set.

| Student | Full command, train | Reference-navigation mask, train | Development: full / mask |
|---|---:|---:|---:|
| Historical BFM, DAgger 5 | 5/89 (31.3%) | 5/89 (33.2%) | 0/20 (10.4%) / 0/20 (10.5%) |
| Frozen BFM + bounded residual | 7/89 (33.1%) | 2/89 (31.1%) | 0/20 (10.4%) / 0/20 (11.5%) |
| Direct regression, final DAgger 3 | 0/89 (13.3%) | 0/89 (13.5%) | 0/20 (5.5%) / 0/20 (5.6%) |
| Action flow, final DAgger 3 | 0/89 (11.8%) | 0/89 (11.8%) | 0/20 (7.4%) / 0/20 (6.4%) |

The historical BFM used 40,000 optimizer updates and a pretrained SONIC decoder. Each new direct-action head received 10,000 initial updates, 5,000 coverage updates and three rounds of 2,000 DAgger updates: 21,000 total, with the inherited encoder but a new action head. This makes regression versus flow a useful local comparison; the historical BFM is a practical baseline with different training and total motor capacity. A negative direct-head result does not falsify flow policies generally.

The original navigation mask supplies reference-derived velocity, yaw rate and height. It contains no destination or obstacle request. Its completion score remains reference tracking. The residual is supervised action-error fitting with a frozen BFM/decoder and a ±0.05 correction limit; it is not residual PPO. The 7-versus-5 full-command result is offset by 2-versus-5 masked completions and no development completions. One seed does not establish a reliable gain.

After DAgger, regression mean full-command train progress increased from 11.0% to 13.3%; flow increased from 9.4% to 11.8%. Flow development progress increased from 5.0% to 7.4%, but still completed no motion. At the initial 10,000-update checkpoint, eight Euler steps with zero initial noise performed better than one step (9.4% versus 6.1% train progress); eight steps with episode-held Gaussian noise reached 7.9%. Zero noise is a diagnostic sampler, not the mean of a learned distribution.

[Progress figure](/home/linjiw/research-data/m2s-flow-context-20260913/flow-dagger-progress.png), [exact plotted data](/home/linjiw/research-data/m2s-flow-context-20260913/flow-dagger-progress.csv), [all tracking results](/home/linjiw/research-data/m2s-flow-context-20260913/tracking-results.csv), and [historical baseline data](/home/linjiw/research-data/m2s-flow-context-20260913/historical-baseline.csv).

## Direct goal/scene context

The actor receives normalized measured history, body-relative start/goal and terminal requirements, plus masked known-map obstacle primitives. **Context-only hides all 79 detailed command components.** These rows still use reference tracking as a motor diagnostic; actual task success is scored separately below.

| Student | Full command, train | Context only, train | Development: full / context |
|---|---:|---:|---:|
| Regression, always-full training | 0/89 (13.2%) | 0/89 (10.8%) | 0/20 (5.6%) / 0/20 (5.4%) |
| Regression, masked training | 0/89 (13.5%) | 0/89 (13.0%) | 0/20 (5.9%) / 0/20 (5.7%) |
| Action flow, masked training | 0/89 (10.7%) | 0/89 (10.9%) | 0/20 (5.7%) / 0/20 (5.7%) |
| BFM tokens + decoder, masked training | 1/89 (25.3%) | 1/89 (23.8%) | 0/20 (9.6%) / 0/20 (8.5%) |

Each context arm received 5,000 updates. The three new action-head arms start from their final DAgger checkpoints; the context-BFM arm starts from the original BFM and preserves its prior/posterior/token adapter and frozen SONIC decoder. Zero-initialized context migration reproduces the original full-command tokens exactly before training. Masked fits use 30% full, 20% root and 50% context-only minibatches of sampled rows. The always-full regression arm is a diagnostic of removing detailed commands without training that removal.

The context-BFM's 23.8% context-only progress is better than the new action heads, but its full-command progress falls from the historical 31.3% to 25.3%. This comparison changes the data and masking schedule as well as adding context. It suggests preserving the motor path and testing replay/curriculum controls; it does not isolate catastrophic forgetting or prove that context itself harms control.

A 512-row training-data diagnostic confirms that the delivered context-only policies cannot read hidden detailed commands: replacing those commands with NaNs changes none of their actions. Shuffling paired context changes the context-BFM actions by 0.0368 RMS and increases teacher-action MSE from 0.00286 to 0.00353. The direct-head arms change by only approximately 0.0010–0.0023 RMS, without a shuffled-context MSE penalty on this sample. This supports stronger context use by the preserved BFM, but it is an offline training-set sensitivity check, not causal navigation success or held-out generalization. See `context-conditioning-audit.json` for all four arms and the bound dataset/checkpoint hashes.

## Independent scene task evaluation

All four student arms achieved **0/4 task successes**, and the teacher achieved **0/4**. These are four training-related synthetic scene probes from two motions, not held-out scene generalization. Goal arrival requires 3D root-to-goal distance within the task tolerance, speed within its limit for 50 consecutive control ticks, no undesired contact and no fall within the deadline. The callback uses actual measured robot state and pair-resolved 200 Hz contact forces. Reference-pose mismatch does not terminate a valid alternate route, and the reference cursor is clamped to prevent reference exhaustion from resetting the robot.

| Teacher task | Ever reached goal | Longest stable hold (ticks) | Maximum undesired force (N) | Stop |
|---|---:|---:|---:|---|
| 00134-clear | yes | 19/50 | 0.0 | deadline |
| 00463-clear | yes | 0/50 | 0.0 | deadline |
| 00463-corridor | yes | 0/50 | 0.0 | deadline |
| 00463-low_beam | no | 0/50 | 1732.7 | contact |

The context-BFM survived the full 410-tick deadline on both clear scenes, ending 0.91 m and 3.92 m from the goal. It contacted the corridor and low beam at ticks 156 and 199. It never entered the goal tolerance on these four probes. The other context students also never reached a goal and stopped for contact or the declared fall guard. The goal-only scorer therefore confirms that partial reference progress is not successful navigation.

These outcomes expose missing task competence in the teacher labels: the teacher can approach several goals, but does not perform the required stable stop, and its beam reference does not avoid the obstacle. Imitation cannot be expected to reproduce a successful stopping/avoidance strategy absent from its supervision. This is evidence about this dataset and teacher, not a claim that goal labels are intrinsically insufficient.

[All scene outcomes](/home/linjiw/research-data/m2s-flow-context-20260913/scene-task-results.csv); each `task-*/task/` directory retains `task-result.json` and `trace.npz`, including 200 Hz contact arrays. The current scorer is limited to upright traversal, with an absolute pelvis-height fall guard. Heading, route efficiency, general ground-contact skill scoring and partial camera observations are not implemented in this pilot.

## The data bottleneck is measurable

Fresh plane teacher collection again completed 88/89 native motions, but only 21 met the stricter whole-demonstration filter. The teacher's native tracking ability therefore remains much stronger than the student's. Native completion and the additional training-data quality filter are different criteria.

The collector stored 30,422 plane rows and retained 10,296. Exact reference crops starting at 25%, 50% and 75% supplied another 25,043 retained rows. All 267 crop files preserve the corresponding serialized joint/root arrays exactly; they broaden intermediate-state coverage without changing joint order. Crop completion is never counted as original whole-motion completion.

The four scene recordings supplied only 177 retained rows: 120 for `00134-clear` and 19 each for the three `00463` scenes. Consequently **only 38 retained rows contain actual obstacle-scene execution**, out of 35,516 rows in the paired-context dataset. Those obstacle prefixes end at approximately 0.38 seconds; the independent teacher beam contact occurs much later. These data do not cover an obstacle response or a successful terminal hold. Uniform episode sampling changes their sampling weight but cannot create the missing behavior.

The support screen uses 0.25 m root XY and 0.10 m mean body error. Whole-qualified trajectories retain their complete valid rows; other trajectories are censored at the first unsupported decision. For example, `00463-clear` completes native tracking with a 0.157 m maximum root error but has 0.114 m mean body error, so it loses almost its entire trajectory under the stricter prefix rule. A useful next ablation should compare explicit label-support tiers while keeping evaluation unchanged; simply increasing epochs repeats these early rows.

The new DAgger rounds retained 22,169 regression query rows and 21,531 flow query rows. Teacher intervention probabilities were 0.8, 0.4 and 0; collection profiles were full, navigation-mask and full. Retained per-round rows fell from 11,585 to 3,369 for regression and 11,924 to 2,714 for flow as teacher assistance decreased. This reflects collection under different mixtures, not an isolated learning curve for student-only recovery. The final direct-head datasets contain 104,908 and 104,270 retained rows respectively. Supported same-state queries are not physically validated recovery sequences.

One attempted changed-goal collection was rejected because its unchanged reference did not match the new request. That failed attempt is preserved in `coverage-failure.json` and documented in `excluded-incompatible-contexts.json`; it contributed no labels. A corrected explicit four-scene schedule and the remaining coverage stages completed successfully. This was a provenance guard working as intended, not a failure silently omitted from the study.

[Coverage audit](/home/linjiw/research-data/m2s-flow-context-20260913/collection-coverage.csv).

## Next experiments justified by these results

1. **Repair the task supervision first.** Obtain executed deceleration/standing continuations and scene-compatible obstacle responses. Requalify them using the unchanged goal/hold/contact scorer. Collect meaningful approach, avoidance, arrival and hold segments; inspect support coverage by phase rather than just total row count. A changed goal must come with its matching successful continuation.
2. **Preserve the BFM motor representation.** Compare continuing the original BFM on the same data with and without the context path, then add nominal/old-query replay and a gradual context-mask curriculum. Keep full-command completion as a retention metric. The current context-BFM code enables this representation path; those additional controls have not yet run.
3. **Make DAgger closer to the studied online recipes.** Compare frequent short collection/update cycles with the current long static aggregates under an equal simulator-transition budget. Add randomized valid intermediate starts, history burn-in, hard-motion sampling and bounded teacher-takeover tests that establish actual recovery. Report intervention rate, retained coverage and unassisted success separately. The current implementation runs bounded aggregate DAgger with deterministic quarter-start coverage; it is not full online BFM replication.
4. **Keep flow as a controlled research branch.** Test a flow model over the preserved motor latent or a joint future-state/action model, after obtaining contiguous successful trajectories. Future-state predictions make waypoint and body-clearance guidance evaluable. Random action rows cannot serve as temporally coherent trajectory supervision. Compare integrated actions and physical success, not only denoising loss.
5. **Escalate residual training only with a suitable base and objective.** The implemented supervised bounded residual is a mixed-result diagnostic. Actuation-aware residual PPO or flow-policy RL requires reward, likelihood/surrogate, previous-action-history and dynamics design; neither has been implemented or claimed here. Once task competence is demonstrated, use matched seeds and report retention on the original motion benchmark.

The earlier 144,000-update, three-seed context study remains unlaunched. This completed pilot validates the implementation and identifies missing positive supervision; it provides no evidence that repeating the current scene data at that larger budget is the best next experiment.

## Reproduction, artifacts and validation

The [research review](FLOW_DISTILLATION_RESEARCH_20260913.md) contains the primary-paper analysis, including BFM, OmniXtreme, BeyondMimic, FPO++, PhysiFlow and ResMimic. The [implementation guide](../../gear_sonic/research/scene_distillation/FLOW_CONTEXT_EXPERIMENTS.md) lists training CLIs, model schemas and native callbacks.

Packet: `/home/linjiw/research-data/m2s-flow-context-20260913`. `head-complete.json`, `research-complete.json`, `context-bfm-complete.json` and `context-bfm-tasks-complete.json` record successful stage completion. Each native stage stores its exact argv, log and output. Fits retain configs, losses, checkpoints and receipts; dataset manifests bind parent collections and shard hashes. `results.json`, `result-input-sha256.json` and the CSV files are regenerated from recorded outcomes by `summarize-results.py`.

The frozen teacher SHA-256 is `afd649cfbbfd28833550e11a0f8c3b7a5f6a05ee8b4021dd0dac97a6f94733ce`. New checkpoint paths and hashes are recorded in `final-artifacts.json`. Source hashes were recorded at stage launch; the action loader gained a backwards-compatible context-BFM dispatch during this research sequence, so this is not a claim of one immutable source tree across every stage. The final source snapshot and validation receipt document the delivered implementation.

Run result reconstruction from the repo root:

```bash
.venv_isaaclab/bin/python /home/linjiw/research-data/m2s-flow-context-20260913/summarize-results.py
```

Focused validation: **60 tests passed**, covering flow paths/integration, masking and geometry, migration identity, frozen residual bounds, data/decoder invariants and independent task scoring. Ruff and Black checks passed for all eight new modules and their test file. Native success/failure results above validate simulator execution; unit tests do not substitute for those measurements. All bounded training/evaluation processes finished; no new long training job remains running.
