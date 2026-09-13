# BFM / navigation-mask evaluation — 2026-09-13

The larger transformer experiment completed all 40,000 updates (10,000 initial + five 6,000-update DAgger fits), plus a 10,000-update MLP control. Its recorded total wall time was 22.8 minutes. No new training was launched for this readout. Eight additional native evaluations completed successfully: 872 episodes, testing rounds 4 and 5 on two more seeds.

## Original matched evaluation

Every row uses 89 train + 20 development motions, evaluation seed 91260, the same plane environment and native reference-tracking scorer. Progress is the fraction of reference duration tracked before termination, not distance-to-goal progress. Completion means reaching the end without native tracking termination; it does not certify strict tracking quality or scene-task success.

| Model | Full train complete / progress | Navigation-mask train complete / progress | Full dev complete / progress | Navigation-mask dev complete / progress |
|---|---:|---:|---:|---:|
| MLP, 10k | 0/89 / 7.5% | 0/89 / 7.3% | 0/20 / 4.3% | 0/20 / 4.4% |
| Transformer, initial 10k | 3/89 / 15.1% | 1/89 / 14.5% | 0/20 / 5.2% | 0/20 / 5.3% |
| Transformer DAgger 1 | 1/89 / 21.8% | 0/89 / 20.6% | 0/20 / 9.4% | 0/20 / 9.7% |
| Transformer DAgger 2 | 4/89 / 27.7% | 3/89 / 28.1% | 0/20 / 9.9% | 0/20 / 10.2% |
| Transformer DAgger 3 | 2/89 / 28.1% | 1/89 / 27.5% | 0/20 / 10.5% | 0/20 / 10.1% |
| Transformer DAgger 4 | 3/89 / 29.9% | 3/89 / 31.1% | 0/20 / 11.8% | 0/20 / 11.4% |
| Transformer DAgger 5 | 5/89 / 31.3% | 5/89 / 33.2% | 0/20 / 10.4% | 0/20 / 10.5% |

The frozen step-1400 teacher previously completed 88/89 train and 14/20 development motions (99.0% / 81.1% duration progress). Its full-reference information is richer than the public student interface, so this measures the remaining distillation gap rather than an equal-input comparison. Teacher source: `/home/linjiw/research-data/m2s-repaired-teacher-4096-step1400-20260912/summary.json`.

## Additional evaluation seeds

Each cell reports completions at seeds 91260, 91261, 91262, followed by mean progress across those seeds. These are repeated evaluations of one training run, not three independently trained models or additional unique development motions.

| Checkpoint | Mode | Train completions out of 89 by seed | Mean train progress | Dev completions out of 20 by seed | Mean dev progress |
|---|---|---|---:|---|---:|
| dagger-4-fit | full | 3, 2, 2 | 30.85% | 0, 0, 0 | 10.81% |
| dagger-4-fit | navigation | 3, 2, 3 | 29.51% | 0, 0, 0 | 10.66% |
| dagger-5-fit | full | 5, 4, 4 | 31.33% | 0, 0, 0 | 10.27% |
| dagger-5-fit | navigation | 5, 5, 1 | 31.75% | 0, 0, 0 | 10.70% |

Round 5 is the latest and stronger train-tracking candidate, but its navigation-mask completion varies from 1 to 5/89. Every development evaluation remains 0/20. The original-seed development drop after round 4 is not a consistent regression in both modes across seeds: full progress is slightly lower on average, while navigation-mask progress is nearly unchanged. Describe development as stalled at low performance, not proven optimization convergence.

## What is working and what is still weak

- Larger transformer plus DAgger substantially extends train tracking duration: final 31.3% / 33.2% versus initial 15.1% / 14.5% (full / navigation mask). The initial transformer also outperforms the MLP under the matched 10k recipe. Capacity, data and later update count are not isolated by the combined final comparison.
- Coverage still matters: final full-command progress is 47.4% on the 21 original nominal-demonstration motions versus 26.4% on the other 68 train motions; navigation-mask progress is 56.6% versus 26.0%. These subsets differ in difficulty, so this is an association rather than proof of the cause.
- Final development rollouts average only about 0.68 s full / 0.70 s navigation-mask before censoring at the original seed. No development motion reaches halfway. The main motor/distillation problem remains in full mode, too.
- Five DAgger collections produced 216,181 total simulator transitions and 40,967 retained query rows. Supported rows by round were 9,035, 9,155, 8,554, 7,674, 6,549. Total transitions include post-failure/reset periods and are not all usable demonstrations. The fit begins with only 21 fully qualified nominal episodes / 6,433 rows; exploratory query support is not verified recovery competence.

## Fixed-row conditioning and decoder audit

Diagnostics use every fifth available row: 1,295 original nominal teacher rows and 1,343 supported round-5 query rows. These are training-related data, not a development action test. The same deterministic command permutation was used across models; hidden/shuffled inputs can be off-distribution.

- On nominal rows, final full-command action MSE is 0.001472; shuffled commands raise it to 0.002620 (about 78% higher). Navigation-mask MSE rises from 0.001457 to 0.002555. The model uses commands; this does not prove goal or obstacle reasoning.
- On round-5 query rows, full-mode MSE falls from 0.016885 for the round-4 collector policy to 0.003680 after round-5 fitting (about 78% lower). Round 5 trained on these rows, so this demonstrates fitting the visited-state labels, not held-out generalization. Initial-transformer MSE on these later states is 0.175598.
- On those same rows, using the privileged posterior barely changes final MSE (0.003683 versus prior 0.003680). Investigate latent/posterior utility, coverage and the token predictor before assuming a larger flow model solves the failure. This is not a conclusive diagnosis of posterior collapse.
- Reconstructing teacher actions through the frozen decoder matches recorded targets within maximum absolute error 1.53e-5 on nominal rows and 1.05e-5 on query rows. No decoder wiring discrepancy was found in this audit.

## Navigation and scene status

The evaluated transformer sees 930D history and masked 79D motion commands. The navigation profile exposes only reference-derived body-frame vx, vy, yaw rate and target height. It contains no final destination or obstacle tokens. Thus 5/89 is a reference-tracking completion count under reduced commands, not a navigation-task success rate. True scene/navigation student success is not measured by this experiment.

The prior corrected scene probes are teacher probes, not this student: 0/4 pass full qualification. Three reach the destination but fail the required terminal hold; the low-beam case also has an undesired obstacle contact. A stationary reference suffix is insufficient evidence of a stable stop. Unexecuted scene/motion pairs remain useful context-learning candidates with explicitly limited provenance.

## Next experiment prepared

Use direct scene/navigation-conditioned masked distillation as the primary architecture. Compare a motion-only baseline, a context model always shown detailed commands during training, and the same context model trained with 30% full / 20% partial / 50% scene-and-goal-only episodes. Keep teacher actions as targets and hide reference trajectory, phase and IDs in the public context-only mode.

Recollect synchronized measured pose and scene/task fields, repair and execute stable stopping continuations, expand supported teacher/recovery coverage, and implement a task scorer that permits valid alternate motions. Start with a 32-update smoke and a small native pilot; the subsequent planned ceiling is 144,000 updates across three arms and three training seeds. Residual and flow-matching comparisons follow a measurable task-control baseline.

Detailed protocol: [SCENE_NAVIGATION_NEXT_EXPERIMENT_20260913.md](/home/linjiw/groot-wbc-sonic-sim-trackb/docs/motion2scene/SCENE_NAVIGATION_NEXT_EXPERIMENT_20260913.md). This is a prepared design and budget; the direct context model, new collection fields and independent task scorer are not yet implemented or launched.

## Reproduction and artifacts

Packet: `/home/linjiw/research-data/m2s-bfm-readout-20260913`.

- `readout.json`: per-checkpoint/profile/seed/split metrics and completed motion IDs.
- `conditioning-audit.json`: fixed-row prior/posterior/shuffle/hidden diagnostics.
- `input-sha256.json`: hashes of evaluator inputs used for the recomputed readout.
- `round-*-seed-*`: exact native argv, logs, receipts and raw metrics for eight new evaluations.
- `tracking-progress.png`: original-seed progress curves; `next-experiment-plan.json`: explicitly non-launchable research plan.
- `run-validation.py`, `audit-conditioning.py`, `summarize.py`, `write-report.py`: packet-local reproduction scripts. The evaluator runner refuses existing run directories; use a fresh packet when rerunning.

Readout and conditioning commands from the repository root:

```bash
PYTHONPATH=. .venv_isaaclab/bin/python /home/linjiw/research-data/m2s-bfm-readout-20260913/audit-conditioning.py
.venv_isaaclab/bin/python /home/linjiw/research-data/m2s-bfm-readout-20260913/summarize.py
python3 /home/linjiw/research-data/m2s-bfm-readout-20260913/write-report.py
```

All 28 original aggregate summary entries were independently matched to raw per-motion results. Training source hashes matched the completed experiment before additional evaluation. No policy source or checkpoint was modified.


## Follow-up experiment completed

The [flow, DAgger, residual and direct-context pilot](FLOW_DISTILLATION_RESULTS_20260913.md) has now completed, including an independent scene-task evaluator and a context-conditioned BFM that preserves the SONIC decoder. The [research review](FLOW_DISTILLATION_RESEARCH_20260913.md) explains the implementation and paper comparisons. This bounded pilot is distinct from the larger three-seed study proposed above; the new results identify missing successful stop/obstacle supervision before that study should be scaled.
