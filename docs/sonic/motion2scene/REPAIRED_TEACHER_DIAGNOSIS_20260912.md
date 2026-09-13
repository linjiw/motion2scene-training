# Repaired teacher degradation diagnosis — 2026-09-12

The current teacher was stopped at the user's request. The last observed training iteration was **13,166**, and the preserved checkpoint is **13,100**. Both training and monitor processes exited. This is an interrupted run, not a completed 32,000-iteration experiment. No replacement training was launched.

## Main finding

**The repaired inputs are substantially easier for the released policy to track. Fine-tuning destroys much of that capability.** Grounding explains almost all of the observed improvement; additional joint/root smoothing has not demonstrated an aggregate completion advantage over grounding alone.

A matched native evaluation covered the same 89 screened training IDs and 20 development IDs for every arm. Each arm used 109 environments, seed 91260, deterministic policy inference, the same plane environment and native termination criteria. Each of eight arms exited successfully, giving 872 clip-policy evaluations with zero optimizer updates. Original motions here are the identical selected IDs, not the full previous 100-motion training set. “Completion” means no native early termination; it is not a stricter tracking-error qualification or navigation/obstacle success.

| Policy | Motion inputs | Train completion | Development completion | Train mean progress | Development mean progress |
|---|---|---:|---:|---:|---:|
| Released SONIC | Original | 35/89 | 6/20 | 47.3% | 37.4% |
| Released SONIC | Grounding only | **84/89** | **17/20** | **95.5%** | 91.1% |
| Released SONIC | Full repair | **83/89** | **17/20** | 94.8% | **93.2%** |
| Previous teacher, step 8000 | Original | 60/89 | 6/20 | 76.5% | 50.8% |
| Previous teacher, step 8000 | Full repair | 55/89 | 9/20 | 72.1% | 61.6% |
| New teacher, step 500 | Full repair | 31/89 | 4/20 | 59.1% | 40.3% |
| New teacher, step 13100 | Original | 2/89 | 0/20 | 21.0% | 11.4% |
| New teacher, step 13100 | Full repair | 28/89 | 3/20 | 50.5% | 36.1% |

Across 109 clips, grounding gains 61 completions and loses one relative to original inputs. Full repair gains one and loses two relative to grounding. Step 500 loses 65 released-policy completions and gains none; step 13100 loses 69 and gains none. This is already a severe early regression, followed by failure to recover, rather than evidence that deterioration began only late. The available checkpoints do not establish monotonically decreasing performance at every iteration.

On the identical surviving prefix of each repaired clip, the mean of per-clip global body-position errors is 122.6 mm for release versus 235.6 mm for step 13100. Post-failure/reset frames were excluded using native progress and 50 Hz expected duration. This controls for unequal failure horizons, but uses short prefixes on failing clips; it is not a full-episode quality certificate. Completion alone must not qualify student labels.

Older development-only reports used seed 91231 and a 20-environment evaluation batch. Their counts (for example previous teacher 11/20) should not be merged with this new matched protocol (9/20). This single-seed experiment establishes a large regression, not a multi-seed performance estimate.

## Why fine-tuning is the leading problem

### Confirmed setup differences

| Setting | Previous teacher | Repaired teacher |
|---|---:|---:|
| Environments | 512 | 128 |
| Steps per environment per rollout | 24 | 24 |
| PPO epochs | 5 | 5 |
| Minibatches per epoch | 4 | 8 |
| Trajectories per minibatch | 128 | 16 |
| Transitions per minibatch | 3072 | 384 |
| Optimizer steps per rollout | 20 | 40 |
| Training motions | 100 original | 89 screened repaired |
| Initialization | Pilot step 200, optimizer and scheduler restored | Released actor/critic, fresh optimizer |

The previous run's `initialization.json` explicitly records 69 restored optimizer-state entries and restored scheduler history. Prior wording describing both runs as fresh-optimizer fits from release was incorrect. The old run also has 200 pilot iterations of ancestry. These runs are not a controlled experiment isolating motion repair.

The new setup makes eight times as many optimizer steps per collected transition, with eight times smaller minibatches, compared with the previous setup. Each sample still participates in five PPO epochs: this is **not eight times as many sample reuses**. More iterations compensate environment exposure but do not restore the old optimizer dynamics. The new 13,166 observed iterations represent about 40.45 million transitions; the old 8,000 iterations represent 98.30 million new transitions, excluding its pilot ancestry.

### Evidence consistent with unstable updates and forgetting

- New first logged approximate PPO KL is **0.661**, with clipping fraction **0.499**; previous run starts at KL 0.0677. Both configure desired KL 0.01. The logged approximate KL and the analytic KL used by the learning-rate controller are different statistics, so their numeric ratio is a warning signal, not a measured violation of a hard constraint.
- New last-100-update mean approximate KL is **0.0518**, clipping fraction **0.398**, reward **2.954**. Previous final-100 means are KL **0.0277**, clipping fraction **0.210**, reward **6.177**. Reward curves are contextual diagnostics because data and initialization differ.
- Adaptive LR reaches its **1e-5 floor**. The controller lowers LR when analytic KL is high but does not impose a hard KL early-stop in this path. A floor alone cannot ensure conservative policy changes.
- Saved optimizer groups contain LR 2e-5 while logged adaptive LR is 1e-5. The scheduler and within-minibatch controller both touch LR; do not infer actual update LR solely from checkpoint groups. A future ablation must log effective LR at optimizer.step.
- Both original and repaired evaluation regress, and training-clip completion collapses too. Ordinary held-out overfitting alone does not explain this pattern. Loss of pretrained tracking capability is the strongest behavioral description.

Small batches, fresh optimizer/critic adaptation, actor/decoder drift, and rollout distribution mismatch remain **candidate mechanisms**, not individually proven causes. The six-way crossing identifies damaged policy capability rather than globally broken repair; it cannot assign a causal percentage to each optimizer setting. More training with the unchanged configuration is unsupported by the observed results.

Relevant primary research: [PPO](https://arxiv.org/abs/1707.06347) motivates multiple minibatch updates, while [Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem](https://arxiv.org/abs/2402.02868) documents forgetting during RL adaptation. Neither substitutes for controlled ablations in our system.

## Does the motion repair make sense?

**Grounding: yes, with strong native evidence. Additional smoothing: plausible but not yet beneficial on aggregate completion. Offline force screening: useful triage, not a physical certificate.**

The repair source is `climb-feasibility-first/tools/sonic_repair.py` plus `build_sonic_repaired_bank.py`. It removes scene vertical translation, bounded-smooths joint positions, and projects/smooths root height relative to robot collision geometry. Horizontal paths, root attitude and timing are preserved. It does not retime trajectories. Its nominal support clearance is 3 mm within a 12 mm band, and specially detected ballistic spans retain a constant height shift.

An independent serialization/fidelity audit of the evaluated 109 clips found:

- Frame counts and FPS unchanged; all audited arrays finite.
- Horizontal root coordinates exactly preserved; root quaternion differences at float32 roundoff (maximum 1.20e-7).
- Exported pose-axis-angle joint magnitudes exactly agree with absolute DOF angles; root pose and quaternion rotations agree within 1.22e-7 rad. This checks magnitudes/root encoding, not every joint-axis sign or all robot kinematics.
- Maximum additional root-height change after grounding is **0.02665 m**. Original-to-repaired height changes can reach **0.926 m**, which includes scene-height removal and should not be interpreted as smoothing deformation.
- Maximum joint change is **0.3315 rad** (~19 degrees). This is material for some clips even though within the configured 0.35-rad fidelity limit; retain per-clip visual/physical checks.

The offline screen uses smoothed finite-difference velocities/accelerations, MuJoCo inverse dynamics with internal constraints disabled, and a friction-pyramid contact-force feasibility program. Nearby collision geometry can supply candidate contact support within 2 cm; these are not necessarily intended stance-foot contacts. It does not establish actual PhysX forces, stance-foot locking, self-collision safety, actuator tracking under noise, or obstacle compatibility. In particular, grounding to a flat plane is inappropriate for intended stair/obstacle support surfaces without scene-aware support geometry. Jump preservation is heuristic, not a validated contact-phase label.

Keep original, grounding-only and fully repaired variants with provenance. Do not discard grounding-only or assume all offline-passing clips are valid downstream demonstrations. Full repair changes only three completion outcomes relative to grounding in this evaluation; inspect those clips before declaring smoothing necessary. Per-clip changed IDs are in `completion-changes.json`.

## Proposed next bounded experiment — not launched

1. **Use released policy on repaired/grounded inputs as the baseline.** Before student recollection, apply the existing strict body/root tracking-error criteria, censor failures, and evaluate scene collision/goal criteria separately. Do not automatically promote all 100 completed repaired episodes to labels.
2. **Isolate optimizer stability on fixed repaired data and release initialization.** At 128 environments, compare the current eight-minibatch setup against one minibatch (128 trajectories), keeping five epochs and matched transition budgets. Evaluate step 0, 10, 50, 100, 250 and 500; stop regressions promptly. One minibatch restores the previous trajectories-per-minibatch without increasing simulation environment count, though update memory still needs a smoke measurement.
3. **Then isolate conservatism.** Test a lower effective LR and floor, effective-LR logging, and a hard analytic-KL stop; keep other settings fixed. Separately test critic warm-up with frozen actor, then supervised anchoring on released-policy successful rollouts or a frozen base decoder plus bounded residual. These are competing ablations, not one combined intervention whose benefit cannot be attributed.
4. **Select by physical evaluation, not duration/reward.** Preserve step-zero and best-qualified checkpoints. Require non-regression on train and development completion plus failure-censored body/root errors; confirm any improvement across multiple seeds. Reserve a fresh held-out test set because this development set has been repeatedly inspected.
5. **Return to scene/navigation distillation only with qualified labels.** Retain a trusted full-command teacher and frozen base motion controller; learn navigation-conditioned command generation and a bounded residual. Scene-aware support and obstacle clearance require their own physical qualification. This plane-tracking diagnosis does not validate navigation.

## Reproducibility and evidence

Packet: `/home/linjiw/research-data/m2s-repair-training-diagnosis-20260912/`.

- `plan.json`, each arm's `command.json`, `receipt.json`, `evaluation.log`, native metrics and NPZ trajectories.
- `policies.json` and `all-motion-input-hashes.json`; preserved checkpoints including stopped step 13100.
- `evaluation-summary.json`, `paired-outcomes.json`, `completion-changes.json`, `common-prefix-errors.json`.
- `motion-fidelity-audit.json`, `encoding-audit.json`, `config-differences.json`, `checkpoint-training-metadata.json`, `training-summary.json`.
- `run-matrix.py`, `run-grounding-ablation.py`, `run-early-checkpoint.py`, `summarize.py`, `audit-encoding.py`, `audit-prefix.py`. Original evaluation scripts retain absolute source paths and are forensic reproduction scripts, not portable package entry points. The matrix launcher refuses an existing output root.
- `evaluation-comparison.png` and `training-comparison.png`.
- Stop receipts are in the source teacher packet: `USER-STOP-REQUEST.json`, `USER-STOPPED.json`, `attempt-1/exit.json`.

All eight native evaluation receipts have exit code zero. The audit used the existing `.venv_isaaclab/bin/python`; no new model training or source dataset mutation was performed.
