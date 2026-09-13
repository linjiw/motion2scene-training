# Full-command motor recovery: completed experiments, September 13, 2026

Follow-up: [contribution report](MOTOR_TO_NAVIGATION_CONTRIBUTION_20260913.md) and [matched navigation task pilot](NAVIGATION_MOTOR_PILOT_20260913.md). Compact results and figure data are preserved in the repository's [motor evidence bundle](evidence/motor-recovery-20260913/README.md).

The selected student completes **88/89 training motions and 10/20 development motions** on the original evaluation seed. Repeating the selected checkpoint gives **85/89 and 86/89 train**, with **10/20 development on both repeats**. The historical BFM checkpoint completed 5/89 and 0/20. This is a substantial full-command improvement, with a changed 114D input interface and pretrained-encoder architecture. It does not establish arbitrary command masking or goal-only navigation.

All bounded stages are finished. No further training was left running by this experiment. The selected checkpoint is the 3,200-update checkpoint from the second online stage; training to 6,400 updates did not improve nominal-start evaluation. Selection used training completions, then training mean progress; development scores were not used to choose checkpoints. Development has been inspected throughout this research and is not an untouched final test set.

## Artifacts and reproduction

Experiment root: `/home/linjiw/research-data/m2s-bfm-motor-recovery-20260913`.

- Selected model: `scale/training/step-003200.pt`.
- SHA-256: `1e1ed0efaf386245afc59fda3f0e7fd1ee13ba23b3ad3c4cfd6fe0892c8b94cf`.
- Teacher SHA-256: `afd649cfbbfd28833550e11a0f8c3b7a5f6a05ee8b4021dd0dac97a6f94733ce`.
- Machine-readable results: `results.json`, `tracking-results.csv`, `stopping-results.csv`.
- Figure and supporting data: `motor-recovery.png`, `motor-recovery-plot.csv`.
- Selection and integrity evidence: `scale-complete.json`, `selected-checkpoint-audit.json`, `paired-motion-audit.json`, `result-input-sha256.json`, `final-artifacts.json`.
- Navigation supervision: `successful-stopping-context.json`, referencing eight hashed executed collections, physical task receipts and NPZ shards.

The [implementation protocol](BFM_MOTOR_RECOVERY_20260913.md) describes interfaces and data tiers. Native stages retain `command.json`, `receipt.json`, source hashes, logs and metrics. Offline stages retain configs, receipts, checkpoints and loss logs. Source changed between the rejected and corrected stages; use per-stage source hashes when interpreting earlier runs. The final source snapshot describes the corrected implementation.

To repeat the selected native evaluation into fresh output directories on this machine:

```bash
PYTHONPATH=. .venv_isaaclab/bin/python - <<'PY'
import json
import subprocess
from pathlib import Path

packet = Path('/home/linjiw/research-data/m2s-bfm-motor-recovery-20260913')
out = packet / 'selected-repeat-new'
out.mkdir(exist_ok=False)
argv = json.loads((packet / 'scale-3200-eval-full/command.json').read_text())
replacements = {
    '++eval_output_dir=': str(out / 'metrics'),
    '++eval_base_dir=': str(out / 'hydra'),
}
argv = [next((prefix + value for prefix, value in replacements.items()
              if argument.startswith(prefix)), argument) for argument in argv]
subprocess.run(argv, check=True)
PY
```

The selected model contains the offline architecture config and a separate `online_config`. Inherited `foundation_architecture`, `transformer`, and mixed `command_curriculum` fields are historical metadata: the effective `motor_architecture=anticipatory` chooses the MLP forecaster and these motor trainers use full commands exclusively. `allow_teacher_prefixes` in the inherited config is also superseded by the motor trainer's explicitly enabled support ablation. Inspect the implementation and recorded manifest, rather than interpreting those inherited fields as active settings. New online stages initialize a fresh optimizer; they are not exact simulator/RNG resumes.

## Native full-command results

Completion means reaching the end of the reference without native early termination. Mean progress is the average fraction of reference duration reached, not a navigation success rate. Evaluation uses nominal starts, fixed native termination, repaired 89/20 splits and no teacher action assistance. The teacher retains its actual future reference; the public students do not receive future reference frames at inference.

| Method, original seed 91260 | Train complete | Train progress | Development complete | Development progress |
|---|---:|---:|---:|---:|
| Historical BFM, 79D | 5/89 | 31.3% | 0/20 | 10.4% |
| Strict support, full-only CVAE, 79D | 5/89 | 34.1% | 0/20 | 10.7% |
| Broader support, full-only CVAE, 79D | 6/89 | 37.0% | 0/20 | 10.2% |
| Broader support, public action/token loss, 79D | 5/89 | 36.3% | 0/20 | 10.5% |
| Best legacy online DAgger, 79D | 14/89 | 43.4% | 0/20 | 11.6% |
| Current-frame extension, transformer, 114D | 21/89 | 55.3% | 0/20 | 17.9% |
| Preserved encoder, current-frame repetition, no learned anticipation | 16/89 | 29.4% | 1/20 | 23.7% |
| Preserved encoder, learned anticipation, 114D | 77/89 | 90.7% | 8/20 | 64.7% |
| **Anticipation + selected online DAgger, 114D** | **88/89** | **98.9%** | **10/20** | **67.4%** |
| Frozen teacher, full ten-frame reference | 88/89 | 99.0% | 15/20 | 82.6% |

The final student matches the teacher's training completion count on this seed, but fails a different motion. It has not demonstrated teacher-equivalent tracking accuracy or development generalization. Each main offline fit received 6,000 updates, batch 256; the anticipatory architecture replaces the learned token prior with a reference predictor and preserves more pretrained structure. Its gain cannot be attributed to the input extension alone or advertised as an equal-architecture BFM reproduction.

| Selected checkpoint evaluation seed | Train complete | Train progress | Development complete | Development progress |
|---|---:|---:|---:|---:|
| 91260, selection seed | 88/89 | 98.95% | 10/20 | 67.36% |
| 91261 | 85/89 | 96.11% | 10/20 | 70.98% |
| 91262 | 86/89 | 96.91% | 10/20 | 69.03% |

These are evaluation-seed repeats of one trained model, not three independent training runs. The old 79D online winner repeated at only 8/89 and 6/89 train, both with 0/20 development. Its original 14/89 result was much less stable.

## What was limiting distillation

**Public command completeness and the pretrained representation matter.** The original command lacks current target joint velocities and complete relative root orientation. Adding those 35 present-time quantities helps, but reaches only 21/89. Preserving the teacher encoder and FSQ quantizer gives a useful pretrained current-reference baseline. Predicting nine missing desired-reference frames from measured history plus current commands then reaches 77/89. The first physical target frame is fixed; a learned residual predicts only the later frames. This is deterministic reference prediction, not joint physical-state/action flow matching.

**Teacher anticipation supplies important information.** Correctly packed teacher probes give 16/89 train and 1/20 development using the current frame repeated, 34/89 and 6/20 with the first two real reference frames, and 88/89 and 15/20 with the full reference. The two-frame probe is a diagnostic supplied with extra future information, not a causal public student. The zero-initialized anticipatory wrapper reproduces the current-only teacher's native scores exactly.

**More labels and more updates alone were insufficient.** Expanding strict teacher support from 35,339 to 75,187 rows and retaining the same nominal/query aggregate produced 122,587 training rows over 822 episodes. The broader CVAE fit increased completion by one motion. Public action/token regression alone did not solve the problem. Online DAgger on the original 79D representation helped modestly; it became far more effective after the motor representation was preserved and missing context was predicted.

This interpretation is consistent with testing BFM's emphasis on online teacher supervision at student-visited states, but our frozen SONIC encoder/decoder design remains an adaptation. The paper's model and training setup differ; these controls do not establish that the paper itself needs our anticipator. [Behavior Foundation Model for Humanoid Robots](https://arxiv.org/html/2509.13780v1).

**A new experimental packing error was caught and excluded.** Native 640D inputs do not contain ten chronological 64D physical frames when simply reshaped. The initial new extension misread later positions as current velocity. Those early horizon/extended-student runs are listed in `invalid-layout-experiments.json` and excluded from results and final selection. They do not explain the historical 79D model's poor results, which used the reference input opaquely. An intermediate orientation audit also rejected a noise-free equality assertion; corrected checks account for the native ±0.05 orientation observation noise. A first stopping export failed on missing body names and was rerun after repair. These are recorded implementation failures, not robot-policy failures.

## DAgger coverage and stopping supervision

The first online control used 256 environments, 1,638,400 transitions and 3,200 updates. The anticipatory stage used **512 environments**, 1,638,400 transitions and 6,400 updates. Both query the teacher on the actual pre-step state and mix fresh queries with nominal teacher replay. Starts are randomized, with a 20% nominal-start mixture. Current-command fields are checked against captured current-frame teacher observations. Burn-in and takeovers are explicit; no scene-success labels are inferred from them.

| Anticipatory online updates | Train complete | Development complete |
|---:|---:|---:|
| 1,600 | 86/89 | 9/20 |
| **3,200, selected** | **88/89** | **10/20** |
| 4,800 | 86/89 | 10/20 |
| 6,400 | 86/89 | 9/20 |

The selected checkpoint has seen 819,200 online transitions. The full stage covers all 89 training motions and all four temporal quarters; quarter row counts are 213,503 / 350,759 / 477,125 / 597,013. The full-stage actual teacher-action fraction is 32.56%, including burn-in and takeovers. These assisted rollouts must not be presented as autonomous evaluation.

Of 21,584 local takeovers, 1,288 satisfied the conservative recovery criterion, 20,190 failed it, and 106 remained censored. The criterion is four stable final ticks within 0.10 m mean body error during a 16-tick takeover, without reset. Failures combine boundary events and failure to recover within that short horizon; they are not all falls. This remains a weak recovery supervision source despite improved normal tracking. Hard-motion mining is not enabled, and only sparse query snapshots are saved beyond the online replay used during training.

Four training motions, 00908 / 00413 / 00976 / 00265, each passed both clear and corridor stopping tasks: **8/8 successful teacher continuations, 2,739 synchronized context/action rows**. Every run held the goal for 50 control ticks with zero measured undesired contact force. The scorer requires 3D goal error at most 0.25 m, speed at most 0.10 m/s, no fall and contact checks at 200 Hz. All successes occurred before the original reference ended, so the evidence supports selecting naturally decelerating motions; it does not isolate a benefit from the appended stationary tail.

The corridor obstacles leave a feasible passage. These demonstrations establish stopping and clearance, not obstacle-induced route selection or low-beam traversal. `scene_task_success=true` records the independent task result; `scene_teacher_qualified=false` preserves the separate legacy strict-reference qualification semantics. A future navigation loader must explicitly validate the independent task receipts, rather than blindly accepting generic prefix eligibility or relabeling all reference clips as successful navigation.

## Next navigation and flow experiment

Preserve this checkpoint as the full-command specialist and regression anchor. The next component should generate desired reference context from measured history, goal, observed geometry and available command masks. The encoder, quantizer and motor decoder can remain fixed while this generator learns. Full-command requests should retain the current specialist path, with an identity-preserving initialization and full-command replay if any shared trainable layers are introduced.

At inference the navigation branch must predict the **current** target as well as future targets: the present specialist requires current pose/velocity/orientation commands and deliberately rejects their absence. Simply applying a navigation mask to this checkpoint is not an implementation of navigation. Reference-motion ID, phase and future targets remain label/provenance fields, never hidden actor inputs. The public task interface should contain localized start/goal, body-frame relative goal, desired terminal heading/speed, valid obstacle tokens and explicit observation availability. Camera support later replaces the known-map geometry input through a separately validated perception adapter; missing observations must not mean free space.

Start with the successful stopping collection as a small positive-control dataset, splitting by source motion family rather than rows or clear/corridor replicas. Obtain new source families before claiming broad task generalization. For avoidance, hold start and goal fixed while changing obstacles and collect successful, genuinely different continuations. Re-query a teacher only where its planned reference is valid for the current scene and state; a tracking teacher with an obstructed stale reference cannot supply useful avoidance labels.

Use deterministic reference prediction as the baseline for a conditional flow model over short desired-reference chunks. The flow variant can represent multiple feasible routes under the same sparse goal/scene context. Train with action consistency through the frozen motor path and task-qualified trajectory labels, execute receding-horizon actions, and evaluate goal hold, contact, route response and motor regressions separately. Predicting future physical robot states would require separately aligned executed-state targets and a dynamics/consistency objective; reference anticipation here does not supply that result automatically.

Prioritize training-family variants and recovery windows corresponding to recurrent training failures (00950 on all seeds; 00163 and 00335 on repeats). Development motions 00187 / 00333 / 00611 / 00623 fail on all three student seeds despite teacher completion on the original seed: use these as diagnostic evidence of the remaining anticipation/generalization gap, not as new training data. A reserved evaluation set is needed after further tuning. Split recovery failure reasons before increasing takeover length or weighting; the current aggregate cannot distinguish unstable teacher queries from slow recoveries. An action-residual PPO model should be a later controlled ablation after these sources of error are measured, because the preserved decoder already supports strong nominal tracking.

## Validation

The selected checkpoint audit verifies tensor-exact teacher encoder and decoder weights, frozen gradients, nonzero trainable-forecaster gradients, exact reconstructed teacher tokens and maximum decoded teacher-action discrepancy of 8.6e-6 on 64 captured same-state rows. It also verifies current-frame parity, absence of future-label inputs from the public prior, rejection of missing required full commands, and hashes of all successful stopping shards/task receipts.

The focused suite passes **69 tests** across motor recovery, scene distillation, foundation navigation, coverage, transformer and action-flow modules. Ruff and Black checks cover the seven changed/new runtime modules and the focused motor test. Logs are in `final-tests.log`, `final-ruff.log` and `final-black.log` in the packet. These checks and the native rollouts support the reported bounded result; they do not claim real-robot deployment or general scene navigation.
