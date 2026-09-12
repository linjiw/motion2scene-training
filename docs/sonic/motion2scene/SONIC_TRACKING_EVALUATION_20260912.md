> 2026-09-12 update: [100-motion experiment, scene implementation, and trajectory-role correction](BFM_SCENE_NAVIGATION_COVERAGE_20260912.md). The final student is not navigation-qualified; use the corrected video/scene receipts linked there.

# SONIC completed-training evaluation and BFM decision

September 12, 2026. Follow-up to [the restart handoff](SONIC_DISTILLATION_HANDOFF_20260911.md).

The 8,000-additional-iteration run completed correctly. Its final checkpoint improves native complete-motion tracking on the training set, but does not improve development completion or matched-prefix position accuracy. **Do not promote this final checkpoint to a broadly qualified BFM teacher yet.** Prepare the foundation interface and collection machinery, then select and qualify the teacher before substantial student training.

## Completion and provenance

Training packet: `/home/linjiw/research-data/m2s-sonic-teacher-8000-20260911`.

Verified `FINAL-RECEIPT.json`: complete, trainer exit 0, iteration 8000. All 195 entries in `FINAL-MANIFEST.json` passed SHA-256 verification. Final checkpoint `tracking-run-1/model_step_008000.pt` has SHA-256 `48b3a1c04cdbbcd9ffe8ad10b2591aff781c253c03c61ccb8683348d862c54ed`.

Measured training: 98,304,000 environment transitions; 393,216,000 environment physics steps; 24,089.24 seconds callback wall time (6.69 hours). These exclude unobserved simulator initialization. All 100 training motions were sampled. No student training took place.

The current CUDA allocation check and `nvidia-smi` both succeeded; the handoff's driver/library mismatch is no longer present in this session. No driver change was performed here.

## Locked matched evaluation

New packet: `/home/linjiw/research-data/m2s-sonic-qualification-20260912`.

`evaluation-lock.json` and `commands.json` were written before either rollout. The original release and final checkpoint were copied into separate evaluation directories, each with the same saved training configuration. Each arm evaluated the same 120 motions in 120 environments, seed 91201, deterministic native G1 inference, 200 Hz physics / 50 Hz control, plane terrain, no obstacles or cameras. Native evaluation removes training pushes; other native evaluation settings are shared. Training schedules were disabled. There was no evaluation step truncation. Both processes exited 0 and produced 120 trajectory files and native metrics each.

The main outcome is completion without a native non-timeout tracking termination over the complete reference. This is a tracking diagnostic, **not** a contact, drift, navigation, or terminal-stabilization qualification. One seeded execution per motion is insufficient to establish robustness. Development motions were excluded from this training run, but this split is not an ancestry-held-out test.

| Split / metric | Original release | Final trained |
|---|---:|---:|
| Training complete motions | 35/100 (35%) | 63/100 (63%) |
| Training mean motion progress | 45.1% | 73.1% |
| Development complete motions | 8/20 (40%) | 6/20 (30%) |
| Development mean motion progress | 43.9% | 45.7% |
| Training common-prefix global body error | 129.4 mm | 133.1 mm |
| Development common-prefix global body error | 116.8 mm | 136.1 mm |

On training motions, 40 changed from failure to completion and 12 regressed. On development motions, one improved and three regressed. These results suggest training-set specialization; they do not establish its cause or statistical significance. Longer survival on some failures explains why development progress can rise while full-motion completion falls.

Position error is mean Euclidean error across the 14 tracked body points. Common-prefix errors compare each pair over the shorter surviving prefix, then average motions equally. They exclude terminating/reset samples. They are conditional on survival, not full-horizon success measures. Native metrics are retained unchanged, but their all-frame error arrays may include resets after failure; do not use those arrays as uninterrupted tracking evidence.

Native checks include reference-dependent height tolerances, anchor orientation and relative foot position. There is no explicit global root-XY termination guard in this configuration. A native success may therefore have substantial global drift: sample 00047's released controller completes while its body-position error grows visibly. Completion alone is not sufficient for scene teaching.

Artifacts: `summary.json`, `per-motion-results.csv`, `common-prefix-errors.json`, per-arm `metrics/metrics_eval.json`, per-motion NPZs, per-arm logs and exit receipts. A partial motion-directory preparation failed before simulation because the two split metadata files shared a filename; it is preserved in the adjacent `-preparation-incomplete` directory. The successful preparation merged metadata. No physical attempt was retried.

## Five motion videos

Open [the local video gallery](/home/linjiw/research-data/m2s-sonic-qualification-20260912/index.html). These are 1280×720, 25 fps rendered replays of **measured Isaac Lab skeleton trajectories**, with reference overlays and error curves; they are not RGB camera footage or robot-mesh renders. Orange is release, blue is trained, dashed gray is reference. After the first failure, the reference freezes and the tracked skeleton disappears; later automatic resets are excluded. Short failed episodes remain short.

The five IDs were selected before evaluation by taking every fourth lexically sorted development ID. All five failed under the trained checkpoint; none was replaced with a more attractive example.

| Motion / video | Release progress | Trained progress | Trained surviving duration |
|---|---:|---:|---:|
| [00047](/home/linjiw/research-data/m2s-sonic-qualification-20260912/videos/motion-00047.mp4) | 100%, complete | 16.3%, failed | 1.30 s |
| [00187](/home/linjiw/research-data/m2s-sonic-qualification-20260912/videos/motion-00187.mp4) | 2.7%, failed | 4.0%, failed | 0.30 s |
| [00379](/home/linjiw/research-data/m2s-sonic-qualification-20260912/videos/motion-00379.mp4) | 6.7%, failed | 18.4%, failed | 1.10 s |
| [00714](/home/linjiw/research-data/m2s-sonic-qualification-20260912/videos/motion-00714.mp4) | 5.6%, failed | 6.8%, failed | 0.34 s |
| [00921](/home/linjiw/research-data/m2s-sonic-qualification-20260912/videos/motion-00921.mp4) | 100%, complete | 34.3%, failed | 2.74 s |

## BFM student: recommended next protocol

Retain the architecture decision in [FOUNDATION_NAVIGATION.md](../../gear_sonic/research/scene_distillation/FOUNDATION_NAVIGATION.md): tracking teacher → scene-independent behavior foundation → scene-and-goal command head through a frozen foundation. The [BFM paper](https://arxiv.org/html/2509.13780v1) uses a masked root/kinematic/joint control interface, a conditional VAE, and online distillation. Its downstream action residual differs from our proposed command-only navigation adaptation. The current eight-command prototype is not full BFM interface fidelity.

1. **Teacher selection and diagnosis.** Before further PPO, lock a modest comparison of the released, 200-iteration, and selected intermediate checkpoints. Reuse development only as development, account for this new selection, and preserve every failure. Add pre-reset termination reason, root XY/yaw error, joint/action traces, and terminal hold measurements. Compare common-prefix error and complete motion outcomes. Fix reference initialization, infeasible motion segments, or optimization issues only when measured evidence identifies them. The current captures do not identify the individual termination cause.
2. **Define the command interface before collecting labels.** Specify body-frame root velocity/yaw/height, keypoint and joint command semantics, masks, timestamps, history, normalization and attainable ranges from qualified executions. The four navigation commands may omit critical turn/crouch/arrival information; test expressibility before freezing them. Preserve the native 930D history and 29-action processing. Keep a Gaussian behavior latent distinct from the 64D SONIC tokens.
3. **Implement a bounded foundation pilot.** Proposed first budget: 1 million teacher queries, at most 2 million student environment transitions, a two-hour wall cap, and no automatic extension. These are proposed engineering limits, not an executed or preregistered training result. Start with a documented qualified subset; retain failed-query and reset accounting. Warm-start from executed teacher states, then use same-state teacher action queries on student-visited states. Train the prior/posterior and token adapter through a fixed selected SONIC decoder using reconstruction and KL. Include declared command-mask profiles. Never substitute reference joint targets for executed teacher action labels.
4. **Qualify public-prior control.** Physically test full-command reconstruction and each intended masked command profile, using the public prior without the privileged posterior. Include turning, lateral movement, slowing, height changes and stabilized arrival, plus any proposed under-beam skills. Lock task-specific tracking/drift/contact/hold tolerances before this stage. Compare to the same selected teacher with equal episode assignments; include stochastic latent timing explicitly. Low imitation loss alone does not pass this gate.
5. **Then learn scene navigation.** Freeze the qualified foundation, proprioception encoder, token adapter and decoder; train only the recurrent command head with residual scale zero. A qualified scene expert must supply different continuations for changed geometry or goals and label the student's current state. The existing 240 geometry scenes have no executed scene-teacher labels. Map-conditioned fitting cannot start from nonexistent labels. Keep latent residuals as a separate ablation.

The immediate useful work is interface/collector implementation plus teacher diagnosis, not another blind long training run. Neither final-trained nor released completion rates support a broad foundation-teacher claim today. A deliberately narrow student pilot remains possible after the subset, commands and teacher queries are qualified. Original A/B/C assignments and eighteen reserved layouts remain untouched.

## Validation

Both physical evaluation commands are preserved verbatim in the packet's `commands.json`. Reproduce the video rendering with:

```bash
.venv_isaaclab/bin/python -m gear_sonic.research.hindsight_training.render_qualification /home/linjiw/research-data/m2s-sonic-qualification-20260912
```

The existing 50 hindsight/foundation tests passed. The added focused tests check failure-prefix rounding/censoring and exact trajectory/order/timing preservation:

```bash
.venv_isaaclab/bin/python -m pytest -q decoupled_wbc/tests/test_hindsight_qualification.py
.venv_research/bin/python -m black --check gear_sonic/research/hindsight_training/qualify.py gear_sonic/research/hindsight_training/render_qualification.py decoupled_wbc/tests/test_hindsight_qualification.py
.venv_research/bin/python -m ruff check --select E,F,I gear_sonic/research/hindsight_training/qualify.py gear_sonic/research/hindsight_training/render_qualification.py decoupled_wbc/tests/test_hindsight_qualification.py
```
