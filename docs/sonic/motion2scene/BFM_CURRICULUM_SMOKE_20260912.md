# BFM command curriculum: implemented and CPU-validated

The training path works, but this small experiment does **not** establish a curriculum advantage or physical navigation improvement. The repaired teacher continues independently on the GPU.

Implemented in `gear_sonic/research/scene_distillation/`:

- `curriculum.py`: explicit weighted command stages, increasing absolute update boundaries, availability-preserving masks, and an episode/occupied-reference-quarter sampler. Missing late phases are reported rather than invented.
- `train.py`: optional `command_curriculum` and `foundation_sampling="reference_quarters"`; existing uniform masking remains the default. Exact optimizer continuation checks curriculum, sampling and dataset identity, and schedules from the saved absolute update.
- `curriculum_smoke.py`: reproducible CPU comparison, hash-bound teacher/data, disjoint within-clip held-out blocks, matched initialization and update budgets, checkpoints and prior-only evaluation.

Run packet: `/home/linjiw/research-data/m2s-bfm-curriculum-smoke-20260912`.

## Executed experiment

Used the old teacher's FP32 collection, bound to its original decoder/checkpoint. No labels from the still-training repaired teacher were mixed in. Inspection of the actual collection identifies whole-qualified clips **00246 and 00770** (these differ from the earlier two-clip pilot). Both were nonterminated and passed the recorded whole-motion error thresholds. Their existing raw arrays contain 721 frames; the old prefix masks retained only 266. The derived dataset restores the recorded whole-qualified support, then splits it into 547 training frames, 142 validation frames, and 32 excluded boundary-gap frames. Source files remain unchanged.

Validation takes a contiguous approximately 20% block within each reference quarter, with two excluded adjacent frames on each side. This is within-clip interpolation: neighboring motion history can remain correlated. It is neither held-out-motion generalization nor a physical rollout. The old student already saw some of this source data, so both arms start from the same fresh seeded model, not that student checkpoint.

Each arm: CPU, 2 threads, batch 32, 240 AdamW updates, learning rate 0.0003, KL weight 0.001, public-prior action weight 1.0, frozen matching SONIC decoder. Both use the same episode/quarter sampling. Uniform uses the existing equal five-profile mixture. Curriculum uses 80 full-only updates, 80 updates with full/root weights 0.25/0.75, then 80 with full/root/navigation weights 0.2/0.2/0.6. This compares these two complete masking strategies; it does not isolate ordering from cumulative profile exposure.

| Held-out prior action MSE, mean over two clips | Initial | Uniform | Curriculum |
|---|---:|---:|---:|
| Full | 0.875349 | 0.175743 | 0.179833 |
| Root/heading | 0.874897 | 0.175966 | 0.178820 |
| Sparse navigation command | 0.875048 | 0.176136 | 0.178860 |

All 480 updates completed, about 6.6/6.5 seconds for the two optimization loops. Physics steps: **0**. Teacher-token/decoder parity is checked on sampled training batches by the existing imitation loss. Uniform wins this single-seed small test; curriculum navigation MSE is about 1.55% higher. Neither checkpoint is promoted.

A post-fit diagnostic reverses command order within each held-out clip while holding proprioception fixed, and separately removes command availability. Shuffling changes actions by only 0.0012–0.0028 RMS, with almost unchanged target error. This is consistent with weak command dependence on this narrow dataset; the commands may also vary too little to make this diagnostic decisive. Low cloning error alone does not establish controllability. Detailed per-clip values are in `command-sensitivity.json`; the exact diagnostic script is archived in the packet.

## Next experiment decision

Keep uniform masking as the baseline. Do not scale this curriculum merely because its optimization loss decreases. After repaired-teacher qualification, collect broader complete walking/turning/stopping episodes, record original reference phase before filtering, and retain interventions across the full duration. For aggregates with several episodes per motion, add motion-ID/source balancing above the current episode sampler before claiming motion-balanced coverage.

Require same-state, meaningfully different commands with qualified teacher continuations to test controllability. Do not assign an unchanged teacher action to an artificially changed command. Compare matched profile exposure and several seeds before attributing gains to curriculum ordering. Evaluate velocity/yaw/height, stability, duration and goal stopping separately from exact-reference tracking.

The existing scene task labels, collision qualification, recurrent public scene/goal head and bounded residual implementation remain the downstream path. Current scene probes are unqualified, so this smoke does not train a scene-navigation policy or run residual PPO on those failed labels. The pending repaired teacher must supply physically qualified continuations first.

## Reproduction and checks

```bash
.venv_isaaclab/bin/python -m gear_sonic.research.scene_distillation.curriculum_smoke \
  --base-config /home/linjiw/research-data/m2s-bfm-100motions-20260912/fit-2/config.json \
  --collection /home/linjiw/research-data/m2s-bfm-100motions-20260912/collection-0/collection.json \
  --output /home/linjiw/research-data/m2s-bfm-curriculum-smoke-20260912-rerun \
  --updates 240

.venv_isaaclab/bin/python -m pytest -q \
  decoupled_wbc/tests/test_bfm_curriculum.py \
  decoupled_wbc/tests/test_bfm_pipeline.py \
  decoupled_wbc/tests/test_bfm_coverage_navigation.py \
  decoupled_wbc/tests/test_bfm_command_quality.py
```

26 tests passed. Scoped Black and Ruff checks passed. Source snapshot in the packet was captured after execution and formatting, not as a prelaunch archive. The packet includes config/data hashes, split indices, saved optimizer/RNG state, metrics, receipts, evaluations and a final file manifest.
