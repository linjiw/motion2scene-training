# Installed generation capability and the next qualification gate

The installed Kimodo pipeline can request longer references without changing the
frozen SONIC tracker. The immediate extension is a **six-second development
carrier**, followed by a measured low-height interval and recovery. This is an
available generation interface, not an established longer-traversal result.
The installed SONIC planner is also present and exposes useful controls, but
needs an offline interface audit and a new simulator bridge before contributing
physical capability evidence.

This assessment used local source, configuration and checkpoint inspection,
CPU-only import checks and ONNX protobuf parsing. It generated no motion and ran
no inference, physics, GPU work or hardware. Current runtime files, reference
registries and locked evaluation layouts were left unchanged. The machine-readable
receipt is
`/home/linjiw/research-data/groot-wbc/m2s-installed-generation-capability-v1/receipt.json`.

## Current ancestry and actual limits

The source41002 neutral reference is a Kimodo-generated straight walk. Its
d040/d055/d070/d085 alternatives are **authored `local_crouch` edits** of that
neutral: fixed root XY, source duration and gait phase, with local leg-joint and
root-height changes. Their names denote requested geometric drops; they are not
separately sampled low-height prior skills and do not guarantee that amount of
executed clearance. The original candidate registration specifies station0.55,
window0.30 and target drops0.040/0.055/0.070/0.085m. Physical qualification is
recorded separately from that reference-only candidate registration.

Every currently registered option has120 source frames at30Hz and199 loaded
frames at50Hz, spanning reference phase0–3.96s. The current high-level adapter
permits one early entry and one return. Neither longer motion nor reentry after
return is qualified. The new two-beam smoke demonstrates two constraints during
one adaptation, not two separate adaptation cycles.

Relevant implementation:

- [Kimodo generator](../../scripts/research/generate_kimodo_motions.py) accepts
  `--duration`, converts it to30Hz frames and caps a single segment at300 frames.
  Its4s default is an experimental choice, not a controller architecture limit.
- [Local adaptation](../../gear_sonic/dataset_generation/local_adaptation.py)
  accepts a station, window and ramp. It holds frame count and route fixed.
- [Retiming](../../gear_sonic/dataset_generation/deployable_retiming.py) can
  resample an authored adaptation with a measured slowdown. This changes frame
  count and timing, so it cannot enter the existing same-phase, equal-duration
  option bank without separate qualification.
- [Motion stitching](../../gear_sonic/dataset_generation/motion_transitions.py)
  searches reference gait/pose/velocity compatibility and blends re-anchored
  clips. It supplies kinematic proposals, not qualified physical joins.

The SONIC configuration uses a10s episode limit and a50Hz reference loader. A6s
candidate fits that existing episode limit. An extension beyond10s would require
an explicit episode-budget change and its own audit.

## Locally available prior assets

Kimodo environment: `.venv_kimodo/bin/python`, importing
`/home/linjiw/kimodo/kimodo` with Torch2.11.0+cu128. The import was checked with
CUDA hidden. The checkpoint is already cached at:

```
/home/linjiw/.cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1/snapshots/3020ad8c419c244e0429d360163730c63c4ed011/
```

Its configuration specifies `G1Skeleton34` and30Hz output; the local model card
states a10s/300-frame maximum. The model weights, configuration and card hashes
are pinned in the receipt. A working prompt cache already contains straight
walking and walk–crouch–recover descriptions:

```
/home/linjiw/research-data/groot-wbc/cg-wbc-v2-shared-seed-confirmatory/taxonomy/prompt_cache.npz
```

`Kimodo.__call__` also implements `multi_prompt=True`, per-segment frame counts,
constraints and overlapping transitions. The installed CLI invokes that path.
The repository's cached-prompt generator currently requests one unconstrained
segment per call. Using multi-prompt output would therefore be an explicit new
generation arm with its own provenance and physical qualification. G1
postprocessing is explicitly disabled in the upstream CLI; the optional
`motion_correction` extension is absent in the checked generation environment.

## Smallest longer-reference experiment

The following is a concrete **future development command**, not an executed
experiment. It reuses the inspected development seed41002; it makes no new-source
or held-out claim. Create a fresh output directory and register its inputs before
generation. The already existing model-cache symlink below resolves the pinned
Kimodo checkpoint and is used only as a model asset.

```bash
capability_root=/home/linjiw/research-data/groot-wbc/m2s-longer-reference-development-v1
mkdir "$capability_root"
printf '%s\n' 'A person walks at a steady pace in a straight line.' > "$capability_root/prompts.txt"
CHECKPOINT_DIR=/home/linjiw/research-data/groot-wbc/m2s-fresh-source-v1/model-cache \
HF_HUB_OFFLINE=1 LOCAL_CACHE=true CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv_kimodo/bin/python scripts/research/generate_kimodo_motions.py \
  --prompts "$capability_root/prompts.txt" \
  --cache /home/linjiw/research-data/groot-wbc/cg-wbc-v2-shared-seed-confirmatory/taxonomy/prompt_cache.npz \
  --out "$capability_root/generated" --model kimodo-g1-rp \
  --duration 6.0 --seeds 1 --seed-base 41002 --steps 100 --device cpu \
  --prompt-design-version longer_development_41002_v1
```

This requests180 source frames. The current exclusive-end loader should produce
299 samples spanning0–5.96s; read the actual loaded bank before selecting the
driver limit. Do not use the wrapper's legacy `max-steps=auto` calculation or pad
the old4s motion. Conversion uses the existing CLI:

```bash
PYTHONPATH=. .venv_kimodo/bin/python gear_sonic/data_process/convert_kimodo_to_motion_lib.py \
  --input "$capability_root/generated/000_a_person_walks_at_a_steady_pace_in_a_straight_li_s0.csv" \
  --output "$capability_root/neutral.pkl" --motion-key longer_development_41002_neutral \
  --source-fps 30
```

Then qualify in this order:

1. Record the complete generated neutral in a fresh sufficiently long empty
   corridor under the existing checkpoint and g1 encoder. Check source/loaded
   clocks, reset-free sensor alignment, stability and achieved progress.
2. Apply the existing bounded `local_crouch` operator to that carrier, first with
   one low-height interval. Keep each generated versus authored artifact
   distinguishable. Record the actual held-low envelope, onset delay and recovery;
   do not scale old transition times in proportion to duration.
3. Find candidate entry/return ticks from loaded-reference compatibility, then
   verify each transition physically from matched approach states. Register the
   successful ticks and finite horizon in a **new** option bank and opt-in adapter.
   Existing code hard-codes the early entry/3.3s return windows and cannot certify
   a longer option merely by loading a longer file.
4. Test one longer overhead constraint, then two separated constraints. Only
   after full recovery is verified should a second independently selected entry
   be qualified. Keep unsuccessful entries, late detections, contact and incomplete
   second encounters in the dataset and acquisition accounting.

If the neutral itself cannot be tracked for6s, that is a motion-execution gate;
another selector or scene constructor cannot remove it. If the longer generated
neutral works but the authored low-height option does not, compare a directly
generated crouch sequence or the installed planner after its separate bridge
audit. Retiming and multi-prompt stitching remain additional proposals whose
feasibility must be measured.

## Installed SONIC planner: exact graph and wrapper contract

The planner file is locally available, despite the earlier search missing the
Hugging Face cache:

```
/home/linjiw/.cache/huggingface/hub/models--nvidia--GEAR-SONIC/snapshots/6733128a3d8a523b1418b06bca3cdf61c8b0987f/planner_sonic.onnx
```

CPU graph parsing confirmed these exact tensors:

| Input | Type | Shape | Meaning to verify/use |
| --- | --- | --- | --- |
| `context_mujoco_qpos` | float32 |1×4×36| Four30Hz whole-body context samples |
| `target_vel` | float32 |1| Positive speed override; nonpositive selects the mode default |
| `mode` | int64 |1| Motion mode; explicit range validation required |
| `movement_direction` | float32 |1×3| World direction of translation |
| `facing_direction` | float32 |1×3| World facing direction |
| `random_seed` | int64 |1| Explicit model random seed |
| `height` | float32 |1| Target root-height keyframe selection for supporting modes |
| `has_specific_target` | int64 |1×1| Enable the four target samples |
| `specific_target_positions` | float32 |1×4×3| World target root positions |
| `specific_target_headings` | float32 |1×4| World yaw targets in radians |
| `allowed_pred_num_tokens` | int64 |1×11| Allowed6–16-token prediction lengths |

Outputs are `mujoco_qpos: float32[1,64,36]` and
`num_pred_frames: int32[1]`. The graph's int32 valid count differs from the int64
description in the local documentation. Only the first validated count of output
frames is usable. Each token represents four30Hz frames: the mask selects
24,28,…,64 frames. This is prediction horizon control, not a guarantee of a
desired traversal or crouch duration.

The active C++ backend is
`localmotion_kplanner_tensorrt.hpp`; `localmotion_kplanner_onnx.hpp` is marked
deprecated. The active default mask is `[0,0,0,1,1,1,0,0,0,0,0]`, allowing
36/40/44-frame outputs. The old ONNX backend and documentation use other defaults.
The cached graph clamps mode above28 and contains29 mode branches; C++ version2
allows0–26 and version1 allows0–19. Its actual behavior catalog must be qualified
against the cached model, not inferred from a newer documentation table.

In particular, mode4 is `IDLE_SQUAT` in the wrapper; mode22 is named
`STEALTH_WALK_2`, while the local documentation calls it crouched walking. A
`height` tensor does not establish arbitrary height control for walk mode, and
`target_vel=0` does not establish a protective stop. Begin with well-understood
mode0/1/2 controls and a separately measured mode4 height sweep; add a moving
low-height style only after its actual behavior and transitions are recorded.

## Bounded offline bridge before inference

The36D layout is `[world_xyz, qw,qx,qy,qz, 29 joint angles]`, in metres/radians,
Z-up, +X forward, +Y left. Use named joints from
`KIMODO_G1_JOINT_NAMES` and the existing
`convert_trajectory_joint_order_to_mujoco` helper. Isaac Lab order is different.
Do not feed `reference_g1_qpos` when claiming achieved-state conditioning: assemble
qpos from actual `root_pos_w`, `root_quat_w` and reordered `dof_pos`.

The deployed context is not simply the last four achieved states. At replan time
it samples the **current planned motion** from `current_frame+2` at50Hz, then four
samples at30Hz. Thus the context spans a future reference window from+0.04 to+0.14s.
It uses linear interpolation for positions/joints and quaternion slerp. Initial
startup instead repeats current joints at a fixed root height0.788740m and
identity orientation. These initialization conventions must not be described as
an executed approach history.

The first offline bridge should expose two named, separately audited context
sources:

- **Deployment parity:** sample the recorded current reference at the exact
  wrapper timestamps. Refuse missing endpoint samples instead of silently
  extending the clip. This establishes API/coordinate/rate parity first.
- **Achieved history experiment:** resample four actual states at30Hz, ending at
  the decision time. This is a changed input distribution and needs qualification.
  The context begins0.10s earlier, so anchor generated timestamps there; the first
  four output frames overlap already observed history. Do not append those frames
  as new future motion or count them as progress. The splice and remaining horizon
  must be audited explicitly before physical replay.

The model outputs30Hz references. The C++ path resamples to50Hz with linear
position/joint interpolation, quaternion slerp and finite-difference velocities;
it uses `floor(valid_count*50/30)` samples and can hold the final endpoint briefly.
The Python MotionLibRobot path uses exclusive-end interpolation instead. A bridge
must choose and record one convention, without resampling twice or silently
borrowing padded output. C++ also blends new planned animations over eight50Hz
frames; those are reference blends, not evidence that a physical robot can execute
the join from an arbitrary achieved state.

No Python ONNX Runtime installation was found in the three checked project
environments; the Isaac environment contains `onnx` for graph parsing. A separate
CPU inference environment or audited deployment backend is therefore needed for
the first inference test. No model is missing and no new motion-model training is
required. The missing piece is the tested context/output bridge and its physical
qualification.

Before invoking inference, validate named-joint permutation with distinct values,
WXYZ quaternion normalization/slerp, causal history bounds, all11 dtypes/shapes,
explicit seed, nonempty binary token mask, finite output and valid-count slicing.
Then run a small offline walk/slow-walk/squat-height matrix from matched contexts,
record actual output lengths and kinematic envelopes, and compare repeated seeded
outputs. Only after that audit should the unchanged SONIC tracker replay full
generated options, measure transition delay/clearance/progress/cost, and qualify
walk→adapt→walk or repeated schedules in fresh development scenes.
