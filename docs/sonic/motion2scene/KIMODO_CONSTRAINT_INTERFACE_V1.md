# Installed Kimodo constraints: input contract and one proposed experiment

The installed prior can condition a new low-height candidate on a named approach, route and explicit source-frame targets. Its ordinary G1 output does **not** exactly preserve these conditions. The practical next experiment is a single constrained sample, followed by measured input-match, native-limit and timing admission. The frozen SONIC tracker and existing option bank remain unchanged.

This audit inspected local Kimodo source at Git revision `1aece8c124d73d255ceff5086d983b844c9f4e94`, including working-tree file hashes. It loaded only the skeleton, converters and condition builder, with CUDA hidden. It performed **zero model loads, inference calls or physics steps**. The [audit receipt](../../../research-data/groot-wbc/m2s-kimodo-constraint-api-audit-v1/audit.json), [prepared native constraints](../../../research-data/groot-wbc/m2s-kimodo-constraint-api-audit-v1/constraints.json) and [experiment proposal](../../../research-data/groot-wbc/m2s-kimodo-constraint-api-audit-v1/proposal.json) retain exact inputs and code snapshots. The proposal is not an executed registration or a qualified motion.

The motivation is specific: the [independent six-second crouch sample](../../../research-data/groot-wbc/m2s-longer-direct-crouch-development-v1/completed.json) violates native knee and waist limits, differs from the neutral approach and realizes its low posture late. Neither text timing nor identical random seeds establish a usable transition. No clipping or physical execution was applied to that rejected candidate.

## Supported native interface

| Input | Exact accepted representation | What it supplies to this model |
| --- | --- | --- |
| `Root2DConstraintSet` / JSON `root2d` | Integer `frame_indices[T]`, `smooth_root_2d[T,2]`; optional `global_root_heading[T,2]` | Smoothed root XZ coordinates and optional heading cosine/sine. A dense path is supported; it is not a pelvis-height or clearance constraint. |
| `FullBodyConstraintSet` / JSON `fullbody` | Python global positions `[T,34,3]`, global rotation matrices `[T,34,3,3]`, optional smoothed XZ; JSON instead stores local axis-angle `[T,34,3]` and root XYZ `[T,3]` | FK joint positions, root height, smoothed path and body heading. **Global joint rotations are deliberately not put into the full-body conditioning mask.** JSON local rotations construct position targets; they do not fix 29 hinge angles. |
| `EndEffectorConstraintSet` / JSON `end-effector` | Same pose fields plus `joint_names` | Selected positions and global rotations, together with root XZ, root height and heading. Valid generic names are case-sensitive `LeftFoot`, `RightFoot`, `LeftHand`, `RightHand`, `Hips`; `Hips` expands to `pelvis_skel`. The hand/foot shorthand classes are also present. |
| Explicit interval targets | Enumerated zero-based source frames | `crop_move(start,end)` uses `[start,end)` and shifts indices. The public JSON has no separate interval-duration, root-height inequality, beam-clearance, hinge-limit or maximum-speed constraint type. Sparse Hips targets are possible through the general end-effector class; continuous low height remains an outcome to measure. |
| Multiple prompts | `prompts=[...]`, `num_frames=[...]`, `multi_prompt=True`, explicit `num_samples=1` | Sequential diffusion segments, with a five-frame default transition. Lists without `multi_prompt=True` mean independent samples. Use an explicit frame-count list and sample count: the installed multiprompt scalar/default paths contain list-length/batch assumptions. |

The authoritative implementations are [constraint classes](/home/linjiw/kimodo/kimodo/constraints.py:75), [feature-mask construction](/home/linjiw/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:222) and [model call](/home/linjiw/kimodo/kimodo/model/kimodo_model.py:380). The installed [G1 example folder](/home/linjiw/kimodo/kimodo/assets/demo/examples/kimodo-g1-rp/02_multi_text_ee_constraint/meta.json) supplies concrete multiprompt durations; full-body and dense-path JSON examples are adjacent. These examples demonstrate supported syntax, not physical qualification in our tracker.

All constraint positions are **Y-up metres**, ground plane XZ. Root Y is absolute pelvis height. Heading is `[cos(theta), sin(theta)]`; `first_heading_angle` is a separate radian scalar or batch vector. Constraints are relative to the initial smoothed-root XZ origin. Use one recorded translation for every constraint field, and restore that translation before exporting to the original route coordinate convention. Do not independently recenter each keyframe.

The model config fixes 30 Hz, G1Skeleton34, separated classifier-free guidance and `motion_mask_mode: concat`. The denoiser replaces observed **input channels** and appends their mask, predicts root/body outputs, and passes these predictions to the diffusion sampler. It does not overwrite final predicted outputs with constraint targets. Increasing constraint guidance therefore changes soft conditioning, not a hard guarantee. See [denoiser](/home/linjiw/kimodo/kimodo/model/twostage_denoiser.py:100) and [denoising step](/home/linjiw/kimodo/kimodo/model/kimodo_model.py:77).

## Exactness, post-processing and transition limits

The standard [G1 CLI explicitly disables post-processing](/home/linjiw/kimodo/kimodo/scripts/generate.py:325), explaining that it does not work well for this skeleton. Although the Python model exposes `post_processing=True`, that is a separate MotionCorrection optimization path, not the current G1 generation contract. Do not invoke it silently to describe neural conditions as exact or to repair a rejected sample.

`MujocoQposConverter.dict_to_qpos` extracts one hinge angle per joint from generated rotations. It does not clamp angles. A separate [projection helper](/home/linjiw/kimodo/kimodo/exports/mujoco.py:432) defaults to limit clamping; using it changes the candidate and must be separately named, preserved and evaluated. Even an angle-valid result is not a feasible transition, stable motion, collision-free envelope or battery-energy result.

Multiprompt generation crops global conditions per segment, conditions each new segment on the previous segment's five trailing frames, then **blends and replaces those trailing representation frames** when post-processing is off. Total source length remains the sum of requested segment lengths. Text-segment boundaries do not guarantee that the desired behavior is already achieved at the boundary, and this API does not accept an untouched external prefix as its generated first segment. The proposed experiment uses one segment to avoid attributing exact prefix preservation to this blending path. The installed best-practice document favors fewer than 20 constrained frames per non-path type; dense root paths are the exception.

## Mapping to the frozen controller

| Boundary | Required conversion |
| --- | --- |
| Kimodo 34 joints → CSV | Native converter emits 36 columns: root XYZ, quaternion WXYZ, 29 named hinge angles. Kimodo `(x,y,z)` maps to MuJoCo `(z,x,y)`, hence Z-up and +X-forward. Use `mujoco_rest_zero=False`, matching all current source exports; the four added end-effector joints are not actuator columns. |
| Recorded achieved state → Kimodo input | Reconstruct root pose and 29 hinge values in the native MuJoCo order, then call `qpos_to_motion_dict`; do not treat the simulator's 30 body origins as a 34-joint Kimodo skeleton. Preserve actual first-episode timing and alignment masks. This audit uses a **planned** neutral reference, not an achieved-state prefix. |
| IsaacLab joints → MuJoCo joints | The explicit `G1_ISAACLAB_TO_MUJOCO_DOF` / `MJ_TO_IL` array selects IsaacLab indices for each MuJoCo output column. The inverse is `argsort`. Copying a 29-vector without this ordering is invalid. |
| 30 Hz source → loaded reference | Source frame `i` is at `i/30` seconds. The installed loader uses quaternion interpolation and `arange(0,(T-1)/30,1/50)`. For 180 source frames this gives 299 loaded frames ending at reference phase 5.96 s. Aligned physical capture currently ends at 5.94 s. |
| Loaded reference → runtime decision | Entry/return timestamps refer to the advanced reference cursor; the same indexed physical row is one 50 Hz tick earlier. Compare options using the same reference phase; retain the physical clock for measured traversal cost. Never align source and sensor arrays by equal row indices. |

See the [named converter contract](../../gear_sonic/dataset_generation/kimodo_motion_adapter.py), [joint ordering](../../gear_sonic/data_process/convert_soma_csv_to_motion_lib.py) and [exclusive-end interpolation](../../gear_sonic/utils/motion_lib/torch_humanoid_batch.py). The source conversion audit here finds maximum hinge round-trip error **0.00018373 rad**, with zero root-position error. Thus even the input bridge is measured rather than assumed bit-exact. No converter correction was applied.

## One minimal, preregisterable experiment

The prepared experiment makes **one CPU diffusion call** on the existing `Kimodo-G1-RP-v1` checkpoint and exact cached crouch prompt, seed 41002, 180 frames, 100 denoising steps, guidance `[2,2]`, and post-processing disabled. Checkpoint SHA256 is `e18c1de73e2ce17a107b06d85155fbbc5debe68eb35455aa5b033e6ddbe056a5`; cache SHA256 is `eea36e67890f77aedcb3c49dcc598282004c60825330a23f5f6b91c0faf96153`. Both were rehashed against the existing registration. The cache contains the exact sanitized prompt including its final period, so no new encoder or model is needed.

The [prepared constraints](../../../research-data/groot-wbc/m2s-kimodo-constraint-api-audit-v1/constraints.json) use the qualified six-second neutral CSV through the native converter:

1. Condition its complete smoothed root path and heading at frames 0–179.
2. Supply 19 full-body keys: frames 0–15 (planned approach through 0.50 s), plus 150, 165 and 179 (recovery targets at 5.00, 5.50 and 5.967 s).
3. Supply four `Hips` targets at frames 60, 75, 90 and 105 (2.0–3.5 s), each 0.085 m below that neutral frame's pelvis. Retain the neutral pelvis orientation and route; the model must infer leg/torso motion. These are pelvis targets, not prescribed body clearance or a continuous hold guarantee.
4. Canonicalize all constraints using the same stored smoothed-root origin; undo that translation before native CSV export. Preserve raw generated 34-joint motion and converted 29-hinge motion separately.

The source-frame schedule is a specified conditioning intervention. Its purpose is to test whether conditioned prior synthesis reduces the approach/timing defects of an independent text sample. It does not promise an executed low envelope. The existing cache has no isolated crouch-walk prompt, so adding a multiprompt crouch segment would require a separately recorded embedding; repeating the complex cached prompt is not an equivalent isolated skill request.

The native public JSON loader and condition builder successfully parsed these exact files: shape **`[1,180,417]`**, 2,717 observed channels, zero full-body rotation channels and 24 Hips rotation channels. This verifies interface construction only. Before generation, freeze the actual runner/import closure and proposal pins. After the single call, retain every failed gate and prohibit automatic resampling, clipping, blending, retiming or padding.

Admission should first inspect all 180 source and 299 loaded samples for finiteness, unit root quaternion and native 29-joint limits at the existing `1e-6` rad tolerance. Report every constraint error, full planned-approach pose/velocity error, route deviation, low-height duration and native body-envelope change. Candidate entry 0.40 s and return 5.50 s are fixed diagnostic requests; compare the existing join limits of 0.05 rad and 0.01 m at those phases. They are not yet qualified commands. If these gates fail, retain the candidate without inserting it into a registry. A subsequent separately registered matched-history empty-scene branch must establish actual entry, stability, low envelope, recovery and no reset. Only then can this become another executable option.

No V2/V3 reserved layout, beam clearance or outcome enters this preparation. Current manuscript performance claims remain unchanged.

Reproduce only the non-generative interface audit in a new directory:

```bash
CUDA_VISIBLE_DEVICES='' .venv_kimodo/bin/python \
  scripts/research/motion2scene_kimodo_constraint_audit.py \
  --out /tmp/m2s-kimodo-constraint-api-audit-reproduction
```
