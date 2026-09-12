# Measured six-second duration-policy comparison

The [completed comparison](/home/linjiw/research-data/groot-wbc/m2s-timed-policy-comparison-v1/result.json) contains **15 actual episodes**, all measurement-admitted, on the same three training layouts with physics seed 8732. The learned 106D student and the one-reference sensor script pass all three with identical choices and times. This is development integration evidence; it does not establish a curriculum advantage, improved duration decisions, or held-out performance.

| Policy | Empty | Short beam | Long passage | Completion |
| --- | --- | --- | --- | --- |
| Learned | Neutral, 5.40 s | Sustained, 4.10 s | Sustained, 4.68 s | 3/3 |
| Scripted sustained | Neutral, 5.40 s | Sustained, 4.10 s | Sustained, 4.68 s | 3/3 |
| Constant short | 5.90 s | 4.06 s | Fail | 2/3 |
| Constant sustained | Finite-horizon hold failure | 4.10 s | 4.68 s | 2/3 |
| Always walk | 5.40 s | Fail | Fail | 1/3 |

Times are measured passage costs and should be compared **within each scene**, because the finish planes differ. The learned policy saves 0.50 s over constant short on the empty scene, matches the script, and misses the available 0.04 s short-beam saving. The constant-sustained empty episode crosses without fall, reset, beam contact or refusal, and returns legally; its required 0.30 s stabilization does not finish before the 5.94 s recorded horizon. It remains a task failure with null success time under the registered protocol. It is not labeled instability, and the episode is not dropped. The earlier sustained teacher passes at seed 8731; that does not replace the new-seed result.

The comparison fixes the reference registry, sensor, legal entry/return, scene, seed and scoring across policies. It uses 17,880 measured physics steps and 4,470 exactly aligned 106D sensor/state packets. All 290,550 rays, measured hit normals and normal-known masks are retained. The three training targets and the three evaluation layouts overlap; a different physics seed alone is not held-out layout evidence. The script selects a single reference family and is a simple baseline; the proposed V3 study still requires a physically validated strong multi-option script.

![Actual development policy outcomes](evidence/traversal-v2/duration_policy_comparison.png)

The [vector figure](evidence/traversal-v2/duration_policy_comparison.pdf), [source receipt](evidence/traversal-v2/duration_policy_comparison.json), and [renderer](../../scripts/research/render_motion2scene_duration_policy_comparison.py) bind every plotted outcome to its actual child capture.

## Independent portable dataset increment

The [211.9 MiB archive](/home/linjiw/research-data/groot-wbc/m2s-timed-policy-development-dataset-v1-aligned.tar.gz), [manifest](/home/linjiw/research-data/groot-wbc/m2s-timed-policy-development-dataset-v1-aligned/manifest.json), and [receipt](/home/linjiw/research-data/groot-wbc/m2s-timed-policy-development-dataset-v1-aligned.receipt.json) preserve these 15 distinct actual recordings separately from the immutable [16-episode qualification/teacher release](TIMED_DATASET_V1.md). Archive SHA-256: `e413a6fb180bd12cd3b2aa3fe0ae39c42f08c5ac09bc74f7694c89d0837adbf3`.

All **3,990 file hashes** pass the portable audit. Each episode includes pickle-free physical arrays, every raw measurement packet with identity fields removed, exact alignment masks, 106D student inputs, actual switches, complete 30-by-36 contact streams, and separate scene/outcome provenance. Configured policy mode/preference/model identity is separate from actual selected options, including no-entry neutral behavior. The parent registration, plan, 15 child outcomes and frozen source closures are bundled without counting their repeated assessments as new episodes. This policy increment contains **zero teacher targets**. It does not impute alternative outcomes at student-visited states. Pretrained weights and reference banks are excluded.

```bash
.venv_isaaclab/bin/python scripts/research/motion2scene_export_timed_dataset.py \
  audit --out /home/linjiw/research-data/groot-wbc/m2s-timed-policy-development-dataset-v1-aligned
```

For a fresh export, supply all 15 child result paths with `--results` and their parent `result.json` with `--suite-results`; use a new output directory. The exporter refuses to overwrite an existing release. The previous 16-episode archive remains unchanged.

## Recorded execution video

The [5.96-second video](/home/linjiw/research-data/groot-wbc/m2s-duration-visual-demo-v1/duration_execution.mp4) shows the earlier three forced-schedule long-passage **teacher** recordings, not the new policy comparison. The walk and short teacher fail; sustained passes. It replays recorded Isaac Lab states and measured beam forces using MuJoCo visual forward kinematics, with zero rendering physics steps. Visual meshes differ from the native collision bodies used for scoring.

The [render receipt](/home/linjiw/research-data/groot-wbc/m2s-duration-visual-demo-v1/receipt.json) binds the original physical sources. An [independent audit](/home/linjiw/research-data/groot-wbc/m2s-duration-visual-demo-v1/independent_replay_audit.json) confirms all 149 frames decode at 25 fps and 1200×520 pixels; the preview and recorded 3.40 s frame were visually inspected. Rendering used CPU Mesa software rendering and created no new traversal evidence.
