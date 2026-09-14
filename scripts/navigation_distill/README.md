# Navigation distillation: stopping tasks, goal/map adapter, supported recovery

These scripts drive the vendored `gear_sonic.research.scene_distillation` navigation stack on a
local teacher and motor student. They produced `workspace/nav-8192` and the
[navigation report](../../docs/sonic/motion2scene/NAV_8192_STOPPING_20260913.md). They add no
model code; every stage is one of the existing callbacks, launched through `eval_agent_trl.py`
with a single-environment `scene_usd` scene.

Set `NAV_PACKET` to the packet directory. Native launches use `.venv_native` with `PYTHONPATH`
unset and a user-owned `TMPDIR`. Each scene launch takes 80–200 s and about 3.8 GB of GPU
memory; two at a time is safe on a shared 32 GB card, three when it is otherwise idle.

```bash
export NAV_PACKET=$PWD/workspace/nav-8192 T=$PWD/scripts/navigation_distill
cd vendor/sonic && export PY="env -u PYTHONPATH PYTHONPATH=$PWD TRL_EXPERIMENTAL_SILENCE=1 ../../.venv_native/bin/python"
```

## Stages

| Stage | Command | Output |
|---|---|---|
| 1. Tasks | `$PY $T/build_tasks.py --motions <repaired train dir> --ledger <teacher packet>/motion-ledger.json --body-reference <hindsight reference.npz> --ids 00908 ... --output $NAV_PACKET/tasks` | `tasks/<id>/{clear,corridor}.{json,usda}`, native motion with a 2 s stationary tail, `manifest.json` |
| 2. Teacher positives | `bash $T/run_stage.sh teacher $NAV_PACKET/collect/teacher-S S - -- tasks/*/*.json` | `StoppingTeacherCallback`: `teacher-episode.npz` + `collection.json` per task; only successes are eligible |
| 3. Motor control | `bash $T/run_stage.sh full <stage> S <motor.pt> -- tasks...` | `FullMotorTaskCallback`: does the frozen motor complete the task with oracle commands |
| 4. nav-v0 | `$PY $T/aggregate_stopping.py --stages collect/teacher-S --output collect/teacher-positive.json`, then `$PY $T/nav_fit_config.py --motor <motor.pt> --manifest ... --out fit/nav-v0-config.json`, then `$PY -m gear_sonic.research.scene_distillation.navigation_motor --config ... --output fit/nav-v0` | goal/map adapter from teacher positives |
| 5. Motor demonstrations | `bash $T/run_stage.sh recovery <stage> S <motor.pt> --takeover-tick 0 -- tasks...` | `MotorRecoveryCollectionCallback` at switch 0: exact-motor executed rows |
| 6. nav-v1 | `$PY $T/aggregate_recovery.py --stages collect/motor-demo-S --output collect/motor-demo.json`, then `nav_fit_config.py --view motor_recovery --task-manifest tasks/manifest.json --warm-start fit/nav-v0/step-006000.pt` | adapter refit on motor-executed demonstrations |
| 7. Learner-prefix recoveries | `python3 $T/plan_recovery.py --tasks tasks/manifest.json --reference collect/teacher-S --fractions 0.15 0.3 0.5 --out plan.txt`, then `bash $T/run_recovery.sh <stage> S <nav.pt> <motor.pt> plan.txt` | navigation drives to the switch tick, the motor takes over; a suffix is admitted only if it earns its own 50-tick hold before the original deadline |
| 8. nav-v2 forks | `nav_fit_config.py --view motor_recovery --warm-start fit/nav-v1/... --recovery-fraction 0.5 --fresh-behavior <sha of nav-v1>` versus `--recovery-fraction 0` | equal-update replay-versus-recovery comparison |
| 9. Evaluate | `bash $T/run_stage.sh nav <stage> S <nav.pt> -- tasks...`; `python3 $T/table.py <stage>...` | unassisted goal/map panel at declared seeds |

`run_stage.sh` and `run_recovery.sh` skip tasks whose receipt already exists, so a stage can be
resumed after a failed launch (a shared GPU can refuse device memory at startup; the runner
records `PROCESS_FAILED` and moves on).

## Where the framework extends

The stack already separates the pieces a new student profile needs; extend at these seams
rather than adding a parallel trainer:

- **Task families.** `build_tasks.py` writes `bfm_known_map_navigation_task_v1` records. A new
  family (goal switch, route switch, posture switch, independently constructed layouts) is a
  new obstacle/goal recipe in that script plus its own `tasks/manifest.json`; every stage
  below reads tasks only through the manifest and `validate_task`.
- **Actor profiles.** `stage_config.py --actor-profile` and `nav_fit_config.py` select the
  public input contract (`nav_goal_map_v1`, `nav_goal_map_localization_v2`). A new profile is a
  new `NavigationInput` field plus its `task_context`/localization producer in
  `navigation_motor_runtime.py`; the recovery loader reconstructs and checks it before rows
  are admitted.
- **Sensor input later.** `task_context` is the only place the known map enters the actor.
  A camera/depth profile replaces that producer with a timestamped belief (visible, unknown,
  free) and keeps the exact map as training-only supervision; `DirectSceneTaskCallback`'s
  scorer and contact sensors do not change.
- **Motor backend.** Every stage takes `--motor`; a different frozen motor (another teacher,
  another distillation run) only needs its checkpoint hash in `ids.json`.
- **Evidence.** Each attempt keeps `task-result.json`, `trace.npz` and (for collections) a
  receipt bound by SHA-256 to task, checkpoint and score, so a later loader can re-derive
  support instead of trusting a flag.
