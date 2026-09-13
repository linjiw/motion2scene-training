# Teacher review: matched evaluation and G1 mesh videos

These tools produced `workspace/teacher-8192-500-review` on 2026-09-13. They compare three
teacher checkpoints on the same motions: the release initialization (step 41,550), the
8192-env × 500-iteration teacher, and the previous 8000-iteration teacher. The arms,
checkpoint hashes and packet paths are listed in `review_common.py`. To review another run,
edit `ARMS`, `PACKET` and `SPLITS` there.

The pipeline has four stages:

1. **Evaluate in Isaac Lab.** `run_eval.sh` launches `eval_agent_trl.py` with
   `gear_sonic.research.hindsight_training.pose_capture.PoseCaptureQualificationCallback`.
   The callback writes the same native `metrics_eval.json` and `<key>.npz` files as
   `TrackingQualificationCallback`. It also writes a `<key>.pose.npz` for every motion,
   holding root pose, joints, reference frame and termination flags per step.
2. **Check each run.** `check_run.py` checks every run (`run_eval.sh` calls it
   automatically). It confirms that captured poses match the native trajectories and that
   MuJoCo forward kinematics reproduces Isaac's 14 body positions: it fails a run above
   20 mm and warns above 5 mm.
3. **Render.** `render_review.py` replays the captured states on the exact training URDF with
   MuJoCo EGL. Each MP4 has three panels, a translucent reference ghost and error/drift
   timelines. Completion, errors and failure times always come from Isaac, never from the
   replay.
4. **Summarize.** `summarize_review.py` writes `per-motion-results.csv`,
   `paired-results.csv`, `summary.json`, `index.html` and the reels.
   `plot_training_curves.py` plots the run's `metrics.jsonl`.

## Usage

Run the Python tools with `.venv_native/bin/python` and `MUJOCO_GL=egl`. Launch Isaac
evaluations one at a time, in tmux on shared hosts.

```bash
PY=.venv_native/bin/python
$PY scripts/teacher_review/stage_review.py          # symlink checkpoints, write evaluation-lock.json
bash scripts/teacher_review/run_sequence.sh previous8000/development trained500/development \
    release/development trained500/train release/train previous8000/train
MUJOCO_GL=egl $PY scripts/teacher_review/render_review.py --motions all --skip-existing
$PY scripts/teacher_review/summarize_review.py
$PY scripts/teacher_review/plot_training_curves.py
```

`stage_review.py` expects `<review>/video-selection-lock.json`. That file fixes the featured
motions before any evaluation output exists. `previous8000/development` runs first as a
reproduction canary against the recorded 11/20 baseline.

To test stages 2–4 without Isaac Sim, run
`make_synthetic_fixture.py --review <scratch dir>`. It fabricates outputs for three motions;
then point `check_run.py --allow-partial`, `render_review.py` and `summarize_review.py` at
the same `--review` directory. Copy `evaluation-lock.json` in first. On the fixture,
`previous8000/development` fails the reproduction check by design, because it holds only
2 of the 20 motions.

The review directory keeps a copy of these scripts under `tools/` for provenance. The tools
locate the checkout from their own path, or from `$M2S_KIT` if it is set.
