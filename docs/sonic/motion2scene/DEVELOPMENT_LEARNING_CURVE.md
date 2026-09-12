# Fixed development learning curve: M8, M16 and M32

The [declared continuation](/home/linjiw/research-data/groot-wbc/m2s-development-learning-curve-20260909-v1/study.json)
adds the missing native M16/M32 evaluation to the existing M8 panel. Declaration
SHA-256: `a81e9f64fdd9d661e0515cdf5e2c126028f6713326d35ef91b42f5f9edcd9c30`.
It was frozen on September 9, 2026, before any M8 evaluation or M16/M32 model was
available. This is an evaluation implementation and allocation, not a result.

The central question remains whether executed-motion construction and complete
continuation teaching improve traversal at matched physical acquisition cost.
This six-context development panel informs that question; reserved evaluation
remains necessary for a held-out claim. The fixed checkpoint comparisons and
their measured acquisition costs will be reported together. Unequal actual
costs must remain visible; equal encounter counts alone are not equal measured
physical cost. Do not interpolate unmeasured policy performance.

| Checkpoint | New learned-policy episodes | New script/fixed episodes | Analysis rows |
| --- | ---: | ---: | ---: |
| M8, existing declaration | 90 | 48 | 138 |
| M16 | 90 | 0; reuse exact M8 references | 138 |
| M32 | 90 | 0; reuse exact M8 references | 138 |
| Total | 270 | 48 | 414, from 318 unique episodes |

Each checkpoint includes all five constructors and all three corpus seeds,
evaluated on the unchanged six development contexts at execution seed 8732.
The 48 shared references comprise the existing strong script and all seven
complete fixed schedules. They are the same captures at later checkpoints and
are counted once, without treating them as new replicates. The continuation
adds at most 214,560 evaluation physics steps; all three panels together assign
at most 379,056. These evaluation maxima are separate from measured acquisition
steps and from the original 320,000-step charged acquisition ceiling per corpus.

M16/M32 evaluation requires all fifteen original acquisition trajectories to
finish M32 and the full original M8 panel to have known outcomes. The gate keeps
later development evaluation from affecting acquisition stopping or candidate
selection. The fixed M8 evaluation remains in its existing position before
acquisition continues. The new runner does not change the active staged worker
or launch itself in the background.

## Preserved experiment and accounting

The [runner](/home/linjiw/groot-wbc-sonic-sim-trackb/scripts/research/motion2scene_development_learning_curve.py)
binds each later policy to its exact assigned checkpoint, cumulative teacher
prefix and common ridge learner (λ=10, measured-tie initialization). Only the
existing observation-curriculum arm uses historical observable-gap weighting.
All other arms retain uniform phase weighting and the scene-wise complete
continuation teacher. There is no model selection or new teacher version.

Preparation checks the native runtime, sensor, command, asset and scoring
contracts against the corresponding M8 collection. The original scene identity,
execution seed, strong-script parameters and seven-schedule repertoire remain
fixed. A resource pause occurs before launching the current assignment. Known
outcomes are reused on resume; unknown or interrupted outcomes are retained
without automatic retry. Partial records and assigned tasks remain visible.

Analysis reports every corpus separately, paired construction differences,
matched finite-bank capability, selection failures and passage time on mutually
successful conditions with their counts. Time remains the existing measured
crossing-plus-stabilization time; it does not become full recovery duration.
Bootstrap teacher, encounter teacher and pre-update student costs come from the
recorded acquisition results. Evaluation cost is deduplicated by exact capture
identity, and proposal computation is linked separately. Missing measurements
remain unknown, not physical failures. Three corpus seeds and six reused
development contexts do not become eighteen independent held-out layouts.

## Executed validation

- 61 focused tests pass across the new runner and existing panel, statistics,
  evaluation-statistics and extension-task suites. They include exact checkpoint
  binding, shared-reference accounting, capacity pauses, idempotent resume and
  preserving unknown outcomes without retry. Black and Ruff pass on new code.
- [Native preparation receipt](/home/linjiw/research-data/groot-wbc/m2s-development-learning-curve-20260909-v1/native_preparation_smoke_result.json):
  existing M1 and M2 policies prepare and verify against the current executable
  source/asset closures, with identical runtime, sensor and scoring contracts.
  They live in separate smoke directories; they never fill future model slots.
- [Cost accounting receipt](/home/linjiw/research-data/groot-wbc/m2s-development-learning-curve-20260909-v1/acquisition_accounting_smoke.json):
  the new reader reproduces 328,992 recorded M2 acquisition steps across twelve
  corpora, including bootstrap. This checks existing receipts; it is not another
  physical replay or independent raw-capture re-audit.
- The declaration check returns `waiting_for_M32_acquisition` for all fifteen
  corpora. No M16/M32 evaluation directory was prepared and no new physics ran.

The first preparation-check invocation failed before creating outputs because
its helper assumed baseline assignments also had a `run_id`. The helper now
selects learned rows correctly; both native preparations subsequently passed.
An initial cost-check invocation similarly used the expanded-plan `seed` key
with an original-plan row; mapping its existing `physics_seed` resolved it.
Neither issue affected the frozen runner, acquisition histories or results.

## Commands

Check readiness now; this performs no physical execution:

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_development_learning_curve.py check \
  --study /home/linjiw/research-data/groot-wbc/m2s-development-learning-curve-20260909-v1
```

After all M32 acquisitions and the original M8 panel finish, run the two fixed
later panels. These commands are supplied but **have not been executed**. The
runner checks its gates and takes the existing acquisition locks itself.
Calling it again after a capacity pause resumes completed known assignments.
An unknown outcome requires inspecting the retained capture, not deleting it
or restarting an experiment under a different output path.

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_development_learning_curve.py run --budget 16 \
  --study /home/linjiw/research-data/groot-wbc/m2s-development-learning-curve-20260909-v1

OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_development_learning_curve.py run --budget 32 \
  --study /home/linjiw/research-data/groot-wbc/m2s-development-learning-curve-20260909-v1

OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_development_learning_curve.py analyze \
  --study /home/linjiw/research-data/groot-wbc/m2s-development-learning-curve-20260909-v1 \
  --out /home/linjiw/research-data/groot-wbc/m2s-development-learning-curve-statistics-20260909-v1
```

Analysis can retain incomplete outcomes after both later panels are prepared;
it does not launch them. Use a fresh analysis output directory to preserve
previous reports. Source hashes are part of the declaration: a changed runtime
or implementation requires an explicit version decision, not silent rebinding.

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python -m pytest -q \
  decoupled_wbc/tests/test_motion2scene_development_learning_curve.py \
  decoupled_wbc/tests/test_motion2scene_development_checkpoint_panel.py \
  decoupled_wbc/tests/test_motion2scene_development_panel_statistics.py \
  decoupled_wbc/tests/test_motion2scene_evaluation_statistics.py \
  decoupled_wbc/tests/test_motion2scene_extension_tasks.py
```

The single next policy-performance experiment remains the original **M8
138-episode closed-loop panel**. The new continuation fills the later evaluation
dependency; it does not replace that checkpoint or justify broader method changes.
