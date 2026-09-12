# Private Kimodo navigation development study

`study.py` audits the existing 1,000-clip Kimodo CSV bank and runs a fixed
100-fit / 20-development reference-geometry comparison. It does not run SONIC,
Isaac Lab episodes, or hardware. Prompt-connected components determine the new
development partition. Joint-limit violations remain visible; clips are not
silently repaired or excluded.

The completed artifact is outside the repository:
`/home/linjiw/research-data/m2s-kimodo-navigation-100-20260911/REPORT.md`.
Open `viewer/index.html` inside that directory for the local mesh viewer.
Do not copy licensed motion arrays into public repository assets or host them.

The study has already completed its four fits / 800-update budget. Its runner
refuses an existing output directory. Do not delete that directory to rerun;
new experiments require a separately dated design and output path.

`render.py` builds a self-contained viewer from the study outputs and local G1
visual mesh assets. It refuses an existing viewer directory. Display mesh
simplification does not change the conservative geometry used for labels.
The viewer shows one sampled obstacle and alternative candidate locations;
it does not demonstrate physical passage or decision-criticality.

Focused validation:

```sh
OMP_NUM_THREADS=2 .venv_isaaclab/bin/python -m pytest \
  decoupled_wbc/tests/test_motion2scene_navigation.py \
  decoupled_wbc/tests/test_motion2scene_source_split.py \
  decoupled_wbc/tests/test_motion2scene_constrained.py \
  decoupled_wbc/tests/test_motion2scene_native_label_roles.py -q
node --check scripts/research/lflh_next/navigation/app.js
```
