# Motion2Scene methods atlas

Open [the local reading copy](index.html), [the complete PDF](research_methods.pdf), or [the editable Markdown](methods_draft.md).

The package contains a detailed introduction, method formulation, three extended captions, an evidence snapshot, and linked implementation sources. It describes the research state recorded on September 13, 2026. It preserves the distinction between the original Motion2Scene traversal studies, earlier masked CVAE experiments, the recovered anticipatory motor, and the newer goal/map navigation branch.

| Figure | Editable vector | Vector PDF | High-resolution image |
| --- | --- | --- | --- |
| Research overview | [SVG](01_research_overview.svg) | [PDF](01_research_overview.pdf) | [PNG](01_research_overview.png) |
| Teacher–student architecture | [SVG](02_teacher_student_architecture.svg) | [PDF](02_teacher_student_architecture.pdf) | [PNG](02_teacher_student_architecture.png) |
| Scene and learning illustration | [SVG](03_scene_and_learning_illustration.svg) | [PDF](03_scene_and_learning_illustration.pdf) | [PNG](03_scene_and_learning_illustration.png) |

The drawings are explanatory schematics, not simulated rollouts or measured result plots. Editable SVGs retain text objects. The HTML uses local figure files and embedded equation SVGs, with no external JavaScript or network dependency. The PDF document retains vector figures and rendered equations. For a slide or manuscript, use the individual vector figures rather than a screenshot of the reading copy.

## Reproduction

Run from the repository root:

```bash
.venv_research/bin/python docs/motion2scene/overview_20260913/draw_figures.py
PYTHONPATH=/tmp/m2s-overview-render-deps .venv_research/bin/python docs/motion2scene/overview_20260913/build_reading_copy.py
```

`draw_figures.py` uses the existing research environment's Matplotlib. `build_reading_copy.py` also requires Markdown and Playwright with Chromium. For this rendering, Markdown was installed only into `/tmp/m2s-overview-render-deps`; project dependencies were not changed. That temporary path is an environment convenience, not an artifact dependency. Another environment can install Markdown normally or provide its own `PYTHONPATH`. PyMuPDF in the same temporary directory was used for PDF inspection only.

## Validation

The figures were rendered and visually inspected. The reading copy was opened in headless Chromium; every image loaded, all local Markdown links resolved, and desktop horizontal overflow was absent. The PDF was inspected for pagination and equation rendering. Scoped Python formatting and lint were checked with:

```bash
.venv_research/bin/python -m black --check docs/motion2scene/overview_20260913/draw_figures.py docs/motion2scene/overview_20260913/build_reading_copy.py
.venv_research/bin/python -m ruff check --select E,F,I docs/motion2scene/overview_20260913/draw_figures.py docs/motion2scene/overview_20260913/build_reading_copy.py
```

No controller code, research data, checkpoint, or running experiment was modified. No simulation or training was launched for these documentation artifacts.
