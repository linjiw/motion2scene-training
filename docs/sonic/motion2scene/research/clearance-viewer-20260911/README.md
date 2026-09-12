# Motion and obstacle studio

Open `index.html` in a desktop browser. This self-contained file embeds Plotly, the recorded capsule motions, and the frozen placement probabilities. It needs no server or internet connection and sends no data anywhere. WebGL must be available.

For HTTP access, from the repository root:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs/motion2scene/research/clearance-viewer-20260911
```

Then open http://127.0.0.1:8765/ on that machine. For an SSH workspace, forward port 8765 to your computer. Serving this folder does not expose the repository root. This is a local tool, not a publicly deployed website.

Choose a target motion and model/seed. Drag to orbit, scroll to zoom, and use the frame slider or Play. The teal body is the recorded motion represented by its collision capsules; the line is its recorded root trajectory. The gray sweep shows capsule endpoints at every twelfth frame for visual context, not a certified continuous swept volume. The purple comparison motion uses the same recorded frame index, not a newly aligned physical encounter.

Each dot is the center of one possible beam, with fixed half-extents (0.15, 1.0, 0.10) m. The two variable parameters are forward position and underside height; this is a 2D distribution displayed in 3D, not a general learned 3D obstacle distribution. Dot color/size show probability. Every one of the 384 inspected coordinate cells remains visible, including zero-probability cells. Click a dot or sample one obstacle to inspect the actual beam box. Only that beam represents the candidate scene. The points are alternatives, not simultaneous obstacles.

Switch raw/constrained to inspect the explicit clearance projection. Classification colors distinguish critical, target-clear, uncertain, and capsule-penetration-witness cells. Displayed clearance values summarize the full recorded motion for that candidate, not only the frame currently shown. The numerical fields retain their original precision; displayed mesh vertices are rounded to 10 micrometers for export. The 50 Hz replay does not replace contact evidence or execute dynamics.

All seven targets and both seeds remain available. Neutral has no critical cells on this domain and is not dropped. No empirical safety or passage claim follows from this viewer.

## Reproduce

```bash
.venv_isaaclab/bin/python -m scripts.research.lflh_next.build_viewer --out /tmp/m2s-viewer-new
.venv_isaaclab/bin/python -m pytest decoupled_wbc/tests/test_motion2scene_constrained.py decoupled_wbc/tests/test_motion2scene_critical_learning.py decoupled_wbc/tests/test_motion2scene_lflh_geometry.py -q
```

The export command requires the trusted original bank and frozen experiment artifacts on this machine, verifies their existing manifest, and refuses an existing output directory. Sharing the generated HTML does not require those source files. Check data and dependency licensing separately before public redistribution.

See `RESEARCH.md`, `outcomes.csv`, `summary.json`, `receipt.json`, and `browser-check.json` for interpretation, complete finite-domain results and measured execution information.
