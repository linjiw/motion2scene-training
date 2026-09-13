# Motion2Scene methods revision

A detailed revision of the supplied teacher–student manuscript, grounded in the local generator, motor, navigation implementations, and recorded results through 13 September 2026.

## Read and reuse

- [Illustrated manuscript PDF](research_methods.pdf): introduction, methods, equations, five detailed captions, current results, and proposed experiments.
- [Offline reading copy](index.html): zoomable vector figures and locally rendered equations; no network needed.
- [Architecture atlas PDF](architecture_atlas.pdf): five original-size vector plates for inspection and presentation.
- [Editable full manuscript](methods_draft.md).
- [Introduction](introduction.md), [Motion2Scene insertion section](motion2scene_section.md), and [captions](figure_captions.md).

| Paper figure | Scope | Editable vector | Preview |
| --- | --- | --- | --- |
| 1 | Integrated research overview | [SVG](fig0_research_overview.svg) | [PNG](fig0_research_overview.png) |
| 2 | Learned hindsight categorical proposer | [SVG](fig1_hindsight_generator.svg) | [PNG](fig1_hindsight_generator.png) |
| 3 | Teacher and full-command motor | [SVG](fig2_teacher_full_command.svg) | [PNG](fig2_teacher_full_command.png) |
| 4 | Navigation/context actor and recovery | [SVG](fig3_navigation_context.svg) | [PNG](fig3_navigation_context.png) |
| A1 | Separate continuous inverse-beam branch | [SVG](figA_inverse_beam.svg) | [PNG](figA_inverse_beam.png) |

Each figure also has a standalone PDF under the same basename. SVG labels remain editable text; the G1 pose illustrations are embedded raster renders. The atlas preserves large figure dimensions so the detailed interfaces can be inspected without shrinking them to manuscript-column width.

## Substantive revisions

1. Adds the actual 21,185-parameter categorical hindsight generator: native features, fixed event anchors, 225 recipes, geometric target distribution, exact loss, and sequential sampling.
2. Separates that dataset-producing branch from the Gaussian inverse-beam diagnostic. Their losses and sampling laws are different.
3. Specifies the 114-value full-command motor, reference forecast, unusual native packing, retained encoder/FSQ/decoder, and trainable/frozen boundaries. Documents the earlier BFM-inspired CVAE separately.
4. Specifies the known-map navigation inputs, optional causal localization, shared obstacle encoding, command prediction, same-backend structured loss, and physically qualified recovery data.
5. Updates the attached draft's proposed recovery section with the completed experiment and its unfavorable confirmation result. Future scene-utility and wider downstream comparisons remain labeled as proposed work.

The figures borrow the clear training/inference organization of the supplied BFM and BFM-Zero examples. They do not import those papers' mechanisms into this implementation or present reference-pose renders as executed behavior.

## Rendering

From the repository root:

```bash
.venv_research/bin/python docs/motion2scene/methods_v2_20260913/draw_architectures.py
PYTHONPATH=/tmp/m2s-overview-render-deps .venv_research/bin/python docs/motion2scene/methods_v2_20260913/build_reading_copy.py
```

The diagram renderer uses Matplotlib and NumPy plus the included assets. The reading-copy renderer additionally uses Markdown, Playwright, and an installed Chromium; `/tmp/m2s-overview-render-deps` is the session-local Markdown installation, not a project dependency. If Markdown is already installed, omit that environment assignment. No LaTeX service, external font, or remote math renderer is required.

`render_reference_poses.py` is an optional provenance/recreation utility. It reads existing native mesh transforms from the research-data path recorded in `assets/pose_provenance.json`; diagram regeneration does not require rerunning it. No physics is simulated for these images.

[Validation](validation.json) records rendering checks. [Input identity](sources/input_manifest.json) and [source bindings](sources/source_bindings.json) identify the attachment and evidence. The original attachment and ongoing research artifacts were not modified.
