# Research paper draft

**[Read the compiled PDF](main.pdf)** · [Editable LaTeX](main.tex)

Working title: **Motion2Scene: Qualifying Motion-Derived Scenes for Humanoid Teacher–Student Learning**.

This is a five-page exploratory systems/failure-analysis draft, not a claim of successful autonomous navigation. Author order, affiliations, target venue and final experiment selection remain to be decided. It is not submitted or peer reviewed.

The draft covers inverse scene construction, geometric versus physical qualification, SONIC teacher adaptation, masked BFM-inspired distillation, DAgger/temporal coverage, scene/goal conditioning and proposed bounded residual learning. Numerical results come from the archived experiment packets, not from speculative future experiments.

Build with Tectonic:

```bash
tectonic paper/main.tex
```

The PDF compiled successfully. Bibliographic links were checked against primary arXiv/PMLR sources. The comparison describes the original BFM method; it does not claim a comprehensive survey of all subsequent BFM work. Before submission, update the literature review, finalize authorship, and complete the discriminating experiments in the [research guide](../docs/RESEARCH_GUIDE.md).

## Claim-to-evidence map

All paths below are relative to the unpacked workspace unless prefixed by `docs/`.

| Draft result | Evidence |
|---|---|
| Hindsight geometric study: contrast/clear/penetration mass | `m2s-hindsight-events-100-20260911/REPORT.md`, `summary.json`, `outcomes.csv` |
| 12,000-update foundation study, 1/100 full and 0/100 sparse | `m2s-bfm-100motions-20260912/REPORT.md`, final native evaluation metrics |
| Additional assistance did not improve standalone completion | `m2s-bfm-assisted-20260912/REPORT.md` and final metrics |
| Curriculum is not superior in the two-clip CPU diagnostic | `m2s-bfm-curriculum-smoke-20260912/results.json`, `split.json`, `command-sensitivity.json` |
| Two scene probes admit zero qualified demonstrations | `m2s-bfm-scene-probes-v2-20260912/*/metrics/scene-qualification-corrected.json`; use corrected receipts |
| Repaired checkpoint progression through step 6,200 | `m2s-repaired-teacher-step6200-comparison-20260912/comparison.json` and bound earlier evaluations |
| Packaging and setup evidence | `docs/READINESS_AUDIT.md`, `docs/validation-evidence/` |

The teacher training may advance after these archived measurements. Refresh the table only after another declared evaluation; live optimizer iteration count is not a new performance measurement.
