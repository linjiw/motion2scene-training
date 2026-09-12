# Static evaluation figures

`scripts/research/motion2scene_plot_evaluation.py` produces PDF and 250 dpi PNG
figures from a `motion2scene_reserved_statistics_v1` result. It uses only NumPy,
Matplotlib and the adjacent standalone statistics module. It does not import the
robot package or open reference motions, controller checkpoints or model files.

```bash
.venv_isaaclab/bin/python scripts/research/motion2scene_plot_evaluation.py \
  --input /path/to/statistics/result.json \
  --expected-sha256 STATISTICS_SHA256 \
  --out /path/to/separate-figure-directory \
  --data-kind reported
```

The output must be a fresh directory outside the statistics input directory.
For synthetic tests use `--data-kind synthetic`; every figure then visibly says
**SYNTHETIC — NOT PAPER EVIDENCE**. An input path identifying synthetic data
cannot be rendered as `reported`. The explicit flag declares data provenance;
this plotting tool does not independently certify physical outcomes.

## Figures and interpretation

- `learning_curves.pdf/.png`: three individual acquisition-corpus panels and
  one aggregate panel, each with the four training arms at M2 and M4. Individual
  horizontal coordinates use the corpus's actual acquisition steps. Aggregate
  coordinates and completion are equal-weight means of the three corpus points;
  actual budgets can differ. Assigned maxima remain 27,416 and 46,488 steps.
- Fixed baseline completion is drawn horizontally. Each baseline contains only
  **36 unique episodes**, visually repeated in the panels, with no corpus
  replication or baseline confidence interval. Each individual learned point
  has 36 assigned episodes; each aggregate has 108 assignments across three
  independently acquired corpora.
- `paired_completion.pdf/.png`: all 34 declared left-minus-right completion
  contrasts. Positive differences favor the left policy. Only supplied crossed
  95% percentile intervals are displayed, with the declared 20,000 draws and
  seed 202609081822. Layout-only sensitivity intervals are not substituted.
  The right column gives known/assigned matched-pair counts. Baseline contrasts
  retain only 36 unique right-hand episodes, despite 108 matched pairings.

Unknown outcomes retain lower/upper completion bounds. They have no midpoint
estimate or confidence interval. Learning curves use triangle endpoints and
dashed edges; paired plots use arrow endpoints and an explicit `bounds` suffix.
Only complete comparison outcomes may have crossed intervals. No successful
time, cost, failure-time imputation or mechanical-energy quantity is plotted.
The figures retain the caveat that only three acquired corpora are available,
M2/M4 share acquisition trajectories, and intervals are descriptive without a
multiplicity correction.

## Validation and provenance

The CLI validates all 972 retained assignments and checks displayed curve means,
budgets, denominators, comparison identities and unknown bounds against them.
A one-draw reconstruction through the standalone statistics helper checks
descriptive quantities only; its intervals are discarded. Every displayed
interval is copied unchanged from the hashed statistics input.

`registration.json` records exact statistics, plot-source and validation-source
hashes, the original statistics registration reference, NumPy/Matplotlib
versions and bootstrap metadata. `result.json` records every displayed point,
caption, denominator and figure hash. PDF/PNG metadata carry full provenance;
the figure footer gives abbreviated statistics and plot-source hashes.

Focused tests cover actual-budget differences, unknown bounds, baseline unique
counts, inconsistent summaries, hashes, synthetic labeling, and both export
formats. Validation examples use only the synthetic statistics archive; they
provide no reserved evaluation or paper performance evidence.
