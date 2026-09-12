# Recorded-input replay controls at M2 and M4

The seven predeclared replay controls still give identical selected schedules
within each corpus at M4. Every control selects a passing recorded branch in
five of the same six development contexts. Replay weights differ from uniform,
but these changes do not improve recorded-input selection at either checkpoint.
No new physical policy execution is reported here.

| Checkpoint | Corpus seed | All seven controls | Selected complete schedule on all six contexts | Eligible nonempty phase cues |
|---|---:|---:|---|---:|
| M2 | 93201 | 5/6 | `prior_splice_e050_r265` | 6/6 |
| M2 | 93202 | 5/6 | `prior_splice_e015_r265` | 6/6 |
| M2 | 93203 | 5/6 | `prior_splice_e050_r265` | 6/6 |
| M4 | 93201 | 5/6 | `prior_splice_e050_r265` | 12/12 |
| M4 | 93202 | 5/6 | `prior_splice_e015_r265` | 12/12 |
| M4 | 93203 | 5/6 | `prior_splice_e050_r265` | 12/12 |

The controls are uniform, coverage only, the implemented 0.8 uniform / 0.2
coverage fallback, historical observable-gap weighting, its ungated counterpart,
refreshed supervised error, and historical-gap times supervised-error weighting.
They use identical teacher data within each corpus, ridge penalty 10, the common
initializer and three declared refits. The completed native M4 control worker
produced 21 final policies using 66 CPU fits. The later verification below
performs no additional fitting.

Every method has zero selected-schedule changes and zero passing-branch difference
relative to uniform. Each time comparison retains the five mutually successful
recorded contexts, with mean difference 0 seconds. These are branch-table
readouts, not newly executed policy passage or time measurements. M2 and M4
reuse nested prefixes of the same three corpora and the same six development
contexts; the 252 verification choices are not independent trials.

Each corpus retains three positive historical phase gaps at both checkpoints.
The M4 prefixes provide 13, 14 and 14 supervised phase rows, respectively, out
of 15 recorded phase slots. All twelve nonempty M4 phase cues per corpus satisfy
the current visibility rule, and gated/ungated weights are identical at every
refit. Thus these data exercise replay priority but provide no evidence for a
benefit from the visibility exclusion. Cue eligibility does not establish that
observations resolve all action-relevant aliasing. Historical gaps remain bound
to their actual pre-update policies. Refreshed residuals remain fitting signals.

The [source-bound verification](evidence/research-progress-20260909/M2_M4_replay_controls/result.json)
checks all 42 saved M2/M4 policies against the same recorded teacher inputs,
reproduces their 252 selected-context readouts and recomputes all 126 weight
recipes from archived physical/residual signals. It verifies common development
teacher identity, corpus/checkpoint assignments, implementation hashes and saved
policy hashes. It adds zero fits and zero physics. The native control runs
audited their training prefixes before fitting; this summary does not repeat a
raw-trajectory audit or recompute the archived fitting residuals.

The [complete CSV](evidence/research-progress-20260909/M2_M4_replay_controls/controls.csv)
retains individual corpus, method and checkpoint results, schedule choices,
paired-time counts and weight differences. The concise table above is also
[generated from those results](evidence/research-progress-20260909/M2_M4_replay_controls/corpus_table.md).
Black and Ruff pass for the new verification script. The command below ran
successfully against the completed artifacts:

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_summarize_checkpoint_controls.py \
  --source /home/linjiw/research-data/groot-wbc/m2s-checkpoint-replay-learning-curves-20260909-v1 \
  --checkpoints 2 4 \
  --out /home/linjiw/research-data/groot-wbc/m2s-M2-M4-replay-control-readouts-20260910-v2
```

Use a fresh output directory for a subsequent invocation. The initial command
stopped before creating output because of a registry-loader keyword mismatch;
the corrected positional call and both completed summary versions are retained.

Replay remains an auxiliary controlled arm in the frozen acquisition comparison.
These results extend its recorded-input null finding through M4. Its independent
closed-loop benefit remains unproven; the original M8 comparison proceeds with
its declared arms, budget and queues unchanged.
