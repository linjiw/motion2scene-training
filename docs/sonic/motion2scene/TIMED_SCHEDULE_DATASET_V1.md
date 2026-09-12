# Portable repeated-schedule dataset interface

The current release is the [42-episode six-context archive](PORTABLE_SIX_CONTEXT_DATASET_V1.md):
12,516 aligned 114D packets, 50,064 physics steps, 18 available phase targets and
17 consequential decisions. It contains the original 35 branches plus the seven
complementary branches, with 28 passages, 14 physical failures and no missing
technical outcomes. The standalone NumPy toolkit is included. Earlier archives
remain unchanged.

Its bundled physical audit deliberately uses the preserved original scorer.
The later upstream-approach correction is separate, with audited compatibility
for all 42 captures. Use the bundled version when reproducing this release;
running a changed current exporter against old captures is not equivalent to
reproducing their recorded source version. The release note provides exact hashes,
the isolated exporter relocation receipt, and tested offline commands.

`scripts/research/motion2scene_export_timed_schedule_dataset.py` is a separate exporter for the new repeated-decision collections. It preserves the old106D exporter and archives. The actual single-branch smoke is exported at `m2s-timed-schedule-smoke-dataset-v2`: 298 aligned 114D packets, 1,192 physics steps, and zero fabricated teacher targets. Its standalone bundled-code audit passed from `/tmp`, recomputing beam forces and all hashes without pretrained references or weights. The first packaging attempt is retained with a bytecode-cache inventory erratum; no physics was rerun.

```sh
.venv_research/bin/python scripts/research/motion2scene_export_timed_schedule_dataset.py export \
  --out /absolute/new-immutable-dataset \
  --collections /absolute/collection/result.json \
  --archive
.venv_research/bin/python scripts/research/motion2scene_export_timed_schedule_dataset.py audit \
  --out /absolute/new-immutable-dataset
```

`--teachers /absolute/training/teachers.json` optionally binds the trainer's target groups. Every supplied matching group must equal an independent physical re-audit. A complete seven-schedule forced development collection produces one measured teacher target per registered phase. An incomplete collection, a single smoke branch, or an evaluation collection does not receive fabricated targets.

The package keeps complete physical trajectories as pickle-free NPZ and reconstructable metadata, causal sensor packets, exact `100+2K` student inputs, per-packet alignment, and actual preaction/postaction command records. For seven schedules the inputs are114D. A full original interface is retained separately under supervision, along with source trajectory/sensor/feature hashes. Student input files exclude geometry, outcome labels, selected postdecision actions, and policy values.

Each episode declares the actual beam count and counterpart order. Full all-body environment pair streams, native per-subject filter mappings, original net-force streams, and beam measurements remain at200Hz. Portable audit reconstructs each beam's worst substep force, checks contact sums and ordering, and recomputes complete passage/course and entry/return outcomes. An unreached second beam remains failure. Complete failed passages retain measurement admission when all clocks and source bindings pass.

Every phase target preserves the recorded neutral feature vector, actual legality, admitted/pass/time arrays, verified continuation option and branch IDs, and portable continuation episode IDs. WAIT is tied to the best measured future complete schedule from the matched recorded history. The portable audit rebuilds the physical/sensor history hashes and recomputes all continuation values using only bundled data. Different past actions or sensor packets invalidate that branch's matching history. Current postdecision action differences are excluded from the preaction comparison.

The archive also keeps source registrations and source-code snapshots, recorded controller/reference hashes, and separate failed-attempt records. Available aborted physical dictionaries are converted to pickle-free arrays; raw contact prefixes and sensor/interface packets remain. Known failed-attempt physics steps contribute to total acquisition cost, while unknown counts remain explicit. No pretrained checkpoint, ONNX model, raw motion-prior file, or whole reference bank is redistributed. Recorded reference commands in actual trajectories remain identified. Upstream asset licenses are not relicensed.

Portable audit verifies the archived measured execution and teaching table. It cannot newly qualify an online controller without the original motion assets and qualification evidence. No learned-policy generalization, hardware transfer, route selection, or repeated adaptation follows from this schema.
