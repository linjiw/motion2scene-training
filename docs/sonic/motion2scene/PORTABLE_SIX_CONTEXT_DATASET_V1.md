# Portable six-context development dataset

The [current archive](/home/linjiw/research-data/groot-wbc/m2s-timed-schedule-six-context-dataset-v1.tar.gz)
packages the original five-context 35-branch acquisition and the complementary
seven-branch acquisition, with the exact six-context teacher targets. It creates
no new physics and preserves the earlier 35-episode release unchanged.

| Content | Count |
| --- | ---: |
| Complete measured teacher episodes | 42 |
| Passed / physically failed episodes | 28 / 14 |
| Aligned 114D sensor/state packets | 12,516 |
| Recorded physics steps | 50,064 |
| Available phase targets | 18 |
| Consequential learning targets | 17 |
| Supervised targets at ticks 15 / 50 / 70 | 6 / 6 / 5 |
| Files verified by the standalone baseline | 1,806 |

These are six inspected development contexts at physics seed 8731, using one
neutral source ancestry and seven qualified complete schedules. They are not an
independent acquisition comparison, a held-out benchmark or a navigation dataset.
The early context's known all-fail final phase remains present without an invented
successful target. Failed physical passages retain complete measurements; there
are zero technical failed attempts in this release.

The archive separates sensor inputs from privileged supervision. It retains
65-ray measurements, two-second history with observed/unknown masks, exact
preaction state and legality, complete physical trajectories, command timing,
every beam's 200 Hz counterpart forces, and verified WAIT continuation IDs.
Raw pretrained weights and complete prior/reference banks are excluded. The
included recorded reference commands remain identified as execution data.

## Reconstruct the baseline

The unchanged NumPy toolkit is included under `tools/portable_baseline`.
From the extracted dataset directory:

```bash
python -m pip install -r tools/portable_baseline/requirements.txt
python tools/portable_baseline/scripts/research/motion2scene_schedule_dataset_baseline.py inspect \
  --dataset . --out /absolute/path/to/new-inspection
python tools/portable_baseline/scripts/research/motion2scene_schedule_dataset_baseline.py fit \
  --dataset . --l2 10.0 --out /absolute/path/to/new-fit
```

Use new output directories. An independent `python -I` run blocked access to the
original repository and other experiment directories, plus Torch, Isaac, Omni,
USD and Joblib imports. It recomputed the 18 actual WAIT/phase tables and fitted
the 17 consequential rows with the unchanged ridge penalty 10.0. All 18 teacher
packet decisions and 84 actual neutral-packet decisions match the original
six-context fitted model. The latter include repeated matched prefixes, not
84 different contexts. Only the weights differ numerically, by at most
4.38854e-17; maximum legal value difference is 1.249e-16. The NPZ files are not
byte-identical. This verifies offline reconstruction, not additional traversal
performance or qualification of absent motion assets.

## Compact archive alternative

The optional [hardlink archive](/home/linjiw/research-data/groot-wbc/m2s-timed-schedule-six-context-dataset-v1-hardlinks.tar.gz)
contains the same 42 episodes and all 2,049 original members. Its 1,808 file paths,
dataset manifest, teacher targets, recorded data and bundled source bytes are
unchanged. Identical files share storage through 1,220 tar hardlinks; their
separate episode paths and provenance identities remain present.

| Storage | Original | Compact | Reduction |
| --- | ---: | ---: | ---: |
| Gzip archive, bytes | 691,924,598 | 223,804,884 | 67.65% |
| Allocated extraction, bytes on this filesystem | 8,224,817,152 | 2,665,664,512 | 67.59% |

The logical file content remains 8,219,911,290 bytes. The compact archive SHA-256
is `1a9c7fb97ae56a33077d322a9bd99a0d6e7f1252fe12af18a81759e4073c8fb6`.
GNU tar 1.34 extraction was verified in a new directory:

```bash
mkdir /absolute/path/to/new-compact-extraction
tar -xzf /absolute/path/to/m2s-timed-schedule-six-context-dataset-v1-hardlinks.tar.gz \
  -C /absolute/path/to/new-compact-extraction
cd /absolute/path/to/new-compact-extraction/m2s-timed-schedule-six-context-dataset-v1
```

The inner directory name and NumPy commands above are unchanged. The original
bundled native auditor reproduced its exact recorded audit, and the bundled
NumPy inspector verified all inventory hashes and 18 teacher tables. These checks
used a separate NumPy-only environment and declared Python-level path/import
guards; they are trusted-reader isolation checks, not an operating-system sandbox.
No model was fitted and no physics was run for this conversion.

Tar filesystem metadata is normalized to time zero, uid/gid zero, file mode
0644 and directory mode 0755; JSON metadata stays byte-identical. Treat the
extracted release as immutable: changing one hardlinked file in place changes
its aliases. Other extractors or filesystems may duplicate the bytes and use
more disk space. The original archive and expanded dataset remain available.
The [conversion receipt](/home/linjiw/research-data/groot-wbc/m2s-deterministic-hardlink-release-v1/result.json)
binds the deterministic repacker, tests, independent member comparison, extraction
measurements and both reader validations.

## Optional current-repository initializer

For the later adopted rule, the **current repository CLI** additionally accepts
`--allow-measured-tie-initialization` on `fit`; it defaults to off and records the
choice in the fit registration. The immutable toolkit bundled in this 42-episode
archive predates that option and keeps its original command above. Use the
current repository driver together with its current pure learning modules, and
a new output directory, to request the amended fit:

```bash
python scripts/research/motion2scene_schedule_dataset_baseline.py fit \
  --dataset /absolute/path/to/extracted-dataset \
  --l2 10.0 --allow-measured-tie-initialization \
  --out /absolute/path/to/new-amended-fit
```

Run that command from the current repository root. The fallback requires a
phase with no consequential support and a complete matched, all-success table
of exactly equal costs; missing or all-failed tables cannot initialize a phase. On this
42-episode release, the [flag-enabled validation](/home/linjiw/research-data/groot-wbc/m2s-portable-measured-tie-interface-v1/six_context_fit/result.json)
still fits 17 consequential targets and initializes zero phases. It verifies the
CLI on existing data, not new traversal performance. The original archive,
physical labels, scorer snapshots and default reconstruction remain unchanged.

## Exact scoring version

The default bundled physical auditor preserves the original pre-upstream-fix
scorer. Its three source SHA-256 values are:

| Source | SHA-256 |
| --- | --- |
| `motion2scene_passage.py` | `0e785d6263e66e12059f734d41a64331fb519cfd14832fe884125e2856b04a06` |
| `motion2scene_course.py` | `2c9d9a32dba18a009c5d7ac9105fa19b08046ec262d2606ccc2f75efebb94c51` |
| `motion2scene_timed_outcome.py` | `fde7f10d8c020fcfd01a6b825aeba2bda03c8b7be3aa53bf8ec433664c904c2b` |

`CRITERION.json` and `provenance/later_upstream_correction` separately bind the
subsequent upstream-approach correction and its recorded-data audit. All 42
included starts satisfy that later check, without changing passage labels or
classifier outputs. The new code is not attributed to the old executions or
substituted into their default portable auditor.

The exporter was relocated into a new copy of the archived 35-release source
closure. Only the exporter mapping from original absolute repository paths to
portable provenance paths changed; the other 68 source files remain identical.
The relocation patch and source-origin receipts are included. The first export
used a guard that forbids reading production Python source, and the final bundled
native audit ran from `/tmp` with private assets and simulator/model runtime
imports forbidden. Both physical and teacher audits passed.

The crossing criterion uses recorded body origins, not a certificate that every
native collider extent crossed the finish plane. Environmental contact, stability
and complete guarded return remain required. Sensor normals are ideal PhysX
queries, and floor/ceiling corridor summaries can combine different cells.
Mechanical work remains unavailable from the implicit actuator's effort estimate.
These limitations remain explicit in the archive's data card.

## Release identity

| Artifact | SHA-256 |
| --- | --- |
| Dataset manifest | `952821b42d73788d868fae10aa7acf7983337329393d7d79819aeb3b67010b70` |
| Gzip archive, 691,924,598 bytes | `8a2b8bcd2a3cd351a1ceb1837ab3318e494af3d3bd6a68238f023d64b2164e25` |
| Final release receipt | `fde2561ef8852fe0b8710687147efd9eb29cb56bf517eb9563c3172f7597c3d2` |
| Guarded bundled native audit | `f58e6287d3619494c12fa3609385603f7c26ae767d8200c9d5814ff3f58fd7fa` |
| Standalone NumPy reconstruction | `2e8d17ad98db0bf4a8fc0eca6cdd8fa836c7f4b8adf0a3a10efcaaef7804b3c6` |

The [final release receipt](/home/linjiw/research-data/groot-wbc/m2s-timed-schedule-six-context-dataset-export-v1/result.json)
links every input, source-isolation record, native audit and model comparison.
The [original 35-episode archive](/home/linjiw/research-data/groot-wbc/m2s-timed-schedule-five-context-dataset-v1.tar.gz)
was rehashed and remains unchanged at SHA-256
`3f171b25b3e0bbd6958fa52fe5cf4f76bdf7293f1ec71c717942c8dd7897e80e`.
