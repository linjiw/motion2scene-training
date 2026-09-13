# navigation-motor-20260913 evidence

Compact recorded experiment evidence supporting the research reports. Large checkpoints, repaired motions, collision assets and raw physics traces remain in `/home/linjiw/research-data/m2s-nav-motor-bridge-20260913` and its bound parent packets. Absolute paths in JSON describe that local asset layout; this directory is not an asset installer.

`artifact-sha256.json` hashes the committed files. CSV line endings are normalized to LF without changing numeric data. A plot is accompanied by its supporting CSV. Recorded stage commands must use fresh output directories when replayed.

See [the navigation pilot](../../NAVIGATION_MOTOR_PILOT_20260913.md). `results.json` records the goal/map branch; `matched-control-results.json` records the same tasks supplied with full current commands. `validation.json` contains the 95-test and formatting/lint results. Neither a low training loss nor entry into the goal region is reported as successful stopping.
