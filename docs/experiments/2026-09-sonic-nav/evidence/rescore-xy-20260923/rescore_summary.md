# XY re-score of the nav-8192 scene episodes

Generated 2026-09-23T17:16:19Z by `scripts/navigation_distill/rescore_xy.py` (sha256 `505cadfeeb39b210`, git `141a230`).
Packet `/home/robotixx/motion2scene-training/workspace/nav-8192`: 497 unique episodes (535 paths including symlinked aliases); 0 without a trace.

Protocol: every row is one recorded single-env episode (50 Hz control rows). Legacy = the recorded `task-result.json` flag (3-D pelvis goal, reference-derived deadline). XY = XY pelvis distance <= 0.25 m and speed <= 0.10 m/s for 50 consecutive rows, completed before any contact > 1 N or fall on the same recorded rollout. The recorded speed is the 3-D pelvis speed, an upper bound on planar speed, so XY is a lower bound; XY-FD uses planar speed from XY finite differences instead. The 'feasible' columns restrict to the 19 task ids of the DAgger panels (19 ids).

**Legacy reproduction:** the 3-D port reproduces all recorded score fields for 497/497 episodes.

## Evaluation panels (unassisted: teacher, full-command motor, navigation adapter)

| panel | seed | episodes | legacy 3-D | XY | XY-FD | XY whole-trace | gained / lost | legacy 3-D /19 | XY /19 | XY: held / reached not held / never reached / contact | median final XY err (m) | median min XY err (m) | median path ratio | repro match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| collect/teacher-91260 | 91260 | 24 | 19 | 19 | 15 | 19 | 0 / 0 | 19/19 | 19/19 | 19 / 1 / 1 / 3 | 0.14 | 0.12 | 1.18 | 24/24 |
| eval/dag-approach-c1-91260 | 91260 | 19 | 4 | 4 | 3 | 4 | 0 / 0 | 4/19 | 4/19 | 4 / 6 / 8 / 1 | 0.50 | 0.23 | 1.28 | 19/19 |
| eval/dag-approach-c2-91260 | 91260 | 19 | 7 | 7 | 3 | 7 | 0 / 0 | 7/19 | 7/19 | 7 / 2 / 8 / 2 | 0.46 | 0.33 | 1.15 | 19/19 |
| eval/dag-uniform-c1-91260 | 91260 | 19 | 3 | 3 | 1 | 3 | 0 / 0 | 3/19 | 3/19 | 3 / 6 / 7 / 3 | 0.50 | 0.24 | 1.24 | 19/19 |
| eval/dag-uniform-c2-91260 | 91260 | 19 | 2 | 2 | 1 | 2 | 0 / 0 | 2/19 | 2/19 | 2 / 9 / 7 / 1 | 0.56 | 0.21 | 1.22 | 19/19 |
| eval/full-s1_3200-91260 | 91260 | 24 | 19 | 19 | 16 | 19 | 0 / 0 | 19/19 | 19/19 | 19 / 1 / 1 / 3 | 0.16 | 0.10 | 1.18 | 24/24 |
| eval/nav-v0-91260 | 91260 | 24 | 0 | 0 | 0 | 0 | 0 / 0 | 0/19 | 0/19 | 0 / 5 / 17 / 2 | 1.01 | 0.52 | 1.08 | 24/24 |
| eval/nav-v1-91260 | 91260 | 24 | 1 | 1 | 0 | 1 | 0 / 0 | 1/19 | 1/19 | 1 / 7 / 12 / 4 | 0.95 | 0.42 | 1.30 | 24/24 |
| eval/nav-v2-recovery-91260 | 91260 | 24 | 4 | 4 | 3 | 4 | 0 / 0 | 4/19 | 4/19 | 4 / 5 / 11 / 4 | 0.86 | 0.38 | 1.14 | 24/24 |
| eval/nav-v2-recovery-91262 | 91262 | 24 | 1 | 1 | 1 | 1 | 0 / 0 | 1/19 | 1/19 | 1 / 2 / 18 / 3 | 1.01 | 0.94 | 0.92 | 24/24 |
| eval/nav-v2-replay-91260 | 91260 | 24 | 2 | 2 | 2 | 2 | 0 / 0 | 2/19 | 2/19 | 2 / 6 / 14 / 2 | 0.92 | 0.40 | 1.31 | 24/24 |
| eval/nav-v2-replay-91262 | 91262 | 3 | 0 | 0 | 0 | 0 | 0 / 0 | 0/3 | 0/3 | 0 / 1 / 2 / 0 | 0.91 | 0.28 | 1.33 | 3/3 |

## Collection panels (assisted: motor demonstrations, learner-prefix recoveries, re-entry probes)

Whole-episode flags mix a learner prefix with a motor or rewound suffix; they are not navigation results. The suffix columns are the receipts that admit DAgger rows.

| panel | aliases | episodes | legacy 3-D | XY | suffix: legacy receipt | suffix: 3-D repro match | suffix: XY | gained / lost | repro match |
|---|---|---|---|---|---|---|---|---|---|
| collect/motor-demo-91260 | – | 24 | 19 | 19 | – | – | – | 0 / 0 | 24/24 |
| collect/motor-demo-91261 | – | 3 | 3 | 3 | – | – | – | 0 / 0 | 3/3 |
| collect/recovery-v1-91260 | – | 76 | 43 | 43 | 43/76 | 76/76 | 43/76 | 0 / 0 | 76/76 |
| collect/reentry-v1-91260 | – | 33 | 6 | 6 | 6/33 | 33/33 | 6/33 | 0 / 0 | 33/33 |
| dagger/dag-approach/c1-recovery | dagger/dag-uniform/c1-recovery | 38 | 32 | 32 | 32/38 | 38/38 | 32/38 | 0 / 0 | 38/38 |
| dagger/dag-approach/c2-recovery | – | 38 | 34 | 34 | 34/38 | 38/38 | 34/38 | 0 / 0 | 38/38 |
| dagger/dag-uniform/c2-recovery | – | 38 | 34 | 34 | 34/38 | 38/38 | 34/38 | 0 / 0 | 38/38 |

## Episodes whose outcome changes (0)

| panel | episode | legacy 3-D | XY | legacy hold | XY hold row | final XY err | final 3-D err | final z - goal z | stop |
|---|---|---|---|---|---|---|---|---|---|

## Legacy reproduction mismatches (0)


## Suffix receipt reproduction mismatches (0)


## Speed-channel sensitivity

Rows within 0.25 m XY of the goal (all episodes, row 0 excluded): rows 32457, median_recorded_mps 0.1001, median_fd_planar_mps 0.0990, median_abs_diff_mps 0.0085, p90_abs_diff_mps 0.0288, correlation 0.9952, rows_below_threshold_recorded 0.4999, rows_below_threshold_fd 0.5042, row_threshold_agreement 0.9573. The 0.10 m/s hold threshold sits at the median in-tolerance pelvis speed, so a 50-row run is sensitive to the speed channel even when the channels agree closely.

Rows within 0.25 m XY but outside 0.25 m 3-D: 169 over all episodes.

## Columns (rescore.csv)

- `xy_hold_row`: 0-based trace row on which the first XY hold completes (`xy_hold_s` = (row + 1) x 0.02 s).
- `first_xy_reach_row` / `first_xy_reach_s`: first row within 0.25 m XY (time to goal).
- `path_xy_m`: XY pelvis path from the task start through every row; `path_ratio` = path / straight start-goal XY distance (reference-derived goal).
- `max_root_height_dev_m`: max |pelvis z - start z|; `final_z_minus_goal_z_m` explains most 3-D vs XY differences.
- `trace`: `present` for every row scored here; `missing` rows keep only the legacy flag.
