# E0/E1 shared-seed v2 result

This result is bound by
`/home/linjiw/research-data/groot-wbc/cg-wbc-v2-shared-seed-confirmatory/CORPUS_MANIFEST_V2_Q3.json`
(SHA-256 `0ad479dad3677c37f07e2dc25b5b89887e10c8fbee08305dbb78a33e62b3d429`).

## Confirmatory acquisition

- Generation: 144/144 references, zero failed, over eight independent shared seeds.
- Q0 joint reachability: 132 pass, 12 fail.
- Q1 severe self-intersection: 144 pass, zero fail.
- Routes: 34 valid straight, 16 valid gentle-left, 16 valid gentle-right, 65 over-turn,
  11 looping/reversal, and 2 insufficient-progress.
- Duck paired peak reduction: median 0.242 m; S0=3, S1=16, S2=4, S3=1.
- Arm-tuck paired peak reduction: median 0.060 m; S0=12, S1=8, S2=4, S3=0.
- Shoulder-turn, step-over and carry-walk remain `not_measured` under the complete
  preregistered predicate. No partial proxy is promoted.

The only prompt-generated duck reaching S3 fails Q0, so zero confirmatory references are
eligible for Q3 under the frozen joint-and-semantic rule.

## Controlled same-carrier ladders

Eight valid straight neutral carriers were transformed at 40, 55, 70 and 85 mm registered
target drops. All 40 derived rungs pass Q0/Q1 and S3 at reference level. Seven of eight groups
are strictly ordered; the eighth saturates at its two deepest levels. Ten of 32 adjacent
reference intervals remain nonempty after 10 mm clearance and 10 mm strike margins. These are
diagnostic intervals only.

## Obstacle-absent Q3

The 24-cell manifest fixes one physics seed and three levels per carrier. It completed all
cells using 0.174 contended GPU-hours:

| level | accepted | rejected |
|---|---:|---:|
| neutral | 8/8 | 0/8 |
| 40 mm | 3/8 | 5/8 |
| 55 mm | 1/8 | 7/8 |

Registered predictions adjudicate as follows:

1. At least 5/8 neutrals survive: **confirmed** (8/8).
2. At least 3/8 groups have neutral and 40 mm survival with a measurable retained crouch:
   **falsified** (2/8 have an achieved localized 40 mm event; 0/8 reaches S4).
3. The 55 mm pass rate does not exceed the 40 mm rate: **confirmed** (1/8 versus 3/8).
4. No ladder is admitted without three retained levels: **confirmed** (zero admitted).

The achieved route instrument is now a visible issue: 22/24 trajectories classify as
`over_turn`, including 7/8 accepted neutral controls, although neutral net/path ratios are
0.992--0.997 and net heading changes are small. The failures come from cumulative curvature
of 1.06--1.40 rad versus the frozen 0.96-rad threshold. The v2 result remains frozen; a new
neutral-control calibration must precede any revised route predicate.

## Gate decision

- Q4 candidates: 0.
- Dataset-eligible critical intervals: 0.
- Obstacle-present preference reversal: not authorized.
- Learned hallucinator: not authorized.

The next task is to calibrate route retention on accepted neutral executions and construct
controller-retained same-carrier ladders. It is not to lower thresholds retroactively or train
LfLH on the reference-only intervals.
