# Complementary nominal development capability

The complementary passage is a physically supported counterexample to both
available always-prior schedules: both fail torso–beam contact, while either
sustained schedule passes in 4.20 seconds. All seven executions have complete
measurements, 298 aligned control/sensor rows, and 1,192 physics steps. No branch
falls or resets; all scheduled returns complete. These are nominal development
results at seed 8731, separate from the original five-context selection.

| Complete schedule | Passage | Peak beam normal force (N) | First force above 1 N (s) |
| --- | --- | ---: | ---: |
| Neutral | Fail | 1074.281 | 2.910 |
| Short, entry 0.30 s | Fail | 166.619 | 2.925 |
| Sustained, entry 0.30 s | Pass, 4.20 s | 0 | None |
| Prior derivative, entry 0.30 s | Fail | 514.955 | 2.835 |
| Prior derivative, entry 1.00 s | Fail | 400.533 | 2.830 |
| Short, entry 1.40 s | Fail | 166.619 | 2.925 |
| Sustained, entry 1.40 s | Pass, 4.20 s | 0 | None |

Forces are the maximum measured body-to-beam normal-force magnitude from the
explicit per-body 200 Hz counterpart stream. Each failed branch's undesired
environment contact is at the torso. Internal contacts and permitted foot–floor
support remain separate. Contact failures are complete physical measurements;
they are not dropped as instrumentation failures.

## Forecast and acquisition audit

The registered development search selected beam `0100`: 0.60 m long, 1.20 m wide,
underside 1.25 m, at route fraction 0.45. The selected sustained option retains
at least 13.201 mm of outer primitive clearance over the finite 81-offset set.
Both prior schedules have negative nominal native-inner clearance: −12.460 and
−14.554 mm. This independently audited selection followed its frozen shortlist
and ranking rules across 2,755 proposals.

The audit recomputed all 14 selected nominal values and the 81 positive-offset
values using every capsule, bypassing the proposal's AABB pruning and clearance
cap. All 95 results agree exactly. The proposal itself recorded 40,514 clearance
queries and 17.653 seconds of search computation; a prior independent nominal
witness added 14 queries. The current re-audit adds 95 queries and zero physics
steps. Only the nominal selected scene has physical evidence. Geometric offset
clearance is not a physical robustness result, and these failures occur during
traversal before the scheduled recovery windows.

## Observable decisions and a verified WAIT continuation

All 14 eligible branch-prefix comparisons match exactly in recorded state,
actions, reference tokens, and causal sensor history. All three legal decision
tables are complete:

| Neutral decision phase | Successful immediate action or continuation | Failing legal alternatives |
| --- | --- | --- |
| 0.30 s | Sustained now; or WAIT toward sustained at 1.40 s | Short now, prior now |
| 1.00 s | WAIT toward sustained at 1.40 s | Prior now |
| 1.40 s | Sustained now | Continue neutral, short now |

The sequence WAIT → WAIT → sustained is backed by the actual late sustained
branch, which shares the required neutral prefixes and passes in 4.20 seconds.
This is an available complete physical solution, not an executed learned policy.

The actual beam surface appears in the first sensor packet; a downward underside
normal appears at capture elapsed 0.04 seconds. The registered command phases
0.30/1.00/1.40 seconds consume packets captured at 0.28/0.98/1.38 seconds in the
sensor's elapsed clock. At the first decision, the current packet has six beam
surface hits but no underside hits; retained earlier observations still provide
ceiling features. The later packets have three and one underside hits.

Replaying all first 70 packets from ideal rays and robot poses reproduces every
114D feature exactly. The 28 sensor-derived corridor fields differ from every
original development context at every legal phase. These differences include
observed ceiling coverage and height, rather than only proprioceptive differences.
For example, at the final phase the complementary scene's 1.5–2.5 m corridor has
ceiling coverage 0.64 versus 0.12 for the original short beam, and a minimum
observed ceiling 15 mm lower relative to the same approach root.

There is an important representation limit: no cell with jointly observed floor
and ceiling falls inside the student's ±0.5 m lateral corridor at any decision
phase. The full map has one such cell at phase 1.00 s, with a 1.25 m gap, but its
lateral coordinate is 0.60 m and it is outside that student corridor. The 114D
summary combines separately observed floor and ceiling samples. Thus this is a
distinguishable finite example with useful timely sensor cues; it is not proof
of dense vertical free-space sensing, general perceptual sufficiency, or noise
robustness.

The previously selected script still chooses WAIT then prior at 1.00 s on these
packets, yielding a failed recorded-branch proxy. Timely information being
available does not mean the existing script or learned policy uses it correctly.
No original model, baseline setting, tuning panel, or 20-episode validation plan
was changed by this audit.

Across the original five contexts plus this separately acquired witness, the
available option oracle covers 6/6 contexts and the best constant covers 5/6.
This posthoc capability accounting explains why adaptation selection can matter;
it does not establish an achieved learned or curriculum improvement.

## Reproduction and evidence

The report is
`/home/linjiw/research-data/groot-wbc/m2s-complementary-capability-audit-v1/result.json`
(SHA-256 `6e986348ebde56bd7174c100fd3c398b634a1c498734f52206f92840b7fed02b`).
`completed.json` binds the sensor-map scope correction and standalone
`complementary_capability.pdf`/`.png` figure. The figure's source is retained.

```bash
.venv_research/bin/python scripts/research/motion2scene_audit_complementary_capability.py \
  --collection /home/linjiw/research-data/groot-wbc/m2s-complementary-schedule-teachers-v1/collection_000/result.json \
  --screen /home/linjiw/research-data/groot-wbc/m2s-complementary-late-screen-v1/result.json \
  --original-baseline /home/linjiw/research-data/groot-wbc/m2s-schedule-baseline-development-selection-v1/result.json \
  --out /absolute/new-audit-folder
```

Three focused audit tests pass, covering uncapped geometry, per-subject counterpart
permutations, and the distinction between sensor and proprioceptive differences.
The new audit module and tests pass Black and Ruff. Original physical artifacts
and production sources remain unchanged.
