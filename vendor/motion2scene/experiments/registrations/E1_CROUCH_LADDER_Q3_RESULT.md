# E1 crouch-ladder Q3 result

Status: **completed; 0/2 edited motions accepted**.

The batch was launched only after commit `9276e62` froze its manifest and predictions. Both cells
completed 199 physics frames at 50 Hz, emitted the success marker and trajectory artifact, and had
exactly zero external contact force. Both nevertheless failed the unchanged Q3
reference-trackability policy on endpoint error.

| Cell | Expected | Observed | Endpoint error | Path p95 | Progress ratio | External force |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `d040` | accepted | rejected | 0.517 m | 0.206 m | 0.901 | 0 N |
| `d055` | no directional prediction | rejected | 0.607 m | 0.234 m | 0.885 | 0 N |

The frozen endpoint threshold is 0.350 m and the path-p95 threshold is 0.250 m. For comparison,
the same carrier's unedited nominal cell in E0-Q3 V8 was accepted with 0.226 m endpoint error,
0.146 m path p95, and 0.960 progress ratio.

## Prediction adjudication

- P1 (both complete, gradeable, zero external force): **pass**.
- P2 (`d040` accepted): **miss**.
- P3 (at least one edit accepted): **miss**.
- P4 (no promotion beyond Q3 and no retry): **pass**.
- Resource prediction: **pass**. Cells took 30.568 s and 29.360 s; total measured cost was
  0.016647 contended GPU-hours. Free memory at launch was 8,082 and 7,822 MiB.

## Interpretation

The deterministic local-crouch operator preserved the reference root path, but SONIC accumulated
terminal route lag as crouch amplitude increased. This is evidence against treating reference-side
kinematic qualification as execution qualification. The two edited motions remain Q0/Q1 only and
must not enter the main Motion2Scene hallucination bank.

A subsequent shallower-amplitude run may localize the controller's feasibility boundary, but it is
a post-result calibration experiment—not a retry or a rescue of these preregistered predictions.

## Frozen evidence

- Run record: `/home/linjiw/research-data/groot-wbc/cg-wbc-v1-pilot/e1_crouch_ladder/q3_v1/run_record.json`
- Run-record SHA-256: `0cf6632f5b381becc91434f303c346e05959810b6fd8d83b56297f817a3fdb00`
- Registered manifest SHA-256: `faee0fee0410dc40a21b2ddeebcaa545b764ec7bc07dcbd461568db0b1bd4924`
