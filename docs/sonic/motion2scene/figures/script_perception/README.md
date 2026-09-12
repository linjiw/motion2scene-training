# Recorded perception-to-decision examples

![Recorded script inputs and neutral execution](perception_to_decision.png)

These are the two previously inspected development contexts requiring disjoint bank responses. All six script traces are retained in the source, and all 16 recorded neutral-phase readouts match the frozen script. The figure uses recorded features and robot states, not generated robot imagery.

| Context | Phase (s) | Delivered packet capture (s) | Decision | Hazard bands |
|---|---:|---:|---|---|
| early_prior_development_0284 | 0.30 | 0.28 | WAIT | 0.75–1.5 |
| early_prior_development_0284 | 1.00 | 0.98 | COMMIT prior_splice_e050_r265 | 0.75–1.5 |
| complementary_late_development_0100 | 0.30 | 0.28 | WAIT | 1.5–2.5 |
| complementary_late_development_0100 | 1.00 | 0.98 | WAIT | 1.5–2.5 |
| complementary_late_development_0100 | 1.40 | 1.38 | COMMIT sustained_e070_r255 | 1.5–2.5 |

Both examples already have different recorded upper-hit bands at the first decision. The plots therefore do not demonstrate an additional-information advantage for WAIT. WAIT preserves a later supported entry while neutral motion continues; final-phase neutral selection has no later entry and is labeled CONTINUE_NEUTRAL in the complete extraction.

Upper-hit zero means no recorded upper hit, not certified free space. Unknown fractions remain explicit. No postcommit features are plotted. States and features were captured before their registered command phase; the separate packet capture timestamps are in the table. Blank time after commitment contains no plotted measurement.

recorded script examples, not learned-policy or reserved evidence. Upper-hit absence does not establish free space; unknown fractions are retained. Surface visibility alone does not prove alias resolution. These traces do not isolate additional information from availability of later schedules; the same-repertoire one-time control remains separate.

Source: `/home/linjiw/research-data/groot-wbc/m2s-M8-script-perception-traces-20260910-v2/result.json`, SHA256 `c962d6c2a42bf41dff2f8fbf77b4a80889678094fbb76f0a5b3d8e74fe0595b8`.

Reproduce with `scripts/research/motion2scene_plot_script_perception.py --result <result.json> --out <new-directory>`.
