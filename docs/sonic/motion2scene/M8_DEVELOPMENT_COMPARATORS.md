# M8 development comparator block

The strong script passes 6/6 assigned development contexts. The seven-schedule bank covers 6/6; the best fixed schedules cover 5/6. These are actual matched executions with seed 8732, not learned M8 policy results.

All 48 assignments completed: 32 passes, 16 failures and **56,780 measured physics steps**. Partial captures have recorded step counts [756]; observed contact establishes task failure in this block, while its unobserved later fall outcome remains unknown. No task outcome is unknown.

| Comparator | Passage | Contact observed | Fall threshold observed / unknown | Time minus script (s) | Mutually successful contexts |
|---|---:|---:|---:|---:|---:|
| script | 6/6 | 0 | 0 / 0 | +0.000 | 6 |
| neutral | 2/6 | 4 | 0 / 1 | +0.020 | 2 |
| prior_splice_e015_r265 | 5/6 | 1 | 0 / 0 | -0.204 | 5 |
| prior_splice_e050_r265 | 5/6 | 1 | 0 / 0 | -0.204 | 5 |
| short_e015_r265 | 3/6 | 3 | 0 / 0 | +0.273 | 3 |
| short_e070_r265 | 3/6 | 3 | 0 / 0 | +0.273 | 3 |
| sustained_e015_r255 | 4/6 | 1 | 0 / 0 | +0.115 | 4 |
| sustained_e070_r255 | 4/6 | 1 | 0 / 0 | +0.115 | 4 |

The early prior-splice schedule's time minus script is -0.204 s on 5 mutually successful contexts. Its failures remain in the passage denominator. This is a passage/time comparison on development data. It does not establish which learner or constructor is better.

| Development context | Measured passing bank schedules | Script choice | Script passage time (s) |
|---|---|---|---:|
| complementary_late_development_0100 | sustained_e015_r255, sustained_e070_r255 | sustained_e070_r255 | 4.54 |
| early_prior_development_0284 | prior_splice_e015_r265, prior_splice_e050_r265 | prior_splice_e050_r265 | 2.72 |
| early_prior_two_beam_development_v1 | neutral, prior_splice_e015_r265, prior_splice_e050_r265, short_e015_r265, short_e070_r265, sustained_e015_r255, sustained_e070_r255 | prior_splice_e050_r265 | 4.12 |
| long_neutral_empty | neutral, prior_splice_e015_r265, prior_splice_e050_r265, short_e015_r265, short_e070_r265 | neutral | 5.40 |
| timed_development_0187 | prior_splice_e015_r265, prior_splice_e050_r265, short_e015_r265, short_e070_r265, sustained_e015_r255, sustained_e070_r255 | sustained_e070_r255 | 4.10 |
| timed_development_0247 | prior_splice_e015_r265, prior_splice_e050_r265, sustained_e015_r255, sustained_e070_r255 | sustained_e070_r255 | 4.68 |

native per-body external counterpart normal force above 1 N; foot/floor support allowed; self contacts retained separately; partial observed contact establishes failure. recorded root height below 0.5 m or upright cosine below 0.5; absence requires complete admitted first-episode measurement.

passage_time_s is all-body-origin crossing plus 0.30 s stabilization; paired only on mutually successful identical conditions. Whole episode capture duration is separate and is not an adaptation cost or independently measured recovery latency.

Complete admitted captures last 5.96 s. That recording duration is separate from crossing plus stabilization and is not a measured energy or recovery cost. The six layouts are reused across comparators; these 48 episodes are not 48 independent layouts. No population confidence interval or superiority claim is made.

The first reader conservatively marked 39 complete single-beam fall outcomes unknown because it looked only for the course-specific whole-episode field. The corrected reader uses the existing single-beam field. Both outputs and the original reader bytes are preserved; passage, time, contact, capability and raw measurements are unchanged. Acquisition separately stopped at a preflight guard because an unused offline selector was added inside the frozen editable package. Moving that helper outside the runtime package restored the exact original identity; the next teacher had no attempt before this recovery. No physical outcome was replaced.

Source: `/home/linjiw/research-data/groot-wbc/m2s-M8-development-comparators-summary-20260910-v2/result.json`; SHA256 `41891e2533d1da581fd99593dac0c2b6da095606d77886072f6c02e712dce795`.

Reproduce this table with `scripts/research/motion2scene_render_development_comparators.py --result <result.json> --out <new-report.md>`. The readout and renderer add no physics.
