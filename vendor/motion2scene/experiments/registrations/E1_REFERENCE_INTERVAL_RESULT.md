# E1 reference critical-interval diagnostic

Status: **1/2 margin-robust reference intervals; 0/2 dataset-eligible**.

This post-Q3 diagnostic applies the analytic Motion2Scene-LfH solver to the deterministic
index-000 crouch ladder. It uses the conservative maximum full-body capsule height over the same
33-frame station window and reserves 10 mm each for target clearance and weaker-motion strike.

| Weaker → target | Raw gap | 10 mm/side interval | Q3 target | Dataset eligible |
| --- | ---: | ---: | --- | --- |
| nominal → `d040` | 56.425 mm | 36.425 mm | rejected | no |
| `d040` → `d055` | 19.394 mm | empty by 0.606 mm | rejected | no |

For the nonempty diagnostic interval, 20 beam underside heights were sampled by drawing one
uniform point from each equal-probability stratum over 1.252300–1.288725 m. This is the critical
distribution: it covers the target-feasible/weaker-colliding band. The same artifact independently
samples thickness (0.04–0.16 m), routewise span (0.20–0.50 m), cross-route span (1.00–2.00 m), yaw
jitter (−0.08–0.08 rad), and one of three materials. Changing these nuisance ranges cannot move the
critical-height stream.

The result validates the analytic solver and shows that the first pair is geometrically
identifiable. It does **not** validate Motion2Scene training data because the target motion was not
execution-qualified. The sampler records these 20 values as diagnostics but reports zero eligible
scene–motion pairs.

Frozen artifact:
`/home/linjiw/research-data/groot-wbc/cg-wbc-v1-pilot/e1_crouch_ladder/analytic_reference_v1.json`

SHA-256: `8f4b391f30d988282f2f3a2fad6e5c08e2baa3b4e958b47b5736cd6ee74f0e9b`
