# One projected and spliced prior candidate

This CPU experiment produced one **authored derivative** of the [rejected, softly conditioned Kimodo sample](CONDITIONED_KIMODO_CANDIDATE_V1.md). It fixes the measured hinge-box violations and matches the qualified neutral reference at the proposed entry and return. Controller execution remains unqualified. The uncorrected neural sample and its rejection are preserved.

The [registration](/home/linjiw/research-data/groot-wbc/m2s-prior-projected-splice-development-v1/registration.json), SHA256 `0198e6c64a2b228f7948d67ca5e31b7e26226ee43ff19faeefe96ca9e126ef4e`, preceded construction. It binds both parent CSVs, the full raw skeleton output, native MJCF, a complete local source snapshot, CPU environment, and one permitted construction. No model inference, physics, sensor acquisition, or reserved-evaluation query occurred.

The explicit operator clamps only violating hinge samples to native joint boxes. Each parent uses its own native horizontal-origin canonicalization. The derivative preserves the neutral source through 1.0 s and from 5.0 s onward; quintic weights blend into the projected prior over 1.0–1.4 s and back over 4.5–5.0 s. XYZ and hinges use affine interpolation; root orientation uses shortest-arc quaternion SLERP. Source rate, frame count, root height, and core trajectory are otherwise retained. No floor-height adjustment is applied.

Measured [result](/home/linjiw/research-data/groot-wbc/m2s-prior-projected-splice-development-v1/result.json):

| Audit | Result |
|---|---|
| Hinge projection | 11 joint samples across 8 of 180 source frames; maximum correction 0.0475143 rad |
| Native limits | Source 30 Hz and loaded 50 Hz values satisfy the joint boxes within 1e−6 rad |
| Splice changes | 94 frames differ from raw canonical prior; 86 differ from projected prior |
| Source prefix/tail | Exactly equal to neutral at all registered preserved samples |
| Entry 0.30 s / return 5.30 s | Joint, pelvis-position, and joint-velocity differences all zero in the native reference |
| Native prefix boundary | Exact through 0.98 s; at 1.00 s one joint differs by 0.000947 rad; tail exact |
| Low interval 2.0–3.5 s | Pelvis drop relative to neutral: minimum 20.7 mm, median 54.3 mm, maximum 95.7 mm |
| Native sole-sphere minimum height | Neutral −32.02 mm; raw prior −29.31 mm; derivative −29.65 mm |

The [boundary diagnostic](/home/linjiw/research-data/groot-wbc/m2s-prior-projected-splice-development-v1/diagnostics.json) reproduces the installed native loader’s small-angle quaternion fallback: it averages the adjacent quaternions independently of interpolation weight. That incorporates the first blended source frame at the 1.00 s boundary. The source prefix is exact, and both requested joins remain exact. The frozen loader and resulting candidate were retained unchanged.

The floor diagnostic transforms the eight native MJCF sole contact spheres using native FK and subtracts their radii from world height. It flags 284 of 299 derivative frames, 284 raw-prior frames, and all 299 neutral frames. Because it also flags the physically qualified neutral, this predicate does not establish a physical failure or uniquely disqualify the derivative. It documents a reference/geometry limitation requiring separate controller and contact evaluation. It covers those spheres, not every robot mesh or cooked PhysX collider.

Quintic blending does not establish dynamic feasibility. Entry-window maximum source joint speed/acceleration is 6.21 rad/s and 78.5 rad/s² for the derivative, versus neutral 7.08/98.6 and raw prior 6.84/93.9. Return-window values are 8.74/226.9, versus neutral 9.17/187.8 and raw 8.98/247.3. Root translation diagnostics, full stages, and per-frame sole heights are retained in the artifact. No actuator-speed or acceleration admission threshold was assumed. The pelvis drop varies with phase; body-origin drops are not collider clearances.

The artifact contains the pickle-free `authored_stages_30hz.npz`, CSV, native 50 Hz arrays, sole-height arrays, and a separately identified SONIC library entry. The [runner](../../scripts/research/motion2scene_project_splice_prior.py) and four focused [tests](../../decoupled_wbc/tests/test_motion2scene_prior_splice.py) cover projection scope, exact preserved regions, quaternion interpolation, invalid inputs, and rotated sphere geometry. Black and Ruff pass. The command in the registration reproduces the frozen operation in a fresh equivalent output location; the original location rejects another attempt. Post-run formatting of the repository runner leaves the executed source snapshot unchanged.

This is a candidate for a separately registered physical qualification, not an addition to the qualified option bank or evidence of improved traversal performance.
