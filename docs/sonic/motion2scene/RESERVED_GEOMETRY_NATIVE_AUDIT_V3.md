# Native materialization of the frozen six-second panel

All **162 V3 scene variants and 216 beams** are now materialized and pass a CPU-only native USD composition audit. These comprise 108 single-beam variants and 54 two-beam variants: every original layout’s nominal geometry and eight stress offsets. No reference route was loaded or used to relocate a beam; the world XYZ, yaw, and full dimensions in the frozen V3 JSON are authoritative.

The [registration](/home/linjiw/research-data/groot-wbc/m2s-reserved-geometry-materialization-v3/registration.json), SHA256 `dc19b76fa4285e7e68a148666bdda304608426361d5eb27528cc7094486bf3d5`, preceded scene authoring. It binds the full runner/dependency closure, V3 lock, untouched V2 lock, and the six-second neutral qualification’s empty room. The original lock hashes remain unchanged. The [result](/home/linjiw/research-data/groot-wbc/m2s-reserved-geometry-materialization-v3/result.json) links every hashed scene and layout definition.

The [native audit](/home/linjiw/research-data/groot-wbc/m2s-reserved-geometry-materialization-v3/native_geometry_audit.json), SHA256 `b8d1108b8fe145ec505ad7d543a4a3897292e09d5067f4b5d0bf7e0ea30da05f`, extracts composed world transforms and cube sizes independently of the authoring routine. Maximum errors from the locked values are:

| Quantity | Maximum error |
|---|---:|
| World center | 4.97e−12 m |
| Full dimensions | 2.22e−16 m |
| Yaw | 8.69e−14 rad |
| Rotation matrix entry | 8.68e−14 |
| Underside height | 2.22e−16 m |

Every cube has the required collision and kinematic rigid-body flags. All seven non-beam room prims preserve their composed transforms, attributes, relationships, and metadata; only the declared scene/split identifiers change. Each scene explicitly identifies the reserved V3 split. No file is labeled a development scene.

The first import attempt could not locate the already-installed USD extension and created no scene or materialization-start marker. Its log is retained. A separately preregistered [environment amendment](/home/linjiw/research-data/groot-wbc/m2s-reserved-geometry-materialization-v3/environment_amendment.json) exposes the existing USD 0.24.5 Python and shared-library paths, pins 288 runtime files, and uses the unchanged frozen materializer. No package installation or simulator startup occurred. The amended command and environment are recorded for reproduction in a separate equivalent study directory.

The [materializer](../../scripts/research/motion2scene_materialize_six_second_evaluation.py) rejects inconsistent redundant world coordinates and requires every frozen variant. Two focused [tests](../../decoupled_wbc/tests/test_motion2scene_six_second_materialization.py), Black, and Ruff pass.

All physical statuses remain `not_run`: **zero robot-clearance queries, physics steps, and outcome queries**. The common implementation and terminal/scoring policy remain pending. Materializing these assets does not adopt the proposed implementation amendment, complete V2’s original evaluation, or qualify repeated course transitions.
