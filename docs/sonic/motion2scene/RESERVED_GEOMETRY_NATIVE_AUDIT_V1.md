# Native audit of the reserved geometry

All **36 layouts and 48 beams** pass a CPU-only composed USD audit. The audit
independently reconstructs each center from the original source-specific neutral
polyline and locked arc-length station, applies the net-heading left-normal
lateral offset, and compares the resulting center/orientation/dimensions with
USD's composed world transforms. It does not use the materializer's placement
helper, robot clearance, a teacher, or physics outcomes.

| Quantity | Maximum absolute discrepancy |
| --- | ---: |
| Composed beam center | 4.947e-12 m |
| Composed dimensions | 2.220e-16 m |
| Composed yaw | 8.564e-14 rad |
| Composed underside | 2.220e-16 m |
| Materialized center vs independent route interpolation | 2.220e-15 m |

Every single/course has exactly one/two beams with enabled collision, rigid body
and kinematic settings. All nine shared-room prims retain their composed
attributes, relationships and transforms; only the intended scene/split identity
fields change. The reserved scene/split metadata, all locked parameter values,
original source assignments and ordering agree. Source assets and the frozen
geometry lock remain unchanged.

The [machine-readable audit](../../../research-data/groot-wbc/m2s-reserved-geometry-materialization-v1/native_geometry_audit.json)
includes all per-beam errors, source/layer hashes, the complete room fingerprint,
and [driver snapshot](../../../research-data/groot-wbc/m2s-reserved-geometry-materialization-v1/native_geometry_audit_driver.py).
Audit SHA-256: `a795887fb3c1651907b03ac747985265b2246091f28c188ad112555abcf4f9cc`.
The [repository driver](../../scripts/research/motion2scene_audit_reserved_geometry.py)
uses the installed USD 0.24.5 libraries without starting Isaac Sim:

```bash
M2S_USD_LIB_DIR=/home/linjiw/isaaclab-install/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
M2S_PY_LIB_DIR=/home/linjiw/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/lib
PYTHONPATH=$M2S_USD_LIB_DIR \
LD_LIBRARY_PATH=$M2S_USD_LIB_DIR/bin:$M2S_PY_LIB_DIR \
/home/linjiw/isaaclab-install/env_isaaclab/bin/python \
  scripts/research/motion2scene_audit_reserved_geometry.py \
  --study /path/to/separate/materialized-study-copy
```

Use a copy containing the materialized registration/layouts/scenes but no existing
audit result or driver snapshot. The driver refuses to overwrite either audit
artifact. Original source paths and hashes named by registration must be present
for this provenance audit.

Recorded audit work is zero physics steps and zero robot-clearance/outcome
queries. Native USD composition does not verify PhysX cooking, passage,
perception or policy performance. No implementation amendment is adopted, and
no reserved evaluation episode has been run by this audit.
