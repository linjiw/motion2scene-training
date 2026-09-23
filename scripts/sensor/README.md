# Sensor geometry (roadmap Phase 0.9)

`mid360_geometry.py` computes where a Livox MID-360 on the G1 head can see. It compares the two
mounts found in the repo: inverted in `main.urdf`, the asset Isaac spawns, and upright in
`decoupled_wbc/.../g1_29dof.urdf`. It uses CPU only and never imports Isaac.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 .venv_native/bin/python scripts/sensor/mid360_geometry.py
# -> workspace/phase0/sensor_geometry/{results.json,summary.md,fig_*.png}  (~1.5 min, 3.3 GB)
```

| Output | Contents |
|---|---|
| Sensor height and gravity-frame elevation window | Standing, executed and reference walking, crouch and crawl |
| Blind floor radius | Field of view (FOV) only, and with exact-mesh self-occlusion |
| Last distance at which an overhead beam is seen | Beam undersides 1.0–1.5 m; static poses and a 0.6 m/s / 10 Hz approach replay |
| Self-occlusion | Blocked share of FOV rays for the exact mesh cast (MuJoCo `mj_multiRay`), the Isaac collision capsules, bounding capsules/boxes and sphere proxies, with per-ray IoU against the mesh |

Inputs, all read-only:
- `main.urdf` and meshes under `vendor/sonic/gear_sonic/data` (or `workspace/vendor/...` from a worktree).
- Executed release rollouts: `workspace/teacher-8192-500-review/eval/release/*/metrics/*.pose.npz`.
- Reference clips: `workspace/m2s-hindsight-dataset-v1-20260911/motions/*/reference.npz`.

The geometry core is numpy-only and needs no MuJoCo. It covers FK, FOV windows, ray/primitive
intersection and the approach replay, and is tested by:

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_mid360_geometry.py
```

`--no-mesh` skips the MuJoCo cast, and `--rays` and `--occlusion-frames` trade accuracy for time.
