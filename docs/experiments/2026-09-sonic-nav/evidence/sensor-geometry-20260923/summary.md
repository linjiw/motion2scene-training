# MID-360 sensor geometry on the G1 (roadmap Phase 0.9)

Generated 2026-09-23T13:37:39 by `scripts/sensor/mid360_geometry.py` at commit `519e1d2627`. CPU only. Numbers come from `results.json` in this directory. To reproduce: `CUDA_VISIBLE_DEVICES= .venv_native/bin/python scripts/sensor/mid360_geometry.py`, which takes about 1.5 min on 4 threads and needs 3.3 GB RAM.

Tags:
- **[C]** computed: URDF forward kinematics plus ray geometry.
- **[M]** measured input: executed release-SONIC physics rollouts from `teacher-8192-500-review/eval/release`.
- **[A]** assumed.

Values are the median, with p5 to p95 in brackets, over frames or simulated approaches. "never" means the edge is never inside the field of view (FOV). Distances are horizontal and measured from the sensor.

## Bottom line

- **Both mounts put the sensor at the same point: 1.217 m above the floor standing [C]; the roadmap had assumed about 1.2 m.** They differ only in orientation. `g1_29dof.urdf`'s +0.40618 m is relative to a `torso_link` frame that sits 0.010 m higher than `main.urdf`'s. Head top: 1.288 m. Executed walking: sensor 1.202 (1.161 to 1.229) m, torso pitch 1.8 (-6.6 to 11.5) deg, roll -0.4 (-6.4 to 6.3) deg [C from M].
- **Inverted mount (`main.urdf`, and Unitree's own description):** covers -54.3 to +4.7 deg straight ahead when level.
  - It sees the floor from 0.87 m ahead, 0.95 m to the side (1.40 m once the shoulders are counted) and 1.03 m behind.
  - Its upper edge is only +4.7 deg ahead. Overheads above the sensor therefore drop out early. While walking, a 1.40 m underside is last seen 2.06 (0.94 to 4.00) m out.
  - Crouched, the upper edge ahead is -21.4 (-33.4 to -0.9) deg, so an overhead ahead is essentially invisible [C].
- **Upright mount (`g1_29dof.urdf`):** covers -9.3 to +49.7 deg ahead. It sees overheads almost until it is underneath them, but it is floor-blind out to 7.4 m ahead standing (6.1 m median while walking) and 14.8 m behind [C]. Ground obstacles near the feet would need another sensor, such as the D435i.
- **Self-occlusion (exact mesh, inverted):** 2.5% of FOV rays standing, 2.5% (2.5% to 3.7%) walking and 8.2% (4.3% to 10.5%) crouched. The blocked rays are a thin band at the lower edge, from the shoulder pitch/roll links. The upright mount blocks 0% [C]. Isaac's collision capsules capture only 0.5% standing.
- **Memory is mandatory for the inverted mount.** Overheads that are walked under need 2.3–5.1 s of registered memory at 0.6 m/s (median; 8.8 s at p95). The floor under the feet needs 1.3 s (1.8 s at p95) [C+A]; see §6.

## 1. Mount survey [C]

| File | Orientation | Offset in parent (m) | Sensor in pelvis frame, zero joints (m) |
|---|---|---|---|
| `workspace/vendor/sonic/gear_sonic/data/assets/robot_description/urdf/g1/main.urdf` | inverted (tilt 2.3 deg) | +0.41618 | (-0.0037, +0.00003, +0.46018) |
| `vendor/sonic/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof.urdf` | upright (tilt 2.3 deg) | +0.40618 | (-0.0037, +0.00003, +0.46018) |
| `vendor/sonic/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf` | upright (tilt 2.3 deg) | +0.40618 | (-0.0037, +0.00003, +0.46018) |
| `vendor/sonic/decoupled_wbc/sim2mujoco/resources/robots/g1/g1.urdf` | upright (tilt 2.3 deg) | +0.41618 | (-0.0037, +0.00003, +0.46018) |
| `/opt/ros/humble/share/unitree_description/urdf/g1/main.urdf` | inverted (tilt 2.3 deg) | +0.41618 | (-0.0037, +0.00003, +0.46018) |

- The inverted rpy (0, 3.101, 3.1415) is Rx(π)·Ry(−0.0406). The sensor z axis points down, tilted 2.3 deg backwards, so the band leans 2.3 deg down at the front.
- The upright rpy (0, 0.0401, 0) tilts the sensor z axis 2.3 deg forwards, so the band is 2.3 deg lower at the front.
- The only asset spawned in Isaac (`main.urdf`) and Unitree's ROS description (`/opt/ros/humble/.../g1/main.urdf`, same as its MJCF site) are inverted. The decoupled_wbc copies are upright.
- FK check: max body error 5.2e-07 m against the MuJoCo-native `reference.npz`, and 1.0e-06 m against Isaac's `ref_track_body_pos`. In the executed rollouts, the foot soles sit on z=0 (p50 -2.0e-05 m).

## 2. Sensor height and gravity-frame elevation window [C]

Windows are in degrees above the horizon, at heading azimuths front, left and back. For walking and crouching the window shows its median; lo and hi also give their p5/p95. A ±5 deg pitch sweep is in `results.json` → `pitch_sweep` and `fig_window_vs_pitch.png`.

| Posture | Frames | Sensor height (m) | Torso pitch (deg) | Mount | Front | Left | Back |
|---|---|---|---|---|---|---|---|
| standing (Isaac init pose) | 1 | 1.217 | -0.0 | inverted | -54.3 to +4.7 | -52.1 to +7.0 | -49.7 to +9.3 |
| standing (Isaac init pose) | 1 | 1.217 | -0.0 | upright | -9.3 to +49.7 | -7.0 to +52.1 | -4.7 to +54.3 |
| walking, executed (release SONIC, physics) | 7627 | 1.202 (1.161 to 1.229) | 1.8 (-6.6 to 11.5) | inverted | -56.2 to +2.9 (lo -66/-48, hi -7/+11) | -52.8 to +6.7 (lo -60/-46, hi +1/+13) | -48.0 to +11.1 (lo -56/-39, hi +3/+21) |
| walking, executed (release SONIC, physics) | 7627 | 1.202 (1.161 to 1.229) | 1.8 (-6.6 to 11.5) | upright | -11.1 to +48.0 (lo -21/-3, hi +39/+56) | -7.4 to +52.0 (lo -14/-1, hi +47/+59) | -2.9 to +56.2 (lo -11/+7, hi +48/+66) |
| walking, kinematic reference | 4549 | 1.216 (1.167 to 1.242) | 10.4 (2.6 to 24.8) | inverted | -64.8 to -5.7 (lo -79/-57, hi -20/+2) | -54.2 to +7.2 (lo -66/-47, hi +1/+14) | -39.5 to +19.7 (lo -48/-26, hi +12/+34) |
| walking, kinematic reference | 4549 | 1.216 (1.167 to 1.242) | 10.4 (2.6 to 24.8) | upright | -19.7 to +39.5 (lo -34/-12, hi +26/+48) | -7.2 to +54.2 (lo -14/-0, hi +50/+65) | +5.7 to +64.8 (lo -2/+20, hi +57/+79) |
| crouched, executed | 1732 | 0.914 (0.822 to 1.026) | 26.2 (5.7 to 39.0) | inverted | -82.8 to -21.4 (lo -90/-62, hi -33/-1) | -67.2 to +7.2 (lo -90/-42, hi -42/+25) | -30.4 to +35.9 (lo -90/-17, hi +15/+49) |
| crouched, executed | 1732 | 0.914 (0.822 to 1.026) | 26.2 (5.7 to 39.0) | upright | -35.9 to +30.4 (lo -49/-15, hi +17/+90) | -8.8 to +64.7 (lo -57/+9, hi +43/+90) | +21.4 to +82.8 (lo +1/+33, hi +62/+90) |
| crouched, kinematic reference | 1021 | 0.757 (0.675 to 0.984) | 45.2 (29.9 to 58.6) | inverted | -90.0 to -39.7 (lo -90/-87, hi -54/-25) | -90.0 to +4.4 (lo -90/-78, hi -63/+37) | -90.0 to +55.3 (lo -90/-19, hi +39/+69) |
| crouched, kinematic reference | 1021 | 0.757 (0.675 to 0.984) | 45.2 (29.9 to 58.6) | upright | -55.3 to +90.0 (lo -69/-39, hi +19/+90) | -15.9 to +90.0 (lo -79/+20, hi +62/+90) | +39.7 to +90.0 (lo +25/+54, hi +87/+90) |
| crawling, executed | 746 | 0.589 (0.540 to 0.650) | 69.7 (63.7 to 78.7) | inverted | -90.0 to -64.4 (lo -90/-90, hi -74/-57) | -90.0 to +49.5 (lo -90/-90, hi -57/+84) | -90.0 to +80.9 (lo -90/-90, hi +74/+88) |
| crawling, executed | 746 | 0.589 (0.540 to 0.650) | 69.7 (63.7 to 78.7) | upright | -80.8 to +90.0 (lo -88/-74, hi +90/+90) | +4.4 to +90.0 (lo -83/+63, hi +90/+90) | +64.4 to +90.0 (lo +57/+74, hi +90/+90) |

The kinematic references lean further forward than the executed gait (median 10.4 vs 1.8 deg). The executed rollouts are what a robot running the release tracker would see.

## 3. Blind floor radius (closest visible floor point, m) [C]

"FOV only" ignores the robot's own body. "+ self-occlusion" also requires a clear ray against the exact visual meshes, on a subsample of frames.

| Posture | Mount | Front, FOV only | Front + self-occl. | Left, FOV only | Left + self-occl. | Back, FOV only | Back + self-occl. |
|---|---|---|---|---|---|---|---|
| standing (Isaac init pose) | inverted | 0.87 | 0.87 | 0.95 | 1.40 | 1.03 | 1.03 |
| standing (Isaac init pose) | upright | 7.43 | 7.43 | 9.91 | 9.91 | 14.81 | 14.81 |
| walking, executed (release SONIC, physics) | inverted | 0.81 (0.52 to 1.09) | 0.79 (0.55 to 1.02) | 0.91 (0.69 to 1.16) | 1.35 (1.12 to 1.65) | 1.08 (0.80 to 1.49) | 1.11 (0.87 to 1.50) |
| walking, executed (release SONIC, physics) | upright | 6.09 (3.10 to 18.80; never in 2%) | 5.58 (3.46 to 14.55) | 9.05 (4.84 to 32.16; never in 4%) | 9.14 (4.45 to 24.87; never in 2%) | 15.82 (5.45 to 115.26; never in 26%) | 20.94 (7.38 to 225.25; never in 29%) |
| crouched, executed | inverted | 0.12 (0.00 to 0.47) | 0.14 (0.00 to 0.47) | 0.39 (0.00 to 1.02) | 0.46 (0.00 to 1.48) | 1.54 (0.00 to 3.18) | 1.75 (0.01 to 3.48) |
| crouched, executed | upright | 1.30 (0.76 to 3.12) | 1.33 (0.75 to 1.96) | 3.88 (0.54 to 39.60; never in 19%) | 4.51 (0.93 to 36.73; never in 21%) | 22.67 (8.66 to 781.58; never in 97%) | never |

## 4. Overhead beam: last distance at which its lower front edge is seen [C]

- **Static pose:** the geometric threshold on the heading line.
- **Approach:** the executed pose sequence is replayed while the robot closes in at 0.6 m/s. Scans run at 10 Hz with a random phase (2000 approaches). The value is the distance of the last scan that still contains the edge; the scan spacing is 0.06 m.
- For a beam below the sensor, the "edge" is the lower edge of its front face, seen from above. The underside itself is never seen from above.
- Standing, the head top is at 1.29 m. Beams below about 1.30 m need a duck, so the standing and walking rows apply to the approach before the duck.

| Underside (m) | Inverted, standing | Inverted, walking approach | Inverted, crouched approach | Upright, standing | Upright, walking approach | Upright, crouched approach |
|---|---|---|---|---|---|---|
| 1.00 | 0.16 | 0.16 (0.10 to 0.22) | 3.06 (0.03 to 11.18) | 1.33 | 0.94 (0.48 to 1.77) | 0.17 (0.03 to 0.52) |
| 1.05 | 0.12 | 0.13 (0.08 to 0.18) | 4.11 (0.36 to 11.48) | 1.02 | 0.72 (0.35 to 1.52) | 0.25 (0.03 to 0.68) |
| 1.10 | 0.08 | 0.10 (0.05 to 0.14) | 7.66 (2.95 to 14.29; never in 29%) | 0.72 | 0.51 (0.21 to 1.17) | 0.34 (0.05 to 0.87) |
| 1.15 | 0.05 | 0.07 (0.03 to 0.11) | 8.17 (3.47 to 14.32; never in 31%) | 0.41 | 0.26 (0.07 to 0.69) | 0.44 (0.05 to 0.99) |
| 1.20 | 0.01 | 0.06 (0.01 to 0.46) | 8.60 (3.91 to 14.35; never in 34%) | 0.11 | 0.07 (0.02 to 0.21) | 0.54 (0.05 to 1.06) |
| 1.25 | 0.40 | 0.59 (0.20 to 1.98) | 9.12 (4.53 to 14.39; never in 37%) | 0.03 | 0.08 (0.03 to 0.13) | 0.63 (0.05 to 1.21) |
| 1.30 | 1.01 | 1.12 (0.48 to 2.69) | 9.52 (5.01 to 14.40; never in 39%) | 0.07 | 0.12 (0.07 to 0.19) | 0.73 (0.05 to 1.31) |
| 1.35 | 1.62 | 1.61 (0.71 to 3.27) | 9.69 (5.50 to 14.44; never in 42%) | 0.11 | 0.16 (0.11 to 0.25) | 0.82 (0.05 to 1.42) |
| 1.40 | 2.24 | 2.06 (0.94 to 4.00) | 9.90 (6.08 to 14.54; never in 46%) | 0.16 | 0.21 (0.15 to 0.31) | 0.91 (0.05 to 1.58) |
| 1.45 | 2.85 | 2.47 (1.17 to 4.51) | 10.29 (6.60 to 14.62; never in 49%) | 0.20 | 0.25 (0.18 to 0.37) | 1.01 (0.05 to 1.71) |
| 1.50 | 3.46 | 2.82 (1.38 to 5.04) | 10.34 (6.89 to 14.60; never in 52%) | 0.24 | 0.30 (0.22 to 0.43) | 1.09 (0.05 to 1.82) |

The approach replays the executed crouch frames at 0.6 m/s. Real crouched transit speed is unknown (roadmap 0.7). The corridor variant, where any point of the edge within |y| ≤ 0.3 m counts, and the kinematic-reference gait are in `results.json`.

## 5. Self-occlusion (share of FOV rays, uniform in solid angle) [C]

The reference is the exact MuJoCo ray cast on the `main.urdf` visual meshes. The head shell and the head collision disk are excluded, because the sensor sits inside them. IoU is per ray, against the exact mesh.

| Mount | Posture | Frames | Caster | Blocked | Within 0.1 m blind zone | IoU vs mesh | Main blockers (share of rays) |
|---|---|---|---|---|---|---|---|
| inverted | standing (Isaac init pose) | 1 | visual_mesh_exact | 0.025 | 0.000 | ref | left_shoulder_pitch 1.1%, right_shoulder_pitch 1.1%, left_shoulder_roll 0.2%, right_shoulder_roll 0.2% |
| inverted | standing (Isaac init pose) | 1 | isaac_collision_capsules | 0.005 | 0.000 | 0.21 | left_shoulder_yaw 0.3%, right_shoulder_yaw 0.3% |
| inverted | standing (Isaac init pose) | 1 | bounding_capsules_boxes | 0.098 | 0.000 | 0.26 | torso 7.7%, left_shoulder_roll 0.9%, right_shoulder_roll 0.8%, right_shoulder_pitch 0.2% |
| inverted | standing (Isaac init pose) | 1 | sphere_proxies_k16 | 0.040 | 0.000 | 0.59 | right_shoulder_pitch 1.5%, left_shoulder_pitch 1.4%, left_shoulder_roll 0.6%, right_shoulder_roll 0.4% |
| inverted | walking, executed (release SONIC, physics) | 48 | visual_mesh_exact | 0.025 (0.025 to 0.037) | 0.000 (0.000 to 0.000) | ref | left_shoulder_pitch 1.1%, right_shoulder_pitch 1.0%, right_shoulder_roll 0.2%, left_shoulder_roll 0.2% |
| inverted | walking, executed (release SONIC, physics) | 48 | isaac_collision_capsules | 0.007 (0.005 to 0.024) | 0.000 (0.000 to 0.000) | 0.27 (0.22 to 0.53) | right_shoulder_yaw 0.4%, left_shoulder_yaw 0.4%, right_hand_palm 0.1%, right_elbow 0.0% |
| inverted | walking, executed (release SONIC, physics) | 48 | bounding_capsules_boxes | 0.098 (0.096 to 0.108) | 0.000 (0.000 to 0.000) | 0.26 (0.26 to 0.34) | torso 7.7%, left_shoulder_roll 0.9%, right_shoulder_roll 0.8%, left_shoulder_pitch 0.2% |
| inverted | walking, executed (release SONIC, physics) | 48 | sphere_proxies_k16 | 0.041 (0.040 to 0.056) | 0.000 (0.000 to 0.000) | 0.60 (0.59 to 0.66) | right_shoulder_pitch 1.5%, left_shoulder_pitch 1.4%, left_shoulder_roll 0.7%, right_shoulder_roll 0.5% |
| inverted | crouched, executed | 24 | visual_mesh_exact | 0.082 (0.043 to 0.105) | 0.000 (0.000 to 0.000) | ref | left_shoulder_roll 1.2%, right_shoulder_roll 1.0%, right_shoulder_pitch 0.7%, left_shoulder_yaw 0.6% |
| inverted | crouched, executed | 24 | isaac_collision_capsules | 0.074 (0.032 to 0.106) | 0.000 (0.000 to 0.000) | 0.66 (0.58 to 0.68) | left_shoulder_yaw 1.4%, right_shoulder_yaw 1.3%, left_elbow 0.7%, right_elbow 0.7% |
| inverted | crouched, executed | 24 | bounding_capsules_boxes | 0.153 (0.107 to 0.193) | 0.000 (0.000 to 0.000) | 0.53 (0.40 to 0.55) | torso 7.7%, left_shoulder_roll 1.2%, right_shoulder_roll 1.0%, left_shoulder_yaw 0.7% |
| inverted | crouched, executed | 24 | sphere_proxies_k16 | 0.111 (0.062 to 0.142) | 0.000 (0.000 to 0.000) | 0.70 (0.68 to 0.74) | left_shoulder_roll 1.6%, right_shoulder_roll 1.4%, right_shoulder_pitch 1.2%, left_shoulder_pitch 1.1% |
| upright | standing (Isaac init pose) | 1 | visual_mesh_exact | 0.000 | 0.000 | ref | - |
| upright | standing (Isaac init pose) | 1 | isaac_collision_capsules | 0.000 | 0.000 | 1.00 | - |
| upright | standing (Isaac init pose) | 1 | bounding_capsules_boxes | 0.000 | 0.000 | 1.00 | - |
| upright | standing (Isaac init pose) | 1 | sphere_proxies_k16 | 0.000 | 0.000 | 1.00 | - |
| upright | walking, executed (release SONIC, physics) | 48 | visual_mesh_exact | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | ref | - |
| upright | walking, executed (release SONIC, physics) | 48 | isaac_collision_capsules | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | 1.00 (1.00 to 1.00) | - |
| upright | walking, executed (release SONIC, physics) | 48 | bounding_capsules_boxes | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | 1.00 (1.00 to 1.00) | - |
| upright | walking, executed (release SONIC, physics) | 48 | sphere_proxies_k16 | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | 1.00 (1.00 to 1.00) | - |
| upright | crouched, executed | 24 | visual_mesh_exact | 0.000 (0.000 to 0.001) | 0.000 (0.000 to 0.000) | ref | left_hand_index_0 0.0%, left_hand_index_1 0.0%, left_hand_middle_1 0.0%, left_hand_middle_0 0.0% |
| upright | crouched, executed | 24 | isaac_collision_capsules | 0.000 (0.000 to 0.002) | 0.000 (0.000 to 0.000) | 1.00 (0.70 to 1.00) | left_hand_index_0 0.0%, left_hand_palm 0.0%, left_hand_index_1 0.0%, left_hand_middle_1 0.0% |
| upright | crouched, executed | 24 | bounding_capsules_boxes | 0.000 (0.000 to 0.001) | 0.000 (0.000 to 0.000) | 1.00 (0.82 to 1.00) | left_hand_index_1 0.0%, left_hand_index_0 0.0%, left_hand_palm 0.0%, left_hand_middle_1 0.0% |
| upright | crouched, executed | 24 | sphere_proxies_k16 | 0.000 (0.000 to 0.001) | 0.000 (0.000 to 0.000) | 1.00 (0.82 to 1.00) | left_hand_index_1 0.0%, left_hand_middle_1 0.0%, left_hand_index_0 0.0%, left_hand_palm 0.0% |

Links that are rigid with the sensor (torso, logo, head) block no ray in any posture of the exact cast. Everything that is blocked comes from the moving arm links.
- A simulated MID-360 therefore needs no torso mask.
- The arm links need a good ray cast. Bounding primitives over-block, and the Isaac collision set, which has no shoulder pitch/roll geometry, under-blocks while standing.

## 6. Implications

**Memory horizon (inverted mount, 0.6 m/s).** The horizon is: last-seen distance + beam depth 0.20 m [A] + body behind the sensor 0.07 m, divided by speed.
- Underside 1.30 m (walk-under): 2.3 s median, 4.9 s at p95.
- Underside 1.40 m (walk-under): 3.9 s median, 7.1 s at p95.
- Underside 1.50 m (walk-under): 5.1 s median, 8.8 s at p95.
- Floor: the ground under the feet was last seen at ≥ 0.81 m (median; p95 1.09 m) in front, so a floor/obstacle layer must persist ≥ 1.3–1.8 s.
- Duck-under beams (≤1.30 m): once crouched, the edge ahead is out of view in most frames (table §4). Memory must span the whole transit, from the start of the duck until the back clears the beam. That is at least (duck lead distance + beam depth + 0.07 m) / crouched speed, where lead distance and speed come from Phase 0.7 [A]. With a 1 m lead at 0.3 m/s it is about 4.2 s.
- **Recommendation:** a registered map (FAST-LIO2-style odometry plus a rolling voxel/elevation memory) that keeps at least 9 s, or 5.4 m of travel at 0.6 m/s. That covers the p95 horizon.
  - The teacher's visibility mask should use this horizon, not the instantaneous FOV.
  - Memory must reach at least 5.0 m ahead: an overhead is last seen that far out at p95.
  - The policy's map crop must extend at least 0.27 m behind the sensor, plus margin, so a beam stays in view until the body clears it.
  - Its forward reach only needs to cover the duck lead distance (Phase 0.7).

**Can the inverted mount see the floor ahead?** Yes. The forward floor becomes visible at 0.87 m standing, 0.81 (0.52 to 1.09) m walking, and closer when crouched. The chest and shoulders never hide the forward floor band (§3); the shoulders only push the side radius out. This is the mount that suits stepping and foothold perception.

**Can it see overheads?** Only far ahead. The upper edge is +4.7 deg ahead when level. While walking it is 2.9 (-6.8 to 11.2) deg, so forward lean often pushes it below the horizon. While crouching it looks down entirely. An underside 0.1–0.3 m above the sensor is last seen at 1.12 (0.48 to 2.69) m (1.30 m) to 2.82 (1.38 to 5.04) m (1.50 m) while walking. After that the robot relies on memory.
- The upright mount sees the same overheads until 0.19–0.43 m (p95).
- But the upright mount is floor-blind within 6.1 m ahead while walking (3.1 to 18.8 m).

**Consequences for the plan:**
- The Phase 4.1 sensor proxy must model the inverted band (−54/+5 deg ahead).
- Phase 4.2's pre-registered signature is expected to hold for the inverted mount: a student distilled from a full-geometry teacher fails on overheads that leave the FOV before arrival.
- The distance at which they leave is the approach column in §4.

## 7. Question for the user

**Is the MID-360 on your G1 mounted inverted, as in `main.urdf` and Unitree's `unitree_description`, or upright, as in `decoupled_wbc/.../g1_29dof.urdf`?** A 10-second check: with the robot standing, look at the raw Livox point cloud in the sensor frame. Floor returns sit at z ≈ **+1.2 m** if the sensor is inverted and at z ≈ −1.2 m if it is upright. Also: is the D435i fitted, and is it pitched about 48 deg down as in `g1_29dof.urdf`? With an upright MID-360, it would be the only near-floor sensor.

## 8. Caveats

- [A] Ray density is uniform in solid angle over the −7..+52 deg band. The real non-repetitive MID-360 pattern (OmniPerception's `mid360.npy`, not on this host) is not uniform. A 10 Hz frame covers about 1 point per deg², so thin bars can be missed in a single frame.
- [A] The head shell is treated as transparent, because the sensor sits inside the `head_link` mesh and its window geometry is not in these files. The 0.1 m blind zone only removes returns; blocked rays still count as blocked.
- [A] Walking uses executed release-SONIC rollouts of Kimodo reference clips, at their own speeds, with the 0.6 m/s approach synthesised on top. The motor student and navigation adapter may walk differently. Crouch frames come from clips that were never qualified for crouch transit.
- [A] Beam: a horizontal bar across the path, depth 0.20 m for the memory horizon. Distances are from the sensor; the toes lead it by 0.28 m.
- The kinematic chain and mount are from `main.urdf`. The upright mount is transplanted through the pelvis frame, and both mounts give the same point to 1e-12 m.

## Figures

![fig_side_view.png](fig_side_view.png)

![fig_window_vs_pitch.png](fig_window_vs_pitch.png)

![fig_overhead_last_seen.png](fig_overhead_last_seen.png)

![fig_self_occlusion_map.png](fig_self_occlusion_map.png)

