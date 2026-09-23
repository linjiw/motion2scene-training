# External anchor check (Phase 1.8) + scoop watch, week 1 — 2026-09-23

Sources: GitHub REST API (`gh api`), Hugging Face API, arXiv abs/HTML/API, project pages. Checked 2026-09-23.
UNVERIFIED = not confirmed from a primary source.

## Task 1: External anchor candidates

| Candidate | Public code | Released G1 weights | Licence | Sim / stack | Policy task interface |
|---|---|---|---|---|---|
| **CAT / HumanoidPF** 2601.16035 | Yes: github.com/GalaxyGeneralRobotics/Click-and-Traverse (last push 2026-07-07) | **Yes.** HF `Axian12138/Click-and-Traverse`: `logs_v1/generalist_v1/checkpoints/005033164800/policy.onnx` (1,032,695 B) + `checkpoints/config.json`; about 30 specialist teachers (`logs_v1/teachers/*/policy.onnx`, 339,955 B each); `logs_v0/` specialists. Repo total about 1.08 GB. Also on Google Drive and Tsinghua Cloud mirrors | **Apache-2.0** (repo LICENSE; the HF card also says apache-2.0) | MuJoCo 3.3.1 / MJX + Brax 0.12.3 + mujoco_playground (JAX), 50 Hz control, 500 Hz sim. ONNX eval scripts `cat_ppo/eval/mj_onnx_{play,test}.py`. Real-robot deploy: FAST-LIO + OctoMap → PF | 29-DoF G1 MJCF (`g1_mjx_feetonly_torque.xml`, 29 joints). **Actuates 12 leg joints only** (`num_act=12`; waist and arms stay at their default pose). Observes 23 joints. Obs is 162-D: gyro, gravity, joint pos/vel, last action, motor targets, 4-D command, foot height, gait phase, plus a 77-D **HumanoidPF** term (guidance, boundary and distance fields sampled at head, pelvis, torso, feet, hands, knees and shoulders). The **goal enters through the guidance field**, a fast-marching (skfmm) geodesic field toward the goal computed over a known voxel map (0.04 m voxels; default crop 3.0x2.0x1.5 m). The command is derived from the field (≤0.75 m/s) |
| **Gallant** 2511.14625 | Partial: github.com/InternRobotics/Gallant. `main` has 4 commits (last 2026-04-24); branch `aa-migrate` was active until 2026-09-15. Training code only; README says "detailed commands and evaluation scripts soon" | **No** (no releases; nothing on HF) | **No LICENSE file** (GitHub licence = null; pyproject says "SEE LICENSE FILE"). The dependency btx0424/active-adaptation also has no licence | active-adaptation on **Isaac Sim 5.1 + Isaac Lab** (`GallantEnvIsaac`), with a MuJoCo sim2sim path. Example launch uses 8 GPUs and 1024 envs/GPU | G1 29-DoF (`g1_waist_unlocked`, wrists in the action scaling). Command `LocoNavigation`: target position (base x,y) + pass/rest time. LiDAR voxel grid (0.05 m) + 6-step proprioception history |
| **MTC** 2609.21107 (Sep 17) | **Not found.** The paper says "we release MTC-Challenge" but gives no URL. GitHub search found nothing. RobotiXX org has no MTC repo | No | n/a | UNVERIFIED (HTML never names a simulator) | Goal command from waypoints. 32x32x40 binary voxel grid (Gallant-style CNN). Privileged **BeyondMimic** teachers distilled by DAgger with "perceptual hallucination" |
| **PASSAGE** 2609.18732 (Sep 16) | **No.** github.com/GalaxyGeneralRobotics/PASSAGE is the project page only; its button reads "Code (Coming Soon)" (repo pushed 2026-09-23) | No | none on the page repo | MuJoCo kinematic validation; tracker = perception-enhanced ScaleBFM (29-DoF G1), planner = flow matching at 6.25 Hz | Elevation map + destination tokens → reference chunks → frozen tracker. Planner-side RL runs under the frozen tracker |
| **LP-NavOA** 2606.23249 | Yes: github.com/shenqiqishi/LP-NavOA-code (created 2026-09-19, one commit, "minimal release") | **No.** README trains everything: backbone about 12.5k iters at 4–8k envs, then the local planner | **Apache-2.0** | mjlab fork (MuJoCo Warp), CUDA 12.8 | mjlab 29-DoF G1. Backbone takes `[heading error, target speed]`; the distilled planner overwrites heading only. Short-range raycasts + body-frame goal direction. No map |
| **GuideWalk** 2606.10449 (v3 Sep 19) | **No.** guide-walk.github.io/GuideWalk is a template page with no code link | No | n/a | Isaac Sim (single RTX 5080, 2048 envs), MuJoCo sim2sim | DWA velocity guidance from target position. Student: elevation-map + depth CNNs + target position. Robot model UNVERIFIED |

### Recommendation: CAT generalist v1 (single anchor)
- **Only candidate with released G1 weights, a permissive licence and a goal-conditioned known-map interface.** HumanoidPF is built from a known voxel map plus a goal point, which is our "known-map goal reaching" setting. Its training distribution explicitly covers ground, lateral and **overhead** obstacles (crouch/hurdle/side teachers), which matches our box and beam tasks.
- **External precedent:** PASSAGE Table I reports "CAT (released, zero-shot)" at 70.3% success and 14.0% collision-free success on held-out PASSAGE scenes. TANGO (2609.09158) uses HumanoidPF as a low-level baseline tracking waypoints 1.2 m ahead. Using the released checkpoint as an anchor is therefore established practice, and the waypoint variant shows how to feed it goals.
- **Port plan (3–5 days):**
  - (a) Days 1–2: run CAT natively. Export our box/beam scenes as obstacle meshes/voxels, run `pf_modular.py` (sdf/bf/gf with our goal), then `mj_onnx_test` in MuJoCo. This gives an anchor number with zero porting risk.
  - (b) Days 3–5, optional Isaac Lab 2.3 port:
    - reimplement the 162-D observation (trilinear PF lookups at the 7 body groups, gait phase, field-projected command);
    - hold waist and arms at `DEFAULT_QPOS`;
    - match the MJX torque/PD model and action scale 0.5;
    - validate the port against native MuJoCo rollouts.
- **Caveats to state in the paper:**
  - legs-only action space (upper body frozen), unlike SONIC's 29-DoF;
  - trained on about 2 m crops (3x2x1.5 m PF grid), so longer maps need a larger PF grid, which is a distribution shift;
  - speed ≤0.75 m/s;
  - low collision-free success out of distribution (14% in PASSAGE's scenes);
  - MuJoCo↔Isaac sim-to-sim gap if ported.
- **Why not the others:**
  - Gallant fits our stack (Isaac Sim 5.1 / Isaac Lab) and has a goal-position interface, but it ships no weights and no licence, and retraining needs about 8 GPUs, which the shared GPU cannot supply in 3–5 days.
  - LP-NavOA has Apache code but no weights, and it is mapless limited-perception navigation, so it is not a known-map anchor.
  - MTC, PASSAGE and GuideWalk have no released code.

## Task 2: Scoop watch (arXiv/GitHub, about 2026-09-10 → 09-23)
Method: arXiv API abstract queries with submittedDate ≥ 2026-09-08, the cs.RO `pastweek` and `new` listings (Sep 17–23), GitHub and HF APIs, web search.

**Pre-emption verdict: nothing found that pre-empts C1 or C2.** Closest items are flagged below.

### (1) RL/DAgger in SONIC token or latent space; frozen SONIC + goal/nav/perception
- **ViBe** 2609.09918 (Sep 9). SONIC with rank-16 LoRA in the **decoder** plus a frozen visual encoder, trained with PPO. This is the "decoder LoRA" arm of C1 only. It compares blind vs adapted and ablates encoders/queries; there is no command/chunk/token comparison. Code link github.com/lok-i/vibe returns **404** (not public). No follow-up found. C1-adjacent (one arm).
- **S³ / Sample-Simulate-Select** 2609.26420 (Sep 22). Frozen SONIC used as a physics verifier for text-to-motion candidates. No goal/nav. Low relevance.
- **Gated Residual Body-Hand** 2609.18763 (Sep 16). Frozen SONIC-based tracker with a bounded residual; teleoperation. Low.
- **RoM-Nav** 2609.19272 (Sep 16). Nav policy over a **frozen locomotion policy** (velocity interface), kickstarted from a dynamics-privileged reduced-order-model teacher. Not SONIC. Low–medium.
- No paper found that runs RL/DAgger directly in SONIC's FSQ token space for navigation.

### (2) Matched comparison of conditioning interfaces for a frozen tracker
- **TANGO** 2609.09158 (Sep 8; follow-up check). Table 4 ablates action space: a 2D trajectory to the Unitree controller + MPC vs whole-body reference chunks to SONIC. Table 5 swaps the SONIC tracker for ScaleBFM. This is **partial overlap with C1 (commands vs chunks)**, but the arms use different low-level controllers, and there are no token or LoRA arms. No code found (no GitHub hit, no link in the paper).
- **PASSAGE** 2609.18732. Reference-chunk interface into a frozen ScaleBFM with planner-side RL; no interface comparison. C1-adjacent.
- **RECAL** 2609.16405 (Sep 14). Scene-geometry cross-attention layer wrapping a blind WBC (Digit V3), compared against alternative geometry encoders. It asks "where scene info enters" but uses no SONIC and no interface arms. Low–medium.
- Background, not new: the SONIC paper (2511.07820, v dated 2026-05-22) already reports FSQ tokens vs SMPL for its VLA (68% vs 27%). Cite it as prior art for the token arm.

### (3) Imitation gap / privileged-teacher asymmetry in humanoid distillation
- **MTC** 2609.21107 (Sep 17). **Reference-tracking (BeyondMimic) privileged teachers → DAgger voxel student** for humanoid clutter traversal. This is exactly C2's setting, but the paper does no imitation-gap analysis (0 hits for "imitation gap" or "asymmetr"). **C2-adjacent. Watch for v2 and the benchmark release.**
- **PLAT** 2609.25754 (Sep 22). A dense-reference tracking expert is distilled via DAgger into a sparse-keyframe student. It names the "transition ambiguity" of under-specified commands (the imitation-gap mechanism) and fixes it with a privileged CVAE latent plus latent residual RL, with SONIC as a baseline. Keyframe tracking, not traversal. **C2-adjacent (mechanism), not pre-empting.**
- **HOTICE** 2609.25363 (Sep 21). HumanoidPF-style fields + specialist→generalist distillation for object transport (CAT lineage). Low.
- **DWMP** 2609.12347 (Sep 11). Teacher-student obstacle traversal with world-model latents. Low.
- Out of window, useful background: AR-OPD 2606.10385 (privileged on-policy distillation, "hindsight leakage").

### (4) Follow-ups and code releases
- **PASSAGE**: code "Coming Soon". **MTC**: none found. **ViBe**: repo 404. **TANGO**: none found.
- **MotionBricks**: official code is at `GR00T-WholeBodyControl/motionbricks` (last commit 2026-05-01; "full release … approximately one month out" per nvlabs.github.io/motionbricks, date UNVERIFIED). Only community ports appeared (e.g. localai-org/motion-bricks.cpp, pushed Sep 21). No new paper.
- CAT lineage: HOTICE (above). TANGO and PASSAGE both use CAT/HumanoidPF as a baseline.

### (5) NVlabs/GR00T-WholeBodyControl, September 2026
- **No releases or tags.** Merged PRs:
  - #267 (Sep 3): SONIC v1.1 per-motor Kp/Kd scaling (ankle-pitch ×1.5) and an **adaptive-sampling fix**. Failures were being charged to the post-reset motion cursor. This matters if we fine-tune with gear_sonic adaptive sampling.
  - #232 (Sep 15): `sonic_release` encoder observation aliases for C++ deploy.
  - #240 (Sep 15): motion_lib → deploy CSV exporter.
  - #213 (Sep 23): reject an empty `--planner-file`.
- **No new planner or navigation features.** Open PR #242 "Planner" (empty body) has been inactive since 2026-08-07.
- HF `nvidia/GEAR-SONIC`: last commit 2026-08-26 (v1.1 tuning docs). Issue #269 (Sep 7): external 29-policy G1 disturbance benchmark includes SONIC.

### Other in-window hits (low relevance)
X-WBC 2609.15213 (cross-embodiment tracker, command tokens); KINO 2609.18869 (keyframe interface VLM→WBC); PredActor 2609.24840 (argues against steering by reference into a separate tracker); CHOREO 2609.22274 (skills as trajectories); PGMT 2609.08511 (perceptive general tracker).

## URLs
- CAT: https://github.com/GalaxyGeneralRobotics/Click-and-Traverse · https://huggingface.co/Axian12138/Click-and-Traverse · https://axian12138.github.io/CAT/
- Gallant: https://github.com/InternRobotics/Gallant (branch aa-migrate) · https://github.com/btx0424/active-adaptation
- MTC: https://arxiv.org/abs/2609.21107 · PASSAGE: https://arxiv.org/abs/2609.18732 · https://github.com/GalaxyGeneralRobotics/PASSAGE
- LP-NavOA: https://github.com/shenqiqishi/LP-NavOA-code · GuideWalk: https://guide-walk.github.io/GuideWalk
- ViBe https://arxiv.org/abs/2609.09918 · TANGO https://arxiv.org/abs/2609.09158 · PLAT https://arxiv.org/abs/2609.25754 · S³ https://arxiv.org/abs/2609.26420 · RECAL https://arxiv.org/abs/2609.16405 · RoM-Nav https://arxiv.org/abs/2609.19272 · HOTICE https://arxiv.org/abs/2609.25363
- GR00T-WBC PRs: https://github.com/NVlabs/GR00T-WholeBodyControl/pull/267 , /pull/232 , /pull/240 , /pull/213 · MotionBricks: https://nvlabs.github.io/motionbricks/
