# Phase 1 design: closed-loop SONIC-planner baseline, task schema v2, open-loop posture tracking

*September 23, 2026. Code line numbers were checked at HEAD `b9c4d69`. Tags: [M] measured, with a receipt in `workspace/`; [C] computed from repo files; everything else is proposed.*

## 1. Goal and scope

**In scope**
- **Phase 0.7 GPU run.** Release SONIC tracks 25 planner posture clips and 120 open-loop goal clips (B1-OL) at 3 DR seeds.
  - This run **has already been executed** (§7). What remains is fixing the readout, registering the run, and deciding the gate.
- **1.1 Task schema v2.**
  - Tasks no longer require a reference clip.
  - XY goal metric.
  - A family registry covering contact whitelists, fall rules, deadlines and success rules.
  - `confirmation` and `test` splits.
  - Scoring and receipts for v1 tasks stay byte-identical.
- **1.2 `PlannerBaselineCallback`.**
  - The frozen tracker (release, or 8192×500) runs in `teacher_mode` and is driven by `DeployPlannerRuntime`.
  - The loop is closed through the measured pelvis.
  - Includes a parity test against native loading.
- **1.3 Physical smoke test.** 4 tasks × 10 seeds × {release, 8192×500} × {nominal, DR}.

**Out of scope, but the design must not block them**
- **1.4 multi-instance harness.**
  - All per-robot state lives in per-env objects: runtime, controller, drift map, lib writer and carrier key.
  - The ORT worker is stateless, so it can be pooled.
  - The writer addresses rows as `length_starts[motion_ids[e]] + t`.
  - The only single-env assumptions are the existing ones: `direct_scene_runtime.py:87` and `scene_env.py:27`.
- **1.5 benchmark.**
  - Schema v2 already carries the F0–F4/E family fields, layout ids, mirror pairs, control goals and an unbounded-obstacle profile.
  - `build_nav_bench.py` ships with `--family smoke` only.
- **Excluded from Phase 1:**
  - the P1 waypoint arm (it sprints; §4);
  - crawl families in scoring (their registry entries exist but stay provisional);
  - the batched planner (I1).

## 2. Driving the frozen tracker from planner output

### 2.1 Choice: carrier clip overwritten in place ahead of the cursor (mechanism a)

**What the policy sees**
- The G1 encoder reads only two inputs:
  - `command_multi_future`: reference joint positions and velocities for 10 frames at cursor + {0, 5, …, 45}; frame 0 is the current frame.
  - `motion_anchor_ori_b_mf`: `quat_inv(measured pelvis) · reference root quat`.
  - Sources: release `config.yaml:74-78`; `commands.py:436-442, 2089-2111, 3499-3517`.
- So reference root XY never reaches the actor. Reference yaw relative to the measured pelvis does.

**Why writing into the motion lib covers everything**
- Every consumer (encoder, critic, terminations, metrics, reset, `body_pos_relative_w`) reads through integer-index getters: `motion_lib_base.py:577-645`.
- There is no interpolation cache on this path; `get_motion_state` is used only by `hindsight_training/validate.py`.
- One consistent in-place write therefore keeps all consumers coherent. It also allows the parity test the roadmap requires.

**Rejected alternatives**
- **(b) An encoder pre-hook or property patch.** It changes only the action path. The critic, terminations and metrics keep reading the stale carrier. It bypasses the native ±0.05 orientation noise and cannot be checked against native loading. Keep it only as a debugging aid.
- **(c) Rebuilding the lib on every replan.** `load_motions` reloads every motion, creates an `mp.Manager`, runs gc and `cuda.empty_cache`, and changes the length metadata (`motion_lib_base.py:1045-1178, 1626-1642`). That is not viable at 10 Hz.

### 2.2 Timing: where the write happens

Isaac Lab steps in this order (`manager_based_rl_env.py`):
1. physics;
2. terminations (`:203`);
3. rewards;
4. reset;
5. `command_manager.compute` (`:232`), where the cursor advances (`time_steps += 1`, `commands.py:3439`);
6. observations (`:238`).

Consequences:
- A row written after `act_inference` and before `env.step` is already visible in the observation that step returns. Latency is zero, the same as the deploy's order: observation → policy → `CurrentFrameAdvancement`.
- The hook is the existing `_record_teacher_query(env, teacher, task, observation, action)` at `direct_scene_runtime.py:174`. It runs after `act_inference` (`:171`) and before `env.step` (`:181`).
- The action path therefore needs **no edit to the base loop**. The alternative, a new `_before_step` hook, is cosmetic.
- Do not use `eval_step`: it runs after `env.step`, which adds one tick of lag.

### 2.3 Index mapping and invariants

Notation:
- `S = (t_S, ψ_S)` places the planner frame in the scene: start XY and yaw. The v2 canonical start makes `S` the identity.
- `Q_e[0:L]` is a CPU shadow stream in the scene frame (MuJoCo qpos, float64), initialised to the carrier rows.
- Invariant: **lib rows == native load of `Q_e`**, on all rows.

Per tick `k`, inside `_record_teacher_query`:
1. Read the cursor: `s = motion_start_time_steps + time_steps`, after the base clamp at `:166-169`.
   - Assert `s == k`. This proves the clamp never engaged.
2. If `k % 5 == 0`, run the controller to get `cmd` (§4).
3. Step the runtime: `f = runtime.step(cmd)` (`planner_runtime.py:619-632`).
   - Assert `S·f == Q_e[k]` to 1e-9. The frame the runtime says is observed at tick `k` must be the row that `obs(k)` was built from.
4. If `runtime.blends` increased, or the IDLE re-adapt edited the held frame (`:603-616`), set
   `Q_e[k+1+j] = S·runtime.motion[min(current_frame+j, len-1)]` for every j up to L.
   - Frames beyond the plan are padded with the plan's last frame, as the deploy clamps.
   - They are never filled with leftover carrier content.
5. Run the writer (§2.4).
6. On every tick, run a cheap GPU check: `lib.dof_pos[k+1]` must equal the Isaac-order joints of `Q_e[k+1]` to 1e-6.

### 2.4 FK and velocities: the loader's own code path

**Window per write event**
- `c0` is the first row whose qpos changed:
  - `s+3` after a blend, because `cross_fade` keeps rebased frames 0–2 unchanged (`w_new = 0` at `f = 2`; `planner_runtime.py:357-379`);
  - `s+1` after an IDLE re-adapt edit.
- `E` is the first row of the constant hold.
- Recompute on FK window `W = [max(0, c0-18), min(L, E+18))`.
- **Write rows `[max(0, c0-9), L)`.**
  - Rows from `E+9` onward are stationary. Every velocity there is exactly 0 (asserted), so they are filled by copying row `E+9`.
- Why 9 rows: `np.gradient` (±1 frame) plus `gaussian_filter1d(σ=2)` (radius 8) (`torch_humanoid_batch.py:727-752`). The forward differences for `dof_vel`, angular velocity and feet (`:449-450`, `motion_lib_base.py:1673-1699`) fall inside that margin.
- Rows behind the cursor get new velocities only, which keeps parity exact. No observation reads them again.

**Call chain, identical to `load_motion_with_skeleton` (`motion_lib_base.py:1916-1926, 2219`)**
1. `entry = qpos_to_sonic_motion_entry(Q_e[W], source_fps=50, canonicalize_horizontal_origin=False)` (`kimodo_motion_adapter.py:144-184`).
   - `Q_e` is already in the scene frame, so the default placement arguments (origin, yaw 0) are the identity.
2. `trans, _ = lib.fix_trans_height(pose_aa, trans, fix_height_mode=cfg)`. This is `no_fix` in release, but the writer calls it anyway.
3. `m = lib.mesh_parsers.fk_batch(pose_aa[None], trans[None], return_full=True, fps=50, target_fps=50, interpolate_data=True, use_parallel_fk=lib.use_parallel_fk)`.
4. `feet_l, feet_r = lib.foot_detect(m.global_translation, 0.0005, 0.05)`.
5. Reorder exactly as the loader does (`:1644-1664`):
   - DOFs: `[:, mujoco_to_isaaclab_dof]`;
   - bodies: `full = x[:, mujoco_to_isaaclab_body]`;
   - quaternions: xyzw → wxyz;
   - sliced tensors: `full[:, lib.body_indexes]`.
- Never use `research/planner/g1_geometry.py` here. It is built from the Isaac URDF with Dex3 hands, not the motion-lib MJCF.

**Tensors written: every per-frame tensor that `load_motions` creates (`:1524-1560`)**
- `dof_pos`, `dof_vel`;
- `body_{pos,quat,lin_vel,ang_vel}_w` **and** their `_full` twins. These are separate storages created by two rounds of advanced indexing, so both must be written.
- `feet_l`, `feet_r`;
- `root_linv_vel_w`, `root_ang_vel_w`, `body_pos_b`.

At bind time, enumerate every lib tensor whose first dimension equals the total frame count, and assert that the set equals this list.

**Binding the tensors**
- Bind in `_begin_task` (`:95`). That is after `set_is_evaluating(True)` (`:91`), which reloads every tensor (`manager_env_wrapper.py:1092-1112`), and after `reset_all` (`:92`).
- Re-read tensors through `lib.<attr>` on every write, and assert that `data_ptr()` is unchanged since binding.

**Cost**
- About 110 rows of CPU FK per blend. The kinematic P0 run made a median of 17 calls per 30 s.
- About 36 rows per IDLE re-adapt tick.
- Budget: ≤5 ms per write, measured in WP3.

**Velocity convention**
- Use the lib's convention: differentiate the blended positions. This is what native loading produces and what the tracker was trained on.
- The deploy instead blends each plan's own velocities (`localmotion_kplanner.hpp:500-516`; `g1_deploy_onnx_ref.cpp:3229-3241`). The difference is about (50/8)·|q_new − q_old| rad/s inside the 8-frame fade.
- Record this as a declared deviation (see open decisions).

### 2.5 Carrier

**How it is built**
- There is one carrier for all canonical-start tasks.
- `runtime.reset()` uses the standing context: `DEFAULT_ANGLES`, height 0.78874, planner seed 1234 (`planner_runtime.py:58-62, 295-307, 513-532`).
- Its IDLE plan (60–73 frames) is followed by a hold of the last frame out to **L = 3100 frames (62 s)**.
- It is saved with `save_sonic_motion_file(key="planner_idle_carrier_v1")` (`kimodo_motion_adapter.py:187-199`), plus `metadata.pkl`.
- It sits alone in its own directory, because `scene_native.sh` loads `dirname(native_motion.path)` (`:15`) with one motion (`:35`).

**Why L = 3100:** 3000 max ticks (`direct_scene_runtime.py:157-159`) + 45 future frames + 2 for the `total−2` clamp (`:166`) + 53 frames of margin. The roadmap's 60 s would freeze the cursor near the end of long episodes.

**Checks and caveats**
- The carrier is built in `.venv_sim` with the same ORT version and thread count.
- `_begin_task` re-runs `runtime.reset()` through the worker and asserts equality with the carrier rows to 1e-5. This doubles as a determinism check.
- The robot spawns from carrier frame 0 with no randomization (`commands.py:3208-3217`). The v2 validator asserts that frame 0 equals `start_xyz`/`start_wxyz`.
- The standing soles start about 3 cm above the floor, so a small initial drop is expected and accepted.

### 2.6 Termination and cursor handling

**Already handled by the base runtime**
- It runs the native terminations, then zeroes both buffers (`:135-151`).
- It clamps the cursor at `total−2` (`:166-169`).
- It raises on any reset (`:182-183`).
- It stops on fall, contact, goal hold or deadline (`:191-202`).

**Added by the subclass**
- In `_begin_task`, it wraps `termination_manager.compute` *before* the base captures it at `:105`. The base then calls the wrapper.
- The wrapper records per-term flags for `anchor_pos`, `anchor_ori_full`, `ee_body_pos` and `foot_pos_xyz` as **diagnostics only**. `anchor_ori_full` (a 25.6° full-3D bound that includes yaw) is expected to fire in 180° turns.
- It also records the measured pelvis quaternion at the same pre-reset boundary as `roots`.

**Resets:** single-env runs have no mid-episode reset; the base raises instead. In 1.4, a per-env reset must restore the pristine carrier rows and re-seed that env's runtime and `Q_e`.

### 2.7 Frame and drift math (SE(2): xy + yaw)

**Poses**
- The measured pelvis after `env.step(k)` is `m_k = ((robot_anchor_pos_w − env_origins).xy, h(robot_anchor_quat_w))`.
  - `h(q)` is the heading of the pelvis x-axis projected onto the floor plane. It equals `yaw_from_quat` when upright and stays well-conditioned in crawls (`planner_runtime.py:681-684`).
- `r_k = (f_k.xy, h(f_k.quat))` is the planned root, in the planner frame, of the frame the tracker was driven with at tick k. Pairing it with `m_k` introduces at most one tick of bias.

**Drift map**
- `D_k = r_k ∘ m_k⁻¹`, so `D(g) = R(Δψ)(g − t_m) + t_r` with `Δψ = wrap(ψ_r − ψ_m)`.
- The scene placement `S` cancels: `S⁻¹ ∘ (S∘r) ∘ m⁻¹ = r ∘ m⁻¹`. Use `runtime.motion` directly, in the planner frame.
- Smooth `D` over the last 10 pairs (0.2 s): take the mean of `t` and the circular mean of `Δψ`.

**What the controller receives**
- `state.xy = t̄_r`, `goal_p = D̄(g_scene)`.
- The offset is therefore `R(Δψ̄)(g − t_m)`: the distance is exactly the measured distance, and the bearing is the measured bearing plus Δψ̄.
- Movement directions map as `θ_p = θ_scene + Δψ̄`.
- Absolute facing targets, such as the sidestep facing hold, map through `S` only: `φ_p = φ_scene − ψ_S`. The tracker closes its own heading lag, so adding Δψ to a facing target would chase that lag.

**Guardrails**
- Never rebase `runtime.motion` onto the robot's pose. That would step the reference yaw, and the encoder sees yaw.
- XY drift is corrected only through the goal mapping. Yaw drift is corrected by the tracker itself.

## 3. Planner process integration

**Placement**
- `DeployPlannerRuntime`, the controllers and the writer run in-process in `.venv_native`. The planner package imports there without ORT, Isaac or torch (verified by the planner reader).
- Only `infer` crosses a process boundary, into a persistent **stateless ORT worker**: `/home/robotixx/GR00T-WholeBodyControl/.venv_sim/bin/python -m gear_sonic.research.planner.ort_worker`.
- The worker is required because `.venv_native` has no onnxruntime, and `OrtPlannerSession` refuses to run unless `CUDA_VISIBLE_DEVICES==''` (`ort_session.py:37-38`).

**Launch**
- `subprocess.Popen(..., stdin=PIPE, stdout=PIPE, stderr=<out>/ort-worker.log, bufsize=0)`.
- Environment: `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`, `PYTHONPATH=$KIT/vendor/sonic`.
  - The last one makes vendor `gear_sonic` shadow GR00T's copy.
  - It overrides the Isaac process's `OMP_NUM_THREADS=2` (`scene_native.sh:59-60`).
- `OrtPlannerSession(intra_op_threads=4, inter_op_threads=1)` (`ort_session.py:30-34`), built once per episode.

**Protocol**
- The worker first sends a handshake: ORT version, provider list, model sha256, and `INPUT_SPEC`.
- Requests are fixed binary frames of 784 B, the 11 tensors in `INPUT_SPEC` order.
- Replies are 9,220 B: a float32 (1, 64, 36) array plus an int32 count.
- The worker sets `sys.stdout = sys.stderr` after duplicating the protocol file descriptor, so stray prints cannot corrupt the stream.
- It exits on stdin EOF. That covers the base's `os._exit(0)` (`:235`) and the `timeout -s KILL` in `scene_native.sh:61`.

**Latency budget [M, planner reader]**
- Pipe overhead is about 0.01 ms. A call takes 28.0 ms median and 29.6 ms max. Startup takes 1.23 s. The worker uses 1.05 GB RSS.
- Calls are **synchronous**, so no sim time passes during a call, matching the deploy's TensorRT latency of 0 ticks (`latency_ticks=0`, `planner_runtime.py:497`).
- Worst case, with a replan on every planner tick, adds 28 ms per 5 ticks: about 11% of wall time at the best-case 49 ms/step. For P0 it is under 1%.
- Log `PlannerCall.inference_s` (`:536-557`).
- Require `late_plans == 0`.
- If the worker times out (5 s per call), the episode fails as an infrastructure failure: rerun at most twice, as in the roadmap's evaluation protocol (§10).

**Replan timing**
- `runtime.step` runs every 50 Hz tick. The planner thread runs when `tick > 0` and `tick % 5 == 0`, with context taken at `cursor+2` (`:559-589`).
- The controller runs *before* `runtime.step` on `k % 5 == 0`, as in `run_controller` (`goal_controllers.py:193-203`).
- Replan intervals follow the deploy: 1.0 s by default, 0.2 s for CRAWLING, 0.1 s for RUN (`planner_runtime.py:265-275`).

**Determinism**
- The planner seed input is fixed at 1234 (`:62, 400, 498`), the deploy default. The planner is therefore deterministic given its context and command.
- Per-episode randomness comes only from `++seed`: the DR draw and observation noise.
- The callback draws no torch RNG and never recomputes observations.
- Before the smoke run, check determinism by running the same episode twice at seed 70610 and diffing `trace.npz` and `planner-trace.npz`.

## 4. Controller P0cl (closed-loop P0)

A new class in `research/planner/closed_loop.py`. It leaves the kinematic `DirectionSpeedStopController` (`goal_controllers.py:85-136`) unchanged, so the kinematic study stays reproducible.

**Inputs:** at each 10 Hz tick, `KinematicState(time, xy=t̄_r, yaw=ψ̄_r, speed=measured planar speed)` and `goal_xy = D̄(g)` (§2.7). It never calls `observe(frames)`, which reads reference frames.

**Speed schedule**
- The same bands as kinematic P0 (`:78-82`):
  - WALK (mode 2, default speed; 0.86 m/s kinematic) above 2.5 m;
  - SLOW_WALK 0.4 above 1.0 m;
  - otherwise SLOW_WALK 0.2.
- Commands come from `locomotion_command(mode, movement_yaw, facing_yaw, speed)` (`planner_runtime.py:187`).

**Aim**
- Keep the kinematic ±5° deadband and the 0.3 m freeze radius, and apply both in the planner frame.
- New rule: re-aim only while the planned heading is steady (|Δψ_r| < 2° over 0.2 s).
  - Without it, the controller chases tracker lag during turns.
  - Every facing change forces a replan (`planner_runtime.py:564-565`).
- The initial command yaw is the first measured bearing mapped through D.

**Stop and hold: a state machine that replaces the permanent latch**
1. **APPROACH → STOP.** Command IDLE when `distance ≤ r_stop[band]`, or when `distance < 0.3` and the goal is more than 90° behind.
   - `r_stop` comes from a **physical calibration**: the `stop_distance_trial` protocol (`goal_controllers.py:265-291`) run in Isaac and measured on the pelvis.
   - The kinematic 0.035 m includes neither tracker lag nor sway.
2. **SETTLE.** Wait for the measured planar speed to stay ≤ 0.10 m/s for 25 ticks.
3. **Decide.**
   - If the XY error is ≤ 0.25 m, **HOLD**: IDLE forever. Scoring ends the episode after a 50-tick hold.
   - Otherwise **RE-ARM**: SLOW_WALK 0.2 toward `D̄(g)`, stopping at `r_stop[SLOW 0.2]`. At most 2 re-arms.
- IDLE is a static mode, so no timer replans occur while stopped (`:570-582`). That matches the deploy.
- `measured_lower_body` is wired to `robot.data.joint_pos[:, G1_ISAACLAB_TO_MUJOCO_DOF][:, 0:12]`, after asserting the joint-name order (`kimodo_motion_adapter.py:23-28`). This makes the deploy's IDLE re-adapt live; the lib rows are then rewritten on every hold tick.

**Sidestep variant**
- `facing_policy="hold"`: the facing target is fixed at the start yaw mapped through `S`, and movement follows the bearing.
- Lateral stop radii are calibrated separately.

**Hook for posture modes (not exercised in Phase 1)**
- `posture_policy(task, measured_pose, distance) -> PostureChoice(mode, height, staged) | None`. It defaults to `None`, meaning the standing bands.
- Crawl entry would use `staged_crawl_schedule` (`planner_runtime.py:660`).
- Each mode needs its own `r_stop` calibration; the controller refuses to run a mode without one.

**Parameters**
- Frozen before the smoke run.
- Tuned only on physics seeds 70610–70612 (collection block), never on dev or confirmation seeds.
- The actor profile is `planner_P0cl_v1`.

**P1 and P1c**
- Excluded from the baseline.
- P1 caches its waypoint once, so under drift it would land at goal + drift. It also sprints: peak 2.2 m/s median, 4.1 m/s max (planner report).
- P1c can be mapped the same way later: waypoint = `D̄(g)`, heading + Δψ̄.

## 5. Task schema v2

**Schema string:** `bfm_known_map_navigation_task_v2`.

**Fields**
- `task_id`
- `split` ∈ {train, development, confirmation, test}
- `family` ∈ {smoke, F0, F1, F2, F3, F3free, F4, E}
- `layout_id`, `layout_seed`, `goal_index`, `control_goal`, `mirror_of`
- `observation_profile` ∈ {`known_map_5_primitives_v1`, `known_map_primitives_unbounded_v1`}
  - the second is for the planner only: it allows more than 5 obstacles and skips `navigation_observation`
- `world_frame`, `anchor_body` (unchanged)
- `start_xyz`, `start_wxyz`, `start_posture="planner_idle_default_v1"`
- `goal_xyz` (z = start z), `goal_metric` ∈ {xy, xyz} (xy for v2 tasks), `goal_heading_rad`: null
- `goal_tolerance_m` 0.25, `terminal_speed_mps` 0.1, `hold_ticks` 50
- `hold_speed_metric` ∈ {xyz, xy}, default xyz
- `success_rule` ∈ {`goal_hold_v1` (longest good run, stop early), `terminal_hold_v1` (run to the deadline; the last `hold_ticks` must all be good)}
- `deadline_ticks`, plus `deadline_rule {kind: geodesic_speed_v1, geodesic_m, speed_mps: 0.4, slack_s: 4.0, posture_changes, posture_change_s: 2.0, sensitivity: {0.3, 0.5}}`
- `contact_profile {id, threshold_n: 1.0, floor_allowed_bodies, obstacle_allowed_bodies: []}`
- `fall_rule {kind, params}`
- `obstacles`, `scene_usd_path`, `scene_usd_sha256`, `source_scene` (generator binding)
- `native_motion` (the carrier binding; the key name is kept so `scene_native.sh:15` works unchanged), `native_motion_role: carrier`, `motion_key`, `carrier_frames`
- `reference`: null or a binding
- `generator {script, commit, config_sha256}`

**Registry, in `tasks.py`**
- Contact profiles (Isaac body names; the 30 names are listed in any nav-8192 `trace.npz` `body_names`):
  - `standing_feet_v1` = {left, right}_ankle_roll_link, exactly the v1 whitelist (`direct_scene_runtime.py:112-116`).
  - `crawl_v1` adds `*_knee_link` and `*_wrist_{roll,pitch,yaw}_link`.
  - `elbow_crawl_v1` also adds `*_elbow_link`, the shoulder links and the thigh links. This profile is provisional: 0.7 geometric candidates include upper arms and thighs, and forces were not measured.
- Fall rules:
  - `pelvis_z_below_v1(0.25)` is v1.
  - `crawl_v1` is provisional. The 0.7 elbow crawl pelvis minimum was 0.14–0.19 m [M], so 0.25 m cannot apply to crawls.

**Validation**
- Signature: `validate_task(task, *, verify_files=True, allow_v2=False)`.
  - v1 keeps its current path at `tasks.py:233-268`, untouched.
  - v2 without `allow_v2` raises the existing "Unknown schema" error. This keeps the v1-only consumers from half-processing a v2 task: `navigation_data`, `navigation_recovery_data`, `train`, `navigation_panel`, `prepare_scene_probe`, `scene_qualification`, `online`.
- `_validate_task_v2` checks:
  - the enums and the 0.25 / 0.1 / 50 success profile;
  - the deadline, recomputed from `deadline_rule`, must equal `deadline_ticks` and be ≤ 3000;
  - `carrier_frames ≥ deadline_ticks + 47`;
  - contact and fall profiles are registered, and every body name is in `G1_BODY_NAMES`;
  - `require_start_goal_clearance` at 0.35 m (`tasks.py:105-124`);
  - `navigation_observation` is called only under the 5-primitive profile;
  - every obstacle shape is one `write_collision_scene` supports (no cylinder; `:177-226`).
- File checks:
  - the scene, source-scene and carrier sha256s;
  - the reference sha256, only if `reference` is not null;
  - the carrier pkl: its key equals `motion_key`, fps is 50, and frame 0 matches `start_xyz`/`start_wxyz` to 1e-6 (xyzw → wxyz);
  - its directory holds exactly one `*.pkl`.
- `task_profile(task)` returns the v1 constants for v1 tasks: xyz metric, 3-D speed, ankle-roll whitelist, 1.0 N, pelvis z < 0.25, `goal_hold_v1`, and the `:214` termination string.

**Exact changes**
- **`tasks.py`**
  - `:14`: add `SCHEMA_V2`, the splits, `G1_BODY_NAMES`, `CONTACT_PROFILES`, `FALL_RULES`.
  - `:233`: add the dispatch.
  - After `:268`: add `_validate_task_v2`, `task_profile` and `task_body_names`. `task_body_names` reads the reference npz for v1 and returns `G1_BODY_NAMES` for v2.
  - `:271-275`: the key becomes `task.get("motion_key") or f"hindsight_{motion_id}"`.
- **`direct_context.py`**
  - `:32-65`: `score_navigation_task` takes its profile from `task_profile`. It uses `goal_distance(metric)`, lifted from `rescore_xy.py:70-77` (`:46`), the profile's threshold (`:53`), and the terminal-hold rule.
  - For v2 only, it appends the root metrics from `rescore_xy.py:155-190`. v1 dicts stay byte-identical.
- **`direct_scene_runtime.py`**
  - `:86`: `allow_v2=True`, then build the profile.
  - `:112-116` and `:184-187`: whitelists and thresholds from the profile.
  - `:139-148`: record the pelvis quaternion and planar speed.
  - `:191`: `fell = fall_check(profile, root, quat)`.
  - `:195, :201, :214`: use profile values.
  - `:220-227`: add `root_wxyz` and `speed_xy` to the trace for v2 only.
- **`scene_env.py:15-17`**: `allow_v2=True`, and `names = task_body_names(task)`.
- **`stage_config.py`**
  - `:21-27`: add `planner=…planner_baseline_runtime.PlannerBaselineCallback`.
  - `:73`: `teacher_mode = a.mode in ("teacher", "planner")`.
  - New flags: `--controller P0cl`, `--planner-onnx`, `--planner-calibration`, `--parity-dump`.
  - A `planner` block with the ONNX path and sha, the worker python, and the calibration path and sha.
  - `actor_profile = "planner_P0cl_v1"`.
  - `collect_to_deadline` is true exactly when `success_rule == "terminal_hold_v1"`. The base allows this in teacher mode only (`:79`).
- **`scene_native.sh:15`**: fail fast when `native_motion` is missing. Today `dirname ''` gives `.`, and the loader silently picks up arbitrary pkls under `vendor/sonic`.
- **Unchanged:** `run_stage.sh`, because v2 results keep every key its `:30` reads, and `build_tasks.py`, which stays v1.

**Acceptance:** all 49 nav-8192 task JSONs re-validate, and `rescore_xy.legacy_mismatches` (`:193-205`) over all 497 legacy traces through the new scorer gives **0** mismatches.

## 6. Parity test and unit tests

**Parity**
- **CPU unit test (`decoupled_wbc/tests/test_planner_lib_writer.py`).**
  1. Load the carrier into a CPU `MotionLibRobot` with the release motion-lib cfg, on the `planner_validate_clips.py` path.
  2. Run `DeployPlannerRuntime` with recorded ORT outputs as a fixture `infer`, through a scripted schedule: IDLE → walk → 180° turn → stop → IDLE hold with synthetic measured legs.
  3. Apply the writer.
  4. Natively load the final `Q_e`.
  5. Compare every per-frame tensor on **all** rows [0, L):
     - positions and quaternions to 1e-6 (quaternions sign-invariant);
     - velocities and feet to 1e-4.
  - With the ±9/±18 margins and the loader's own code path, no seam rows should exceed tolerance. Any that do are listed as seam rows and fail the test. The roadmap allows excluding them, but only after inspection.
- **In-sim parity.**
  - With `--parity-dump`, `_complete_task` writes `planner-lib.npz` (the lib rows [0, L)) and `planner-qe.npy`.
  - `scripts/research/planner_parity_check.py` then natively loads `Q_e` on CPU and applies the same tolerances.
  - The smoke gate requires this to pass on every episode.
- **Optional:** compare the 640-D encoder input offline, via `TrackingCommand.create_offline` + `set_motion_state` (`commands.py:654-826, 3689-3700`).

**Unit tests (CPU)**
1. Writer covers the set of per-frame lib tensors.
2. Stale-handle refusal after a reload.
3. Tick mapping `f_k == Q_e[k]` over 3000 ticks.
4. Carrier equals `runtime.reset()`.
5. `DriftMap` algebra:
   - `D(m) = r`;
   - `S` cancels;
   - circular mean;
   - `h(q)` at 80° pitch.
6. P0cl with measured = reference and re-arm disabled issues the same commands as `DirectionSpeedStopController` on the 200-goal study. Run under `.venv_sim`, comparing against `goal_roots_P0_x1.npy`.
7. P0cl re-arm: an undershoot triggers one re-arm, and the cap holds.
8. v2 validation: good and bad cases, including a short carrier, frame 0 ≠ start, a bad deadline, and an unknown profile.
9. v1 unchanged: 49 tasks and 497 traces.
10. v2 scoring: XY metric, terminal hold, whitelist, fall rules.
11. ORT worker (skipped if `.venv_sim` is absent):
    - echo round trip;
    - stdout-pollution guard;
    - exit on EOF;
    - sha-mismatch refusal.
12. `scene_native.sh` guard: exits non-zero when `native_motion` is missing.

## 7. Phase 0.7 open-loop tracking run

**Already executed [M]**
- `workspace/phase0/run_07_tracking.sh` ran at HEAD `b9c4d69`: 6 launches, all `exit_code 0`, finished 18:25:48Z (`tracking07/done.log`). Walls were about 42 s (posture) and 69 s (goal), with a 5.9 GB GPU peak.
- It used exactly this parametrization of `run_eval.sh`. At `:26-38`, `split=custom` needs `MOTION_DIR`, `NUM_ENVS` and `OUT_DIR`.

```bash
KIT=/home/robotixx/motion2scene-training; P=$KIT/workspace/phase0/planner; O=$KIT/workspace/phase0/tracking07
for s in 92610 92611 92612; do
  SEED=$s MOTION_DIR=$P/clips      NUM_ENVS=25  OUT_DIR=$O/posture-release-$s bash $KIT/scripts/teacher_review/run_eval.sh release custom
  SEED=$s MOTION_DIR=$P/goal_clips NUM_ENVS=120 OUT_DIR=$O/goal-release-$s    bash $KIT/scripts/teacher_review/run_eval.sh release custom
done
```

Configuration of this run:
- Staged release checkpoint and `config.yaml` (sha `e6bdab3f…`).
- `NUM_ENVS` equals the clip count, and `sort_motion_keys` (`:60`) makes env i the i-th sorted stem.
- Native release terminations, with all 5 active (`evaluation.log:155`).
- DR startup events stay on and `push_robot` is removed (`eval_agent_trl.py:156-164`).

**Readout [M]** (untracked `vendor/sonic/scripts/research/planner_tracking_readout.py`):
- **Posture clips: 75/75 completed, with 0 termination flags recorded.** This contradicts the review reader's prediction that `anchor_ori_full` would kill the crawls.
- Head top, p10 over time:

  | Mode | Clips | Head top p10 (m) |
  |---|---|---|
  | 8 (crawl) | direct / staged | 0.62–0.67 |
  | 14 (elbow crawl) | direct / staged | 0.29–0.31 |
  | 22 (stealth walk 2) | 1 | 1.02–1.05 |
  | 18 (stealth walk) | 1 | 1.20–1.22 |

- Elbow-crawl pelvis minimum was 0.14–0.19 m.
- Crawl floor-contact candidates: hands and knees (mode 8); plus forearms, upper arms and thighs (mode 14).
- **Goal clips (B1-OL):**
  - P0: 120/120 completed; final XY error to the kinematic endpoint 0.85 m median, 1.13 m p90; **0/120 within 0.25 m**.
  - P1: 114/120 completed; the 6 failures are `foot_pos_xyz` or `ee_body_pos` terminations; 2/120 within 0.25 m.
  - P1c: 120/120 completed; 1/120 within 0.25 m.
  - Conclusion: closed-loop replanning is required, which is the second half of the 0.7 gate.

**Remaining work (CPU only)**
1. **Fix the readout.** The gate needs the head-top **maximum over the steady window** (last 3 s), as in `posture_metrics` (`planner_kinematic_study.py:441-541`). p10 and min are optimistic for an overhang.
   - Also report XY error to `goal_xy`, not only to the kinematic endpoint.
   - Then commit the script.
2. Command:
   `cd vendor/sonic && CUDA_VISIBLE_DEVICES= PYTHONPATH=. ../../.venv_native/bin/python scripts/research/planner_tracking_readout.py --runs $O/posture-release-926{10,11,12} --manifest $P/clips/manifest.json --out $O/readout-posture-v2.json`
   and the same for goal clips with `goal_clips/manifest.json`.
3. **Metrics per clip × seed:**
   - native completion and first failing term;
   - pelvis minimum and steady height;
   - steady head-top maximum;
   - geometric floor-contact candidates (ε 0.03 m);
   - final and maximum root drift;
   - for goal clips, final XY error to the goal and final planar speed.
   - Aggregate per mode with Wilson intervals (`scripts/experiments/eval_stats.py`).
4. **Gate decision:** one of modes 8, 14, 18 or 22 must keep the steady head top ≤ 1.10 m with ≥ 80% completion.
   - This provisionally passes for 8, 14 and 22: all completed (6/6, 6/6, 3/3), with p10 far below 1.10 m, but the steady maximum is still pending. Mode 18 fails.
   - n=3 for mode 22 has a Wilson lower bound of about 0.44. Replicate copies are optional.
5. **Record the run** in `experiments/registry.jsonl` as a post-hoc entry with a declared deviation: it was not pre-registered. Rescan `seeds.yaml`, which burns 92610–92612.
6. **No relaxed-termination profile is needed.** Keep it only for P1 references.

## 8. Smoke protocol (1.3)

**Tasks** (v2, family `smoke`, split `development`)
- Common setup: canonical start (0, 0, z₀) with yaw 0; zero obstacles, i.e. a floor-only USD from `write_collision_scene`, so the filter count is 1; `standing_feet_v1`; `pelvis_z_below_v1`; the shared carrier.

| Task | Goal (x, y) | Controller | Rule | Deadline (ticks) |
|---|---|---|---|---|
| `smoke-idle-10s` | (0, 0) | constant IDLE | `terminal_hold_v1` | 500 |
| `smoke-walk3-stop` | (3, 0) | P0cl | `goal_hold_v1` | 575 (3/0.4 + 4 s) |
| `smoke-turn180-walk3` | (−3, 0) | P0cl | `goal_hold_v1` | 575 |
| `smoke-sidestep-1m` | (0, 1) | P0cl, facing hold 0 | `goal_hold_v1` | 325 |

**Grid**
- Seeds: dev **92613–92622** (fresh; 92610–92612 were used by 0.7).
- Trackers: `NAV_PACKET=workspace/nav-release` (release) or `workspace/nav-8192` (8192×500).
- DR: default. Nominal: `M2S_EXTRA_ARGS='++eval_remove_events=[physics_material,add_joint_default_pos,base_com,randomize_rigid_body_mass,push_robot,compliance_force_push]'`. This is the prior nominal launch string (`scene_native.sh:47-49`).
- Total: 320 episodes.

**Launch** (one chain per cell)
```bash
NAV_PACKET=$KIT/workspace/nav-release bash scripts/navigation_distill/run_stage.sh planner \
  $KIT/workspace/phase1/smoke/release-dr-$S $S - --controller P0cl --parity-dump -- $KIT/workspace/phase1/smoke/tasks/*.json
```

**Order**
1. Bring-up at seed 70610 (§9 WP7).
2. The gate cells: release × DR × {idle, walk}, 20 episodes.
3. The rest.

**Wall time [C]**
- Fitted scene-task cost: 0.193 s/step + 38.8 s startup, over 223 nav episodes. The 0.7 runs on a free GPU were much faster.
- Estimate: about 2 min per episode, so about 45 min for the gate cells and about 11 h serial (≈5.5 h in two chains) for all 320.

**Gate (pre-registered before launch)**
- In-sim parity passes on every episode.
- Release under DR: idle **10/10** and walk-and-stop **≥ 8/10**.
- Reported but not gated: turn and sidestep, all 8192×500 cells, and the nominal cells.
- **Kill rule (roadmap):** if walk-and-stop fails in ≥ 5/10 DR seeds after parity passes:
  1. First replay the same commands through the MuJoCo deploy path.
  2. If the failure is real, record it, drop B3 and I1, and continue with I2, I3 and I4.

**Reported per episode**
- success;
- time to goal, final and minimum XY error, path ratio;
- maximum |t_r − t_m| drift and |Δψ|;
- blends, calls, `late_plans` and p99 inference time;
- re-arms;
- native-term diagnostic flags;
- maximum contact force.

## 9. Implementation plan

- **Worktrees.** Every package is developed in its own worktree under `.claude/worktrees/` (ignored by git).
- **Why this matters.** `run_stage.sh` starts a **new process per task**. Editing `tasks.py`, `direct_context.py`, `direct_scene_runtime.py`, `scene_env.py`, `stage_config.py` or `scene_native.sh` on `main` would change the remaining episodes of any live scene-task chain.
- **When to merge.** Merges into `main` happen only when `pgrep -f 'gear_sonic/(eval|train)_agent_trl.py'` finds nothing. New files are always safe.
- **Running Isaac from a worktree.** Symlink the git-ignored `.venv_native` and `workspace` into it. Then check that `gear_sonic.__file__` resolves to the worktree, since `scene_native.sh:7, 59` sets `PYTHONPATH`.

| WP | Content | Files | Depends on | Effort | Parallel? |
|---|---|---|---|---|---|
| 0 | Close out 0.7: readout fix, commit, registry, seed rescan, STATUS row | `planner_tracking_readout.py`, `registry.jsonl`, `seeds.yaml` | — | 0.3 ED | yes (CPU) |
| 1 | Schema v2, profiles, scorer, 49/497 acceptance | `tasks.py`, `direct_context.py`, `direct_scene_runtime.py`, `scene_env.py`, tests | — | 1–1.5 ED | worktree A; merge-gated |
| 2 | ORT worker, client and protocol tests | new `planner/ort_worker.py`, `ort_client.py` | — | 0.5 ED | worktree B |
| 3 | `MotionLibWriter` and CPU parity test | new `planner/lib_writer.py`, `test_planner_lib_writer.py` | — | 1.5–2 ED | worktree C |
| 4 | `closed_loop.py` (DriftMap, h(q), P0cl, posture hook) and tests | new file | — | 1 ED | worktree D |
| 5 | Carrier builder and `build_nav_bench.py --family smoke` (USD, tasks) | new scripts (carrier built in `.venv_sim`) | 1, 2 | 0.5 ED | after 1–2 |
| 6 | `PlannerBaselineCallback`, stage_config mode, scene_native guard | new `planner_baseline_runtime.py`; `stage_config.py`, `scene_native.sh` | 1–4 | 1–1.5 ED | merge-gated |
| 7 | Bring-up at seed 70610: timing, determinism rerun, in-sim parity, physical `r_stop` calibration (3 bands + lateral × 3 IDLE phases × 2 seeds, 70610–70611) → freeze parameters | runs only | 5, 6 | 0.5 ED + ~1 GPU-h | — |
| 8 | Pre-register the smoke gate, run it, report | registry, report | 7 | 0.5 ED + ~6 h wall | 2 chains |

Total: about 7–9.5 ED and about 7 GPU-h. Packages 0–4 can run in parallel.

## 10. Risks and fallbacks

**Timing, reloads and indexing**
- **Writing after `env.step` adds one tick of lag.** Enforced by the hook placement (§2.2) and the `f_k == Q_e[k]` assertion.
- **`set_is_evaluating` reloads the lib.** Bind after `:92` and assert `data_ptr` on every write.
- **Writing only `body_*_w` or only `_full` leaves inconsistent consumers.** The tensor-coverage test catches this.
- **Seam errors from windowed velocities.** Handled by the margins and the full-row parity test. Fallback: rewrite the whole stream from `c0-18` to L on every blend, at about 3000 rows of FK; measure whether it still fits the budget.
- **Future window reading leftover carrier rows.** Prevented by hold-padding to L. The carrier must be at least deadline + 47 frames.
- **Envs sharing a motion id in 1.4.** Each env gets its own carrier key: `override_num_motions_to_load = num_envs`.

**Planner, controller and ORT**
- **Planner frame drifts from the robot.** Corrected by D (§2.7).
- **Heading jitter makes P0 replan every 0.1 s.** Handled by the deadband, the steady-heading gate and 0.2 s averaging. Fallback: raise the deadband to 10°.
- **Kinematic stop radii are invalid in physics.** Handled by the calibration in WP7 and the re-arm loop. Fallback: SLOW_WALK 0.2 only within 1 m.
- **IDLE re-adapt is untested in closed loop.** A joint-order mistake would corrupt the reference. Guarded by the name-order assertion. Fallback: `measured_lower_body=None`, as in the kinematic study, declared as a deviation.
- **ORT worker hazards:** a stdout print corrupting the protocol, GR00T's `gear_sonic` shadowing ours, orphan processes, or a crash. Mitigated by the handshake, the stdout redirect, exit on EOF, and a loud failure capped at 2 infrastructure reruns.
- **Memory:** 1.05 GB per worker. This matters for the 1.4 pool.
- **Joint-velocity deviation from the deploy at seams.** Declared. Resolved by the MuJoCo replay if the kill rule fires.

**Environment and experiment hygiene**
- **Running from a worktree picks up `main`'s code.** Check `gear_sonic.__file__` before any GPU launch.
- **The 0.7 run was not pre-registered, and its readout uses p10.** Register it post hoc and recompute the steady maximum before any gate claim.
- **8192×500 is sha-bound to its own packet.** Use its packet `ids.json` unchanged. The carrier is tracker-agnostic.

## 11. Open decisions

Listed in `open_decisions`. The most consequential:
- the joint-velocity convention;
- the hold-speed metric;
- the turn and sidestep deadline allowances;
- the P0cl re-arm policy;
- the crawl fall and contact rules;
- where the ORT process lives.


---

## Decisions taken for implementation (September 23, 2026)

Open questions resolved by the project lead (Claude, under the user's standing approval of the roadmap). Recommended defaults are used unless noted.

1. **Seam joint velocity.** Differentiate the blended positions (the motion-lib load convention, so parity is exact). Deviation from the deploy's per-plan velocity blend is declared.
2. **Crawl terminations.** Keep the native terminations. Phase 0.7 showed 0/75 posture clips terminated, so there is no relaxed profile.
3. **0.7 gate statistic.** The steady-window maximum head top, over one pre-declared window (t > 2 s). The committed readout script is extended to output it, and the numbers are regenerated.
4. **0.7 seeds 92610–92612.** A registry entry is added post hoc with a declared deviation; the seeds count as used.
5. **Hold speed.** 3-D pelvis speed (v1 parity) for success; planar speed is recorded alongside.
6. **Smoke deadlines.** Run with `max_steps` above the deadline and score post hoc at the declared deadline and at the 0.3 and 0.5 m/s sensitivities. The smoke gate uses the 0.3 m/s sensitivity. The sidestep and the turn are reported, not gated.
7. **P0cl stop policy.** The closed-loop B1 arm uses settle → re-arm (at most 2 re-arms). Re-arm counts are reported. The latched variant is kept only as a compatibility flag for the kinematic-equivalence test.
8. **Planner seed.** Fixed at 1234 (deterministic planner). Only physics and observation noise vary with `++seed`.
9. **Initial planner context.** The standing context and one sha-bound carrier (declared deviation from the deploy's measured-joint context).
10. **Benchmark start pose.** Canonical: origin, yaw 0, one carrier. Heading diversity comes from goal bearings.
11. **Crawl fall rule and whitelists.** Provisional; they are decided in Phase 1.5 after contact forces are measured.
12. **ORT placement.** A `.venv_sim` stdin/stdout worker process; `.venv_native` is not modified. It must follow the critique's fd-level stdout guard and explicit environment.
13. **Hook point.** Reuse `_record_teacher_query`; do not edit the base action loop.
14. **Bring-up and calibration seeds.** Collection block 70610–70612. Smoke gate seeds come from the dev block, starting at 92620.
15. **Idle success rule.** `terminal_hold_v1`: run to the 500-tick deadline; the last 50 ticks must be within 0.25 m XY and at ≤0.1 m/s, with no fall and no contact.
16. **Smoke grid.** Gate cells first: release × DR × {idle, walk 3 m and stop} × 10 seeds, i.e. 20 episodes per task pair, 40 in total. The full 160-episode grid (not 320) is deferred to the Phase 1.4 multi-instance harness.
17. **v2 opt-in.** `ACCEPTS_V2` is a class attribute, `False` by default and `True` only in `PlannerBaselineCallback`. Legacy callbacks never see v2 tasks.

## Critique findings that the implementation must address

- **[major]** The carrier, Q_e and runtime initialisation cannot meet the §2.3 invariant `S·f == Q_e[k]` to 1e-9 together with the §2.5 rule that `runtime.reset()` may differ from the carrier by up to 1e-5. The carrier pkl stores float32 values (`root_trans_offset`, `pose_aa` rotvec, `root_rot` xyzw), while the runtime frames are float64 blends of float32 planner output at fractional 30 Hz positions. Two outcomes follow. If Q_e comes from the pkl, the 1e-9 check fails within the first few ticks. If Q_e comes from the in-sim `reset()`, rows that are never rewritten differ from the lib by up to 1e-5, so the 1e-6 all-rows parity fails.
  - *Fix:* Save the carrier's float64 50 Hz qpos as a sha-bound .npy next to the pkl, built by the same function. In `_begin_task`, call `reset()` and set `runtime.motion` to that array. Initialise Q_e from the same array padded to L. Use the `reset()` re-run only as a determinism probe: record its max difference and require bitwise equality or declare the deviation. Never feed its output into Q_e.
- **[major]** As specified, both parity tests (the CPU test via the `planner_validate_clips.py` path, and `planner_parity_check.py`) build `MotionLibRobot` from the raw `config.yaml` motion_lib_cfg. That config lacks the mappings TrackingCommand injects in-sim: `mujoco_to_isaaclab_{dof,body}`, `body_indexes_data` and the `isaaclab_to_mujoco_*` keys. The lib then takes the no-reorder branch: `_full` aliases the sliced tensor, `body_quat_w` stays xyzw, and DOFs stay in MuJoCo order. The CPU test would validate a different layout from the one the writer targets in Isaac, and the in-sim check would fail every episode or have to duplicate the writer's reorder logic.
  - *Fix:* Build CPU libs with the same injection as `TrackingCommand.create_offline` (commands.py:703-720: G1Converter mapping plus `body_indexes_data` from `cfg.body_names`). At bind time in Isaac, dump `lib.m_cfg` keys and values and have both checkers assert they match. Add `assert lib.body_pos_w_full.data_ptr() != lib.body_pos_w.data_ptr()`.
- **[major]** The bind-time coverage assertion ('enumerate every lib tensor whose first dimension equals the total frame count; assert the set equals this list') fails on the first run. `load_motions` also creates per-frame tensors that are independent of qpos: `_motion_aa`, all zeros because the carrier entry has no 'beta'. Because the release config sets `smpl_motion_file: dummy`, `smpl_data` is `[None]*N` rather than None, so zero-filled `_motion_smpl_poses`, `_motion_smpl_joints` and `_motion_smpl_transl` are also created.
  - *Fix:* Split the coverage set into rewritten tensors and declared constants (`_motion_aa`, `_motion_smpl_*`). Assert that the constants are all-zero and have unchanged `data_ptr`, include them in parity, and fail on any unlisted per-frame tensor.
- **[major]** The smoke deadlines from `geodesic/0.4 m/s + 4 s` do not fit P0cl's own speed bands, which put the last 1.0 m in SLOW_WALK 0.2 at 0.166 m/s measured. Walk-and-stop, the gated task that also triggers the kill rule, has about 1 s of slack even with perfect tracking, so tracker lag, SETTLE or any re-arm (which takes 2 s or more) causes a deadline failure that could be misread as a baseline failure. The 1 m sidestep lies entirely inside the 0.2 band and cannot finish within 325 ticks.
  - *Fix:* Pre-register deadlines derived from the controller's band speeds plus measured tracker lag, or gate on the 0.3 m/s sensitivity (700 ticks for walk3, 383 for the sidestep). Run with `max_steps` above the deadline and score at the declared deadline and the sensitivities post hoc, so every failure is attributed to either time or accuracy. Either make re-arm fit the budget or drop it from the smoke controller.
- **[major]** The launch passes a single `--controller P0cl` to every task in the chain, and schema v2 has no field for choosing a controller per task. The smoke table needs constant IDLE for the idle task and a facing hold for the sidestep. With plain P0cl, the idle task (goal equals start) takes its IDLE facing from `atan2` of an offset of about 0, which is an arbitrary direction; the facing change forces a replan and the robot turns in place. The sidestep goal (0,1) becomes turn 90° then walk forward, and it would still be reported as 'sidestep', so the results are silently mislabelled.
  - *Fix:* Carry a per-task baseline control spec, either in a stage-dir map keyed by `task_id` or in a v2 field excluded from `navigation_request_sha256`, with controller, facing_policy and yaw. Alternatively, run separate chains per controller. In P0cl, keep the start yaw whenever the initial distance is below the freeze radius.
- **[minor]** The Phase 1.4 claim 'each env gets its own carrier key: `override_num_motions_to_load = num_envs`' is wrong for eval. Evaluation loads `_num_unique_motions`, the number of distinct pkl stems in the directory, and assigns `motion_ids = arange(num_envs) % _num_motions`. With one carrier file, every env shares the same rows, so a per-env writer would corrupt the other envs. The override is read only on the training path.
  - *Fix:* For 1.4, write N copies of the carrier under distinct file stems (for example `planner_idle_carrier_v1_e{i}`) into the motion directory. Assert `curr_motion_keys` has length `num_envs` and that `length_starts` are distinct.
- **[minor]** In directory mode the motion lib keys motions by filename stem, globs `**/*.pkl` recursively excluding `metadata.pkl`, and ignores the key inside the pkl. The validator's check 'pkl key equals motion_key' therefore does not bind what gets loaded, and 'directory holds exactly one *.pkl' conflicts with the planned `metadata.pkl`. The base runtime also never asserts which key was actually loaded.
  - *Fix:* Require the filename stem to equal `motion_key` and count pkls recursively excluding `metadata.pkl`, or drop `metadata.pkl`, since the 0.7 clip directories have none. In the planner `_begin_task`, assert `env._motion_lib.curr_motion_keys == [motion_key]`.
- **[minor]** The writer call chain has a shape error. `fk_batch(pose_aa[None], ...)` returns batched (1,T,J,3) tensors, and the loader squeezes them through EasyDict before calling `foot_detect`. The design calls `lib.foot_detect(m.global_translation, …)` without squeezing, so it would difference along the size-1 batch axis.
  - *Fix:* Apply the loader's exact squeeze to every fk output before `foot_detect` and the reorder. Better, factor the post-fk block of `load_motion_with_skeleton` into a helper that both the loader and the writer call.
- **[minor]** Calling `fix_trans_height` 'anyway' on a window is wrong for any mode except `no_fix`: it computes the height offset from the first frame of whatever sequence it receives, here the window's first row rather than stream frame 0.
  - *Fix:* Assert `lib.fix_height == FixHeightMode.no_fix` at bind time and skip the call. If another mode is ever needed, compute the offset once from stream frame 0 and apply it as a constant.
- **[minor]** Unit test 6 (P0cl with measured equal to reference reproduces `DirectionSpeedStopController` on the 200-goal study) cannot pass. P0cl uses t̄_r averaged over 10 pairs, a 0.1 s lag of about 8.6 cm at WALK, and the new steady-heading gate changes when it re-aims. The claim that 'distance is exactly the measured distance' also holds only for the unsmoothed pair.
  - *Fix:* Add compatibility flags (no smoothing, no steady gate, latch instead of the state machine) for the equivalence test, and test the new features separately. State the smoothed-distance lag explicitly and let the r_stop calibration absorb it.
- **[minor]** The facing mapping contradicts itself. The design says absolute facing targets map through S only, because adding Δψ̄ would chase tracker lag. But forward walking uses `locomotion_command(mode, θ_p)` with `facing_yaw=None`, which sets facing equal to movement, i.e. θ_scene + Δψ̄.
  - *Fix:* Choose one convention for forward walking, either facing = θ_scene + Δψ̄ (consistent with D) or facing = θ_scene, then document it and test it in the DriftMap unit tests.
- **[minor]** The ORT worker's stdout guard (`sys.stdout = sys.stderr`) only protects Python-level prints; native libraries that write to fd 1 would still corrupt the binary protocol. The worker would also inherit the Isaac process environment, including any library paths Kit sets.
  - *Fix:* In the worker, do `proto = os.dup(1); os.dup2(2, 1)` before any imports and write frames only to `proto`. Launch with an explicitly built `env=` (PATH, HOME, `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS`, `PYTHONPATH`; no `LD_LIBRARY_PATH`, `LD_PRELOAD` or `PYTHONHOME`). The handshake should report `sys.executable`, `onnxruntime.__file__` and `gear_sonic.__file__`.
- **[minor]** Setting `allow_v2=True` at direct_scene_runtime.py:86 opens v2 tasks to every subclass: teacher, full, nav, recovery and reentry. Those subclasses read v1-only fields late. For example, StoppingTeacher reads `task['motion_id']` in `_complete_task`, after a full episode, and navigation_ablation reads `task['reference']`.
  - *Fix:* Make v2 an opt-in class attribute (base `ACCEPTS_V2 = False`, `PlannerBaselineCallback` True) and pass it through to `validate_task`.
- **[minor]** The v1 acceptance test is weaker than the claim of byte-identical results. `legacy_mismatches` compares only the 9 LEGACY_FIELDS with atol 1e-9. Rescoring stored traces cannot detect a change in the runtime's per-tick stop decision (fall, contact threshold, goal stop), because each trace is already truncated at the old stop.
  - *Fix:* Also compare full `task-result.json` dicts, excluding only fields known to differ, for all 497 traces. Add a test that replays each trace's per-tick roots, speeds and forces through the new stop logic and asserts the same stop tick and reason.
- **[minor]** §7 is stale and the 0.7 gate numbers are not reproducible from committed code. The readout script is already committed (`dc0548f`), and the committed report and STATUS already state steady-phase (t > 2 s) median/max head-top values (STEALTH_WALK_2 max 1.064 m), but the committed script outputs only the minimum and p10. The design's 'last 3 s' window also disagrees with the report's 't > 2 s'.
  - *Fix:* Add the steady-window maximum to the committed script with one pre-declared window, regenerate the readouts, and reconcile the report and STATUS with the regenerated numbers.
- **[minor]** The worktree recipe (symlink `.venv_native` and `workspace`) misses a git-ignored symlink. `vendor/sonic/gear_sonic/data` points to `../../../workspace/vendor/sonic/gear_sonic/data` and holds the robot MJCF/URDF and scenes. A fresh worktree has no such path, so the motion-lib skeleton load (`assetRoot: gear_sonic/data/assets/...`) fails.
  - *Fix:* Recreate the `gear_sonic/data` symlink in each worktree. Run pytest with `PYTHONPATH=<worktree>/vendor/sonic`, and assert `gear_sonic.__file__` in the tests as well as before GPU launches.
- **[minor]** Several tolerances are fragile. Positions at 1e-6 absolute are about 1 ULP of float32 beyond 4 m, which the 1.5 goals reach at up to 8 m. The feet flags are binary thresholds, so a 1-ULP difference in position can flip a flag, an error of 1.0 against the 1e-4 tolerance. The per-tick check compares float32 `dof_pos` after an axis-angle to quaternion round trip against float64 Q_e.
  - *Fix:* Compare the per-tick GPU row against the writer's own CPU-computed float32 row, which should be bitwise equal. Use a relative tolerance or 1e-5 for positions beyond 4 m. Require exact equality of the feet flags only where parity is bitwise, and otherwise list and inspect the flips.
- **[minor]** The design understates the single-env assumptions (it lists only :87 and scene_env:27), and its episode arithmetic is wrong. The grid of 4 tasks × 10 seeds × 2 trackers × 2 dynamics settings is 160 episodes, not 320, so the 11 h and 5.5 h wall-time estimates are doubled.
  - *Fix:* List all single-env sites in the 1.4 hand-off, index env explicitly in the new code, and correct the episode count and wall-time budget.

Critique summary: The core mechanism holds up against the code. The frozen tracker is driven by writing planner rows into the motion lib ahead of the cursor, from `_record_teacher_query`. Where the design breaks is in the details: the carrier's float precision, the layout of the parity harness, the list of tensors to rewrite, the smoke-task deadlines and controller wiring, the Phase 1.4 multi-env claims, and a stale Phase 0.7 section. I read the code without modifying anything and launched nothing on the GPU.

**Claims I checked and found correct (at least 17):**
1. The G1 encoder reads only `command_multi_future_nonflat` and `motion_anchor_ori_b_mf_nonflat` (release config.yaml:75-78). The encoder choice is g1 only (config.yaml:286-289, scene_native.sh:24). The decoder takes tokens plus `actor_obs`, so reference root XY never reaches the actor.
2. `motion_anchor_ori_b_mf` is `quat_inv(robot_anchor_quat)·ref_root_quat` as a 6-D representation (commands.py:2089-2105, observations.py:1022-1043). It carries ±0.05 noise (config.yaml:567-570).
3. Future frames: `frame_skips = 0.1//0.02 = 5.0` (commands.py:422, 436-442). Indices are clipped at `motion_num_steps-1` (commands.py:3499-3517). So "cursor + {0,…,45}" is right.
4. Isaac step order: terminations at manager_based_rl_env.py:204 (the design says :203), commands at :232, observations at :238. `CommandTerm.compute` runs `_update_metrics` then `_update_command` (command_manager.py:152-167). `time_steps += 1` is at commands.py:3439. `body_pos_relative_w` is recomputed every step by `_resample_command([])` after the increment (commands.py:3453, 3403-3421). Observations are lazy properties, so a write made before `env.step(k)` is seen by obs(k+1) with zero lag.
5. Hook placement: `act_inference` :171, then `_record_teacher_query` :174, then `env.step` :181. `_begin_task` (:95) runs before `original_compute = manager.compute` (:105), so wrapping the termination manager works.
6. In eval, `motion_start_time_steps=0` and `time_steps=0` (commands.py:3061, 3069-3074), and reset randomization is skipped (commands.py:3209-3240). So s==k holds.
7. `set_is_evaluating` reloads the lib (manager_env_wrapper.py:1092-1112), and `reset_all` runs after it.
8. `_full` and the sliced tensors are separate storages (motion_lib_base.py:1644-1664).
9. `cross_fade` uses `blend_start = gen_frame - current_frame = 2`, so `w_new = 0` for f ≤ 2 and c0 = s+3 (planner_runtime.py:366-373, 586).
10. The ±9/±18 margins are exactly sufficient. `gaussian_filter1d(σ=2)` has radius 8, `np.gradient` reaches ±1, angular velocity and `dof_vel` are forward differences (torch_humanoid_batch.py:728-752, 449-450), and `foot_detect` is a forward difference (motion_lib_base.py:1673-1699). Copying row E+9 over a constant hold is exact.
11. Protocol sizes of 784 B and 9,220 B follow from INPUT_SPEC/OUTPUT_SPEC (motion2scene_planner_adapter.py:23-39). The CPU guard is at ort_session.py:37-38. `.venv_native` has no onnxruntime; `.venv_sim` has ORT 1.23.2, and numpy is 1.26.4 in both, so pickles are compatible.
12. L=3100 is correct, and a 60 s carrier would trip the `total-2` clamp (direct_scene_runtime.py:166).
13. The `dirname ''` hazard is real (scene_native.sh:15, 55). The loader globs `**/*.pkl` recursively (motion_lib_base.py:391-395).
14. The nominal-dynamics string matches prior launches (`workspace/confirm-v1/eval/probe-approach-c2-nominal-*`).
15. Phase 0.7 receipts: 75/75 posture, 0 flags; 354/360 goal clips, 6 flagged; 5 active terminations (evaluation.log:155); `done.log` reads 18:25:48Z.
16. The deploy blends per-plan velocities (g1_deploy_onnx_ref.cpp:3229-3241) that are forward differences ×50 (localmotion_kplanner.hpp:500-516).
17. Every other cited planner_runtime, goal_controllers, kimodo adapter and motion-lib line range matches the code.

**Stale or wrong:**
- HEAD is `e4cd005`, not `b9c4d69`.
- The 0.7 readout script and report were already committed in `dc0548f`.
- 4 tasks × 10 seeds × 2 trackers × 2 dynamics settings = 160 episodes, not 320.

**Main risks, detailed in the issues:**
- The carrier and Q_e cannot meet the 1e-9 check as written.
- The CPU parity test would load the lib without the Isaac mappings.
- The tensor coverage assertion fails on zero-filled SMPL and `_motion_aa` tensors.
- The smoke deadlines leave about 1 s of kinematic slack for walk-and-stop and make the sidestep infeasible.
- One `--controller` flag cannot express the idle and sidestep variants.
- The 1.4 carrier-per-env mechanism is not how eval assigns motions.

Nothing found silently changes legacy v1 behaviour, provided `allow_v2` is scoped and the v1 acceptance test is strengthened.
