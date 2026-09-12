# Sensor history and repeated legal decisions

`motion2scene_observation_history.py` implements an opt-in sensor representation for
Motion2Scene. It accepts immutable range rays in world coordinates and delivered
capture times, retains a finite static-scene history, and reprojects that history
into a gravity-aligned egocentric grid at each decision. Root translation and yaw
change the projection; they do not change the stored observations.

Each horizontal cell has separate floor and ceiling heights and separate
observed/unknown masks. Upward-facing returns below their sensor origin supply
floor candidates; downward-facing returns above their origin supply ceiling
candidates. The conservative highest floor and lowest ceiling are retained.
An occupied endpoint without a suitable normal remains occupied without creating
a floor or ceiling label. Both surfaces can occupy the same horizontal cell.
The vertical gap between two observed surfaces is not a free-space certificate.

The grid also retains three-dimensional occupied endpoint evidence, sparse free
ray samples, and unknown voxels. A valid no-return observes only its sampled ray
segment; invalid or missing depth pixels must be omitted. Returns stop free-space
sampling before the hit. Occupied evidence dominates retained free samples and
expires with the history. Dynamic obstacles require another clearing/motion
model. The history must be reset between episodes.

The current simulator sensor casts 65 rays from one body-mounted position:
body offset `(0.2, 0, 0.4)` m, 4 m range, elevations
`[-75,-45,-20,-10,-5,0,2.5,5,7.5,10,15,30,60]` degrees and azimuths
`[-30,-15,0,15,30]` degrees. Sensor origins and directions use the full body
quaternion. Nearest returns preserve occlusion; only the robot is excluded by
identity. Normals are **ideal PhysX collision-query normals**, not estimated
normals from a noisy real depth sensor. The earlier 35-ray empty-scene smoke is
retained separately and does not establish thin-beam visibility for this fan.

`transition_timing_eligibility` checks the first relevant observation against
sensing latency, an empirically supplied transition duration, margin, obstacle
arrival forecast, and actual legal entry times. A late or absent observation
stays ineligible. The helper is a constructor/teacher timing screen: it does not
provide obstacle parameters to the student or imply that stopping is supported.
The runtime records evidence from which first visibility can be audited; it does
not yet estimate transition duration or obstacle arrival itself.

## Runtime and collection

The opt-in classes are
`motion2scene_closed_loop_execution.ClosedLoopEnvCfg` and
`ClosedLoopRecorderCfg`. They retain the frozen tracker and two equal-duration,
same-phase reference libraries. Decisions recur at control ticks, while switching
still requires the existing entry phase `[0.2,0.4]` s or return phase
`[3.3,3.5]` s and the existing joint/root reference-jump guards. A refused command
retains the current motion; it is not a stop. These classes support one encounter,
not independently reset multi-obstacle courses or general navigation.

Modes `scripted`, `always_walk`, `always_adapt`, and `learned` share the same sensor,
feature representation, and legality masks. `forced` provides exact-tick teacher
replays with `encounter_action` 0/1 and `forced_entry_time_s`. The binary library
can use a separately qualified reference such as d085, but the binary policy does
not select among several alternate references online.

For the 29-DOF robot the student receives 100 values: four corridor bands with
floor/ceiling fractions and heights, upper occupancy, lower sampled free evidence,
and unknown fraction; measured gravity, velocities and joint state; and clock,
active skill, observation age, and legality. This compact summary loses some
distinctions retained by the full grid. Unknown surface values have explicit
zero observed fractions. No beam parameter, object identity or physical outcome
label is a student input.

The learned NPZ contract has schema `motion2scene_history_linear_v1`, ordered
`feature_names`, normalization `mean`/`std`, `weights[D,2]`, `bias[2]`, and
`classes=[0,1]`. Loading verifies the artifact SHA256, dimensions, finite values,
positive standard deviations and exact feature names. Illegal actions are masked
before selection.

`scripts/research/motion2scene_collect_history.py` prepares a new immutable
manifest from named existing scene cells, then executes and analyzes it:

```bash
PYTHONPATH=.:scripts/research .venv_research/bin/python \
  scripts/research/motion2scene_collect_history.py prepare \
  --out /absolute/path/new-acquisition \
  --template /absolute/path/template.json --cells scene_a scene_b \
  --modes forced_walk forced_adapt --entry-time-s 0.3
PYTHONPATH=.:scripts/research .venv_research/bin/python \
  scripts/research/motion2scene_collect_history.py run \
  --out /absolute/path/new-acquisition
```

Every rollout exports the sensor rays, measured state, features, legal masks,
requested/applied actions and transition guards in `reactive_interface.json`,
plus `closed_loop_features.npz`, physical contact traces and the loaded reference
bank. The current implicit-actuator runtime logs its PD effort estimates and
marks actual actuator effort unavailable; those estimates are not reported as
measured mechanical work or battery energy.

Result rows retain passage failures and first-episode scoring. Matched teacher
records require exact physical approach prefixes, direct pre-action state and
feature agreement, and correct guarded command execution. Admission is separate
from passage: a valid failed branch is necessary negative evidence. Neither-pass
conditions and exact passage-time ties do not create a preferred-action target.
Both training and evaluation must keep the sensor and feature configuration fixed.

The first teacher labels compare committing to walking with a fixed adaptation
schedule. In the repeated runtime, action 0 can instead mean waiting for a later
legal entry. Consequently this first binary fit is an integration imitation
experiment, not a validated optimal timing teacher or dataset aggregation.
Learning when to wait requires measured future schedule branches from matched
states, including the continuation after each action.

## Validation and current evidence

The new perception/policy tests cover dual surfaces, unknown regions, occlusion,
world registration after translation/yaw, stale/delayed packets, callback failures,
legal timing, model schema/hash checks and masking. The existing ray-observer and
action-contract tests also pass: 21 tests in total at initial integration.

The initial 35-ray forced-walk empty-scene runtime completed 199 recorded frames
with 100 finite features per frame, observed floor in every frame, no invented
ceiling, zero delivered observation age, and no reference switches. This smoke
checks runtime integration only.

The dense 65-ray paired corpus `m2s-history-d085-acquisition-v1` completed four
physical branches (3,184 physics steps; 167.7 seconds recorded acquisition time).
It supplies two admitted teacher states with exactly matching physical prefixes,
pre-action robot state and 100-dimensional features at the 0.30 s decision:

| Development condition | Walk | d085 at 0.30 s | Teacher |
| --- | --- | --- | --- |
| Absent obstacle | Pass, 2.44 s | Pass, 2.86 s | Walk, lower measured passage time |
| New contrast scene | Fail | Pass, 2.96 s | d085, verified successful branch |

The contrast's first upper occupancy is delivered at phase 0.02 s and its first
ceiling surface at 0.22 s. At the 0.30 s decision the absent and contrast examples
differ in ceiling fraction/height, upper occupancy and unknown fraction. The
sensor therefore distinguishes these two conditions while switching remains
legal. This does not establish a general remedy for observation aliasing.
The absent scene's far wall produces upper returns after 1.98 s; an upper return
is generic occupied evidence, not an object-identity detector.

`m2s-history-d085-policy-evaluation-v2` completed eight actual episodes using
learned, scripted, always-walk and always-adapt policies on these same two layouts
with a different simulation seed, 6,368 recorded physics steps and 323.85 seconds
of recorded acquisition time:

| Policy | Passage | Absent passage time | Contrast passage time |
| --- | --- | --- | --- |
| Learned | 2/2 | 2.46 s | 2.98 s |
| Scripted | 2/2 | 2.46 s | 2.98 s |
| Always walk | 1/2 | 2.46 s | Failure |
| Always adapt | 2/2 | 2.88 s | 2.98 s |

The learned and scripted policies retain walking in the absent scene and request
d085 at 0.20 s in the contrast scene, returning at 3.30 s. They match the constant
adaptation policy's passage and save 0.42 s on the absent condition. The learned
policy does not outperform the strong script. These are development integration
results; they must not be described as unseen-layout generalization or evidence
that the curriculum improves learning. `runtime_audit.json` checks feature trace
shape and state preservation across requested reference switches; its original
all-packet causal interpretation is superseded by the chronology audit below.

### Sensor chronology correction

The installed loader produces 199 samples from the 120-frame, 30 Hz source:
its 50 Hz interpolation excludes the source endpoint. The recorded reference
spans 0 through 3.96 s. Isaac records the physical state and contacts, then
advances the command and captures the sensor packet. The original 200-step
driver limit therefore records 199 valid physical rows but resets the robot
before capturing the final same-index sensor packet. The final physical row
remains valid; its paired sensor packet belongs to the reset state.

`m2s-sensor-chronology-audit-v1/result.json` preserves the raw artifacts and checks
all 31 modern binary/multi episodes: each has 199 valid physical rows and exactly
198 eligible sensor packets. The final packet is ineligible because its phase
wraps to 0.02 s and its recorded robot state differs from the physical row. Every
earlier packet matches root position/quaternion and recorded joint state exactly;
its command phase is one control tick later than the physical reference phase.
The 0.20/0.30/0.40 s teacher packets and reported physical passage labels are
unaffected. Use `audit_sensor_alignment` and its explicit per-packet mask when
exporting or learning from these captures. New bounded collectors stop one tick
earlier and avoid the post-recording reset. A zero physical `reset_count` alone
does not establish that every sensor packet is aligned.

The retained v1 attempt failed before sensor/trajectory recording because its
loader compared a bare digest with the canonical `sha256:` artifact format. The
loader was corrected and regression-tested; the policy and training labels were
unchanged for v2. Its separate 10.33 s initialization cost remains recorded and
is not counted as a passage failure.
