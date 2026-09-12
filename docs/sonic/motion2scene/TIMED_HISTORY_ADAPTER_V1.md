# Timed sensor-history adapter, development v1

This interface has completed nine full development teacher episodes with
298 aligned control records and 1192 physics steps per episode. The input six-second bank is now qualified under the separately
registered environment-contact criterion at
`m2s-environment-contact-qualification-v1/registry.json`
(SHA-256 `719fbf50c8a8060365fc10cb175bc50ebedb420719b3912a90f2c8a5b03d58ed`).
All three new30×36 contact recordings reconciled exactly to independently ordered
net forces and measured zero undesired environment force. Their inherited
self-contact is separately reported; original strict non-foot net-contact
failures remain intact. This is an explicit new contact criterion, not satisfaction
of the previous strict criterion.

The nine completed development teachers are preserved at
`m2s-timed-duration-teachers-v1/result.json` (SHA-256
`de1c280f2024f022c1eec88db2d55810c9a6f67627664d511d30810762fa40c3`).
All nine measurements are admitted independently of passage. The three groups
have exact shared physical prefixes, 106D features and 65 raw rays at entry.
For neutral/short/sustained, empty passes 3/3, the short beam passes 2/3, and the
long passage passes 1/3. There are 10,728 recorded physics steps and 174,330 ray
queries, with no physical reruns. These are development capability and teacher
measurements; no held-out or actual learned-policy performance follows from them.

The neutral short-beam branch saw the beam surface before entry, but first saw
the underside at 0.42 s, after the 0.28 s delivered entry capture. The long-beam
branch saw both at the first capture. Early generic upper occupancy therefore
must not be described as early ceiling visibility in every scene.

Two postexecution analysis corrections were registered in
`m2s-timed-duration-teachers-v1/analysis_correction_registration.json`:
read native `physics:collisionEnabled` rather than CollisionAPI presence, and
serialize the visibility boolean as a Python boolean. Original registrations,
source snapshots and all nine raw episodes remain unchanged; corrected analysis
has a separately bound source snapshot. No physical criterion or threshold was
relaxed by these fixes.

## Runtime and evidence gate

`motion2scene_timed_history_execution.TimedHistoryCommand` subclasses
`LongScheduleCommand` through its `_observe_schedule` and `_requested_schedule`
hooks. The recorder and environment subclass the new explicit-counterpart
`EnvironmentContactRecorder` and `EnvironmentContactEnvCfg`. The command parent
still loads the complete bank, checks actual frame counts and
reference jumps, preserves physical state when changing libraries, enforces the
single-entry schedule, aborts before reference wrap, and records beam contacts,
all-body normal-force sums, foot-to-floor forces, and partial aborted captures.
The new recorder additionally retains every robot body against 36 exact
robot/floor/four-wall/beam counterparts and an independently checked native
body/filter mapping. Self-contact remains a separate measured report. The
original strict non-foot net-force diagnostic is retained, not used as a
substitute for environment contact.
The frozen tracker, existing K5 registry, and existing phase policies are unchanged.

The new adapter accepts the initial ordered bank `neutral`, short, sustained,
with both adaptation entries at tick 15 (0.30 s). Schedule identifiers and exact
return ticks come from the SHA-bound timed definition. For the current request,
short returns at tick 265 (5.30 s), sustained at tick 255 (5.10 s). No 3.30 s
return rule is inherited. No alternate-to-alternate switch, second entry,
reference padding, or protective stop is supported.

Every online mode, including simple baselines, must load a registry rebuilt by
`load_verified_registry` from complete SHA-bound physical evidence. Experimental
requests are allowed only through the explicit forced qualification path. The
collector is stricter: it requires a verified registry for every mode, including
forced teacher collection. It cannot turn a raw request into an online registry.

Modes are `always_walk`, `constant_short`, `constant_sustained`,
`scripted_sustained`, `learned`, and explicit `forced`. After entry, the chosen
option follows its complete declared return schedule. The scripted baseline
selects sustained adaptation only when an upper occupied endpoint was actually
observed in a policy corridor band. This is a development heuristic, not an
obstacle identity detector or a free-space guarantee.

## Observation and model contract

Each command tick issues exactly the same existing 65 body-mounted PhysX rays:
13 elevations and five azimuths, 4 m range, offset (0.2, 0, 0.4) m from the robot
root. Environmental identities are not selected; only the robot is excluded.
Hit normals are ideal PhysX query normals, not noisy reconstructed depth normals.
Missing normals cannot establish a floor or ceiling.

The existing causal floor/ceiling history retains independent floor and underside
surfaces per horizontal cell, observed and unknown masks, occupied endpoints,
and sampled free space. Its initial maximum age is 0.5 s / 26 frames, with zero
configured delivery delay. Robot pose transforms past measurements into the
current egocentric frame. The finite ray history remains sparse and cannot
resolve every perceptual ambiguity. Compact features lose information relative
to the raw grid; raw rays permit reconstruction and separate representation tests.

`motion2scene_timed_history_policy.expected_feature_names()` supplies exactly
106 names: the established 100 sensor/proprioception/binary-legality summaries,
three active-option indicators, and three option-legality indicators. The policy
receives no beam dimensions, physical outcome labels, or teacher scene state.
Raw 65-ray packets, normal-known masks, current robot state, legality, features,
and decision values are retained each tick. Floor/ceiling/occupied counts are
recorded separately. Both decision capture and final row reuse the exact
pre-action feature cache.

The optional value model NPZ schema is `motion2scene_timed_history_value_v1`, with
exact feature names, three option IDs, full timed-request digest, integer entry
tick 15, classes 0–2, mean/std vectors of length 106, a 106×3 matrix, and three
biases. `fit_timed_value_policy` reuses measured physical-regret regression with
default ridge coefficient 1e-6. It accepts only complete legal actual teacher
branches at the common neutral entry. Missing branches and illegal actions do
not become failed outcomes. Its fit report concerns the recorded finite table;
actual policy execution is still required to establish downstream performance.

## Reusable collector

`scripts/research/motion2scene_collect_timed_history.py` provides `prepare`, `run`,
and `analyze`. Preparation binds the actual verified registry, original request,
frozen controller, neutral motion, one scene definition, policy if used, runtime
seed, source closure, and full episode budget before physics. Every option must additionally bind a
complete qualified `environment_contact_audit`; the declared neutral self-pair
set is recovered from that SHA-bound evidence. It clones one
selected matched template cell. It never edits the scene or a locked layout.

A scene definition is an explicitly new development artifact:

```json
{
  "schema": "motion2scene_timed_history_scene_v1",
  "split": "development",
  "scene_id": "YOUR_REGISTERED_SCENE_ID",
  "scene": {"path": "/absolute/path/scene.usda", "sha256": "sha256:..."},
  "beam_collision_enabled": true,
  "beam": {
    "center_xy_m": [2.0, 0.0], "yaw_rad": 0.0,
    "length_m": 0.5, "width_m": 1.2,
    "thickness_m": 0.1, "underside_m": 1.25
  }
}
```

These numbers illustrate the schema; this document registers no physical scene
or solvability claim. The existing scene artifact must already have its declared
geometry. Analysis audits its actual imported collision flag, dimensions,
position, and orientation. Disabled-beam scenes retain their explicit collision
flag and a geometric downstream scoring plane.

Once a real verified registry exists:

```bash
.venv_research/bin/python scripts/research/motion2scene_collect_timed_history.py prepare \
  --out /absolute/new/development/run \
  --registry /absolute/verified/registry.json \
  --request /absolute/original/qualification_request.json \
  --scene-definition /absolute/new/development/scene.json \
  --template /absolute/matched/template/manifest.json \
  --cell neutral --policy-mode forced --seed 8731
```

Run and analyze use the same `--out` and registered source hashes. Failed attempts
remain present and are never automatically retried. Resource preflight pauses
only unlaunched cells. Partial physical counts are retained; sensor queries from
an interrupted callback remain explicitly unknown rather than being inferred
from the number of completed rows.

For an actual 299-frame bank, driver `max_steps=299` requests 298 captured rows
and 1192 physics steps. Sensor row i shares physical state with trajectory row i;
its post-update command phase is 0.02 s ahead of the trajectory reference phase.
The existing alignment helper checks every packet against recorded root, joint,
gravity and phase fields. Tick14 and tick15 receipts include causal capture/delivery times, observed ceiling
counts, upper occupied bands and known-normal counts. Those are sensor evidence
without obstacle identity; they do not independently establish a learnable
clearance decision. A separate privileged offline `audit_decision_visibility`
associates actual nearest ray endpoints with the already audited beam surfaces
and underside, and checks their causal delivery by entry. Other room hits do not
count as seeing that beam. This audit adds no truth to student inputs and does
not remove late-visible cases from the measured outcome table. The collector rejects incomplete alignment, validates
the full selected schedule independently, and preserves all raw physical data.

Passage alone is insufficient: the output also requires whole-horizon stability,
no undesired measured contact under the declared contact audit, and successful
completion of the exact return. Contact force scope is normal-force vectors for every declared body/counterpart
pair, reconciled against independently ordered net forces. Foot-floor support
is allowed; other floor contact and all wall/beam contacts use the registered
1 N criterion. Self-contact and pairs absent from the qualified neutral report
are reported separately. These are not individual contact points or friction.
The original strict net-force diagnostic remains in each result. Mechanical work is null for the implicit actuator; estimated PD effort
is retained. Passage time, total captured duration, and switch count are separate.

A forced collection creates three actual complete episodes. The resulting
`result.json["teacher"]` contains the one entry feature row, 3-action legality,
`passed`, `passage_time_s`, and `admitted` arrays, option identities, and prefix
receipts. Admission requires identical physical prefixes and entry observations
and is independent of physical success. Failed, fully measured branches can
therefore remain informative teacher outcomes. This single decision has complete
schedule outcomes; it does not approximate the value of waiting at an earlier
unqualified decision tick.

## Validation and remaining work

The focused contract, observation, alignment, readout and collector tests pass,
including regression tests for actual disabled-placeholder collision attributes
and visibility JSON serialization. Synthetic test registries remain software
fixtures, not evidence of robot capability. Black and Ruff checks pass. The new
106D sensor/recorder integration has now been exercised in all nine complete
physical episodes above. A fixed-lambda value fit on the three development
contexts is available separately; actual student execution is required before
making a learned traversal-performance claim. Repeated neutral decisions and
multi-beam collection use the separate new114D adapter described in
`TIMED_SCHEDULE_SENSOR_ADAPTER_V1.md`; those changes are not retroactively part
of these nine episodes.
