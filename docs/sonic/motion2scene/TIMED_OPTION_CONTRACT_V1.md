# Finite motion schedules and authored duration candidates

The new opt-in contract makes an option a complete entry/maintenance/return
schedule on a finite reference bank. It supports configurable ticks and longer
equal-duration references without changing the existing199-frame K5 registry or
its phase policies. It does not authorize clip wrapping, repeated entry, direct
adaptation-to-adaptation switching, arbitrary duration or protective stopping.

[Pure implementation](../../gear_sonic/dataset_generation/hallucination/motion2scene_timed_options.py)
and [tests](../../decoupled_wbc/tests/test_motion2scene_timed_options.py) contain no
simulator imports. Eleven focused and adjacent tests pass, as do scoped Ruff and
Black checks. No GPU work or physical qualification was performed by these helpers.

## Request versus verified registry

`validate_request` accepts only an explicit development qualification request.
References bind motion-file hashes and distinguish generated ancestry from
authored local-crouch edits of the neutral parent. Each option has its own identity,
reference, exact entry/return ticks, maintenance interval and recovery endpoint.
Two tested entries and two tested returns do not imply that all four combinations
are executable: each complete combination needs separate evidence.

`assert_loaded_bank` checks actual loaded identities,50Hz rate and equal frame
counts against the request. Anticipated source resampling is not a substitute for
this measured bank. `guard_before_reference_advance` rejects an increment that
would resample/wrap the clip. With299 loaded samples, the bounded driver should
capture298 physical rows and298 aligned command packets; its last physical
reference phase is5.94s and its last command/sensor phase is5.96s. The existing
one-tick sensor/physical phase offset remains explicit.

`TimedOptionState`, `legal_timed_actions` and `apply_timed_request` enforce one
entry per episode and forbid direct switching between adaptations. An experimental
request can drive only an explicitly marked forced qualification. Online use
requires `load_verified_registry`, which rebuilds the registry from bound evidence.
A completed recovery does not reset the entry count and therefore cannot silently
permit a second adaptation.

At an option's exact return tick, neutral is the mandatory request. Reference-jump
limits remain0.05rad at every joint and0.01m at the root. A refused return is recorded
as a failed request while the active reference remains unchanged. If execution
advances, the next tick raises a missed-return error. The qualification runtime
must preserve partial raw data and mark the episode incomplete before aborting.
Aborting a simulation is not a physically supported stop behavior.

## Evidence required for online promotion

`validate_evidence` checks a full physical audit against the request's canonical
digest, controller and motion hashes. It requires every pre-wrap command tick,
the complete200Hz step count, exact active-reference timeline, matched approach
through entry, precisely paired switches and state/clock-preserving jump guards.
Maintenance and the entire remaining recovery interval must appear in that timeline.
The neutral branch and every proposed online schedule require their own evidence.

The certificate binds trajectory, command interface, contact, clock, prefix and
geometry artifacts. Contact/pose/geometry audit results are upstream measurements;
the pure validator verifies their binding and complete timeline, not the simulator
or mesh calculations themselves. The long-schedule recorder supplies all-body net
contact-force streams and separately filtered foot-to-floor forces at200Hz. The
certificate must state that measurement scope and its audited thresholds. Net
body forces do not identify every contact pair and may combine opposing contacts;
stream completeness must not be described as a proof that every possible collision
was absent.

The height trace is the executed outer collision-envelope height above a declared
support plane, with its geometry audit bound. A requested maximum height can be
checked at every maintenance sample. An explicit null maximum is allowed for first
empty-scene execution qualification: it records the achieved envelope without
inventing a clearance requirement. Such a result can qualify a reference schedule
while supplying no proof of a chosen low-clearance duration. Compare the measured
trace with the neutral before claiming that the longer authored profile produces
a longer useful low-height response.

`make_verified_registry` checks the evidence-file hashes and their underlying
artifact hashes before promotion. Current K5 registries use their existing schema;
none is rewritten or automatically upgraded by this module.

## Six-second development carrier and retained proposals

The already qualified neutral is:

```
/home/linjiw/research-data/groot-wbc/m2s-longer-reference-development-v1/neutral.pkl
```

Its SHA-256 is
`ef33c2139f7a6278d670c012edc9db70671d08b6842e8757d15dc6a6b61b3990`.
The separate `m2s-longer-neutral-qualification-v2/result.json` records299 loaded
samples,298 aligned physical/sensor records and1192 physical steps without reset,
wrap or fall. This establishes one longer neutral carrier; adaptations, repeated
entries and complete overhead courses remain separate qualification gates.

[The authored-profile CLI](../../scripts/research/motion2scene_author_timed_profiles.py)
registered two edits before invoking `local_crouch`: station0.55, requested drop
0.085m, route half-windows0.18/0.48, existing ramp0.45, range retention0.94,
maximum joint excursion0.98rad and zero waist use. It retained all180 source
frames at30Hz, root XY and root orientation. No motion was padded or retimed.
The profile parameter is a **half-window in route progress**, not a duration in
seconds or a full window width.

The first authoring attempt stopped before either operator call because ordinary
linear interpolation of source joint angles differed from the real SONIC loader
by1.046mrad. The loader interpolates joint quaternions and converts them back to
axis-angle. The failed attempt is preserved in `m2s-authored-duration-profiles-v1`
with zero authored candidates and zero physics. A fresh V2 registration used native
CPU FK; its neutral result matches the actual loaded neutral bank exactly.

| Authored proposal | Source profile core | Reference silhouette drop | Return compatibility |
| --- | --- | ---: | --- |
|Short, half-window0.18|3.133–4.233s|84.99mm|0.30s entry /5.30s return, zero reference jump|
|Initial sustained, half-window0.48|2.233–5.233s|73.81mm, excursion capped|No legal return inside the6s horizon|
|Narrower sustained, half-window0.30|2.767–4.633s|75.94mm, excursion capped|0.30s entry /5.10s selected return|

These core intervals and drops describe authored reference geometry. They are not
executed low-clearance durations or traversal results. Independent checks against
the native robot MJCF found no static joint-limit violations in any of the three
authored references. That static check does not qualify dynamics or contacts.

The0.48 profile reaches the source endpoint. At the requested5.30s return its
reference jump is0.83888rad/0.06620m, and no later candidate return tick through5.94s
meets the guard. That ineligible reference and its full audit remain in
`m2s-authored-duration-profiles-v2`; it was not repaired by truncation, padding,
silent recovery or relabeling it as physically tested.

## Fresh narrower proposal and exact qualification request

[The followup CLI](../../scripts/research/motion2scene_author_sustained_followup.py)
registered exactly one new half-window0.30 proposal after the0.48 rejection. All
other operator parameters and the qualified source were fixed. Before authoring,
it also registered a deterministic return selection: the earliest native-reference
compatible50Hz tick at or after the later of5.0s and maintenance-end plus0.20s,
with at least0.30s recovery before the last recorded physical phase5.94s.

The complete33-tick audit selected tick255/5.10s, with joint jump0.04957223rad,
root jump0.00201106m and42 physical reference ticks/0.84s remaining. No physical
outcomes were used to select this tick. The short profile keeps its original
0.30/5.30s schedule.

The concrete experimental request is:

```
/home/linjiw/research-data/groot-wbc/m2s-authored-sustained-followup-v1/qualification_request.json
```

Request SHA-256:
`37c8c301756a3d4d5ec940a7acc50251611753134eba055e4259c1a156a06d18`.
It binds the neutral plus short/sustained motion files, controller, authoring
registration and deterministic tick audit. Short maintenance samples are ticks
157–211; sustained samples are139–231. Both record recovery through tick297, and
the physical acquisition checks the declared15-tick minimum recovery budget.
Both height bounds are explicitly null pending actual envelope measurements.

The short candidate's SHA-256 is
`b645d901bf6d3916d9204197b3c2fe09aeed6a9baaab5dba847d7e47237c746b`;
the new sustained candidate's is
`e3a35314be3b264b3da383240cc4cdb3bf06197392910251e1242fbc06564cb2`.
`audit.json` verifies the33 copied implementation sources, motion hashes, exact
route preservation, native joint limits and deterministic selection without more
authoring or physics. V2 authoring took4.207s for two profiles; the one-profile
followup took2.746s, using CPU FK and the existing authored operator.

The next integration uses a new forced long-schedule environment with the same
three loaded references in every neutral/short/sustained branch. It must preserve
all RNG streams during additional library loading, validate the actual bank,
invoke these guards at each command tick, record all physical streams and retain
any incomplete episodes. Promote only the schedules whose matched full-episode
physical evidence passes. The current phase-conditioned K5 learner is not a
policy for this new bank, and neither a second adaptation nor general indoor
navigation is supported by this request.
