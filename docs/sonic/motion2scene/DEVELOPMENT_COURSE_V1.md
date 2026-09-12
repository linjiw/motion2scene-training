# Finite-reference development course adapter

`motion2scene_collect_course.py` registers, runs and scores fresh development
courses containing multiple overhead constraints along the existing reference
route. It reuses the frozen five-reference controller and the existing sensor
policy. Beam geometry is used only for authoring, simulation, contact
instrumentation and scoring; the student receives sensor history, robot state,
reference phase and legal-option information.

The current reference bank supports one adaptation entry at a qualified early
tick and recovery at 3.30–3.50 s. It does not support a second entry after
recovery, route choice, reference extension or a verified protective stop.
Consequently, two beams can test continued traversal during one authored
adaptation. They cannot establish repeated obstacle-conditioned transitions or
general indoor navigation.

## Authoring and measurement contract

The collector binds the source, controller, all reference files, registry,
definition, scene and implementation to SHA-256 records. It authors each beam's
actual length, width, thickness, underside and pose and checks every composed
native collision transform at capture start. It rejects unexpected trailing
template primitives instead of silently dropping geometry. Only explicit
`development` definitions are accepted; no locked evaluation layout is read or
changed.

Each beam has a separate filtered contact sensor for all 30 robot bodies. Direct
PhysX forces are captured at 200 Hz and checked against the 50 Hz cached values.
The largest substep force vector per body and beam is retained for scoring.
Contacts from every beam remain in scope through whole-course completion,
including a collision with an earlier beam after its initial crossing. If any
constraint remains incomplete, all first-episode contacts remain in scope.

Success requires all body origins to cross every beam in order, sustained upright
clearance under the existing passage criterion, no beam contact above 1 N, legal
commands, a valid initial approach and completion within the finite reference
budget. Body-origin crossing is a stated geometric scoring approximation; force
measurements provide the physical contact check. Failure reports distinguish
unfinished constraints, falls, contact, invalid commands, reset and reference
exhaustion. An unreached second beam cannot become a course success.

The actual loader returns 199 reference samples spanning 0–3.96 s from the
120-frame, 30 Hz source. The collector uses `--max-steps 199`, producing at most
198 synchronized physical/sensor rows under this driver. The command refuses a
further clock advance before the inherited code can resample and reset the
robot. This abort is an execution audit, not a protective robot stop. Source
motions are never padded or extended.

## Registered smoke and retained attempts

The fresh course `development_two_beams_043_057` places beams at route fractions
0.43 and 0.57, with lengths 0.12 and 0.18 m and undersides 1.30 and 1.27 m. Its
registered action is d085 at 0.30 s. An empty-execution geometry forecast gives
61.3 and 43.3 mm minimum enclosing-envelope clearance for the two beams. This
finite recorded-pose screen is not a physical passage label.

The registered execution is
`/home/linjiw/research-data/groot-wbc/m2s-course-two-beam-smoke-v4/result.json`.
Its one d085 episode passes both beams: their sustained crossing finishes are
frames 121 and 153, giving a course time of 3.04 s. The complete force capture
has shape `(792, 2, 30, 3)` and every measured beam force is zero. Cached/direct
contact agreement is exact, and both composed geometry transforms match their
definitions within 3e-12. All 198 sensor packets match their physical states
exactly and remain on the first reference pass. Entry at 0.30 s and recovery at
3.30 s preserve robot state and the reference clock; the run ends at reference
phase 3.96 s without wrapping. Successful acquisition costs 48.94 s of wall time.

This is an admitted physical integration result on one fresh development
course. It establishes that the existing finite d085 adaptation can clear two
constraints of different lengths. It does not establish a learned-policy gain,
repeated adaptation, robustness across courses, or held-out generalization.

Earlier attempts remain intact. V1 failed at initialization because the first
registration expected 200 loaded reference samples (12.33 s, no recorded motion).
V2 reached the first physics substep and rejected an incorrectly indexed direct
PhysX contact matrix (17.74 s, no complete recorded control row). The direct view
is now normalized separately from the cached sensor tensor. V3 was registered
but did not launch because available GPU memory was below the existing 7,500 MiB
floor; v4 incorporates the shared collector's clock audit and source validation.
These infrastructure attempts are not course failures or passage successes.
Their combined 30.074 s wall cost is preserved in v4's `prior_attempts.json`,
alongside explicit absence of complete trajectory rows and the v2 first-substep
lower bound inferred from the exception stack. The two failed initializations
are not silently removed from acquisition accounting.

## Reproduction

Run a prepared manifest with:

```bash
PYTHONPATH=.:scripts/research .venv_research/bin/python \
  scripts/research/motion2scene_collect_course.py run \
  --out /home/linjiw/research-data/groot-wbc/m2s-course-two-beam-smoke-v4
```

Use `prepare --help` for fresh parameters and `analyze` to summarize completed
rows. The collector retains incomplete attempts and requires a fresh manifest
after an infrastructure failure. Pending cells after a GPU preflight can be
resumed without rerunning completed work.
