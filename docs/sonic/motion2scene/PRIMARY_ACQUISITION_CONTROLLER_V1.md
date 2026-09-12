# Serial primary acquisition controller

`scripts/research/motion2scene_run_primary_acquisition.py` implements the adopted
five-group acquisition sequence. **Primary recovery acquisition is in progress**
under [adoption V2](/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-adoption-v2/adoption.json),
SHA-256 `f75a952aa1e3c07f4be44c25723e25f9916eeb5db67de54c9dd74543dbfd8226`.
All [12 configuration checks](/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-adoption-v2/configuration_check.json)
completed before the first dispatch on 9 September 2026. The repaired launch
reached Isaac, loaded the tracker and executed the first bootstrap recording.
Completed primary models and curriculum comparisons remain pending.

The [recovery dispatcher](/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-dispatch-recovery-proposed-v4/dispatch.py)
uses the same tested source bytes and 60 boundary order, with a new explicitly
bound execution root. Its [execution binding](/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-dispatch-recovery-proposed-v4/execution_binding.json)
pins adoption, registration and source. The original pre-simulator scene-resolution
failure, unknown assessment and 1,192-step conservative reservation remain
preserved. The [reserved V5 evaluation proposal](TRAVERSAL_PROTOCOL_AMENDMENT_V5_PROPOSED.md)
is separately unadopted and unexecuted; primary adoption does not authorize it.

The controller reads a separate adopted-plan artifact, checks the frozen common
114-dimensional learner and its exact source closure, and checks an explicit
runtime declaration. The runtime declaration binds the registry, qualification
request, template and cell, controller/collector/scoring source closures, native
asset list, environment inventory, replay rule, and execution contract. Creating
this declaration does not adopt a plan. The adopted current declaration is
`m2s-primary-acquisition-common-freeze-v3/runtime.json`, with its matching
`common_learner.json`. It explicitly binds
`m2s-native-runtime-freeze-v5/{environment.json,runtime_assets.json}` and the
repaired collector/launcher source closure. The exact explicit USD path is
resolved and hash-checked by the real launcher before a launch intent is written.

For each run, the controller executes seven fresh bootstrap teachers and fits
M0. In rounds 1–4 it commits M(r−1) and its strictly earlier training collections
before creating either current collection directory, executes one learned
student, records the measured student release, and executes all seven assigned
teachers in their frozen order. Baseline arms fit their uniform common learner
on the accumulated teacher groups. The observation curriculum first performs
an explicitly recorded CPU-only uniform audit fit at `<round>/unweighted_audit`.
It then re-audits historical student gaps against the actual matched teacher
prefixes, verifies replay weights, and fits the weighted common learner at the
registered M(r) slot. This is five fitted models for a baseline run and nine CPU
fits, including four auxiliary audits, for an observation run. Every arm has
the same 39 assigned physical slots; replay adds no physical execution.

The replay receipts retain each gap's generating model and age. They do not
claim that an old gap measures the final model, that replay discovers new
states in this finite one-entry bank, or that weighting improves performance.
WAIT targets come from complete future schedules in the independently audited
teacher, and unknown or causally unavailable gaps receive no regret priority.

The common nonblocking `flock` serializes all arms and seeds. Before each new
cell, runtime preflight checks the explicit native cache, assets, editable
sources, and actual interpreter/package inventory. Free-memory preflight then
checks the adopted 7,500 MiB threshold. Only then is an exclusive intent written
and the process-group runner called with the 375-second limit. Complete actual
attempts are independently re-audited on resume and are never run again.

An interrupted intent with no process receipt pauses without retry; it reserves
1,192 steps and does not assert that physics started. Unknown outcomes preserve
any measured prefix and reserve the remaining finite horizon. Measured physical
failures, including admitted early failures with a nonzero exit, remain assigned
outcomes and allow acquisition to continue. Unknown infrastructure outcomes
pause before subsequent labels. Partial preparation, fit, or replay directories
are preserved and require explicit repair; the controller never overwrites them
or silently starts a replacement attempt.

Each round writes an immutable completion prefix, and M4 adds a final completion
receipt with model and training-result references. Historical prefix receipts
remain unchanged after later directories exist; this matters because the plan's
prefix validator correctly rejects future directories at the original binding
boundary. Immutable collection, fit, replay, launch, assessment and budget
receipts make resumed stages reviewable. Actual physics counts, unresolved
reservations, remaining assigned slots, CPU fit attempts and auxiliary fits are
reported separately. The maximum assigned physical budget is 46,488 steps,
leaving 3,512 of the declared 50,000 steps unallocated. No automatic retries use
that remainder. Recovery retains the predecessor's separate 1,192-step reservation
for `seed93203_analytic_contrast`, giving at most 47,680 allocated steps for that
corpus, still below 50,000. It is not imputed recorded physics or an extra label.

The first command checks the actual adopted recovery configuration without
creating execution directories or launching simulation. The second is the
recorded dispatcher command already coordinating this acquisition. The underlying
per-corpus `run` interface resumes its fixed slots and reports retained pauses
with exit code 75; the dispatcher preserves the registered interleaving.

```bash
.venv_isaaclab/bin/python scripts/research/motion2scene_run_primary_acquisition.py check \
  --plan /home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-plan-recovery-proposed-v4/plan.json \
  --adoption /home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-adoption-v2/adoption.json \
  --run-id seed93201_observation_curriculum

.venv_isaaclab/bin/python \
  /home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-dispatch-recovery-proposed-v4/dispatch.py \
  --adoption /home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-adoption-v2/adoption.json
```

`describe-runtime --help` lists the explicit request/template/asset/environment
arguments used to create the frozen runtime declaration. Its output must be
bound into adoption only after the common learner, runtime and controller
sources are final. It performs no model fitting, asset mutation, or simulation.

The following historical implementation validation preceded primary adoption.
It uses file-backed synthetic results with the real causal plan APIs. Those
focused tests exercise all 39 slots, observation audit/replay ordering,
interruption after a persisted attempt, unresolved dispatch, resource resume,
measured failures versus unknown outcomes, model/invocation tampering, exact
source closure binding, and the common lock. A separate read-only check of the
existing `collection_000/forced_neutral` recording produced an exact match to its
published row (1,192 recorded steps); that check added zero physical steps.
An installed-bank registration-only check also prepared all seven branches in
a temporary directory, verified the 332 runtime artifacts (110 source files
and 222 declared assets), and passed native environment preflight. It created
no rollout directories and no primary execution directory. That check exposed
and then verified the separate preflight fix for nondeterministic ordering of
same-name installed distributions: full package records are compared as a
multiset with duplicate records retained. The immutable V3 environment itself
was unchanged. That historical combined controller/plan suite passed 25 tests; the only
warnings come from an inherited escaped character while parsing source closure.

```bash
.venv_isaaclab/bin/python -m pytest -q \
  decoupled_wbc/tests/test_motion2scene_primary_controller.py
.venv_research/bin/python -m black --check \
  scripts/research/motion2scene_run_primary_acquisition.py \
  decoupled_wbc/tests/test_motion2scene_primary_controller.py
.venv_research/bin/python -m ruff check \
  scripts/research/motion2scene_run_primary_acquisition.py \
  decoupled_wbc/tests/test_motion2scene_primary_controller.py
```
