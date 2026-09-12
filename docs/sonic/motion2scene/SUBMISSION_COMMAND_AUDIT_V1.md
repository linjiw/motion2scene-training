# Motion2Scene submission command audit v1

Post-hoc validity audit, September 8, 2026. This protocol is written after the
nominal results were inspected and before any execution in this audit. It is not
part of either original primary study. Neither original study is changed.

## Specific gap and permitted inference

Across the 144 nominal executions and the parent's seed-8511 traversal executions,
eight of the 36 nominal conditions have no d040 execution. They are carrier 41001,
layouts 02/03/06/08/09, and carrier 41002, layouts 02/09/11. Walking is already
recorded everywhere. Outcomes for always-d040 and a two-command outcome oracle
cannot be completed by treating unexecuted commands as failures or successes.

Execute exactly these eight conditions, each with a constant walk request and a
constant d040 request: 16 executions, no new layout, seed, carrier, command,
generator, fitted selector, scorer, or adaptation timing. The repeated walk is a
measurement control for historical reuse. Both commands use the existing fixed
readout adapter; its constant-action JSON branch is named `privileged_geometry`
in code, but these audit policies consult no geometry or outcome labels.

## Frozen implementation and admission

Before execution, `motion2scene_submission_command_audit.py prepare` creates an
immutable registration, two constant policies, and eight two-cell manifests.
Each copies its exact nominal condition, controller, references, scene, physics
seed, capture horizon, and execution settings. Only policy identity, audit IDs,
and output paths change. Dependencies and the original manifest are hash pinned.
Decision time remains 0.30 s; the legal return window remains 3.3--3.5 s. The
original first-episode contact-qualified passage scorer remains authoritative.

The existing measurement admission must pass for both commands and their paired
prefix. The submission analysis additionally compares captured states, packets,
all 214 features, state/action/token histories, banks, physical configuration,
and scorer hashes against historical rows. Same-command passage must agree;
trajectory differences and their magnitudes are reported. A mismatch prevents
exact offline reuse; it is not removed from the results or retried to obtain a
preferred answer. No later policy-dependent decision is introduced.

## Predictions and reporting

1. Repeated walking reproduces historical passage in all eight conditions.
2. Matched recorded prefixes, full decision inputs, banks, and configuration agree
   exactly with the original condition (apart from explicitly enumerated policy
   metadata). Any failure invalidates exact outcome lookup for that condition.
3. No directional prediction is made for the eight unobserved d040 outcomes.
   Report all outcomes, including falls, resets, denied requests, and returns.

Report full-bank always-walk and always-d040 only when all necessary outcomes and
admission checks exist. Report both-pass unnecessary requests and both-fail
requests/refusals separately, without a composite endpoint. A hindsight outcome
oracle is distinct from the registered privileged geometric predictor. Offline
reuse is described as conditional on the measured history and deterministic
frozen runtime; hidden simulator/controller state is not claimed to be snapshotted.

## Budget and stop rule

Serial execution, two cells per batch, 375 s per-cell timeout, 7500 MiB free-memory
floor and the existing rolling 8/24 contended GPU-hour daily/weekly ceilings.
Sixteen cells reserve at most 1.667 hours in total; each two-cell batch must pass
the rolling gate. Actual process wall time is recorded separately from historical
study costs. Hash mismatch, measurement rejection, infrastructure failure, or a
changed command outcome stops subsequent batches. No automatic retries or refills.

The user explicitly authorized targeted new simulation to close a validity gap.
This audit resolves the simplest-policy comparison only; it supplies no new
source transfer, robustness, or confirmatory primary result.
