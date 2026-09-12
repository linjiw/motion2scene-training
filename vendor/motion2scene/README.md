# Motion2Scene

Motion2Scene studies the inverse complement of scene-conditioned humanoid motion:

> Given a physically executable humanoid motion and matched alternatives, synthesize a
> distribution of minimal scene constraints under which that motion is the preferred traversal
> behavior, then use those scene-motion pairs to learn the forward decision.

The intended loop is:

```text
executable motion
  -> critical constraint distribution
  -> synthetic scene-motion data
  -> scene-conditioned skill/event policy
  -> verified execution
```

The project is deliberately separate from the Scene2Motion/SONIC repository. It consumes pinned
artifacts from that stack without copying its research history. The governing research brief is
[`docs/DESIGN_PLAN.md`](docs/DESIGN_PLAN.md); the live claim/evidence ledger is
[`docs/CLAIM_EVIDENCE_MAP.md`](docs/CLAIM_EVIDENCE_MAP.md). The exact fresh-corpus prompt grammar,
measured funnel, and factorized beam sampler are summarized in
[`docs/E0_E1_DATA_DISTRIBUTION.md`](docs/E0_E1_DATA_DISTRIBUTION.md).

The adopted post-E0 gate is recorded in
[`docs/E0_TO_E3_QUALIFICATION_PLAN.md`](docs/E0_TO_E3_QUALIFICATION_PLAN.md). It freezes the
shared-seed confirmatory acquisition, S0--S4 semantics, route gate, Q4 protocol, ordinal ladders,
and the requirement for physical preference reversal before learned hallucination.

## Current status

The confirmatory `cg-wbc-v2-shared-seed-confirmatory` corpus is complete: 144/144 Kimodo
references across eight shared seeds, six body modes and three routes. Q0 passes 132/144 and Q1
passes 144/144. Same-seed paired semantics find 1/24 duck references at S3 and 0/24 arm-tuck
references at S3; shoulder-turn, step-over and carry remain fail-closed `not_measured` because
their complete predicates are not implemented.

Controlled same-carrier construction produces 7/8 ordered reference duck ladders and 10/32
nonempty adjacent beam intervals after 10 mm target and strike margins. The 24-cell obstacle-absent
Q3 test accepts all eight neutral carriers, 3/8 40 mm crouches and 1/8 55 mm crouches. The frozen
achieved-state audit promotes 0/16 adapted executions to S4 and therefore admits zero Q4
candidates. Seven of eight accepted neutral controls fail the absolute straight-route curvature
threshold. A preregistered 8/8 neutral-control calibration finds median 0.975 path-length retention,
4.6 cm cross-track RMSE and 0.111 rad heading error, while absolute achieved validity varies from
0/8 to 8/8 across the frozen measurement-scale grid. This diagnoses cumulative curvature as a
scale-sensitive retention instrument; the frozen result is not relabeled, and a relative rule
still requires held-out validation.

The fresh eight-carrier validation is now complete: 8/8 reference admissions, 7/8 tracker
survivors and 5/7 survivors passing relative route retention. This misses the registered 80%
criterion. A shared-clock construction produced eight matched three-level reference ladders
(24 references), but its physics pilot remains gated by that failed validation. The
[latest result](experiments/registrations/E1_ROUTE_RETENTION_HELDOUT_V1_RESULT.md) reports all
failures, route plots and artifact hashes.

The hash-bound result is
[`experiments/registrations/E0_E1_SHARED_SEED_V2_RESULT.md`](experiments/registrations/E0_E1_SHARED_SEED_V2_RESULT.md).
No learned hallucinator, obstacle-present preference reversal, or dataset-utility claim is
authorized yet.

Source integration is pinned in [`configs/e0_cg_wbc_v1_pilot.json`](configs/e0_cg_wbc_v1_pilot.json).
The registered Q3 reconciliation is
[`experiments/registrations/E0_Q3_PILOT_V8_RESULT.md`](experiments/registrations/E0_Q3_PILOT_V8_RESULT.md).
