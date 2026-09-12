# Motion2Scene execution-state recheck — September 11, 2026

Execution remains **BLOCKED** at the original incomplete attempt; pilot utility remains **INCONCLUSIVE** and Phase 3 stays closed. This is a fresh verification of an existing audit, not a new preregistration, simulation run, manuscript rewrite, or claim of having performed the earlier audit. All 288 acquisition and 171 validation outcomes were accessible before this recheck. No research agents or physics jobs were launched.

## Current evidence

The authoritative checkout is `/home/linjiw/groot-wbc-sonic-sim-trackb`, commit `05b15a0970af13701a61ae7bfea615bd5f3578e4`. Its identity is supported by the assignment launch commands and the earlier bound runtime audit. The separately named `/home/linjiw/motion2scene` is not substituted. Isaac Lab is installed locally; the pasted statement that this runtime lacks it is stale. No active pilot process was found. Working-tree status and a contemporaneous GPU snapshot are retained here; the GPU snapshot does not measure historical episode utilization.

All **70,173 input-artifact bindings** and **59 audit/manuscript bindings** match their September 11 audit SHA-256 values, with no newly missing files or changed bytes. Approximately 21.36 GB were hashed. This verifies current consistency with that audit, not original preregistration chronology. The historical source drift documented in the earlier audit remains unresolved; matching the audit does not repair those older discrepancies. The original manifest digest also matches its `.sha256` companion.

The current allocation and outcomes remain:

| Stage | Assigned | Scored | Passes | Physical failures | Interrupted, NA | Unrun, NA | Recorded steps | Known driver wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Acquisition | 288 | 288 | 158 | 130 | 0 | 0 | 326800 | 17197.1011 |
| Validation | 200 | 171 | 73 | 98 | 1 | 28 | 192788 | 9926.1078 |

Aggregate physical counts combine different policies and teacher roles and are not the pilot utility estimand. Horizon/censoring flags overlap failures; see the earlier censoring tables. Known driver wall time totals 7.5342 hours plus unknown interrupted-attempt cost; it is not GPU-active time. The 488-row assignment table retains missing outcomes as NA.

The exact missing terminal artifact remains:

`/home/linjiw/research-data/groot-wbc/m2s-support-validation-panel-20260910-v1/episodes/episode_171/rollouts/learned_neutral/attempt.json`

Episode 171 has a launch, manifest, initialization log and collision inventory, but no terminal receipt, captured trajectory/contact NPZ, or result. Its log ends after motion loading. Actual steps, exit state, cause and wall time remain unknown. A filename search of the accessible research-data, separately named motion2scene checkout and `/tmp` found no alternative episode-171 terminal/result/NPZ artifact. This does not rule out an external backup.

The unchanged `execute_cell` in `scripts/research/motion2scene_run_extension_tasks.py` raises `unfinished attempt retained; no automatic retry` for this state. Do not erase that guard or synthesize a receipt. The current sequential runner cannot reach unrun assignments 172–199 unchanged. Recover authentic evidence or adopt a dated recovery/continuation amendment with missing-cost handling before any continuation. No unfavorable physical outcome is eligible for a technical rerun.

## Manuscript and decision boundary

The existing [conference draft](../../submission/traversal_method_v2.pdf) already contains the requested analytical framing, development null, retrospective limitations, incomplete pilot status and missing Screened Uniform comparison. Its LaTeX and PDF match the archived audit's bindings. `pdfinfo` confirms four pages, including references; the source is anonymous and identifies AI assistance in the acknowledgments. No new manuscript edit was needed to make its status agree with the present receipts. The earlier archive, patch and claim ledger remain intact.

The official [ICRA 2027 call](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/), revisited during this recheck, states September 15, 2026 at **23:59 PST**, eight total pages, double-anonymous review and disclosure of generated content identifying the system and affected sections. The [PaperPlaza start page](https://ras.papercept.net/conferences/scripts/start.pl) was also accessed; the retrieved content did not resolve the actual submission cutoff/timezone. Final portal-time verification remains outstanding. No manuscript was submitted or uploaded.

| Hypothesis or task | Decision | Next permitted action |
|---|---|---|
| Gen-2 incremental passage prediction over Screened Uniform on the recorded panel | CONTRADICTED FOR THE REGISTERED PREDICTION | Retain 5/6 versus 5/6 at each seed; do not infer population equivalence |
| Broadening B/C versus A; guided-half value C versus B | INCONCLUSIVE | Resolve the incomplete execution within an explicit recovery boundary; p and confidence intervals remain NA |
| Pilot superiority over contemporaneous Screened Uniform | BLOCKED/UNTESTED | Original 200-row panel lacks this comparator; use only a separately registered, budgeted amendment |
| Positive-support/complementarity causal mechanism | INCONCLUSIVE | Retain as a post hoc explanation consistent with finite comparisons |
| Finite six-context bank coverage and scripted passage | SUPPORTED FOR THE REGISTERED ESTIMAND | Preserve all contexts and forced/scripted outcomes |
| Phase 3, transfer, realistic sensing, sequential skills, dataset reuse and hardware claims | BLOCKED/UNTESTED | Satisfy original utility/adoption/implementation/censoring gates and separately register each expansion |

No new contrast, threshold, exchangeability assumption, non-inferiority margin or experimental allocation was adopted. The eighteen reserved layouts were not opened for outcomes, scoring, or tuning.

## Reproduction and receipts

- Fresh verification: `python3 docs/motion2scene/audit/20260911-state-recheck/verify.py`.
- Fresh tests: `.venv_isaaclab/bin/python -m pytest decoupled_wbc/tests/test_motion2scene_support_pool.py decoupled_wbc/tests/test_motion2scene_support_validation_panel.py decoupled_wbc/tests/test_motion2scene_passage.py -q` — **67 passed**.
- Prior complete assignment, aggregate, paired, censoring, proposal-cost and claim tables: [existing audit](../20260911-pilot-boundary/README.md).
- Recovery and missing-comparator proposals, still unadopted: [proposed amendments](../20260911-pilot-boundary/PROPOSED_AMENDMENT.md).
- This boundary's `verification.json`, source script, test log and state snapshots are bound by `manifest.json` and `manifest.sha256`.
