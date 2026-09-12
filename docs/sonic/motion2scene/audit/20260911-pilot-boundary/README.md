# Motion2Scene audit boundary — 11 September 2026

**Execution: BLOCKED. Manuscript reconciliation completed as a working revision. Phase 3 remains closed.** No physics was launched, retried, or duplicated by this audit. No registration, model, scorer, raw receipt, or reserved layout was modified.

The authoritative local execution checkout is `/home/linjiw/groot-wbc-sonic-sim-trackb`, identified by the actual assignment commands, model paths, source closures, and passing frozen runtime preflight, rather than its name. The checkout commit is recorded in [commit.txt](commit.txt); its pre-existing changes are preserved in [initial-git-status.txt](initial-git-status.txt) and [initial-working-tree.patch](initial-working-tree.patch). The current manuscript is `submission/traversal_method_v2.tex`, as identified by the project manuscript notice and submission README; `paper.tex` is a historical generation. Both original sources and PDFs are archived here.

## Execution boundary

| Stage | Assigned | Scored passes | Scored physical failures | Interrupted, outcome NA | Unrun, outcome NA | Recorded steps | Known driver wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| New acquisition | 288 | 158 | 130 | 0 | 0 | 326,800 | 17,197.1011 |
| Validation | 200 | 73 | 98 | 1 | 28 | 192,788 | 9,926.1078 |

Counts are assignment reconciliation, **not complete pilot utility results**. The prefix is inherited and excluded from new acquisition cost. Each arm has 96 new assigned episodes, read from nine concrete runs at seeds 93201, 93202, and 93203; 84 teacher and 12 student episodes per arm. All evaluation assignments use physics seed 8732. Tables were derived after 171 validation outcomes were accessible. This is a dated exploratory audit, not original preregistration.

The missing terminal receipt is:

`/home/linjiw/research-data/groot-wbc/m2s-support-validation-panel-20260910-v1/episodes/episode_171/rollouts/learned_neutral/attempt.json`

That assignment has `launch.json`, a rollout log, and a collision inventory. It lacks a completed or aborted trajectory/contact capture and `result.json`. The log ends during initialization; actual simulator steps, exit status, termination cause, and elapsed wall time are unknown. No pilot runner or native evaluation process was present at inspection. Do not infer zero cost from missing telemetry. Do not reconstruct a terminal receipt from an assumed timeout or file modification times.

`execute_cell` in `scripts/research/motion2scene_run_extension_tasks.py` refuses an existing unfinished launch with “unfinished attempt retained; no automatic retry”. The registered sequential panel therefore cannot pass episode_171 unchanged. Episodes 172–199 are genuinely unrun but cannot be dispatched by bypassing this stop. Recover an authentic terminal/raw receipt if one exists outside this checkout; otherwise a separately dated recovery/continuation amendment is required. No retry is authorized by this report. Nonzero exits in 32 acquisition and 31 validation receipts already establish physical failure and must never be retried as technical failures.

Known total driver time is 27,123.2089 seconds (7.5342 hours), **plus unknown interrupted-launch cost**. It is not measured GPU-active time. Maximum assignment step ceilings are not measured cost. [gpu-snapshot.txt](gpu-snapshot.txt) contains one contemporaneous GPU reading, not historical per-episode telemetry. [runtime-identity.json](runtime-identity.json) records Python and installed distribution versions. The attempted `pip freeze` failed because this interpreter has no pip; distribution metadata was collected instead.

## Evidence integrity and limits

[Input artifact manifest](input-artifact-manifest.json) binds the checked source references, assignments, checkpoints, raw trajectories, contact streams, and scorer results by SHA-256. [verification.json](verification.json) preserves every detected discrepancy; [source-drift.json](source-drift.json) deduplicates them. A present hash proves current byte identity only, not an earlier registration date.

Two historical source references differ from the current checkout:

- An old bootstrap learning-source hash differs; its recorded analysis snapshot matches the expected bytes.
- The nested legacy `implementation.rollout_driver` field expects `0a48eed8d00336ec17a29be761ab6ebfbe0c9f444da71e1cd72fb9159c6d1eb2`, whereas the current wrapper hashes to `24d291ecf1155f85c696009f5712c71919a3cace951a172ef837d1a07bfaa1db`. No matching historical wrapper archive was located in the traversed bindings. The separate executable runtime closure passes the frozen preflight; this does **not** repair or validate the stale historical field. Full historical source-ancestry verification remains BLOCKED on that version.

The collection manifests' current executable closures and frozen models pass the existing preflight, including an unrun assignment. Prefix specifications and inherited M4 result references are identical across A/B/C **within each seed**, as recorded in [prefix-verification.json](prefix-verification.json). This does not imply all seeds share identical training data. [frozen-settings.json](frozen-settings.json) records phases, 114 feature names, sensor settings, seven schedules, limits, and source identity. No learner or simulation settings changed.

The audit verifies existing scorer bindings and physical evidence; it is not an independent second physics experiment. All scored full and aborted contact captures are checked for 0.005 s cadence and recorded step length. For aborted captures, contacts are under `raw_artifacts`, not the full-result fields. The exact scorer remains unchanged. Body origins must cross the downstream face plus 0.1 m; upright is root height ≥0.5 m and negative projected-gravity z ≥0.5, continuously over 0.30 s. At 50 Hz the stabilization window is 16 samples. Per-body normal-force vector norms are maximized over bodies and time (and relevant beams); four-substep maxima preserve 200 Hz peaks for control-aligned passage scoring. A pass also requires the declared whole-horizon recovery/contact/stability checks. The 1 N rule is a simulation convention, not a protective stopping threshold.

[Physical diagnostics](physical-diagnostics.json) retain classifications and physical events. [Censoring summary](censoring-summary.json) records 32/31 partial captures with verified failure and 14/43 finite-horizon events for acquisition/validation. These flags overlap failures and are not disjoint additional outcomes. No failure is excluded as censored. Episode_171 is an unresolved technical interruption with unknown physical outcome.

## Tables and mechanism diagnostics

- [All 488 assignment outcomes](assignment-outcomes.csv), including NA cells, source result paths, proposal channel, candidate, seed, schedule, measured cost, and exit code.
- [Aggregate outcomes](aggregate-outcomes.csv) for every acquisition corpus and every registered evaluation policy, including all fixed schedules and the script.
- [Censoring/event flags by assignment](censoring-outcomes.csv) and [all terminal attempt receipts](attempts.json).
- [Paired comparisons](paired-comparisons.json): B−A, C−A, C−B per matched acquisition seed. Wins/losses/ties apply only to available pairs; missing pairs are explicit. Full-panel bounds allow each missing difference to range from −1 to +1. These are deterministic identification bounds, **not confidence intervals**.
- [Every validation context's bank/script status](validation-bank.json). No favorable subset is selected. No observed passing branch with unrun branches means unknown solvability, not established unsolvability.
- [Training diagnostics](training-diagnostics.json): passing response sets, all-fail encounters, minimum cover of solvable encounters, shared versus added encounters, consequential fitted decisions and excluded indices. Fitted-decision counts come from model receipts; per-encounter causal-target usability is not inferred from solvability and remains NA where not separately audited.
- [Selected proposals](selected-proposals.json), [all 3,840 candidate eligibility/rejection rows](proposal-eligibility.csv), and [measured proposal cost](proposal-cost.json). The original candidate and queue artifacts retain rejected proposals and gate membership. Shared clearance work is measured once, not attributed to every arm. No per-arm proposal runtime is invented.

The validation generation manifest reports zero clearance queries, zero physics steps, seed 940910, and ten task-geometry strata. Inspection of generation code finds no outcome-driven scoring or rejection: outcome-bearing files supply the existing neutral route/bank references, not labels for choosing contexts. The manifest and source bindings support the stated construction rule, but no contemporaneous query-execution trace establishes an independently measured zero-outcome-query count. This distinction is retained. The reserved geometry/outcomes were not inspected by this audit; only the already declared validation sample was read.

No defensible exchangeability scheme, independent-unit count for a randomized test, quantitative utility decision threshold, or multiplicity procedure was located in the pilot registration. **p = NA**, confidence intervals = NA, permutation space and attainable resolution = NA. The available structure is three paired acquisition seeds crossed with ten shared sampled contexts at one physics seed; neither 200 executions nor nine policies are independent replicates. The original named contrasts are preserved; new calculations here are exploratory. No equivalence or non-inferiority follows from ties or nonsignificance.

The contemporaneous Screened Uniform comparator is absent from the 200 assignments. Its comparison is BLOCKED/UNTESTED. Historical six-context results are never substituted. See [proposed amendment](PROPOSED_AMENDMENT.md); nothing in that proposal changes the registered panel or opens a new budget.

## Decision log

| Hypothesis or claim | Decision | Claim change / next permitted action |
|---|---|---|
| Executed contrast improves the recorded Gen 2 six-context passage score over Screened Uniform | CONTRADICTED FOR THE REGISTERED PREDICTION | Observed difference 0 at all three seeds; state finite-panel null. This label does not assert the historical prediction had an independently verified preregistration date. |
| Population superiority, equivalence, or non-degradation from the development tie | INCONCLUSIVE | No such population claim; no adopted non-inferiority margin. |
| Replay improves the recorded development decisions | CONTRADICTED FOR THE REGISTERED PREDICTION | Report measured null despite changed fitted parameters. |
| Positive-support filtering is the sole cause of the placement advantage | BLOCKED/UNTESTED | Explain as consistent with comparisons, not identified causality. |
| Negative-screen population precision, or complementarity causally prevents collapse | BLOCKED/UNTESTED | 2/2 is selection-conditioned; 2/7 recall is retrospective; collapse association post hoc. |
| Complete-continuation teacher independently improves decisions | BLOCKED/UNTESTED | Teacher was common to the five arms. |
| B or C improves physical passage over A | INCONCLUSIVE | Incomplete pilot; preserve assignments and resolve execution block without rerunning failures. |
| C improves over B because the guided half adds value | INCONCLUSIVE | Incomplete pilot; diversity alone is insufficient. |
| B/C improves beyond contemporaneous Screened Uniform | BLOCKED/UNTESTED | Comparator absent; separate budgeted amendment only. |
| Bank covers the six development contexts; script passes all six | SUPPORTED FOR THE REGISTERED ESTIMAND | Finite development evidence only; retain all 42 forced and six scripted executions. |
| Dataset reuse utility, source-held-out transfer, sensing robustness, sequential skills, hardware safety | BLOCKED/UNTESTED | Separate evidence and adoption gates required; no Phase 3 launch. |

Before Phase 3: resolve receipt and lineage gaps, complete manuscript/evidence review, and establish the registered physical-utility gate. A positive point estimate alone cannot create a missing decision rule after outcome access. All eighteen-layout adoption gates, implementation lock and terminal-censoring policy still apply. Restore source-ancestry-held-out transfer before generalization; separately register sensing/noise/dropout/timing and the proposed 70-degree field of view; compare 100/260/500 ms delays under matched sensing. Only then qualify sequential decisions with matched aggregation/fresh labels and every intended transition, reporting whole-chain success. No crawling, stopping, or hardware claim is authorized.

## Manuscript and checks

The [revised PDF](../../submission/traversal_method_v2.pdf) is four pages including references, anonymous, with AI-use disclosure. [Claim-to-artifact ledger](CLAIMS.md) and [source patch](manuscript.patch) make the revision reviewable. The source no longer imports the old supplement. The existing preregistration and September 10 decision document remain unedited; their overstatements are corrected by this dated audit and manuscript.

Validation: 67 focused support-pool, validation-panel and passage tests passed; Tectonic built the PDF; `pdfinfo` confirmed four pages. The reconciliation asserts 288+200 unique allocation rows, common prefixes, command identity, and contact cadence/length. Reproduce using the repository's `.venv_isaaclab/bin/python` on `reconcile.py`, then `diagnostics.py`. The audit code writes only this audit directory. It does not rescore or overwrite experimental receipts.

Official ICRA 2027 instructions, checked September 11, state September 15, 2026, **23:59 PST**, an eight-page total limit, double-anonymous review, and disclosure of AI-generated content. The page explicitly says PST; do not silently replace that with daylight-saving Pacific time. PaperPlaza's actual cutoff/timezone display still requires verification before final submission. No submission was performed. [Official call](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/).

Audit-script lint tools Ruff and Black are not installed in the frozen interpreter or current PATH; no environment packages were changed. Python compilation and the explicit reconciliation assertions passed. The PDF has only underfull-line warnings, with no overfull boxes; its first page was visually inspected.
