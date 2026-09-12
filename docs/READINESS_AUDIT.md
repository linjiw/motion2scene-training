# Repository readiness audit

**Verdict: ready for dataset inspection, visualization and continued CPU research; conditionally ready for native GPU training.** A clean fresh-machine Isaac installation and physical training smoke remain unverified. Do not describe the repository as fully certified for every training workflow.

## Executed validation

| Check | Result |
|---|---|
| Base environment installs `[view,test]` | Passed |
| Fresh CPU environment renders reference NPZ | Passed |
| Repeated extraction verifies all five bundles | Passed; identical files skipped, changed files rejected |
| Package and vendored BFM tests, launched from package root | 31 passed |
| `doctor --profile teacher` in base/view environment | Correctly exits 1 |
| `doctor --profile teacher` with existing native dependencies and relocated inputs | Discovery/input checks pass; no simulator launch |
| Native dependency command, dry-run against existing native environment | Resolved; no packages changed |
| New installer shell syntax and existing-environment refusal | Passed |
| Primary README/setup/runbook/design links | No missing local targets |
| Scoped Black, Ruff and Git whitespace checks | Passed |

Prior relocated integration evidence still applies: 89/20 native CPU motion loading, 48 student optimizer updates, 20 generator updates, 200 generated tasks, and local clone/LFS verification. Those workflows' training implementations were not changed in this audit.

## Fixes made

- Added workflow-specific readiness checks with nonzero failure status. A base environment no longer appears training-ready merely because the CLI runs.
- Made extraction repeatable and recoverable for unchanged files, while refusing changed/unverified files. Robot asset links bind explicitly to the chosen extraction workspace.
- Reject invalid training/smoke budgets before creating output directories.
- Added a native setup script that refuses an existing environment, verifies the Isaac Lab revision, constrains key dependencies and probes the final CUDA build after dependency installation.
- Pinned SMPLSim to the commit used in the observed environment. Original vendor hashes and intentional amendments are recorded separately.
- Fixed a vendored test that depended on the current directory, ensuring it reads the vendored wrapper rather than another checkout.
- Added a setup/troubleshooting guide and research runbook distinguishing new fits, BFM continuation, teacher continuation, evaluation and scene qualification.

## Remaining integration limits

1. The working native environment has four declared dependency conflicts, detailed in [SETUP.md](SETUP.md). Captured constraints are not a complete, conflict-free lock. A dry-run against an existing environment is not a clean installation test.
2. No fresh native installation or additional teacher GPU run was performed. Full GPU readiness requires an isolated install, loader validation and a bounded physical train/evaluation smoke. The active source teacher was not modified.
3. A portable teacher crash-resume command and one-command teacher evaluation are not implemented. The [research runbook](RESEARCH_RUNBOOK.md) explains the current manual integration boundary. Exact BFM optimizer continuation exists in its native fitter.
4. The repository is local; another machine needs a published Git/LFS remote or an explicit transfer of the actual bundles. There is no remote clone URL yet.
5. Scene proposals and offline fits are not evidence of successful physical scene navigation. Qualification gates remain intact.

Evidence is retained in `validation-evidence/readiness-audit/`. The original data bundles and teacher/student checkpoint bytes were not changed by this audit.
