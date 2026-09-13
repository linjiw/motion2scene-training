# Repository readiness audit

**Verdict: ready for dataset inspection, visualization and continued CPU research; conditionally ready for native GPU training.** A clean native installation later passed on one second host ([2026-09-12](#fresh-installation-2026-09-12)), where a bounded physical training smoke is also recorded; physical evaluation from a fresh install remains unverified. Do not describe the repository as fully certified for every training workflow.

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
2. At this audit, no fresh native installation or additional teacher GPU run was performed (see the 2026-09-12 section below for a later one). Full GPU readiness requires an isolated install, loader validation and a bounded physical train/evaluation smoke. The active source teacher was not modified.
3. A portable teacher crash-resume command and one-command teacher evaluation are not implemented. The [research runbook](RESEARCH_RUNBOOK.md) explains the current manual integration boundary. Exact BFM optimizer continuation exists in its native fitter.
4. At the audit snapshot the repository was local. The subsequent publication was initially private and is now public at the owner’s request. Remote/LFS verification and visibility records are in `docs/publication/`.
5. Scene proposals and offline fits are not evidence of successful physical scene navigation. Qualification gates remain intact.

Evidence is retained in `validation-evidence/readiness-audit/`. The original data bundles and teacher/student checkpoint bytes were not changed by this audit.

## Fresh installation, 2026-09-12

A second host (Ubuntu 22.04, glibc 2.35, one RTX 5090 on driver 590.48.01, uv-managed CPython 3.11.14) installed this checkout from scratch. The GPU was shared throughout with other jobs (rendering and training processes, including a concurrent 2048-environment SONIC training under the same account), which held about 20 GB and ~98% utilization. Evidence is in `validation-evidence/fresh-install-20260912/`.

| Check | Result |
|---|---|
| LFS objects, `m2s verify`, `m2s unpack` of all five bundles | Passed; 3.5 GB extracted and hash-verified in 34 s |
| Base `.venv` with `[view,test]`, `doctor --profile view`, reference render, package tests | Passed; 5 tests |
| `scripts/setup_native.sh` at `a87d6e2` | **Failed**: `flatdict==4.0.1` source build imports `pkg_resources`, removed in setuptools 82 |
| `scripts/setup_native.sh` with the setuptools<81 build constraint, in a new environment | Passed end to end; `2.7.0+cu128` probe passed; `uv pip check` reports exactly the four documented conflicts |
| `doctor --profile teacher --cuda` and `--profile student` in `.venv_native` | Passed |
| Package tests plus vendored BFM and scene-distillation tests in `.venv_native` | 60 passed |
| `generator-smoke --updates 20`, `student-smoke --updates 240`, `scene-tasks` | Completed; student action MSE 0.875 → 0.175 (uniform) and 0.179 (curriculum); 200 tasks |
| `teacher-prepare`, 128 environments, 500 and 32,000 iterations | Completed; 89 train / 20 development clips loaded, 958 source files verified |
| `teacher-train` 500-iteration smoke, EULA recorded in the environment, launched through `scripts/m2s_native.sh` | Running at the time of writing: Isaac Sim launched headless, CUDA preflight passed, release actor/critic loaded strictly with an empty optimizer, 89 resident motions at 128 environments, PPO iterations logging; completion is recorded in a follow-up |

Problems found and fixed during this installation:

- The installer needed the setuptools build constraint.
- The README's `.venv_native/bin/pip install -e .` fails, because `uv` environments have no `pip`, and it was redundant.
- The runbook's 10-iteration smoke cannot complete; checkpoints and the receipt need 500.
- On this shared host, `/tmp/isaaclab/logs` belonged to another user, so an unmodified Isaac Lab launch would fail its logger setup.
- The ROS Humble `PYTHONPATH` from `~/.bashrc` crashes pytest and shadows venv packages.

`scripts/m2s_native.sh` and the setup guide now cover the last two.

Measured teacher throughput on the shared GPU was about 20 s per iteration at 128 environments, compared with about 3.2 s on the source machine (12,572 iterations in 11.0 h, [step-12,500 report](teacher-status/step12500/REPORT.md)). At that rate a 32,000-iteration packet would reach the 48 h wall cap long before completing, so size long runs from measured throughput on the actual device.
