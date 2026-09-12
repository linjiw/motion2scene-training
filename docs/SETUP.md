# Setup, readiness and troubleshooting

## 1. Choose the workflow

| Workflow | Environment | Readiness evidence |
|---|---|---|
| Inspect/verify/unpack data | Python 3.11.8+ base CLI | Fresh environment and clone tested |
| Reference plots and corpus viewer | Base CLI plus `[view]` | Fresh environment render tested |
| Student/generator CPU learning | Existing compatible native dependencies | Relocated CPU training tested |
| Teacher training and physical evaluation | Python 3.11 + Isaac/CUDA | Loader tested; fresh GPU bootstrap/launch not certified |

Run commands from the repository root. Native commands use the Python interpreter running `m2s`; installing only the base CLI does not install PyTorch or Isaac. Every `doctor` result states its scope. It checks module discovery and input presence, not every import, compatibility constraint, or file hash. Use `verify` for bundles and `unpack` for extracted-content validation.

## 2. Get a working copy and data

The complete repository is hosted privately at `github.com/linjiw/motion2scene-training`. Authorized GitHub users can clone it with:

```bash
git clone git@github.com:linjiw/motion2scene-training.git motion2scene-work
cd motion2scene-work
git lfs pull
python3.11 -m venv .venv
.venv/bin/pip install -e '.[view,test]'
.venv/bin/m2s --workspace "$PWD/workspace" unpack
.venv/bin/m2s --workspace "$PWD/workspace" doctor --profile view
.venv/bin/python -m pytest -q tests
```

For another machine, use a remote containing both Git history and LFS objects, or transfer the working checkout with its actual bundles. A Git source ZIP or `git bundle` alone does not include LFS data. The private repository requires GitHub access; ask the owner to grant collaborator access if authentication succeeds but cloning is denied.

Use the same `--workspace` for every invocation. It is a global option and appears **before** the subcommand. Rerunning `unpack` validates existing files and fills missing files. Changed files are rejected rather than overwritten. To deliberately restore changed research input, keep a copy and extract into a new workspace. Extracting robot assets binds the vendored asset link to that workspace; do not switch it while a run from this checkout is active.

## 3. Native environment

Install `uv`, then use `scripts/setup_native.sh` as shown in the README. It refuses an existing environment, checks the Isaac Lab revision, pins the observed key versions and SMPLSim commit, installs dependencies, and only then enforces/tests PyTorch 2.7.0+cu128. It does not set EULA acceptance. Read and accept required upstream terms yourself before launching Isaac.

The new script passed shell syntax checking, its existing-environment refusal was exercised, and its training dependency command resolved successfully in **dry-run mode against the existing native environment**. That is not a clean environment resolution or full installation test.

The observed working source environment reports four metadata conflicts:

| Package requirement | Observed installed version |
|---|---|
| isaacsim-kernel: numpy==1.26.0 | numpy 1.26.4 (SONIC pins this) |
| isaacsim-kernel: click==8.1.7 | click 8.5.0 |
| isaacsim-kernel: psutil==5.9.8 | psutil 7.2.2 |
| fastapi: starlette>=0.40.0,<0.46.0 | starlette 0.49.1 |

Training has run in that source environment, but this does not make the requirements consistent. `native-constraints.txt` captures key observed versions; it is not a complete lock. The installer records `dependency-check.txt` and preserves any reported conflicts. Neither a successful install nor `doctor` alone certifies native readiness. Do not repair the actively training source environment in place; use an isolated environment for compatibility work. The vendored SONIC package metadata is amended to pin SMPLSim to the observed commit. A test source locator was also made independent of the working directory; original hashes and amendments are recorded in the audit.

## 4. Common failures

- **Bundle missing/changed:** run `git lfs pull` in a clone whose remote has the LFS objects, then `m2s verify`.
- **Missing module:** use the environment for that workflow. A base environment is expected to fail `doctor --profile teacher` with exit code 1.
- **Missing repaired motions/checkpoint:** unpack the needed bundle into the selected workspace.
- **Robot assets bound to another workspace:** unpack `robot-assets` for the intended workspace when no run from this checkout is active.
- **Existing output/attempt:** choose a new run directory. Output identity is intentionally immutable.
- **CUDA probe failure:** inspect driver, device support and final PyTorch CUDA build before starting training.
- **Native dependency conflicts:** retain the report; do not declare a fresh GPU installation validated until loader and a bounded physical training/evaluation run pass.

No fresh native installation or second teacher GPU run was launched during this audit. The current source teacher remains untouched.
