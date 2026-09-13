# Setup, readiness and troubleshooting

## 1. Choose the workflow

| Workflow | Environment | Readiness evidence |
|---|---|---|
| Inspect/verify/unpack data | Python 3.11.8+ base CLI | Fresh environment and clone tested |
| Reference plots and corpus viewer | Base CLI plus `[view]` | Fresh environment render tested |
| Student/generator CPU learning | Existing compatible native dependencies | Relocated CPU training tested |
| Teacher training and physical evaluation | Python 3.11 + Isaac/CUDA | Loader tested; fresh install and CUDA probe passed on a second host; physical evaluation not certified |

Run commands from the repository root. Native commands use the Python interpreter running `m2s`; installing only the base CLI does not install PyTorch or Isaac. Every `doctor` result states its scope. It checks module discovery and input presence, not every import, compatibility constraint, or file hash. Use `verify` for bundles and `unpack` for extracted-content validation.

## 2. Get a working copy and data

The complete repository is public at `github.com/linjiw/motion2scene-training`. Clone over HTTPS without repository membership:

```bash
git clone https://github.com/linjiw/motion2scene-training.git motion2scene-work
cd motion2scene-work
git lfs pull
python3.11 -m venv .venv
.venv/bin/pip install -e '.[view,test]'
.venv/bin/m2s --workspace "$PWD/workspace" unpack
.venv/bin/m2s --workspace "$PWD/workspace" doctor --profile view
env -u PYTHONPATH .venv/bin/python -m pytest -q tests
```

If `python3.11 -m venv .venv` fails with `ensurepip ... returned non-zero exit status 1` (the partial `.venv/bin/python` reports `No module named 'encodings'`), `python3.11` is a symlink to a uv-managed interpreter; recreate the environment with `uv venv --clear --seed --python 3.11 .venv`. `env -u PYTHONPATH` keeps host Python paths (for example a sourced ROS distribution, whose pytest plugins crash collection) out of the environment.

For another machine, use a remote containing both Git history and LFS objects, or transfer the working checkout with its actual bundles. A Git source ZIP or `git bundle` alone does not include LFS data. The public HTTPS clone does not require collaborator access; Git LFS must still be installed and able to download the data objects.

Use the same `--workspace` for every invocation. It is a global option and appears **before** the subcommand. Rerunning `unpack` validates existing files and fills missing files. Changed files are rejected rather than overwritten. To deliberately restore changed research input, keep a copy and extract into a new workspace. Extracting robot assets binds the vendored asset link to that workspace; do not switch it while a run from this checkout is active.

## 3. Native environment

Install `uv`, then use `scripts/setup_native.sh` as shown in the README. It refuses an existing environment, checks the Isaac Lab revision, pins the observed key versions and SMPLSim commit, installs dependencies, and only then enforces/tests PyTorch 2.7.0+cu128. It also installs this package into the new environment (a `uv` environment has no `pip`). Build-time setuptools is constrained below 81 through `requirements/build-constraints.txt`, because Isaac Lab's pinned `flatdict==4.0.1` source build imports `pkg_resources`, which setuptools 82 removed. The installer needs HTTPS access to pypi.org, pypi.nvidia.com, download.pytorch.org and github.com, plus the file and CDN hosts those indexes redirect to (for example files.pythonhosted.org and download-r2.pytorch.org). It is not resumable: after a failure, delete the partial environment you created and rerun.

It does not set EULA acceptance. Read the [NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html) and accept it yourself before launching Isaac, either per launch with `OMNI_KIT_ACCEPT_EULA=YES` or once per environment by writing `yes` to `.venv_native/lib/python3.11/site-packages/isaacsim/kit/EULA_ACCEPTED`. Without either, Isaac Sim prompts on stdin; `teacher-train` redirects that prompt into `attempt-1/training.log`, so the run hangs or exits and uses up its packet.

Run native commands through `scripts/m2s_native.sh`, which forwards its arguments to `.venv_native/bin/m2s` with:

- `PYTHONPATH` unset. A sourced ROS Humble shell shadows `pinocchio` and the vendored `scripts` package.
- `TMPDIR=$HOME/.cache/m2s/tmp` (override with `M2S_TMPDIR`). Isaac Lab writes its log under `tempfile.gettempdir()/isaaclab/logs`, and on a shared host `/tmp/isaaclab` can belong to another user, which fails the launch.
- `ISAACLAB_USD_CACHE_DIR=$HOME/.cache/m2s/isaaclab-usd`, which keeps robot USD conversions separate from other checkouts.

When starting training through `tmux`, run the wrapper inside the tmux command, and pass any other variable (such as `OMNI_KIT_ACCEPT_EULA=YES`) inline there too: a tmux server does not inherit variables exported after it started. Environment creation downloads the ground-plane USD from NVIDIA's public content S3 bucket, so the launch host needs network access.

In the original readiness audit, the script passed shell syntax checking, its existing-environment refusal was exercised, and its training dependency command resolved in **dry-run mode against the existing native environment**; that was not a clean installation test. On 2026-09-12 the installer, with the build constraint, completed end to end in a new environment on a second host, and `uv pip check` reported exactly the four conflicts below; see [the readiness audit](READINESS_AUDIT.md#fresh-installation-2026-09-12). Unpinned transitive dependencies can still drift.

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
- **`Failed to build flatdict==4.0.1` / `No module named 'pkg_resources'`:** the installer predates the build constraint; use the current `scripts/setup_native.sh` in a new environment directory.
- **`.venv_native/bin/pip: No such file or directory`:** expected; the installer already installed this package. Use `uv pip install --python .venv_native/bin/python ...` for deliberate additions, with `-c requirements/native-constraints.txt`, then re-check the `+cu128` torch build.
- **Teacher launch hangs, or exits right after start:** read the end of `attempt-1/training.log`, which captures the trainer's stdout and stderr. Look for the EULA prompt; a `PermissionError` on `/tmp/isaaclab/logs/...` (launch not run through `scripts/m2s_native.sh`); a `PermissionError` on `/tmp/isaaclab_app_launcher.lock`; or `ValueError: No collision prim found` (ground-plane USD not downloaded; check network access). The SONIC train and eval scripts hard-code that lock outside `TMPDIR` to serialize simulator start-up. With `fs.protected_regular=2` only its owner can reuse it, so on a shared host its owner or an administrator must remove it. Isaac Lab's own log path is printed on the `Logging to file:` line; through the wrapper it is under `~/.cache/m2s/tmp/isaaclab/logs/`.
- **`attempt-1` ends `incomplete` in a short smoke:** checkpoints are written every 500 iterations and the receipt requires one, so use an `--iterations` value that is a multiple of 500.
- **Shared GPU:** `launch.py` only requires 2 GiB of free memory. 128 environments need a few GiB on top of the Isaac Sim context, and throughput drops when other jobs saturate the device, which counts against the 48 h wall cap. `launch.py` sets `CUDA_VISIBLE_DEVICES=0` for the trainer, so it always uses physical GPU 0.
- **Disk budget:** `teacher-prepare` copies the 469 MB release checkpoint into both `<output>-parent/inputs` and `<output>/inputs`. Training keeps every checkpoint (about 450 MB each, every 500 iterations) plus `last.pt`, roughly 31 GB for a 32,000-iteration packet. A failed save is only printed to `training.log` and training continues, so the run ends `incomplete`. Check `df -h` before launching.
- **Stopping a teacher run:** Ctrl-C or `tmux kill-session` stops only `m2s` and the launch monitor. The trainer runs in its own process session, so it keeps the GPU and no `exit.json` is written. Signal its process group instead: take `pid` from `PACKET/attempt-1/process.json`, confirm with `ps -o pid,args -p PID` that it is `train_agent_trl.py`, then `kill -TERM -- -PID`. The Isaac Sim trainer can ignore SIGTERM; if it is still running after about 20 s, use `kill -KILL -- -PID` (what `launch.py` does at the wall cap). The monitor then writes `exit.json` with state `incomplete`.

The original readiness audit launched no fresh native installation or teacher GPU run. The 2026-09-12 fresh installation on a second host ran the installer end to end and a bounded 500-iteration teacher smoke through `scripts/m2s_native.sh`; see [the readiness audit](READINESS_AUDIT.md#fresh-installation-2026-09-12). Neither touched the source teacher.
