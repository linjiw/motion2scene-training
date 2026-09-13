# Motion2Scene Training Kit

A separate, self-contained research checkout for SONIC teacher training, BFM student distillation, Motion2Scene scene/obstacle generation, and dataset visualization. It contains a working-tree snapshot of both research codebases, actual datasets and selected checkpoints in Git LFS, plus portable commands that create new local configurations.

**Research status:** the full-command motor reaches 88/89 training and 10/20 development completions. The latest navigation continuation fit completes 4/8 tasks on its first evaluation seed and 2/8 on confirmation, versus its parent's 4/8 and 1/8. Contact failures, seed sensitivity and a motion-sampling confound remain. Expanded motor/navigation evaluation is running. Read the [latest research update and log](docs/RESEARCH_UPDATE_20260913.md), [methods and architecture package](docs/sonic/motion2scene/methods_v2_20260913/README.md), and [full research report](docs/DISTILLATION_RESEARCH.md).

Read the [readiness audit](docs/READINESS_AUDIT.md) before setting up native training.

## Project and research overview

The intended loop is **executable motion → critical scene proposals → physical qualification → teacher labels → masked student → scene/goal control**. A scene that merely clears a reference is not yet a useful navigation demonstration. The package preserves each qualification boundary and the negative results that motivate the next experiments.

Read the [distillation research report](docs/DISTILLATION_RESEARCH.md), [detailed research guide](docs/RESEARCH_GUIDE.md), [paper draft (PDF)](paper/main.pdf), and [paper evidence map](paper/README.md). The report separates measured motor recovery from the proposed extensible navigation-context framework. Autonomous scene-navigation and residual-learning gains remain proposed work.

## Install and unpack

Use Python 3.11.8+ and Git LFS. The repository is public. Clone it with its Git LFS objects, then:

```bash
git clone https://github.com/linjiw/motion2scene-training.git
cd motion2scene-training
git lfs install
git lfs pull
python3.11 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/m2s catalog
.venv/bin/m2s verify
.venv/bin/m2s unpack
.venv/bin/m2s doctor
```

The base CLI uses only the standard library. `unpack` writes under `workspace/` and verifies every extracted file. Rerunning extraction verifies and skips identical files, fills missing files, and refuses changed files. To use another disk, pass `m2s --workspace /your/data/path unpack`; supply the same workspace to subsequent commands. Unpack individual bundles by name when desired. Archives are about 2.2 GB compressed; allow at least 8 GB for extraction, in addition to Git/LFS storage and simulator installation.

This is an **editable research checkout**, not a small wheel containing the data. A wheel installation needs `M2S_PACKAGE_ROOT` pointing at the checkout. The unpacker links the vendored robot-data directory to the chosen workspace. Historical records retain original paths for provenance; portable commands generate new local records instead of modifying historical hashes.

See [step-by-step setup and troubleshooting](docs/SETUP.md). `doctor --profile view`, `--profile student`, and `--profile teacher` exit nonzero when required inputs or modules are missing. A base-profile pass does not imply training readiness.

## Native training environment

The tested native stack uses Linux, Python 3.11, Isaac Sim 5.1.0, Isaac Lab v2.3.2 and PyTorch 2.7.0+cu128. The package bootstrap targets NVIDIA Blackwell; it is not a claim of validation on every GPU. `uv` is required (`python -m pip install uv`). The installer refuses existing environments and checks the final CUDA build after all dependency installs. Licensed simulator/model prerequisites remain subject to their original terms.

```bash
bash scripts/setup_native.sh "$PWD/.venv_native" "$PWD/external/IsaacLab"
.venv_native/bin/pip install -e .
.venv_native/bin/m2s doctor --profile teacher --cuda
```

**Native GPU readiness remains conditional:** the observed working environment has four declared dependency conflicts, and the fresh installer has not been executed end-to-end. Details and the exact evidence are in [SETUP.md](docs/SETUP.md).

The bootstrap downloads the simulator rather than bundling its installation. Read its EULA instructions; this package does not accept agreements on behalf of another user. The full upstream SMPL corpus (31 GB on the source machine), Kimodo model weights, and licensed simulator binaries are not included. The 120-motion research corpus and repaired native teacher clips are included, so those full upstream corpora are not needed for the packaged teacher/BFM paths. The independent Motion2Scene package can also be installed with `pip install -e vendor/motion2scene`.

## Prepare and train a repaired teacher

```bash
.venv_native/bin/m2s teacher-prepare --output "$PWD/workspace/new-teacher" --num-envs 128 --iterations 32000
.venv_native/bin/m2s teacher-train "$PWD/workspace/new-teacher"
```

Preparation verifies repaired manifests, uses all 89 screened training clips and all 20 development clips for loader validation, creates local configuration paths, and runs the real CPU motion loader. Training initializes from the released SONIC checkpoint with a fresh optimizer. The launch is bounded, writes checkpoints/receipts and refuses to reuse an existing attempt. The CLI starts the training monitor only; evaluation is a separate declared experiment. No final repaired checkpoint is bundled while its source training remains live.

## Student smoke training and scene generation

Use the native Python environment for the SONIC decoder/TRL dependencies, even though this smoke trains on CPU:

```bash
.venv_native/bin/m2s student-smoke --output "$PWD/workspace/student-smoke" --updates 240
.venv_native/bin/m2s generator-smoke --output "$PWD/workspace/generator-smoke" --updates 20
.venv_native/bin/m2s scene-tasks --output "$PWD/workspace/navigation-tasks"
```

`generator-smoke` trains the preserved conditioned obstacle scorer against the bundled 100-training-motion geometry labels for a bounded number of CPU updates; it does not run physics.

The student smoke recreates the matched uniform/curriculum experiment with unchanged label bytes and a relocated manifest. It never mixes labels from different teachers. It reports offline action errors, not physical navigation success. `scene-tasks` exports 200 train-only scene/goal tasks with actual collision USD geometry from the bundled 100-motion corpus. This materializes the existing generated proposals; it does not retrain a learned generator or qualify their physical feasibility.

The complete generator, acquisition, qualification, BFM, recurrent navigation and residual PPO implementations remain available under `vendor/sonic/gear_sonic/` and `vendor/sonic/scripts/research/`. Those historical entry points have experiment-specific arguments; see the workflow map rather than blindly replaying archived absolute-path commands.

## Visualize data

```bash
.venv/bin/pip install -e '.[view]'
.venv/bin/m2s view workspace/m2s-hindsight-dataset-v1-20260911/motions/00490/reference.npz --output workspace/reference.png
python3 -m http.server 8000 --directory workspace/m2s-hindsight-dataset-v1-20260911
```

The built-in `view` command plots root trajectory and height from a reference NPZ. Open `http://localhost:8000/viewer/` after starting the server. The bundled dataset includes its original interactive viewer and scene assets; the corpus root contains its gallery files. Skeleton replay renderers and existing rendered videos are also included in the research bundle. A reference preview is not a policy execution.

## Contents and provenance

- `src/m2s_package/`: portable orchestration, secure extraction and CPU preview.
- `vendor/sonic/`: Python control/training/data generation, scripts and install helpers from the dirty source working tree; original licenses retained.
- `vendor/motion2scene/`: the separate inverse-scene research project, plans and registrations.
- `docs/sonic/`: original design/evidence documents, including limitations and retractions.
- `bundles/`, `manifests/`: real LFS data archives and SHA-256 inventories; no placeholder dataset links.
- `tests/`: package-level checks; upstream tests remain with the vendored sources.

For new experiments and continuation limits, read [the research runbook](docs/RESEARCH_RUNBOOK.md).

See [dataset coverage](docs/DATASETS.md), [workflow map](docs/WORKFLOWS.md), and [validation](docs/VALIDATION.md). This repository preserves original license notices; dataset/model assets do not acquire a new blanket license merely by being bundled. Repository destination: `github.com/linjiw/motion2scene-training`. Public access does not replace asset-specific license terms.

## Documentation index

| Document | Purpose |
|---|---|
| [Setup](docs/SETUP.md) | Environments, data extraction, readiness and troubleshooting |
| [Readiness audit](docs/READINESS_AUDIT.md) | Executed checks and remaining native integration gaps |
| [Distillation research](docs/DISTILLATION_RESEARCH.md) | Current evidence, related work, extensible student framework and navigation-context roadmap |
| [Research guide](docs/RESEARCH_GUIDE.md) | Problem, interfaces, methods, experiments and next milestones |
| [Research runbook](docs/RESEARCH_RUNBOOK.md) | Concrete workflow and continuation boundaries |
| [Dataset inventory](docs/DATASETS.md) | Bundle contents, provenance and asset terms |
| [Paper](paper/README.md) | Compiled draft, editable source and claim-to-evidence map |
| [Contributing](CONTRIBUTING.md) | Testing, experiment records and contribution practice |
