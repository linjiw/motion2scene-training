# Validation performed

The package was installed editable into a newly created Python 3.11 virtual environment. The base CLI needs no third-party runtime dependencies. All five bundles were verified and extracted into `/home/linjiw/m2s-portability-validation`, separate from the source data tree.

Native integration checks used the existing Python 3.11 Isaac/SONIC dependency environment, but launched code from `vendor/sonic` and consumed the relocated data. This validates source/data relocation; it is **not** a fresh-machine GPU installation test. The new native installer was not executed and no second teacher GPU run was started.

- Teacher preparation, 128 environments / 10-iteration configuration: completed; actual CPU loader verified 89 training clips and 20 development clips, 435,897 finite values. Optimizer updates and physics steps: zero. The prepared bounded configuration was not launched.
- Student smoke: 24 updates per uniform/curriculum arm, 48 actual CPU updates; completed with matched teacher decoder. This shortened run tests packaging, not improvement over the research baseline.
- Learned scene generator: 20 actual CPU updates using the preserved 100-training-motion tensor labels; completed. Physics steps: zero.
- Scene export: 200 training tasks with collision USDs, generated from the relocated corpus.
- Visualization: a reference trajectory/height PNG was rendered at `examples/reference-preview.png`.
- 28 tests passed across package extraction and vendored BFM pipeline/coverage/command metrics/curriculum tests. Scoped Ruff and Black checks passed for the new package code.

Receipts are retained in `validation-evidence/`. The running source teacher was not modified. Large datasets remain LFS content, so a normal Git-only archive is insufficient: distribute the LFS objects too, or use a clone with `git lfs pull` from a remote that contains those objects.

Useful checks after installation:

```bash
m2s verify
m2s doctor
python -m pytest -q tests
```

The CPU viewer optional dependencies and native GPU bootstrap are declared separately. Unsupported simulator/GPU combinations may need environment changes; the package does not claim cross-platform native physics validation.

## Git/LFS distribution check

A fresh local clone of commit `34d7493` automatically materialized the LFS objects. Running the cloned package against its own repository verified all five data archives. This checks local Git/LFS distribution; no remote repository has been created or uploaded.
