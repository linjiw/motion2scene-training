# Research sync validation

Executed against the destination checkout on September 13, 2026.

- Package root: `.venv/bin/python -m pytest -q tests` — 5 passed.
- Vendored SONIC root: `PYTHONPATH=/home/linjiw/motion2scene-training/vendor/sonic /home/linjiw/groot-wbc-sonic-sim-trackb/.venv_isaaclab/bin/python -m pytest -q decoupled_wbc/tests/test_navigation_continuation.py decoupled_wbc/tests/test_navigation_recovery.py decoupled_wbc/tests/test_navigation_motor.py decoupled_wbc/tests/test_motor_recovery.py decoupled_wbc/tests/test_scene_distillation.py decoupled_wbc/tests/test_foundation_navigation.py decoupled_wbc/tests/test_bfm_coverage_navigation.py decoupled_wbc/tests/test_transformer_foundation.py decoupled_wbc/tests/test_action_flow_distillation.py decoupled_wbc/tests/test_bfm_pipeline.py decoupled_wbc/tests/test_bfm_curriculum.py decoupled_wbc/tests/test_bfm_command_quality.py decoupled_wbc/tests/test_hindsight_qualification.py` — 120 passed in 3.42 s.
- Vendored SONIC root: `/home/linjiw/groot-wbc-sonic-sim-trackb/.venv_research/bin/ruff check gear_sonic/research/scene_distillation/navigation_continuation.py gear_sonic/research/scene_distillation/navigation_panel.py gear_sonic/research/scene_distillation/navigation_motor.py gear_sonic/research/scene_distillation/navigation_recovery.py gear_sonic/research/scene_distillation/navigation_recovery_data.py gear_sonic/research/scene_distillation/navigation_takeover.py decoupled_wbc/tests/test_navigation_continuation.py decoupled_wbc/tests/test_navigation_recovery.py decoupled_wbc/tests/test_navigation_motor.py` — passed. Running from the outer package root initially produced import-classification errors; rerunning at the vendored project root resolved discovery without source edits.
- All 679 copied-file SHA-256 values match the sync manifest.
- `git diff --check` — passed before staging.

Native physics and training were not rerun. Reported physical results are preserved source evidence. Existing bundle contents and archived paper/main.pdf are unchanged; the new methods PDF is linked from the latest update.

- `.venv/bin/m2s verify` — all five bundles verified.
- Staged `git diff --cached --check` reports trailing whitespace in source-generated SVG path data. These artifacts retain exact source hashes. Excluding SVGs leaves two source-preserved extra blank lines at EOF in `methods_v2_20260913/introduction.md` and `motion2scene_section.md`. These formatting findings do not affect code or artifact interpretation.
