# Six-second development dataset

The immutable [16-episode package](/home/linjiw/research-data/groot-wbc/m2s-six-second-development-dataset-v1-aligned/manifest.json) and [210.4 MiB archive](/home/linjiw/research-data/groot-wbc/m2s-six-second-development-dataset-v1-aligned.tar.gz) contain **4,768 physical rows, 19,072 physics steps, and all 4,768 aligned sensor packets**. The [exporter](../../scripts/research/motion2scene_export_timed_dataset.py) independently re-audits the portable arrays; all **1,619 file hashes** pass. Three complete teacher decisions map exact 106D entry inputs to the nine measured scene branches. This is a development dataset, not a learned-policy result or held-out benchmark.
The separate [15-episode policy increment](TIMED_POLICY_DEVELOPMENT_V1.md) now
contains the actual new-seed comparison; this original 16-episode release remains immutable.

Archive SHA-256: `9e63d4946e8d52783c57b4ebb43105f13f671ae47d6da705172040d0dd372a95` (220,619,920 bytes). The adjacent `.receipt.json` records the completed audit and archive identity. Earlier datasets remain unchanged.

The seven captures comprise three original strict-contact qualification failures, one counterpart diagnostic, and three new environment-contact qualification executions. The new three executions also have strict-contact assessments. These are preserved as additional criteria on the same episodes, giving six strict assessments, three environment assessments, and one diagnostic assessment across seven actual recordings. Distinct physical captures remain distinct even when deterministic bytes match.

Every episode exports pickle-free physical arrays, causal sensor measurements with identity fields removed, explicit per-packet alignment/confidence, actual switch/state logs, and separate scene/contact/outcome supervision. The diagnostic preserves its four subjects and 35 native counterpart columns. The later environment captures retain complete 200 Hz normal-force arrays of shape `[1192,30,36,3]`, exact subject/counterpart mappings, net contact streams, and the original strict audit. The exporter recomputes native mapping, net-force reconstruction, physical/control clock correspondence, and sensor/physics alignment from the portable arrays. It preserves the declared environment criterion rather than relabeling the original failures.

The seven qualification/diagnostic episodes have sparse rays and measured state; no 214D, 100D, 110D, or 106D input is invented for them. The nine timed records use the exact **106D** schema, 65 ideal PhysX rays with explicit normal-known masks, and 0.5 s/26-frame history. `student_inputs.npz` uses an exact whitelist. Geometry, physical labels, and teacher outputs live separately. Actual actions derive from observed switches, including staying neutral, rather than the configured preference.

Complete timed teacher tables map each option to its exported physical episode and exact tick-15 input. Passage labels, measured time, legality, alignment eligibility, and admission remain distinct. These targets describe one complete forced schedule per option; no variable entry-time or wait-continuation interpretation is implied. Failed or incomplete attempts retain their provenance; partial sensor packets without a validated physical pair are excluded from training admission.

| Development scene | Neutral | Short | Sustained |
| --- | --- | --- | --- |
| Empty | Pass, 5.36 s | Pass, 5.86 s | Pass, 5.92 s |
| Short beam, length 0.10 m | Fail | Pass, 4.04 s | Pass, 4.06 s |
| Long passage, length 1.00 m | Fail | Fail | Pass, 4.64 s |

All nine branches are measurement-admitted, including three physical failures. Passage time is compared within a scene because finish planes differ. The nine records contain 2,682 exact sensor/physics pairs, 10,728 physics steps and 174,330 ray measurements. Command entry is tick 15/reference phase 0.30 s; the same packet's physical elapsed time is 0.28 s. Short and sustained return at reference phases 5.30 and 5.10 s, respectively. The complete 298-row episode is checked for stability and actual return.

The package retains the original collection registration, each corrected analysis snapshot and the parent `analysis_correction_registration.json`. Corrections distinguish an actual enabled collision from mere USD API presence and serialize NumPy visibility booleans; no physical rerun, raw-array change or contact-threshold change occurred. Original strict-contact failures and the separately declared environment-contact assessments remain explicit.

Raw full-skeleton priors, model weights, and whole loaded reference banks are excluded. Logged reference commands remain identified among recorded trajectory signals. Measured actuator effort is unavailable; retained implicit-PD effort estimates are not mechanical work or battery energy. Original manifests, actual attempts, result criteria, source hashes, runtime snapshots, and exporter source closure are included for inspection. Every episode remains development-only.

Validation:

```bash
.venv_isaaclab/bin/python -m pytest -q \
  decoupled_wbc/tests/test_motion2scene_timed_dataset.py \
  decoupled_wbc/tests/test_motion2scene_sensor_alignment.py
.venv_isaaclab/bin/python scripts/research/motion2scene_export_timed_dataset.py inspect
.venv_isaaclab/bin/python scripts/research/motion2scene_export_timed_dataset.py \
  audit --out /home/linjiw/research-data/groot-wbc/m2s-six-second-development-dataset-v1-aligned
```

Ten focused tests pass, including reset/alignment, source identity, multiple assessments, actual action versus preference, normal masks, and the exact input whitelist. Black and Ruff pass.

The completed release used the following command. To reproduce it, choose a new output directory:

```bash
M2S_DATA=/home/linjiw/research-data/groot-wbc
.venv_isaaclab/bin/python scripts/research/motion2scene_export_timed_dataset.py export \
  --out "$M2S_DATA/m2s-six-second-development-dataset-v1-aligned" --archive \
  --results \
  "$M2S_DATA/m2s-long-schedule-qualification-v3/result.json" \
  "$M2S_DATA/m2s-neutral-contact-counterpart-diagnostic-v1/result.json" \
  "$M2S_DATA/m2s-environment-contact-qualification-v1/environment_result.json" \
  "$M2S_DATA/m2s-timed-duration-teachers-v1/empty/result.json" \
  "$M2S_DATA/m2s-timed-duration-teachers-v1/short_beam/result.json" \
  "$M2S_DATA/m2s-timed-duration-teachers-v1/long_passage/result.json"
```

This command requires completed result files and refuses to overwrite an existing version. Counts, teacher admission, and the archive hash above come from the completed export, including its portable-data re-audit.
