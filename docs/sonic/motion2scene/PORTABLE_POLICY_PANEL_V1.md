# Portable actual six-context policy panel

The [policy archive](/home/linjiw/research-data/groot-wbc/m2s-six-context-policy-dataset-v1.tar.gz)
preserves the completed 36-assignment development comparison. It complements the
[42-episode teacher release](PORTABLE_SIX_CONTEXT_DATASET_V1.md). It adds no
physics and produces no teacher targets from unmatched policy executions.

| Recorded content | Count |
| --- | ---: |
| Assigned policy attempts | 36 |
| Complete measured episodes | 35 |
| Partial, independently verified physical failures | 1 |
| Passed / failed / unknown outcomes | 29 / 7 / 0 |
| Physical control rows, including the partial attempt | 10,619 |
| Actual 114D sensor/state packets | 10,618 |
| Recorded 200 Hz physics steps | 42,476 |
| Teacher targets | 0 |
| Inventoried portable files | 5,974 |

All six layouts were development data. The comparison uses physics seed 8732
and is a data-extension and script-retuning check, not independent acquisition
evidence or a held-out generalization result. The
[completed comparison](SIX_CONTEXT_POLICY_COMPARISON_V1.md) reports the policy
outcomes and measured time regressions. The dataset preserves all original
assignments, models' hash identities, script parameters, commands, sensor
histories, complete trajectories, and physical failures.

## Two source versions

The top-level schema is `motion2scene_policy_panel_portable_v1`. Its children
retain the existing `motion2scene_timed_schedule_portable_dataset_v1` schema:

| Child | Assigned attempts | Scoring version |
| --- | ---: | --- |
| `children/original20_pre_fix` | 20 complete | Original scorer, before the upstream-start correction |
| `children/extension16_post_fix` | 15 complete + 1 partial | Scorer used after the upstream-start correction |

Each child bundles its own unchanged scoring modules and exact physical passage
receipts. The wrapper never substitutes one scorer into both batches or forces
their different passage dictionaries into one default representation.
`SCORING_VERSIONS.json` records the exact source hashes. Original batch plans,
the separate upstream correction and compatibility receipts, and the combined
physical audit remain under `provenance`.

The initial packaging attempt completed the original 20-episode child, then
stopped when a strict source-read guard rejected a runtime-asset identity check
outside the exporter's import closure. That attempt is retained separately.
The second packaging registration preserves the completed child unchanged and
copies 36 declared Python runtime-asset files after verifying their original
SHA-256 values. Only `pathlib` identity reads are redirected to those frozen
bytes. Imported code remains in the original archived source closure. Neither
scorer nor any original recording changed.

## Partial attempt

The complementary always-walk run has **189 physical rows, 188 sensor packets
and 756 physics steps**. Its all-body counterpart stream independently shows
385.2182922363281 N of undesired beam contact before the runtime abort. This is
a known physical traversal failure with incomplete measurement, not a completed
episode or an unknown outcome. Passage time, whole-episode time and mechanical
work remain null.

The partial recording is converted to numeric NPZ plus reconstructable metadata.
Its original sensor/interface packets, all-body net forces, per-body counterpart
forces and native filter ordering are preserved. The top-level auditor recomputes
contact completeness and the frozen attempt classifier from these arrays, then
checks the exact recorded outcome, counts, costs and phase availability. It does
not pad the missing packet or assign teacher labels to later phases.

A separate [partial-input check](/home/linjiw/research-data/groot-wbc/m2s-six-context-policy-dataset-export-v2/partial_sensor_validation.json)
verifies all 188 feature vectors are finite and 114-dimensional, and all 188
packets exactly match their own physical prefix under the bundled alignment
helper. The remaining physical row is retained without a sensor counterpart.

## Standalone audit

The 580 MB gzip archive expands to approximately 7 GB. It contains `DATA_CARD.md`,
the audit CLI and both frozen source-version children. The standalone check was
tested with Python 3.11.16 and NumPy 2.4.6. With Python and NumPy available, extract
the archive into an existing destination directory:

```bash
tar -xzf /absolute/path/to/m2s-six-context-policy-dataset-v1.tar.gz \
  -C /absolute/path/to/datasets
cd /absolute/path/to/datasets/m2s-six-context-policy-dataset-v1
python -I motion2scene_export_policy_panel.py audit --out .
```

The auditor launches one isolated process per child. The tested native audit
blocks the original project and other experiment directories, plus Torch,
Isaac, Omni, USD, ONNX Runtime and Joblib imports. Both processes resolve all
43 imported project modules inside their respective archived source closure.
Complete episodes have their clocks, sensor alignment, geometry, all-beam
counterpart forces, guarded schedules and passage costs recomputed. The partial
attempt receives its separate native contact/outcome audit. Every portable slot
is also checked against the completed independent 36-run audit and original
policy invocation.

The archive excludes pretrained weights, raw motion-prior files and full loaded
reference banks. Recorded reference commands remain execution data. It does not
newly qualify absent robot/motion assets, certify full collider crossing, provide
route navigation, establish hardware transfer, or demonstrate repeated
adaptation. Sensor normals remain ideal PhysX queries, and mechanical work is
unavailable from the inherited implicit actuator estimate.

## Release identity

| Artifact | SHA-256 |
| --- | --- |
| Dataset manifest | `ea3d03555ed3f705dcbeee6ec433498239d38015111ec4429d0cfb40f2e5f9f8` |
| Native audit | `c094a935194ffafff88d7e047725750964b7a3215e2ed837d0f6636ba9152d81` |
| Successful export registration | `8799800ea87a6c16c45be1fbe0d02b3437751efc823c7ab885ff38e97a7f89d9` |
| Gzip archive, 580,257,707 bytes | `1b70c244a25d2de1843b6377a508ac9b9d5bc9b7da16a3e733f752f0e18c6ff0` |
| Final release receipt | `b360ac318237beec316c1455b88119c2778d9ef1e6c0f322defffa06baed0ae0` |

The [native audit](/home/linjiw/research-data/groot-wbc/m2s-six-context-policy-dataset-v1/audit.json)
contains both source-version receipts and the independently reconstructed partial
contact failure. Earlier teacher and policy artifacts remain unchanged.
The [final release receipt](/home/linjiw/research-data/groot-wbc/m2s-six-context-policy-dataset-export-v2/result.json)
binds the archive, source registrations, both successful native audits and the
partial-input check. Ten focused wrapper tests and Ruff pass.

A separate [extraction check](/home/linjiw/research-data/groot-wbc/m2s-six-context-policy-dataset-usability-v1/result.json)
matches all archive file names and sizes to the extracted inventory, verifies
149 selectively extracted files, and runs the extracted CLI's help successfully.
The sampled complete inputs load as 298×114 arrays; partial inputs load as
188×114 arrays alongside 189 preserved physical rows. No physical audit was
repeated for this packaging check.
