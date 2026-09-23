# Included datasets and models

| Bundle | Files | Compressed bytes | Contents |
|---|---:|---:|---|
| repaired-motions | 959 | 72,126,563 | 89 screened train clips, 20 development clips, all 100 candidates, repair frames and ledger |
| research-datasets-scenes | 6782 | 1,025,813,741 | Motion2Scene data, scene/obstacle tasks, BFM labels, evaluations and historical manifests; excludes duplicate model checkpoints, archives and logs |
| robot-assets | 814 | 431,851,024 | SONIC robot models and local assets; original asset notices apply |
| checkpoints | 3 | 794,456,368 | Released initialization, previous completed teacher, and best measured BFM student; repaired-teacher checkpoints excluded |
| learned-scene-models | 41 | 19,467,408 | Learned scene/event/shape generators, probabilities and tensor labels; experimental qualification status preserved |

The archives contain actual source data, not LFS pointer stubs. Every archive and extracted file has a SHA-256 manifest. Historical absolute paths remain provenance and must not be treated as executable local configuration. The portable CLI creates fresh local configs or hash-bound derived manifests.

Research data includes the original 100 train/20 development Motion2Scene corpus and scenes, repaired v1/v1.1 variants, BFM teacher queries and DAgger data, assistance records, scene qualification failures, evaluation artifacts, and prior research packets. Repaired-teacher training output is excluded. Repeated teacher/student checkpoint histories, simulator installations, full SMPL corpus, unrelated teleoperation installers/shared libraries and duplicate archive/log files are not exported. Selected model checkpoints and all discovered non-teacher/non-BFM scene-study tensor checkpoints are included.

Code LICENSE files and `vendor/sonic/legal/` notices are preserved. The release checkpoint, robot assets and source motion data keep their original asset terms; this repository does not replace those terms with an inferred dataset license.
