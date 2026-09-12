# Sensor sensitivity: causal interventions and physical evaluation

The traversal policy uses range returns and measured robot state. This study
tests three specific departures from ideal sensing while retaining the same
motion bank, tracker, 114-dimensional policy interface and physical passage
criterion. It is a simulation sensitivity study, not a depth-camera transfer
experiment.

## Interventions

Channel dropout independently removes each fan channel at each capture. A lost
channel contributes no free-space or surface evidence; a valid ray with no hit
still contributes free-space evidence. Range noise adds a zero-mean Gaussian
perturbation to each detected surface distance. Distances outside the sensor's
valid interval become missing returns instead of clipped surfaces. Surviving
surface normals and robot pose remain ideal.

Fixed latency delays the complete corrupted packet. The world-registered history
is updated only when that packet arrives, using its original capture time and
world-space ray origin. History is summarized relative to the current robot
pose. The input includes observation age; neither clean rays nor future packets
enter the student features.

Random draws are indexed by corruption seed, capture frame and fixed fan-channel
index, with separate streams for dropout and range error. This pairs corruption
across policies and severity levels without making random-number consumption
depend on the robot's motion or the number of surface hits.

The separate implementation is
[sensor perturbations](../../scripts/research/motion2scene_sensor_perturbations.py),
[runtime](../../scripts/research/motion2scene_perturbed_execution.py), and
[episode collection/scoring](../../scripts/research/motion2scene_sensor_episode.py).
Primary acquisition continues with its original ideal-sensor runtime. The
separate sensor recorder marks its own schema and effective latency explicitly;
it does not reinterpret the original zero-delay protocol.

## Completed recorded-input experiment

Replaying the six development contexts with zero perturbation reconstructs all
18 nominal decision vectors exactly. With each of 10% or 30% dropout, 1 or 3 cm
range-noise standard deviation, and 40 or 100 ms latency, all 18 vectors change.
The existing six-context ridge policy selects passing recorded branches in all
seven conditions. Under each nonzero condition, the selected short-passage
schedule changes from late sustained to late prior; other complete selections
remain unchanged. The corruption seed is 95001.

These are branch-table proxies computed from recorded neutral prefixes, not
new physical policy executions. They verify the intervention path and motivate
a controlled physical check of the short-passage decision change. They do not
establish robustness or a benefit from adding noise.

[Recorded experiment and results](/home/linjiw/research-data/groot-wbc/m2s-sensor-replay-development-20260909-v1/result.json)
bind the source data, policy and implementations. `source_snapshot.py` in that
directory preserves the executed replay driver before a formatting-only edit.

## Fixed native development smoke study

Four episodes use the existing short passage, existing six-context policy and
physics seed 8732: nominal sensing, 30% dropout, 3 cm range noise and 100 ms
latency. Each perturbation is applied separately with corruption seed 95001.
The study uses 4,768 maximum physics steps, accounted separately from acquisition.
Its scene was selected to exercise the three interventions and investigate the
recorded short-passage ranking change—not to rank methods on held-out tasks.

Each complete capture must reproduce every recorded student feature exactly
from its corrupted and causally delivered raw ray packets. Physical scoring
retains the original measured-contact, stability, passage, legal-schedule and
return criteria. All attempted outcomes are retained; incomplete or unknown
outcomes are not automatically retried.

The [fixed smoke study](/home/linjiw/research-data/groot-wbc/m2s-sensor-native-smoke-20260909-v2/study.json)
ran between intact acquisition batches. Its orchestration paused only the
primary dispatcher, allowed the active batch to finish, and resumed dispatch
after all four episodes completed.

All four episodes pass the original physical criteria, with zero measured beam
normal force and legal return to neutral. All 1,192 control-row feature vectors
reconstruct exactly; the study records 4,768 physics steps.

| Sensing condition | Committed schedule | Passage time |
| --- | --- | ---: |
| Nominal | Sustained, entry 1.40 s | 4.10 s |
| 30% channel dropout | Sustained, entry 1.40 s | 4.10 s |
| 3 cm range-noise standard deviation | Sustained, entry 1.40 s | 4.10 s |
| 100 ms packet latency | Prior, entry 1.00 s | 3.60 s |

[Complete physical results](/home/linjiw/research-data/groot-wbc/m2s-sensor-native-smoke-20260909-v2/result.json)
retain the fixed assignments, policy choices, trajectories and contact records.
Dropout and range noise do not reproduce the recorded-prefix proxy's change to
the prior schedule; those teaching captures used physics seed 8731, whereas this
physical policy study uses 8732. The latency condition does change commitment
and reduces measured passage time by 0.50 s on this context. This is not a reason
to add latency as a policy improvement: it identifies a timing-sensitive cost
decision that the learner should resolve from timely observations.

## Broader evaluation

After the common learner and checkpoint are selected on development data, the
same axes should be applied to the frozen learned policy and strong script,
using paired physics and corruption seeds. Include nominal counterparts and
report assigned passage counts, failures, and times on mutually successful
conditions. Do not choose the evaluated checkpoint from perturbed outcomes.
This study does not validate perception from rendered depth images, correlated
sensor failure, noisy robot localization, or hardware execution.
