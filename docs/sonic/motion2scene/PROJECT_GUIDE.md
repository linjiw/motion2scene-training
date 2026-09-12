# Motion2Scene traversal project

Motion2Scene connects controller-executed motion alternatives to training scenes
and sensor-based traversal decisions. The current task is overhead traversal
along a fixed approach, using a frozen SONIC tracker. Seven qualified schedules
offer three legal entry times, two authored adaptation durations, and a separately
qualified projected/spliced prior sample. Each encounter permits one adaptation
and its mandatory return. Stopping, steering, and repeated adaptation cycles are
not yet qualified.

## Current method and evidence

The [option-extension study](OPTION_EXTENSION_STUDY.md) now compares four fresh
prior samples with four authored duration settings under the same repair and
qualification budget. All eight references are prepared; their 17-episode
physical qualification is separate from the unchanged primary teaching bank.

The [information-consistent teaching study](INFORMATION_CONSISTENT_TEACHING.md)
now tests delayed access to recorded scene summaries: a 1.00 s reveal retains
six-context finite-policy capability, whereas a 1.40 s reveal reduces it to five.
This exposes a concrete difference between scene-wise continuation success and
an observation-consistent decision policy. The separate
[sensor sensitivity study](SENSOR_SENSITIVITY_STUDY.md) implements causal dropout,
range noise and latency, reproduces all 18 nominal development decision vectors,
and completes four native short-passage checks. All pass; latency changes the
selected option and passage time on that context. Broader physical evaluation
remains separate.

The [expanded construction study](EXPANDED_DEVELOPMENT_STUDY.md) now compares
commanded and executed envelopes on 3,840 common candidates and defines matched
five-arm teaching curves through 32 encounters. Its physical continuation is
queued after the original M4 acquisition; its completed geometric results are
reported separately from traversal outcomes.

The [September 9 decision-learning experiments](DECISION_LEARNING_RESULTS_20260909.md)
test separate feasibility/time learning, observation-consistent continuations,
and fixed-data replay controls. Both learner families select passing recorded
branches on 4/6 whole-context holdouts; replay controls do not improve that count.
These completed CPU studies motivate broader acquisition before final evaluation.
The [revised manuscript](submission/traversal_method_v2.pdf) develops the method
and reports its measured scope.

A complete development table contains six contexts and all seven schedules:
42 episodes and 50,064 recorded physics steps. An early beam requires the prior
option; a complementary passage requires a sustained authored option. Every
context has a passing option, while the strongest constant schedule passes five
of six. This measures available capability on inspected development scenes.
Actual learned-policy and curriculum gains on reserved layouts remain pending.

The complementary passage was constructed from executed geometry before its
seven physical outcomes. Both sustained schedules pass in 4.20 s with zero
measured beam force, while both prior schedules fail. Its underside is measured
before the legal decisions; waiting until the 1.40 s entry phase retains a
physically verified sustained continuation. The geometric 81-offset screen is
separate from this nominal physical verification.

The [five-context experiment](SEVEN_SCHEDULE_DEVELOPMENT_EXPERIMENT_V1.md)
documents the original 35 branches, common learner selection and baselines.
The complementary scene remains a separate follow-up; it was not added to that
selection panel. These [recorded early-beam trajectories](/home/linjiw/research-data/groot-wbc/m2s-prior-capability-visual-v1/schedule_execution.mp4)
and [complementary-passage trajectories](/home/linjiw/research-data/groot-wbc/m2s-complementary-capability-visual-v1/schedule_execution.mp4)
show actual Isaac Lab states rendered with MuJoCo forward kinematics. Rendering
adds no physical simulation and does not replace the native contact records.

## Reuse the dataset

The current [42-episode archive](/home/linjiw/research-data/groot-wbc/m2s-timed-schedule-six-context-dataset-v1.tar.gz)
contains 12,516 aligned 114-dimensional sensor/state inputs, 50,064 physics steps,
and 18 available phase-specific teacher tables (17 consequential decisions).
All 42 episodes have complete measurements: 28 passages and 14 physical failures.
Failed alternatives, unknown-space masks, exact command timing, native contact
measurements and file provenance are retained. Privileged scene geometry and
outcome labels are separate from student inputs. All data are development data;
this release is not a navigation or held-out evaluation benchmark.

An optional [compact archive](PORTABLE_SIX_CONTEXT_DATASET_V1.md#compact-archive-alternative)
preserves every file path and byte while reducing the download to 223.8 MB and
the measured allocated extraction from 8.22 GB to 2.67 GB. The original archive
and the same baseline quickstart remain available.

The NumPy baseline toolkit is included in the archive. It requires NumPy;
it does not require Isaac Lab or model weights. From the extracted dataset directory:

```bash
python -m pip install -r tools/portable_baseline/requirements.txt
python tools/portable_baseline/scripts/research/motion2scene_schedule_dataset_baseline.py inspect \
  --dataset . \
  --out /absolute/path/to/new-inspection
python tools/portable_baseline/scripts/research/motion2scene_schedule_dataset_baseline.py fit \
  --dataset . \
  --l2 10.0 --out /absolute/path/to/new-fit
```

Output directories must be new. The tool verifies inventory hashes, reconstructs
legal WAIT continuations from physical branch labels, and fits the same
phase-specific ridge policy. An isolated reconstruction verified 1,806 files
and reproduced all 18 teacher-packet and 84 recorded neutral-packet decisions.
Those 84 checks include repeated matched histories, not 84 independent contexts.
The largest weight difference was 4.39e-17; the model archives are not byte-identical.
The [verification receipt](/home/linjiw/research-data/groot-wbc/m2s-portable-schedule-model-comparison-six-context-v1/result.json)
records the exact comparison. Native-asset qualification requires the separate
simulator audit; the NumPy reconstruction does not claim to repeat it.

The bundled physical audit uses the original recorded scorer snapshot. A later
upstream-approach correction is supplied as a separate receipt; all 42 original
starts pass that added check with unchanged outcomes. It is not presented as code
used during the original executions. The
[release note](PORTABLE_SIX_CONTEXT_DATASET_V1.md) gives exact hashes and criteria.
The [original 35-episode archive](/home/linjiw/research-data/groot-wbc/m2s-timed-schedule-five-context-dataset-v1.tar.gz)
remains unchanged.

The separate [actual policy archive](/home/linjiw/research-data/groot-wbc/m2s-six-context-policy-dataset-v1.tar.gz)
contains all 36 development assignments: 35 complete measured episodes and one
partial verified contact failure. It retains 10,619 physical rows, 10,618 sensor
packets and 42,476 physics steps. The partial attempt contributes 189 physical
rows, 188 packets and 756 steps; its passage cost stays null. This policy panel
adds no teacher targets, so use the 42-episode archive above for the baseline fit.

Its original 20-run and subsequent 16-run children keep separate frozen scorer
versions. Their native audits reproduce the original passage receipts, while a
separate partial audit reconstructs contact and outcome from preserved arrays.
The [policy release note](PORTABLE_POLICY_PANEL_V1.md) gives extraction instructions,
exact hashes, scoring criteria and the standalone audit command. From that
extracted policy dataset directory:

```bash
python -I motion2scene_export_policy_panel.py audit --out .
```

## Method and implementation

[![Implemented observation-curriculum rounds](/home/linjiw/research-data/groot-wbc/m2s-method-overview-figure-v2/method_overview.png)](evidence/traversal-v2/method_overview.pdf)

The figure begins after each corpus's bootstrap. Its 81-offset construction is
geometric, and replay uses historical gaps from the recorded generating models.
No positive eligible gap gives 0.8 uniform plus 0.2 coverage; this implemented
flow does not establish a curriculum advantage.

```mermaid
flowchart LR
  A[Prior references and legal schedules] --> B[Matched controller executions]
  B --> C[Executed body envelopes]
  C --> D[Solution-preserving scene proposals]
  D --> E[Physical alternatives and sensor histories]
  E --> F[Passage-first continuation teacher]
  F --> G[Common sensor policy]
  G --> H[Actual student execution]
  H --> I[Verified observable teacher-student gap]
  I --> F
```

| Component | Entry point | Contract |
| --- | --- | --- |
| Qualified schedules | `motion2scene_prepare_schedule_bank.py` | Executed entry, adaptation and return; no assumed raw-sample feasibility |
| Scene construction | `motion2scene_prepare_acquisition_pools.py` | Positive clearance over finite offsets, nominal negative contrast, complete query accounting |
| Sensor and execution | `motion2scene_collect_timed_schedules.py` | 65 native rays, two-second floor/ceiling history, unknown masks, legal decisions |
| Physical teacher and learner | `motion2scene_train_timed_schedules.py` | Matched recorded prefixes, complete legal continuations, passage before measured time |
| Curriculum replay | `motion2scene_build_timed_replay.py` | Actual student outcomes, causal current observations, uniform/coverage/verified-gap mixture |
| Independent acquisition | `motion2scene_run_primary_acquisition.py` | Bootstrap confined to its acquired corpus, student before current teachers, fixed slots/budgets and immutable reuse receipts |
| Portable learning | `motion2scene_schedule_dataset_baseline.py` | NumPy reconstruction from released observations and physical labels |

Entry points are under `scripts/research/`; runtime and pure algorithm modules
are under `gear_sonic/dataset_generation/hallucination/`. The
[acquisition controller guide](PRIMARY_ACQUISITION_CONTROLLER_V1.md),
[sensor interface](TIMED_SCHEDULE_SENSOR_ADAPTER_V1.md),
[replay specification](TIMED_SCHEDULE_REPLAY_V1.md), and
[dataset format](TIMED_SCHEDULE_DATASET_V1.md) contain detailed contracts.

Analytic construction remains primary. The existing conditional learned
proposal produced fewer accepted geometric proposals and used more search
work than the analytic comparator. Learned/adversarial generation must establish
an acquisition or downstream benefit before becoming a performance claim.

## Current research boundary

The common 114D learner uses the development-selected ridge penalty 10.0 in
every constructor arm. The completed six-context comparison contains 36 actual
policy assignments at seed 8732: the updated learner and updated script each
pass 6/6; the original learner, original script and constant prior pass 5/6;
walking passes 2/6. The updated policies take an additional 0.50 s on both the
short and long passages. The learner matches the updated script's passage times
except for a 0.02 s saving on the empty scene. These inspected layouts establish
a development data-extension effect, not a curriculum or generalization gain.

The [combined audit](/home/linjiw/research-data/groot-wbc/m2s-six-context-policy-comparison-audit-v1/result.json)
reproduces all archived outcomes and decisions: 35 complete measurements and one
partial known contact failure, zero unknown outcomes, and 42,476 recorded physics
steps. The partial walk attempt retains its 756 steps and null successful time.

The [adopted continuation and first-model receipt](PRIMARY_ACQUISITION_TIE_CONTINUATION_V1.md)
fix four arms, three separately acquired corpora per arm, and 468 assigned
episodes, at most 557,856 physical steps. All 12 configurations passed checks.
At 04:34:26 UTC on September 9, the dispatcher completed the first analytic M0
from its original seven bootstrap captures. Two phases use consequential labels;
the exactly tied all-success 1.00 s table initializes zero regret under the newly
adopted common rule. This is not a downstream traversal result.

The original startup failure, failed fit and seven physical outcomes remain
preserved. An archived original-interpreter audit admits reuse only within the
same corpus, charging its 8,344 steps once. Other corpora acquire their own
bootstrap; no development model supplies initialization. The applied source
passes 179 tests and is bound by common freeze V4 and native/environment freeze
V6. These software checks are separate from measured traversal evidence.

The [V8 ledger](/home/linjiw/research-data/groot-wbc/m2s-traversal-acquisition-ledger-v8.json)
is a fixed snapshot through that original bootstrap: 231 complete captures and
one partial study failure, 234,380 study steps, plus 796 instrumentation steps.
A separate 1,192-step pre-simulator reservation has zero recorded physics.
Later primary acquisitions are outside this snapshot. The reserved 972-episode
V6 evaluation and reporting V3 remain separately unadopted and unexecuted until
all 24 primary checkpoint models and the remaining gates are complete.

The [continuation watcher is adopted](/home/linjiw/research-data/groot-wbc/m2s-reserved-continuation-adoption-v1/registration.json)
and running. Its [05:17 UTC launch receipt](/home/linjiw/research-data/groot-wbc/m2s-reserved-continuation-adoption-v1/launch_receipt.json)
records four completed M0 boundaries, an active primary dispatcher, and zero
dependent stage intents. This authorizes the guarded continuation workflow;
reserved evaluation still requires its separate adoption after all 60 primary
boundaries, all 24 checkpoint models, and all nine gates validate. At launch,
primary tool session 74390 used PID 740247; watcher session 83194 used PID 773589.

For current progress, read the mutable [watcher log](/home/linjiw/research-data/groot-wbc/m2s-reserved-continuation-adoption-v1/run.log)
and run the [read-only primary monitor](/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-monitor-v2/monitor.py):

```bash
tail -n 40 /home/linjiw/research-data/groot-wbc/m2s-reserved-continuation-adoption-v1/run.log
python3 /home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-monitor-v2/monitor.py
```

The launch receipt is a fixed snapshot; the log and monitor show later progress.
An unknown attempt or nonresource stage failure stops automatic continuation.

The [working manuscript](submission/traversal_method_v2.pdf) records implemented
interventions and measured development outcomes. The original
[diagnostic manuscript](submission/paper.tex) remains unchanged. Applied
mechanical work is unavailable from the current implicit actuator; passage time,
contact, stability and switching are reported separately. No battery-energy,
hardware-transfer, unseen-motion-source, or general navigation result is claimed.
