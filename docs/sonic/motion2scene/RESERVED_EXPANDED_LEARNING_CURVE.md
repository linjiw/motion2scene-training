# Reserved nominal comparison for the expanded acquisition study

The [immutable proposal](/home/linjiw/research-data/groot-wbc/m2s-reserved-expanded-curve-proposal-20260909-v1/proposal.json)
and [complete assignment preview](/home/linjiw/research-data/groot-wbc/m2s-reserved-expanded-curve-proposal-20260909-v1/assignments_preview.json)
cover the actual five-arm M8/M16/M32 study. Proposal SHA-256:
`eff63828bceac394f62f4a62d9897e639a131106ec65f44f506714bdc3766584`.
This is a proposed allocation and implemented execution path. **No reserved
policy has been prepared or executed, and this proposal is not an adoption.**

The old V6 proposal and its 972-episode M2/M4 assignment remain unchanged. The
new comparison retains the same 18 V3 layouts, execution seeds 94301/94302,
all 162 stored world-space variants, seven-schedule repertoire, sensor and
physical scoring rules. The eight stress offsets remain reserved and disabled,
as in the previous nominal scope. It makes no stress-robustness or new-carrier
claim. No robot clearance or outcome was queried on reserved geometry.

| Component | Policies | Layouts × execution seeds | Unique episodes |
| --- | ---: | ---: | ---: |
| Five arms × three corpora, M8 | 15 | 18 × 2 | 540 |
| Same corpora, M16 | 15 | 18 × 2 | 540 |
| Same corpora, M32 | 15 | 18 × 2 | 540 |
| Strong script and all seven fixed schedules | 8 | 18 × 2 | 288 |
| Total | 53 | 36 conditions per policy | **1,908** |

The maximum allocation is 2,274,336 evaluation physics steps. Actual measured
steps, unknown reservations and unexecuted slots remain separate. The 288
baseline episodes are shared across checkpoint analyses and counted once.
Including all seven fixed schedules provides matched finite-bank capability
without selecting a baseline using reserved outcomes. M32 passage is the final
checkpoint outcome; M8/M16/M32 provide the nested learning curves. This count
is derived from this proposal, not a universal requirement for Motion2Scene.

## Implementation and measurement

The [runner](/home/linjiw/groot-wbc-sonic-sim-trackb/scripts/research/motion2scene_reserved_learning_curve.py)
uses the existing reserved protocol validator, native collector, batch preparer
and charged-attempt controller. It does not edit their frozen sources. It binds
all 45 policies to their actual assigned acquisition results, teacher prefixes,
common ridge λ=10 and original weighting rules. The teacher remains scene-wise
complete continuation; the information-consistent extension does not supply
these models. The script is the same development-selected parameter artifact
used by the M8 panel.

Inventory freezing requires all fifteen M32 acquisition boundaries, the completed
M8 panel and the complete 318-episode development learning curve. The development
summary is regenerated from the exact assigned result receipts before freezing.
This is a receipt/summary check, not another raw-capture audit. Model files and
acquisition-cost receipts are rebound at launch preflight. An earlier checkpoint
cannot fill a missing future slot. Adoption cannot change the proposal's method,
assignment, scoring or reporting rules.

Before the first launch, the runner checks every child in the complete assignment,
including the exact forced-schedule identity. This supplements the unchanged
reserved validator's effective command/model checks. Physical execution shares
both acquisition locks, retains the 7,500 MiB GPU floor and 30 GiB disk floor,
and runs one assigned episode at a time. An episode pause boundary changes only
scheduling. Known attempts are re-audited on resume; an unresolved attempt is
retained and charged without an automatic retry. Byte-identical native collision
inventories may share storage; their paths and bytes remain available.

The [statistics reader](/home/linjiw/groot-wbc-sonic-sim-trackb/scripts/research/motion2scene_reserved_curve_statistics.py)
uses JSON and NumPy without importing the robot stack or reading model weights.
It preserves all 1,908 rows, including `not_run` and `technical_missing`, verifies
their assignment/model/acquisition metadata, and writes 45 per-corpus curve
points. Costs are the recorded cumulative bootstrap, teacher and pre-update
student physics, not assigned maxima or evaluation repeats. Proposal computation
remains available separately through the acquisition plan.

Finite-bank capability uses all seven measured schedules on each of the 36
conditions. Unknown bank outcomes produce explicit bounds. Bank-unsolvable
conditions stay in policy denominators, and a policy-only success is reported
as a repeat-level disagreement. The script and fixed schedules are ordinary
shared comparators, not independent copies for each corpus or checkpoint.

Reporting includes the four construction/replay contrasts at each checkpoint,
each arm against the script and all seven fixed schedules, and each arm's two
successive checkpoint changes: 142 descriptive comparisons. M32 passage and
the construction comparisons address the core claim; the remaining comparisons
support interpretation. There is no multiplicity-adjusted significance claim.
The existing 20,000-draw paired corpus-by-layout bootstrap, seed 202609081822,
keeps both execution seeds together and reuses the same resampling weights.
Three corpus seeds do not supply broad population coverage. Intervals are
withheld when assigned paired outcomes are unknown.

Time comparisons use only mutually successful matched conditions and report
their counts. Time means crossing plus stabilization, with full-horizon return
and recovery still required for task passage. Contact/fall flags remain distinct:
undesired environment normal force above 1 N excludes permitted foot-floor
support; the recorded fall/upright threshold uses root height 0.5 m and minus
projected gravity Z 0.5. Other failures, including invalid initial approach,
are not renamed collisions or falls. No failed-success time or energy is imputed.

## Evidence produced now

[Validation receipt](evidence/research-progress-20260909/reserved_curve_validation.json)
binds the proposal, assignment preview, source/test files and actual negative
preflights. All 106 focused tests pass, including 21 new tests. The existing
40 invalid-escape warnings come from parsing an inspected source file; Black
and Ruff pass on the four new files.

Synthetic full-matrix tests exercise charged pause/resume, unknown attempts,
late-child corruption detection, exact fixed schedules, cost metadata,
bank-unsolvable denominators, shared-baseline counts and the full 20,000-draw
statistics/CSV path. Synthetic outcomes are software validation only. The real
current study fails inventory freezing and adoption at its missing M32 gates,
and its unadopted proposal fails preparation, before creating output directories.
All six preserved predecessor fields compare equal. No physics was consumed.

## Reproduction and remaining execution

From the repository root, the readiness check has been executed:

```bash
M2S_RESERVED_ROOT=/home/linjiw/research-data/groot-wbc/m2s-reserved-expanded-curve-proposal-20260909-v1

OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_reserved_learning_curve.py check \
  --protocol "$M2S_RESERVED_ROOT/proposal.json" \
  --out "$M2S_RESERVED_ROOT/execution"
```

The following steps are implemented but **unexecuted**. First finish the
[development curve](DEVELOPMENT_LEARNING_CURVE.md). Then bind its actual complete
statistics and every acquired model:

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_reserved_learning_curve.py freeze-inventory \
  --protocol "$M2S_RESERVED_ROOT/proposal.json" \
  --development-statistics /home/linjiw/research-data/groot-wbc/m2s-development-learning-curve-statistics-20260909-v1/result.json \
  --out "$M2S_RESERVED_ROOT/frozen_inventory"
```

Before adoption, the collaborator must review and assemble `gate_evidence.json`
with `reviewed: true`, `evaluation_outcomes_inspected: false`, and a
`gate_evidence` object. Each of the following existing validator keys requires
a nonempty list of actual `{path, sha256}` artifact references:

- `seven_schedules_physically_qualified`
- `exact114_runtime_smoke_complete`
- `two_beam_full_horizon_runtime_qualified`
- `terminal_and_failure_scorer_tested`
- `protocol_validator_integrated`
- `strong_multi_option_script_development_validated`
- `all_comparison_models_and_thresholds_frozen`
- `complete_source_closure_and_environment_frozen`
- `no_reserved_outcomes_inspected`

No such completed evidence bundle is asserted here. Missing development/model
evidence must be completed, not marked passed from a plan or test count. The user
has requested this protected research workflow; the evidence bundle records
scientific preconditions and does not itself require another permission request.

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_reserved_learning_curve.py adopt \
  --protocol "$M2S_RESERVED_ROOT/proposal.json" \
  --inventory "$M2S_RESERVED_ROOT/frozen_inventory/inventory.json" \
  --runtime-freeze /home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-common-freeze-v4/runtime.json \
  --gate-evidence "$M2S_RESERVED_ROOT/gate_evidence.json" \
  --out "$M2S_RESERVED_ROOT/adoption"

OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_reserved_learning_curve.py prepare \
  --protocol "$M2S_RESERVED_ROOT/adoption/protocol.json" \
  --out "$M2S_RESERVED_ROOT/execution"

OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_reserved_learning_curve.py run \
  --protocol "$M2S_RESERVED_ROOT/adoption/protocol.json" \
  --out "$M2S_RESERVED_ROOT/execution" --max-new-episodes 36

OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_reserved_curve_statistics.py \
  --input "$M2S_RESERVED_ROOT/execution/result.json" \
  --out "$M2S_RESERVED_ROOT/statistics"
```

The 36-episode pause boundary is scheduling only; repeating the run command
continues the same 1,908 assignments. Unknown attempts require inspecting the
retained evidence. Do not delete or rename an experiment to repeat them. The
statistics reader also accepts a paused `evaluation_status` event and retains
the full denominator. Use a fresh statistics output directory for each report.
Preparation and adoption retain partial files on errors; inspect them before
continuation. No worker has been started for this proposed evaluation.

Validation command actually run:

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python -m pytest -q \
  decoupled_wbc/tests/test_motion2scene_reserved_learning_curve.py \
  decoupled_wbc/tests/test_motion2scene_reserved_curve_statistics.py \
  decoupled_wbc/tests/test_motion2scene_reserved_runner.py \
  decoupled_wbc/tests/test_motion2scene_evaluation_protocol.py \
  decoupled_wbc/tests/test_motion2scene_development_learning_curve.py \
  decoupled_wbc/tests/test_motion2scene_evaluation_statistics.py
```

The single next performance experiment remains the 138-episode M8 development
panel. This implementation removes a later execution/reporting incompatibility;
it does not establish the core performance hypothesis or complete the project.
