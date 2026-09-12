# Proposed V4 checkpoint evaluation

**Proposed only: no adoption, reserved clearance query, or reserved physical
execution.** The [new specification](TRAVERSAL_PROTOCOL_AMENDMENT_V4_PROPOSED.json)
has SHA-256 `966d45ed93066a73a945a55f33c769459fe4c1736a455d4eec4ba581d40de2fd`.
The V3 proposal, its erratum, and both geometry locks remain unchanged.

V4 adds a second available checkpoint to the proposed nominal comparison so
the experiment can measure learning across acquisition budgets. M2 and M4
come from the same serial acquisition trajectory. There are still three
independent corpus seeds per constructor, not six independent checkpoint
corpora. The same three baselines are evaluated once and compared with both
checkpoints.

| Checkpoint | Completed acquisition rounds | Assigned acquisition episodes | Maximum assigned acquisition steps |
| --- | ---: | ---: | ---: |
| M2 | Bootstrap and rounds 1–2 | 23 | 27,416 |
| M4 | Bootstrap and rounds 1–4 | 39 | 46,488 |

These are assigned maxima. Learning-curve plots must use the actual measured
step totals from each checkpoint's recording receipts and disclose any
difference. Both checkpoints remain within each corpus's 50,000-step cap.
Saving and evaluating M2 adds no training physics and does not implement the
retained 150,000/450,000-step proposals.

The primary matrix is **24 learned checkpoints + 3 fixed baselines**, each on
all **18 nominal layouts × two physics seeds = 36 episodes**. This gives
**972 assignments**, at most **1,158,624 recorded evaluation physics steps**.
All eight stress variants remain preserved but disabled. No layout, source
ancestry, route placement, physics seed, or finite-horizon rule is changed.
Source ancestry is reused; the evaluation cannot establish motion-source
holdout, general navigation, or hardware transfer.

The [972-row preview](/home/linjiw/research-data/groot-wbc/m2s-evaluation-v4-proposal-preview-v1/assignments.json)
and [24-checkpoint inventory template](/home/linjiw/research-data/groot-wbc/m2s-evaluation-v4-proposal-preview-v1/inventory_template.json)
are non-executable. They contain future canonical paths, not invented model
artifacts. Their [receipt](/home/linjiw/research-data/groot-wbc/m2s-evaluation-v4-proposal-preview-v1/result.json)
binds the proposal, primary plan and source snapshot. The future execution
directory was not created. Policy/layout/seed enumeration is sorted before
the original deterministic shuffle seed `202609081821`; ordering is never
changed from outcomes.

The wire schema remains `motion2scene_reserved_evaluation_protocol_v3`, which
identifies the preserved V3 world geometry and sensor/command interface.
`execution_scope_revision=4` identifies this checkpoint amendment. The frozen
validator already supports this combination because its 36-episode denominator
is per policy. The new runner enforces the complete 24+3 policy inventory and
972 assignments. No validator or collector dependency was edited.

The [runner](../../scripts/research/motion2scene_run_reserved_evaluation.py) refuses
proposed protocols and incomplete model inventories. Every model must occupy
its primary plan's exact M2/M4 slot and bind its training result, common
penalty, source closure, registry and actual qualified model schema. M4 must
have a complete final acquisition receipt. M2 must have the exact historical
round-2 prefix and agree with the M2 model committed before round 3. Its fit
must use precisely bootstrap plus rounds 1–2; M4 uses bootstrap plus rounds
1–4. Identical model bytes across genuinely distinct corpus executions remain
valid; duplicate original capture slots do not.

Baseline inputs are generic pinned policy definitions and actual development
validation artifacts. The runner hardcodes neither a five-context script
selection nor a particular constant option. The selected script and constant
must be frozen from development and pass the protocol's actual validation
gate before adoption. Pending models and baseline validation are blockers,
not substituted with development models.

An adopted `execution_runner` block must bind the following fields:

| Field | Required content |
| --- | --- |
| `schema` | `motion2scene_reserved_execution_v4` |
| `inventory` | Hash-bound `FROZEN_COMPLETE` checkpoint inventory |
| `output_directory` | Exact future evaluation directory |
| `shared_lock_path` | Primary execution root's `.primary-acquisition.lock` |
| `template` | Hash-bound qualified neutral collector template |
| `runtime_assets_declaration`, `environment` | Explicit native assets and environment artifacts |
| `implementation` | Complete current closure of the new runner |
| `batch_implementation` | Complete current closure of the existing batch collector |

The protocol must also bind the collector's full actual runtime/scoring
artifacts, exact 114 feature names, schedules, sources and controller, and
hash-bound evidence for every adoption gate. The common penalty must be filled
from the adopted common learner; this proposal deliberately leaves it pending.
The runner does not create adoption or infer gate completion from existing
filenames.

`prepare` validates the complete inventory and then uses the existing batch
collector to publish every actual command before physics. `run` wraps each
registered single-episode child with the tested serial acquisition controller's
runtime preflight, exclusive intent, measured-attempt audit and conservative
budget reservation. The shared nonblocking lock prevents concurrent primary
and evaluation dispatch. A resource pause remains unlaunched. An unresolved
intent or unknown outcome pauses with its assigned denominator retained; it is
never automatically repeated. A requested pause after a number of episodes
does not narrow the registered panel.

Reports preserve successes, `verified_task_failed`, technical missing and
not-run assignments separately. For each policy, the completion bounds are
`S/N` and `(S+M)/N`; the measured-only fraction has its own denominator. Known
failure events retain their recorded names and classifications. Category
counts can overlap when multiple criteria fail. Failure success-times remain
null; all actual steps and unresolved reservations are separately counted.

The declared initial-upstream requirement exposed a real single-beam scorer
gap. A separate staged repair adds the same strict initial body-origin test
used by courses. Its agreed classification is `invalid_initial_approach`, a
verified **task-criterion** failure when complete source/geometry/measurement
admission is available, not an invented collision or fall. It remains in the
assigned and measured denominators, with no retry. Missing or invalid evidence
retains the existing unknown classification. The production scorer/classifier
remain frozen during the current development batch; adoption requires the
versioned repair, tests, recorded-data re-audit and refreshed source/runtime
freeze. None of the 45 independently re-audited development recordings violates
the upstream condition, so their passage/time outcomes do not change.

```bash
# Read-only; requires actual completed checkpoints and a separately adopted protocol.
.venv_isaaclab/bin/python scripts/research/motion2scene_run_reserved_evaluation.py check \
  --protocol /absolute/path/to/ADOPTED_V4.json --out /absolute/path/to/evaluation

# Writes all972 registered commands; still performs no physics.
.venv_isaaclab/bin/python scripts/research/motion2scene_run_reserved_evaluation.py prepare \
  --protocol /absolute/path/to/ADOPTED_V4.json --out /absolute/path/to/evaluation

# Executes or resumes the fixed assignments only after adoption.
.venv_isaaclab/bin/python scripts/research/motion2scene_run_reserved_evaluation.py run \
  --protocol /absolute/path/to/ADOPTED_V4.json --out /absolute/path/to/evaluation
```

Twenty new CPU tests pass, including actual file-backed M2/M4 causal provenance,
complete pairing, partial-inventory rejection, late-child tamper detection
before the first dispatch, pause/resume without duplicate attempts, and failure
denominators/categories. Black and Ruff pass. The tests use synthetic capture
files and never launch a simulator or inspect a reserved outcome.
