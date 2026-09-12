# Motion2Scene: manuscript packages

The current constructive working draft is
[traversal_method_v2.pdf](traversal_method_v2.pdf), with its
[main LaTeX source](traversal_method_v2.tex) and
[compact historical/data/protocol supplement](traversal_method_v2_supplement.tex).
It is organized around the implemented seven complete schedules, 114D causal
sensor/state input, matched future-continuation teacher and historical physical-gap
replay. Fixed candidate queues and optional future learned/adversarial geometry
are explicitly distinguished. Mechanical work is unavailable in the current
actuator instrumentation.

The September 9 revision emphasizes execution-conditioned teaching and adds the
[completed decision-learning comparisons](../DECISION_LEARNING_RESULTS_20260909.md):
separate feasibility/time prediction, exact observation-consistent continuation
groups, and four fixed-data replay controls. Both learner families and all replay
controls select passing recorded branches on 4/6 whole-context holdouts. The
main text now defines coverage weighting and distinguishes passage time from
full recovery; initialization chronology is in the supplement. The reserved
handoff is paused while these development choices and broader acquisition are
resolved. The paper remains a working research manuscript, not a completed
held-out performance claim.

The measured capability table contains all 42 forced schedules across six
development contexts: the option union passes 6/6 while the best constant passes
5/6. The completed [36-episode policy comparison](../SIX_CONTEXT_POLICY_COMPARISON_V1.md)
shows the six-context learner and retuned script each passing 6/6, versus 5/6 for
the five-context learner, original script and strong constant. Both updated
policies take 0.50 seconds longer on the short and long passages. All outcomes
are retained, including one 756-step known walking failure. These are development
data-extension results; independent curriculum and reserved-layout improvement
remain unestablished.

The [adopted continuation](../PRIMARY_ACQUISITION_TIE_CONTINUATION_V1.md)
completed its first primary M0 at 04:34:26 UTC on September 9. Its original seven
bootstrap captures are independently re-audited and reused in their original
corpus only. Two phases use consequential labels; an exactly tied all-success
table initializes the 1.00 s phase under the disclosed common amendment. The
failed original fit and startup attempt remain preserved. M2/M4 traversal and
curriculum comparisons remain pending. The current
[V6 reserved proposal](../TRAVERSAL_PROTOCOL_AMENDMENT_V6_PROPOSED.md) and
[reporting V3](../TRAVERSAL_PROTOCOL_REPORTING_CLARIFICATION_V3_PROPOSED.md)
remain separately unadopted and unexecuted.

The integrated [method overview](../evidence/traversal-v2/method_overview.pdf)
shows implemented post-bootstrap rounds, fixed geometric candidates and
historical generating-model gaps. Its caption distinguishes this mechanism
from the measured development capability figures.

The [42-episode portable dataset](../PORTABLE_SIX_CONTEXT_DATASET_V1.md) contains
12,516 aligned 114D packets, all 18 continuation tables, 28 passed and 14 failed
physical episodes. Independent standalone fitting reproduces all 18 teacher-table
decisions and 84 archived neutral actions using the bundle and NumPy. The
[V6 evaluation proposal](../TRAVERSAL_PROTOCOL_AMENDMENT_V6_PROPOSED.md) retains
V4's 972 assignments and separate adoption gates. The current
[reporting clarification](../TRAVERSAL_PROTOCOL_REPORTING_CLARIFICATION_V3_PROPOSED.md)
preserves the previously fixed paired corpus/layout resampling. The separate
[36-episode policy recording release](../PORTABLE_POLICY_PANEL_V1.md) includes
all actual comparison episodes and the shorter known failure, with original
source-version provenance and no fabricated teacher targets.

The previous working draft and PDF are preserved in an
[immutable local archive](/home/linjiw/research-data/groot-wbc/m2s-working-manuscript-before-current-method-restructure-v1/manifest.json).
Earlier binary/106D studies and aligned releases remain in the supplement and
their original evidence documents. The
[recorded duration video](/home/linjiw/research-data/groot-wbc/m2s-duration-visual-demo-v1/duration_execution.mp4)
uses recorded states, with no new dynamics during rendering.

The existing **[paper.tex](paper.tex) and [paper.pdf](paper.pdf) are preserved as
the completed diagnostic snapshot**. The inventory and reproduction notes below
refer to that snapshot.
The original result documents, registrations, selectors, controller, generators,
and evaluation records are preserved unchanged. This is a local evidence package,
not a submission receipt. The provenance files contain local identifying paths;
only the anonymous PDF and video are candidates for review upload.

- [Eight-page paper](paper.pdf) and [LaTeX source](paper.tex).
- [Anonymous video](motion2scene-anonymous.mp4): recorded-state visualization,
  with a useful contrast and a missed adaptation; no new dynamics for rendering.
- [Current evidence audit](CURRENT_AUDIT.md), including the compact protocol table,
  definitions, deviations, comparator results, cost boundary, and claim restrictions.
- [Pinned-record manifest](evidence/manifest.json): configuration hashes, controller
  hashes, result paths, protocol inventory, all 854 seed-level rows, and pin checks.
- [Manuscript claim map](evidence/manuscript-claims.json): each result/figure family
  linked to analysis pointers, scripts, protocols, and underlying row identifiers.
- [Recomputed analysis](evidence/analysis.json), [post-hoc diagnostics](evidence/diagnostics.json),
  and [submission gates](SUBMISSION_GATES.md).

The central interpretation is finite-bank learning utility. Analytic and learned
contrast construction pass 24/36 and 22/36, compared with 14/36 for uniform and
target-only construction. Always-d040 and the scripted rule also pass 24/36.
Analytic reduces unnecessary requests from 14/14 to 8/14 both-pass conditions;
learned construction makes 6/14 but misses two of ten available rescues.
The nominal bank was partly observed before registration. These findings do not
establish learned-generator superiority, equivalence, robust deployment, or
generator-level source-held-out transfer.

## Reproduction

Run at the repository root. Raw records remain in the research-data tree named
by the manifest; the commands below do not launch physics or change a primary fit.
The diagnostic script creates separate sensitivity models under this directory.

```bash
PYTHONPATH=.:scripts/research .venv_research/bin/python scripts/research/audit_motion2scene_submission.py
PYTHONPATH=.:scripts/research .venv_research/bin/python scripts/research/motion2scene_submission_diagnostics.py
PYTHONPATH=.:scripts/research .venv_research/bin/python scripts/research/render_motion2scene_submission.py
MUJOCO_GL=egl PYTHONPATH=.:scripts/research .venv_research/bin/python scripts/research/render_motion2scene_submission_video.py
tectonic docs/motion2scene/submission/paper.tex --keep-logs
PYTHONPATH=.:scripts/research .venv_research/bin/python scripts/research/package_motion2scene_submission.py
```

The separately frozen [command-audit protocol](../SUBMISSION_COMMAND_AUDIT_V1.md)
and its receipts under `m2s-submission-command-audit-v1` document the sixteen new
matched executions. They are **post hoc**, were run solely to complete eight
missing command pairs, and cost 0.137223 rollout-process hours. Do not relaunch
them as a routine reproduction check.

Run the impacted tests:

```bash
PYTHONPATH=.:scripts/research .venv_research/bin/python -m pytest -q \
  decoupled_wbc/tests/test_motion2scene_submission_audit.py \
  tests/research/test_motion2scene_nominal_contract.py \
  tests/dataset_generation/test_motion2scene_passage.py \
  tests/dataset_generation/test_motion2scene_action_contract.py \
  tests/dataset_generation/test_motion2scene_icra_compare.py \
  tests/dataset_generation/test_motion2scene_icra_readout.py \
  tests/dataset_generation/test_motion2scene_empirical_ambiguity.py
```

Original acquisition proposals and labels, plus all primary outcome definitions,
remain authoritative. New explanatory mathematics, visibility/aliasing analysis,
corpus-removal sensitivities, and simple command-lookup policies are post hoc.
