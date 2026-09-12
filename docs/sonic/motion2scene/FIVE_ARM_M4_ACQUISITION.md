# Five-arm M4 acquisition evidence

The completed raw-record audit covers 585 acquisition captures and 696,536 measured physics steps. All twelve original corpus reports and four original arm summaries equal the earlier M4 audit exactly.

There are three corpora per arm and four post-bootstrap encounter assignments per corpus. Costs include each corpus's bootstrap, pre-update students and all teacher branches. Adaptation required means neutral fails and at least one complete schedule passes.

| Arm | Physics steps | Bank-solvable | Adaptation required | Best fixed |
| --- | ---: | ---: | ---: | ---: |
| Screened uniform | 139,464 | 12/12 | 2/12 | 12/12 |
| Target-only | 139,464 | 12/12 | 4/12 | 12/12 |
| Reference contrast | 138,680 | 6/12 | 6/12 | 6/12 |
| Executed contrast | 139,464 | 12/12 | 12/12 | 12/12 |
| Executed + replay | 139,464 | 12/12 | 12/12 | 12/12 |

Reference construction retains six bank-unsolvable tasks in its twelve-task denominator. No task outcome is unknown; four partial captures establish failure through observed contact, while their unobserved later fall/recovery status remains unknown.

Execution-conditioned construction yields more physically solvable adaptation tasks here. The adaptation counts alone do not establish better teaching than target-only. Every corpus still admits one fixed schedule covering all its solvable tasks. These are construction-specific training scenes, not common-set or held-out policy results. Contrast/replay preserve candidate order and have identical teacher outcomes across all three corpora.

[Measured M1/M4 response figure](evidence/research-progress-20260909/five_arm_M4/acquisition_responses.pdf) · [Per-corpus CSV](evidence/research-progress-20260909/five_arm_M4/M4_by_corpus.csv) · [Audit](evidence/research-progress-20260909/five_arm_M4/result.json)

The figure shows individual corpora and corpus means against actual acquisition steps. Its endpoints contain different acquired tasks; it is not a policy learning curve. No confidence interval or significance claim is inferred from repeated episodes.

| Arm | Corpus seed | Bank-solvable | Adaptation required | Best fixed | Physics steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Screened uniform | 93201 | 4/4 | 1/4 | 4/4 | 46,488 |
| Screened uniform | 93202 | 4/4 | 0/4 | 4/4 | 46,488 |
| Screened uniform | 93203 | 4/4 | 1/4 | 4/4 | 46,488 |
| Target-only | 93201 | 4/4 | 1/4 | 4/4 | 46,488 |
| Target-only | 93202 | 4/4 | 1/4 | 4/4 | 46,488 |
| Target-only | 93203 | 4/4 | 2/4 | 4/4 | 46,488 |
| Reference contrast | 93201 | 3/4 | 3/4 | 3/4 | 46,488 |
| Reference contrast | 93202 | 2/4 | 2/4 | 2/4 | 46,096 |
| Reference contrast | 93203 | 1/4 | 1/4 | 1/4 | 46,096 |
| Executed contrast | 93201 | 4/4 | 4/4 | 4/4 | 46,488 |
| Executed contrast | 93202 | 4/4 | 4/4 | 4/4 | 46,488 |
| Executed contrast | 93203 | 4/4 | 4/4 | 4/4 | 46,488 |
| Executed + replay | 93201 | 4/4 | 4/4 | 4/4 | 46,488 |
| Executed + replay | 93202 | 4/4 | 4/4 | 4/4 | 46,488 |
| Executed + replay | 93203 | 4/4 | 4/4 | 4/4 | 46,488 |

Geometric computation is reported separately below. Full pools include rejected and unselected candidates. Both reference screens reuse the expanded candidates; they are not new proposal draws. The retained v1 screen produced no acquisition queue and remains charged as actual search work.

| Recorded search component | Candidates evaluated | Clearance queries | Component seconds |
| --- | ---: | ---: | ---: |
| Original shared pool | 480 | 114,848 | 55.045 |
| Expanded shared pool | 3,840 | 967,424 | 457.371 |
| Reference v1, no acquisition queue | 3,840 | 1,157,424 | 544.260 |
| Reference v2, active queue | 3,840 | 1,157,424 | 542.700 |

These are the recorded search-component timings, not end-to-end proposal wall time. Shared materialization, kinematics setup and other overhead are not inferred from these columns. The full pools support later checkpoints too; no unsupported per-arm or M4-only amortization is applied.

Reproduce the independent audit with the original expanded plan and a fresh output directory:

```bash
.venv_isaaclab/bin/python scripts/research/motion2scene_response_diversity.py \
  --plan /home/linjiw/research-data/groot-wbc/m2s-expanded-acquisition-plan-20260909-v1/plan.json --budget 4 --out /tmp/m2s-five-arm-M4-audit
```

Regenerate this report and its evidence from the recorded source paths with ` .venv_isaaclab/bin/python scripts/research/motion2scene_render_five_arm_m4.py`.
The acquisition figure uses the unchanged `motion2scene_render_acquisition_evidence.py`; its source-bound output manifest also retains the previously measured WAIT control, with no new WAIT experiment.
