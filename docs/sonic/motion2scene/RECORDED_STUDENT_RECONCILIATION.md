# Reference M5 student reconciliation

The v7 acquisition coordinator ended with tool exit code 143; its child's OS
exit and the termination cause are unavailable. The assigned seed-93201 M5
reference student nevertheless saved a native completion manifest, 298 recorded
control frames, 298 sensor packets and all 1,192 physics steps. No simulator
remained at recovery. The original invocation, scene and M4 generating policy
are hash bound in the reconciliation.

The unchanged incomplete-attempt scorer establishes no physical failure event
and cannot admit passage without its required process-exit receipt. The task
outcome therefore remains **unknown**, and all 1,192 measured steps stay charged.
The legacy row status `failed_attempt` refers to infrastructure admission; it
must not be counted as a measured task failure. No OS success code, wall time,
passage time or repeated execution is invented.

The continuation adapter applies only to this declared reference-arm student.
It returns the re-audited saved collection instead of dispatching it again and
keeps accounting's process-exit status explicitly unknown. Other assignments
use the original controller. All seven current teacher schedules still execute
before fitting. Reference-arm learning uses uniform complete-teacher targets;
this student's missing outcome does not alter targets or fitting weights.
The original replay auditor accepts the measured capture with
`task_outcome_admitted: false` and no missing-cost entries.

The native runtime, collector, scoring rules, scene-wise teacher, ridge learner,
candidate identities/order, earlier files and frozen plans remain unchanged.
This is a declared execution-recovery exception, not a method improvement.

The [declaration](evidence/research-progress-20260909/recorded_student_recovery/reconciliation.json),
[independent student audit](evidence/research-progress-20260909/recorded_student_recovery/student_reaudit.json),
and [progress receipt](evidence/research-progress-20260909/recorded_student_recovery/M4_audit_and_M5_recovery_progress.json)
are archived beside this report. Operational evidence is under
`/home/linjiw/research-data/groot-wbc/m2s-expanded-and-capability-stages-20260910-v8/`:
`reconciliation.json`, `student_reaudit.json`, `validation.json`, `launch.json`
and the original-sequence stage receipts. The reconciliation adds zero physics.
Validation preserves all 120 previously reported assessments, audits 150 current
assessments, and passes the original 222-artifact native preflight.

Fifteen tests passed:

```bash
.venv_isaaclab/bin/python -m pytest -q \
  decoupled_wbc/tests/test_motion2scene_resume_recorded_student.py \
  decoupled_wbc/tests/test_motion2scene_resume_expanded_acquisition.py \
  decoupled_wbc/tests/test_motion2scene_expanded_acquisition.py
```

The existing v8 coordinator already runs the command below. Use it only after
checking its PID/start ticks and all descendants, never as a duplicate:

```bash
.venv_isaaclab/bin/python scripts/research/motion2scene_resume_recorded_student.py \
  --reconciliation /home/linjiw/research-data/groot-wbc/m2s-expanded-and-capability-stages-20260910-v8/reconciliation.json \
  --repair /home/linjiw/research-data/groot-wbc/m2s-expanded-and-capability-stages-20260910-v7/repair.json \
  --plan /home/linjiw/research-data/groot-wbc/m2s-expanded-acquisition-plan-20260909-v1/plan.json \
  --adoption /home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-adoption-v3/adoption.json \
  --until-budget 8
```

No learned M8 or reserved result is implied by this recovery. The single next
discriminating experiment remains the original M8 common-set development panel:
90 learned assignments plus 48 already completed comparator assignments.

The reference M5 continuation has since completed. Its seven teacher outcomes
and saved targets were independently re-audited; the ridge fit retains the
unchanged M4 prefix and uniform per-phase weighting. Both prior schedules pass;
the other five schedules fail. The encounter costs 9,144 measured steps including
the unknown student, with no retry.
[Completion evidence](evidence/research-progress-20260909/recorded_student_recovery/reference_M5_complete.json)
binds these measurements to the generating and newly fitted models.

The original response-diversity reader rejects unknown student outcomes even
when their costs are measured. A separate reader now admits only this declared
recording, keeps its outcome as unknown, and enumerates it explicitly in the
report. Other collection checks and teacher response statistics remain under
the original implementation. Fourteen relevant tests pass, and an
[actual-record check](evidence/research-progress-20260909/recorded_student_recovery/unknown_reader_validation.json)
reproduces the original rejection and verifies unchanged rows and cost.
The command below is **unexecuted as a full M8 audit**; it requires all M8
checkpoints and a fresh output directory:

```bash
.venv_isaaclab/bin/python scripts/research/motion2scene_response_diversity_recorded_student.py \
  --plan /home/linjiw/research-data/groot-wbc/m2s-expanded-acquisition-plan-20260909-v1/plan.json \
  --reconciliation /home/linjiw/research-data/groot-wbc/m2s-expanded-and-capability-stages-20260910-v8/reconciliation.json \
  --budget 8 --out /tmp/m2s-five-arm-M8-audit
```
