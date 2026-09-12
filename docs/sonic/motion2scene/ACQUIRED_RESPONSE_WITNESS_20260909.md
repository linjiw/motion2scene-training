# Acquired response witness — 9 September 2026

The measured acquisition pool now contains a pair with disjoint successful
responses. This is **interim acquired-pool capability evidence**, not a balanced
M3 construction comparison, learning curve, or newly executed policy result.

The analysis retains all completed M2 encounters and adds all three assigned
executed-contrast round-three encounters. It independently re-audits the new
seven-branch teachers and actual pre-update students from their archived records:
24 acquisition episodes and 28,608 recorded physics steps. The prior M2 audit
accounts for 328,992 steps including bootstrap. The combined audited prefixes
therefore account for 357,600 steps, with no additional physics run by this
analysis. Repeated geometry/seed pairs are merged only when every recorded bank
outcome agrees. Failed schedules remain in the tables; there are no unknown bank
outcomes in this completed subset.

| Measured quantity | Finding |
|---|---|
| Unique acquired geometry/seed conditions | 20 |
| Bank capability | 20/20 |
| Best fixed schedule coverage | 19/20 |
| Minimum schedules covering the pool | 2 |
| Disjoint passing-set pairs | 1 |
| Individual executed-contrast M3 corpora | A fixed prior schedule covers 3/3 in each |

At execution seed 93201, target-only candidate `primary_candidate_93201_0155`
(round one) admits only the two sustained schedules. Executed-contrast candidate
`primary_candidate_93201_0017` (round three) admits only the two prior-splice
schedules. All other schedules fail the unchanged physical scoring contract.
The three new pre-update students pass their assigned encounters, but those
historical outcomes remain attached to their generating M2 policies. They are
not common-set policy evaluations.

The first recorded decision features differ in occupied distance bands. Scene A
has an upper hit in the 1.5–2.5 m band; scene B has one in the 0.75–1.5 m band.
This difference is present at 0.30 s, before the illustrated commitments at
1.00 s and 1.40 s. Exact perception vectors also differ at the later neutral
phases. No tolerance or new memory rule was introduced. This observation is
not evidence of robust discrimination under perturbations, nor of a trained
policy learning that discrimination.

[Recorded-state figure](evidence/research-progress-20260909/acquired-witness/acquired_response_witness.pdf)
shows the first-phase features, two measured response alternatives, and all seven
bank outcomes. Frames are selected after outcome inspection at the failed
displayed schedule's peak-force control interval; the same index is used for
both schedules within a scene. Rendered poses use the repository's existing G1
visual meshes and MuJoCo forward kinematics only. Contact outcomes come from
IsaacLab recordings, not mesh inspection or a new simulation. The figure exports
all 114 first-phase feature values and the complete schedule table as CSV.
Existing repository and upstream asset licensing remains applicable; this work
does not redistribute model weights or relicense meshes.

The new finding changes the statement that one fixed sustained schedule covers
the entire observed acquisition pool. It does **not** change that statement for
the completed M2 pool, nor establish response diversity within the three
executed-contrast M3 training corpora. A complementarity-aware queue modification
is therefore still deferred until the registered M8 development evidence.

## Evidence and reproduction

The source-bound analysis is stored at
`/home/linjiw/research-data/groot-wbc/m2s-acquired-response-witness-20260909-v1/result.json`
(SHA-256 `bda1faf11357a6622b292754dcc12a9a9e579a03b32e2a84e62e3390360c393b`).
Its registration freezes the complete previous audit and all three new assigned
teacher/student/model references before re-audit. The figure receipt records
trajectory/contact hashes, joint mapping, rendered frame indices and visual XML
hashes. Source snapshots and small result files are mirrored under
`evidence/research-progress-20260909/acquired-witness/`.

```bash
OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python \
  scripts/research/motion2scene_acquired_response_witness.py \
  --plan /home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-plan-tie-proposed-v5/plan.json \
  --previous-audit /home/linjiw/research-data/groot-wbc/m2s-response-diversity-M2-20260909-v1/result.json \
  --out /tmp/m2s-acquired-witness-reproduction

OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv_isaaclab/bin/python \
  scripts/research/render_motion2scene_acquired_witness.py \
  --audit /tmp/m2s-acquired-witness-reproduction/result.json \
  --out /tmp/m2s-acquired-witness-reproduction/figure

OPENBLAS_NUM_THREADS=1 .venv_isaaclab/bin/python -m pytest -q \
  decoupled_wbc/tests/test_motion2scene_acquired_response_witness.py \
  decoupled_wbc/tests/test_motion2scene_response_diversity.py
```

Use fresh output directories. The original audit and render commands above were
executed using the source-bound data directory; the `/tmp` reproduction commands
are equivalents. Eleven focused tests pass. The new analyzer, renderer and test
file pass Black and Ruff. The first-phase comparison explicitly separates tiny
proprioception differences from perception differences and preserves unavailable
phases. Acquisition sources, candidate ordering, learner, teacher, and reserved
evaluation remain unchanged.

The single next decisive experiment remains the **138-episode M8 common-set
closed-loop development panel** after the required original M4 and expanded M8
prefixes complete. The central matched-cost held-out claim is unresolved.
