# Seven-option baseline development selection

The registered selection compares 72 sensor-script settings and all seven constant
complete schedules on the same five development contexts. It uses the 35 forced
physical branches already acquired for teacher construction. It does not execute
a new controller, consume new physics steps, or establish held-out performance.

The immutable rules are in
`/home/linjiw/research-data/groot-wbc/m2s-schedule-baseline-development-selection-v1/registration.json`
(SHA-256 `452adb1441fefaa47b08f7686a642310bab8ed555205fd5b5202bf51f3dd7b84`).
They were registered before inspecting the remaining early-obstacle and course
results. Earlier 106D development results and the two 114D integration smokes were
already known. The settings and their order come from the earlier frozen
`m2s-schedule-script-offline-review-v1` registration.

`implementation_registration.json` binds the 67-file Python import closure and
its snapshots (SHA-256
`c9ef93c54ec3288f6ee0f264432c303fc18292888c74c7a699909560acd0b908`).
No collector, runtime, or existing policy was modified for this analysis.

## Causal script evaluation

The independent teacher audit reconstructs the actual eligible neutral sensor
and robot-state history at ticks 15, 50, and 70. The script receives only the
common named 114D features and legal-action mask. Each setting consumes these
packets until its first nonneutral choice; later forced-neutral observations are
then ignored. The selected recorded branch must match the neutral branch's
physical state, actions, motion tokens, and causal sensor history exactly through
that entry. Its actual complete-branch outcome and passage time supply the score.

For an all-WAIT setting, the recorded neutral branch supplies the score. A missing
required phase remains unknown unless an independently measured fall, reset,
refused transition, or state mutation is indexed strictly before that phase
while all earlier script choices waited. An untimed global force maximum cannot
establish that exception. These rules retain measured physical failures while
preventing missing observations from becoming synthetic labels.

Every constant uses its own actual complete forced schedule. This includes
neutral, the short and sustained schedules at their qualified entry times, and
the two qualified entry schedules of the authored prior derivative. Constant
selection does not compare a neutral action with a privileged future teacher
continuation.

## Registered ranking and missing evidence

All seven outcomes must be independently known for a context to supply the
relative-time oracle. For successful branch time `t`, the regret is
`(t - minimum successful time) / max(maximum successful time, 1e-12)`.
A known failure has regret 1; an all-fail context assigns regret 1 to every branch.
Absolute passage times from different scene-specific finish planes are not pooled
as if they described one identical traversal task.

Candidates rank first by the number of failed selected branches across all five
contexts, then by mean relative-time regret. Exact script ties retain
`DEFAULT_CONFIG` when it is among the tied settings, otherwise the original
setting order. Constant ties retain the verified option order. A candidate with
any unknown context cannot win. All 35 assigned branches remain in accounting,
including audit failures and unknown measurements; unavailable physics cost is
reported explicitly.

The CLI checks that all five result files exist before opening any result. It
independently re-audits every branch, binds each result to its child registration,
and rejects duplicated physical episode identities. A successful selection emits
`script_parameters.json` in the exact SHA-checked runtime schema, the preferred
constant option ID, all candidate/context scores, and the source evidence. An
incomplete or invalid panel emits no selected policy.

```bash
.venv_research/bin/python scripts/research/motion2scene_select_schedule_baselines.py run \
  --out /home/linjiw/research-data/groot-wbc/m2s-schedule-baseline-development-selection-v1
```

The selected scripted controller still requires actual execution at the planned
development validation seed. This finite-branch calculation does not establish
closed-loop generalization, stopping capability, repeated adaptation, or primary
held-out performance.

Validation: 16 focused baseline/script tests passed; the new selector, pure
ranking helper, and tests pass Black and Ruff.

## Completed offline selection

The once-only re-audit admitted all 35 branch outcomes, with no unknown branch or
unaccounted physics stream: 41,720 existing physics steps and zero new physics
steps. `result.json` has SHA-256
`53a14be93b3dce6eb0f220f4139dce3c394862c37cc59f5d2670bfbeced5d832`.

Setting index 6 is selected from 15 exactly tied best settings; the provisional
default is not among these ties. The settings are observed free height 1.28 m,
one immediate prior band, three later prior bands, and one minimum sustained
hazard band. The exact runtime artifact is `script_parameters.json`, SHA-256
`96535f57293aad7adaeef7cbb323249b9131fa72fd80191ca045e8a846e42b83`.
The preferred constant schedule is `prior_splice_e015_r265`.

All five selected-script branches and all five selected-constant branches pass.
The script waits at tick 15 in every context. At tick 50 it selects
`prior_splice_e050_r265` in the four obstacle contexts; it remains neutral at all
three decision phases in the empty context. Every used prefix was independently
matched to the corresponding actual forced branch.

| Development context | Script branch time (s) | Constant branch time (s) |
| --- | ---: | ---: |
| Empty | 5.36 | 5.36 |
| Short beam, `0187` | 3.62 | 3.62 |
| Long beam, `0247` | 4.50 | 4.48 |
| Early beam, `0284` | 2.72 | 2.74 |
| Two-beam course | 4.12 | 4.12 |

Mean relative-time regret is 0.00086207 for the script and 0.00145985 for the
constant. These small differences derive from two 0.02-second changes in the
recorded development branches. A strong constant already passes the entire
panel, and its empty-context passage time equals neutral's. No arbitrary
adaptation penalty was introduced. These results provide selected baselines for
subsequent actual execution; they do not demonstrate a learned policy's passage
advantage or a general efficiency benefit from adaptive selection.
