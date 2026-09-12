# Proposed amendment for the qualified traversal runtime

This is a **proposal**, paired with
[its exact candidate settings](TRAVERSAL_IMPLEMENTATION_AMENDMENT_V2_PROPOSED.json).
It is not an adopted implementation lock or an evaluation result. The original
[geometry/comparison recipe](TRAVERSAL_EVALUATION_LOCK_V2.json) remains unchanged:
`7898862a0a02ab7d65a5e21f245c15a77b82a74c86245e88a611ad48ffd58dfc`.
Development qualification, sensor, course-smoke and policy results informed this
proposal. This drafting task performed no reserved-layout rollout or teacher query.

The current runtime can support a narrower first panel: the **eight originally
assigned source-41002 single passages**, under the two original physics seeds,
for 16 nominal episodes per policy. Source IDs, geometries, perturbations and
seed assignments stay as originally recorded. The other 56 episodes of the
original 72-episode recipe remain explicitly `not_run`. Report this as a partial
source-specific panel, never as the full original benchmark or source transfer.
The corresponding eight-offset stress panel has 128 episodes and a separate
denominator. No scene is removed because its clearance or passage appears poor.

| Item | Original recipe | Proposed supported interpretation |
| --- | --- | --- |
| Source qualification | Three development carriers | Source 41002 only; retain other assignments as not run until their own options qualify |
| Sensor and features | Initial 35 rays, 100D binary interface | 65 ideal PhysX rays, 0.5 s/26-frame history, exact named 110D features for five options |
| Options | Pending expanded interface | neutral, d040/d085 at phase 0.20/0.30/0.40; d055/d070 at phase 0.30; one entry and mandatory return |
| Episode horizon | 8 s single / 12 s course | 199 loaded reference samples; driver limit 199; 198 aligned recorded rows; reference ends at 3.96 s, physical recording at 3.94 s |
| Course gate | Supported repeated transitions and horizon | Still unmet; one authored adaptation across two constraints is a separately scoped exploratory alternative |
| Learner | Initial binary imitation | Final common fitter remains pending; current ridge value imitation is a development candidate |

The source restriction and shorter horizon are **substantive protocol changes**.
They need an explicit, separately frozen amendment before evaluation. The 65-ray
calibration and five-option schema complete previously pending runtime details;
they must be identical for every constructor and policy comparison. The current
references are authored `local_crouch` edits of a Kimodo neutral motion, executed
by frozen SONIC, not separately sampled low-height prior skills.

The exact first-panel layout IDs are `locked_v2_single_01`, `_04`, `_07`, `_10`,
`_13`, `_16`, `_19`, and `_22`. They follow the original source assignments alone.
The originally assigned source-41002 courses are `_01`, `_04`, `_07`, `_10`.
A separate decision could retain these as an eight-episode exploratory panel
(two seeds) for one adaptation through both constraints. That would amend the
original repeated-transition task. The single fresh development smoke does not
qualify those reserved compositions or justify pretending this gate was met.

The final runtime registration must bind the controller, references, registry,
robot, scene authoring, sensor mount/rays, history, feature names, legality guards,
scoring and complete code dependencies to hashes. The current registry permits
joint-reference jumps up to 0.05 rad and root-reference jumps up to 0.01 m, with
return requested at reference phase 3.30 s inside the 3.30–3.50 s guard. It supports
one neutral-to-adaptation-to-neutral sequence. Repeated entry, adaptation-to-
adaptation switches, holding, padding, retiming, steering and stopping remain
unqualified. All raw sensor rows must pass the recorded state/clock alignment
audit; sensor phase is one 0.02 s tick after the same-index physical clock.

Scoring must add complete-episode no-fall/no-reset and actual legal-return checks
to the existing body-origin crossing and force rules. `score_passage()['pass']`
alone does not implement that full contract. Retain the 0.1 m downstream margin,
0.3 s upright hold and 1 N beam-contact threshold. A physically attempted episode
that cannot complete within the finite reference ends in failure. An unlaunched
or invalid infrastructure attempt contributes its recorded cost and an invalid
status; it is not a physical failure or success. A process abort is not a verified
protective stop.

Retain the original acquisition budgets (50,000/150,000/450,000 physics steps),
three acquisition seeds, uniform/target-only/analytic/proposed arms, and
0.2 uniform + 0.2 coverage + 0.6 verified-gap mixture. Count every teacher/student
branch and failed acquisition. Pin the final learner and its normalization,
regularization and replay rules after development work finishes, then use the
same settings across constructors. Keep always-walk, scripted, and constant
adaptation; if both d070/d085 constants are used, report them separately. Their
preferences and scripted rules must not be selected from reserved outcomes.

Observation-aware curriculum attribution still needs an empirically specified
transition-delay/margin admission rule. An obstacle observed before the last
*tested* entry is not proof of a measured physical deadline. That calibration,
the common learned policy, the evaluation runner and the amended denominator
must be frozen separately before reserved execution. This proposal supplies the
concrete settings and remaining implementation gates; it supplies no held-out
performance evidence.
