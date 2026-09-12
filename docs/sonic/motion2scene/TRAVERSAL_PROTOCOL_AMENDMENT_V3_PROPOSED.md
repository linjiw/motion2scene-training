# Proposed V3 execution and scoring amendment

**Review only; not adopted and no evaluation authorized.** The [exact proposal](TRAVERSAL_PROTOCOL_AMENDMENT_V3_PROPOSED.json) has SHA-256 `e2055154144cba54d9751205dff0b9cbf427175ebc61e39a4762995511954239`, recorded in its [companion](TRAVERSAL_PROTOCOL_AMENDMENT_V3_PROPOSED.sha256). It preserves the [V3 fixed-world geometry lock](TRAVERSAL_EVALUATION_LOCK_V3.json), SHA `509817600075888ef5cf681c1033c9dcd6796d7a08561e30ee4157049aff5181`, and the original V2 lock. No clearance, physical outcome or feasibility query was made on the reserved geometry to write this proposal.

The amendment resolves the proposed runtime, finite horizon and primary comparison scope. It does **not** complete the original V3 full experiment matrix or the original V2 72-episode recipe. Adoption requires a new artifact with all implementation gates satisfied and an adoption receipt bound to its exact specification. This proposal remains immutable.

## Primary comparison and retained follow-ups

Every primary checkpoint evaluates all **18 nominal layouts × two original physics seeds, 94301/94302 = 36 assigned episodes**: 24 single-beam episodes and 12 two-beam episodes. No early, difficult, unobservable, physically impossible or horizon-limited layout is removed. All 162 nominal/stress USDs remain bound to their original world poses and hashes.

| Primary component | Fixed scope |
| --- | --- |
| Constructors | Uniform, target-only, analytic contrast, analytic observation curriculum |
| Independent acquisition corpora | Seeds 93201/93202/93203 for each constructor |
| Initial budget | 50,000 actual acquired physics steps per corpus; 12 learned checkpoints |
| Fixed baselines | Always-walk; one constant schedule selected on development only; a strong multi-option sensor script validated and frozen on development |
| Nominal evaluation | 15 policies × 36 episodes = **540 episodes**, at most **643,680 recorded physics steps** |
| Acquisition cap | 4 constructors × 3 corpora × 50,000 = **600,000 steps**, before separately disclosed shared qualification/calibration costs |

The 150,000/450,000-step checkpoints, learned proposal and analytic-resampling arms, all eight stress variants, isolated sensing/replay/aggregation ablations and a complete seven-schedule capability oracle are **planned unexecuted follow-ups**, not part of primary adoption. The JSON retains these axes explicitly. It enables only the nominal variant; the validator rejects stress execution under a nominal-only adopted protocol. Stressing all 15 primary policies would add 4,320 episodes and up to 5,149,440 recorded physics steps. A follow-up needs its own frozen protocol and disclosure of any earlier nominal inspection.

All six constants and the existing three one-reference scripts remain useful development baselines. Only a single development-selected constant enters the primary panel. Selecting the best constant separately for each reserved scene would use privileged test information and is forbidden. Learned gains over one-reference scripts alone do not establish curriculum superiority. The primary panel also does not contain a complete finite capability oracle; such a claim needs all required actual schedules, separately counted.

## Common executable interface

The proposal pins the seven-schedule registry at `m2s-prior-splice-seven-schedule-analysis-v2/registry.json`, SHA `a33af401fe9e6949baaddf63a67a5e37e8f62f87758e1ca373f926f9a3214c3c`, and its request, controller and four reference identities. All seven qualification captures now pass the separately declared environment-contact criterion; their original strict self-contact failures remain preserved. The registry contains six adapting schedules plus neutral:

| Index | Complete schedule | Entry tick / reference phase | Required return |
| ---: | --- | --- | --- |
| 0 | Neutral | No entry | Stay neutral |
| 1 | Short, early | 15 / 0.30 s | 265 / 5.30 s |
| 2 | Sustained, early | 15 / 0.30 s | 255 / 5.10 s |
| 3 | Projected/spliced prior, early | 15 / 0.30 s | 265 / 5.30 s |
| 4 | Projected/spliced prior, later | 50 / 1.00 s | 265 / 5.30 s |
| 5 | Short, later | 70 / 1.40 s | 265 / 5.30 s |
| 6 | Sustained, later | 70 / 1.40 s | 255 / 5.10 s |

At most one adaptation is allowed. Entry may be chosen only at its exact qualified phase, subject to the same joint/root jump guards (0.05 rad / 0.01 m). Return is mandatory and automatic. No reentry, adaptation-to-adaptation switch, arbitrary holding, stopping, route choice or learned return is introduced.

Neutral at tick 15 means **wait**, retaining neutral and schedules 4/5/6 as possible complete continuations. At tick 50 it retains neutral and schedules 5/6. At tick 70 it is terminal commitment to walking. Teacher targets for waiting require every expected continuation, including observed failures, with the same seed and exact preaction physical/controller/sensor history. An absent or unadmitted later branch cannot become an optimistic complete waiting target. Constant policies name a complete schedule and wait until its entry; actual actions always come from switch logs.

Every arm uses the same **114 named features**: 100 sensor/state/phase values, seven preaction active indicators and seven legality indicators. The schema pins 65 ideal PhysX rays, a single root-body offset `[0.2, 0, 0.4]` m, 4 m range, 50 Hz, zero configured delay/noise, and a 2.0 s/101-frame history. Floor and ceiling have separate observed/unknown masks; the grid is `[-1,-1.5,-1.5]` to `[4,1.5,1.5]` m at 0.2 m resolution. This is distinct from the completed 106D, half-second-history experiment. No scene ID, beam geometry, outcome, teacher state or source ID enters the student.

The phase-specific ridge fitter uses the same regularization, `1e-6`, in every constructor arm. Normalization, model weights, phase masks and all heads must be fitted from development/acquisition data and frozen before any reserved outcome. A missing measured phase head blocks registration; it does not trigger a silent test-time fallback. The value objective does not imply an optimal or lexicographically reliable policy under partial observability.

## Baseline development gate

The new [multi-option script](../../gear_sonic/dataset_generation/hallucination/motion2scene_schedule_script.py) is a CPU implementation candidate. It uses near/far upper-occupancy bands, available floor–ceiling gaps and current legality. An observed sufficiently high gap can dismiss an upper hit; unknown surfaces retain the hazard cue. Early/middle decisions can choose the prior-derived reference; the final decision chooses short or sustained according to the extent of observed hazard.

The proposed finite tuning grid contains 90 configurations: free-height threshold `{1.20,1.25,1.30,1.36,1.40}` m, immediate near-band count `{1,2}`, later near-band count `{1,2,3}`, and sustained hazard-band count `{1,2,3}`. Select using a named complete development corpus, passage first and measured time second, with canonical parameter ties. Finite offline replay requires exact matched neutral histories and complete physical continuations; the chosen script then requires actual single/two-beam development execution. Freeze the parameter artifact and source before adoption.

The current script never chooses the early authored short/sustained schedules. Its adequacy as the strong comparator is therefore still a development gate, not an established fact. If that restriction matters in development, revise the script and tuning grid before adoption. No threshold, rule or constant may change from reserved outcomes. The three existing one-reference scripts remain explicitly simpler comparators.

## Course scoring and terminal outcomes

All policies run one continuous first episode with **299 loaded reference frames and 298 recorded control rows / 1,192 physics steps**. Physical samples span **0–5.94 s**. Command/reference phase is one 0.02 s tick ahead; entry packets at ticks 15/50/70 describe physical elapsed times 0.28/0.98/1.38 s. Report the 5.94 s recorded span separately from 5.96 s of integrated control duration. No scene-specific extension, padding, reference wrap or reset continuation is allowed.

For each enabled beam, every recorded body origin must be at least 0.10 m beyond its downstream edge for 16 consecutive 50 Hz samples, a 0.30 s hold. Root height must be at least 0.50 m and minus projected-gravity z at least 0.50. The robot must start upstream of every beam; two-beam holds must complete in beam order. This is a body-origin criterion, not a claim about all collider extents. Both beams are scored from the same continuous episode, including contact with an earlier beam after its first crossing.

Success additionally requires stability, no reset, no undesired environment contact, legal commands and actual neutral return throughout the full recorded horizon. Adaptations must include at least 15 recorded ticks after their qualified return. The environment criterion records all 30 bodies against exact robot/floor/wall/beam counterparts at 200 Hz. Foot–floor support is allowed; other environment normal-force norms above 1 N fail. Self-contact and the historical strict net-force diagnostic remain separate. Mechanical work is unavailable with current actuator instrumentation and remains null.

**Incomplete crossing, hold or recovery at 5.94 s is a task failure**, with separate `finite_horizon_*` flags. Its time to successful completion is right-censored, but the assigned episode stays in the success denominator. A robot that crosses without enough remaining hold time is not relabeled unstable. The interim development constant-sustained result at seed 8732 demonstrates why these categories matter; it is not a reserved outcome. Successful passage cost is the physical time at completion of the final beam's hold, conditioned on eventual full-horizon validity; failure success-times are null, with crossing and last-observed times retained.

## Missing evidence, pairing and adoption blockers

Task status, measurement admission, infrastructure status and supervision eligibility are separate. A source-bound observed reset, fall, contact, illegal transition, missed return or finite-horizon failure remains a **physical task failure** even if later sensor history becomes ineligible or the process then aborts. A startup/recording error without such a verified outcome is **technical missing**, not a made-up physical failure or success. Every assigned slot and attempt remains in the report.

For `N` assigned episodes, `S` successes and `M` technical missing/not-run, report completion bounds `[S/N,(S+M)/N]`, physical-failure counts, and the separately labeled measured-only fraction `S/(N-M)`. An incomplete panel is not a completed benchmark. At most one separately recorded retry is proposed for a preclassified transient infrastructure error, with identical seed/model/scene/settings. A verified physical failure fixes the outcome and is never replaced by a favorable retry. Systematic code faults suspend the panel and require a disclosed amendment.

Compare policies on the same layout and seed. Cluster uncertainty at the 18-layout level, report the six-course stratum separately, and retain variability across the three acquired corpora. Paired success-time differences use only jointly successful assignments with their denominator. Late-visible/unobservable cases remain in evaluation, not removed by the training timing gate. Source 41002 ancestry is reused throughout; neither the six-second carrier nor the derived prior establishes source holdout, raw-prior feasibility, navigation or hardware transfer.

The following remain necessary before adoption:

1. A real 114D seven-bank runtime smoke and a continuous two-beam/full-horizon development qualification. Option qualification alone does not verify the new sensor/runtime path or a complete two-constraint execution.
2. The same explicit initial-approach and finite-horizon categories for single and course scoring. The current course scorer checks upstream approach; the single scorer does not expose that check. Preserve verified physical failures when supervision is unavailable.
3. Integration of the standalone protocol validator, a complete runtime/native-asset/environment closure, a frozen baseline selection and all 12 primary model artifacts.
4. Actual development validation of the strong multi-option script and an exact adopted specification receipt. Neither the geometry lock nor this proposal provides that receipt.

## Validator and integration hook

[motion2scene_evaluation_protocol.py](../../gear_sonic/dataset_generation/hallucination/motion2scene_evaluation_protocol.py) provides `validate_reserved_execution(scene_definition, context)`. It rejects proposals, pending gates, changed adoption specifications, missing models, omitted layouts, substituted scene variants/geometry, unenabled stress variants, altered seeds, registry/request/controller/model identities, runtime/scorer substitutions, changed feature order/horizon and altered script parameters. Its adoption receipt hashes the canonical entire protocol except the receipt reference, avoiding a circular hash.

The collector is intentionally unchanged while its development smoke uses frozen sources. After that smoke, call the validator before reserved preparation and execution, and independently before analysis. Build `context` from the **actual command/manifest** with:

```python
context = {
    "physics_seed": actual_seed,
    "policy_id": registered_policy_id,
    "mode": actual_mode,
    "preferred_option_id": actual_preferred_option,
    "preferred_reference_id": actual_preferred_reference,
    "model": actual_model_artifact_or_none,
    "script_parameters": actual_script_parameter_artifact_or_none,
    "registry": actual_registry_artifact,
    "request": actual_request_artifact,
    "controller": actual_controller_artifact,
    "runtime_artifacts": complete_actual_runtime_artifacts,
    "scoring_artifacts": complete_actual_scoring_artifacts,
    "feature_names": actual_feature_names,
    "reference_frames": actual_loaded_frames,
    "recorded_control_steps": declared_or_actual_recorded_steps,
}
validate_reserved_execution(scene_definition, context)
```

Do not populate this context by simply copying the expected protocol fields. Actual command/seed/scene success-manifest checks, native imported geometry, physical source binding and contact/sensor chronology remain independently required. The validator does not replace those audits or classify outcomes from missing recordings.

Validation: **23 focused adversarial tests pass**; Black and Ruff pass. The actual proposed JSON is intentionally rejected. The [builder](../../scripts/research/motion2scene_propose_v3_protocol.py) is CPU/read-only with respect to all source records and refuses to overwrite a proposal. No collector dependencies were changed and no simulator was launched.
