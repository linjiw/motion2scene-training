# Motion2Scene research brief and paper completion plan

Dated September 9, 2026. Operational snapshot: 18:17:41 UTC (14:17:41 Eastern).

**The system works as a bounded simulation research platform. We have a substantial method draft and real closed-loop development evidence. The central ICRA claim—better perceptive traversal from execution-conditioned teaching at matched acquisition cost—is still unresolved.** The most valuable remaining work is comparative evidence, particularly against target-only construction, constant schedules, and the strong sensor script.

Read the [full working paper](submission/traversal_method_v2.pdf) and [editable LaTeX](submission/traversal_method_v2.tex) for the manuscript, including abstract, introduction, related work, equations, measured results, limitations, references, and supplement. The draft is a long-form research document; it is not yet a submission-ready conference PDF. This brief explains the current argument, evidence, and completion decisions. Historical binary-controller results must remain distinct from the current seven-schedule system.

## 1. What is working now?

The frozen SONIC tracker executes a qualified reference schedule in Isaac Lab. The high-level learner observes native simulated rays and robot state, waits or commits at legal phases, and completes one adaptation and mandatory recovery. Geometry construction, complete branch labeling, portable model fitting, native policy deployment, sensor interventions, provenance, and acquisition accounting all have implemented paths and development records.

The newest saved [read-only monitor snapshot](evidence/research-status-20260909/primary-monitor.json) reports:

| Original acquisition experiment | Snapshot |
| --- | ---: |
| Complete assessed episodes | 251 / 468 assigned |
| Known passing / failing episodes | 192 / 59 |
| Recorded physics steps | 299,192 |
| Additional steps beyond the historical V8 ledger | 290,848 |
| Completed bootstrap M0 models | 12 / 12 |
| Completed one-encounter M1 models | 12 / 12 |
| Completed two-encounter M2 models | 8 / 12 |
| Completed four-encounter M4 models | 0 / 12 |
| Unassessed execution intents | 1 |

These are mixed student and forced-teacher acquisition episodes, on different constructor-selected scenes. The 192/251 fraction is **not a learned-policy success rate**. One unassessed intent has unknown recorded-step accounting; it is not a measured failure. The monitor reads controller receipts rather than recomputing all physical scores. The saved snapshot is intentionally fixed while acquisition continues.

The staged sequence is original M4 → five-arm M8 → 138 matched development policy/schedule episodes → 204 fixed-task option-extension episodes → M16/M32. Its [sequence manifest](/home/linjiw/research-data/groot-wbc/m2s-expanded-and-capability-stages-20260909-v2/sequence.json) records those dependencies. Reserved evaluation remains unexecuted in the inspected status records. No experiment was launched or modified to prepare this brief.

## 2. The research question and proposed contribution

The useful question is: **Can controller-executed alternatives identify training encounters that teach a sensor policy to choose a successful, efficient continuation more effectively than simpler construction methods?**

Three difficulties motivate the method. Commanded motion can differ from achieved body geometry. A scene can defeat every available response, in which case selection learning cannot solve it. A scene may have a successful response that cannot be selected from information available before commitment. The method links executed geometry, complete physical outcomes, and causal observations to address these different failure modes.

The strongest candidate contribution is execution-conditioned teaching for a fixed humanoid repertoire. The scientific claim is about the usefulness of acquired data and continuation supervision. Connecting Kimodo to SONIC is already demonstrated by the [official Kimodo project](https://research.nvidia.com/labs/sil/projects/kimodo/). Inverse scene construction also has direct precedent in [Learning from Hallucination](https://www.cs.utexas.edu/~xiao/Research/LfH/LfH.html). These primary-source checks support a focused positioning, not an exhaustive novelty claim.

An appropriately bounded paper can contribute an execution-based construction method, a finite continuation teaching interface, and an empirical explanation of where they help or fail. Replay and prior-generated option extension should remain secondary claims unless their own controlled comparisons demonstrate value.

## 3. Method in plain language

**Execution interface.** Four references provide seven complete schedules: neutral; short adaptation entered at 0.30 or 1.40 s; sustained adaptation entered at 0.30 or 1.40 s; and a projected/spliced prior derivative entered at 0.30 or 1.00 s. Short and sustained variants are authored edits. The prior derivative has explicit generation and repair provenance. Qualification covers the complete transition and recovery, rather than just a low pose. The robot cannot currently steer, stop, choose another route, or chain arbitrary adaptation cycles.

**Scene construction.** Execute the alternatives in the qualified approach and reconstruct their swept body geometry. Search for an overhead obstacle with at least 1 cm positive clearance for one schedule across 81 finite placement offsets and nominal interference for another. These geometric conditions propose useful tasks; the obstacle can change subsequent motion, so every admitted schedule still needs actual simulation labeling. This is not a continuous robustness or safety certificate.

**Observation.** Sixty-five ideal mounted rays and normals contribute to a two-second floor/ceiling history with explicit unknown masks. The learner receives 114 features, including state, phase, active-schedule indicators, and legality. Privileged scene coordinates and branch outcome labels are excluded from deployed inputs. The reference phase is one 0.02 s tick ahead of the associated recorded physical state; entry and cost clocks must remain explicit.

**Teacher.** Execute all seven complete schedules from matched prefixes. At a decision, compare the successful continuations before their measured passage times. WAIT means preserving later legal choices, so its label cannot simply be the outcome of walking forever. Missing outcomes remain unknown. Both-fail tables do not receive invented successful targets.

**Learner.** Fit separate normalized ridge heads at the three decision phases, using negative measured regret as the target and a development-selected penalty of 10. Mask illegal actions at runtime. A narrow all-success, equal-time bootstrap rule supplies zero-regret initialization at an otherwise unsupported phase. Regression scores are not calibrated success probabilities and do not guarantee the teacher's passage-first preference.

**Acquisition and replay.** Each new encounter first executes the previous student, then seven teachers, then fits the next model. The observation-curriculum arm uses historical verified teacher–student gaps with uniform and coverage mass; it shares its scene queue with executed contrast. This order prevents current-scene labels from contaminating the pre-update outcome. The observation-consistent teacher extension solves shared actions within identical-information groups; it is a separate development extension, not the teacher currently used by every acquisition arm.

**Outcome.** Success requires the declared crossing and stabilization, legal recovery, and acceptable contact and stability throughout the recorded horizon. Crossing alone is insufficient. Passage time and complete adaptation duration are different quantities. Applied actuator torque is unavailable, so mechanical work and energy savings cannot be claimed.

## 4. Completed results and their meaning

| Experiment | Measured finding | Supported conclusion and boundary |
| --- | --- | --- |
| Six-context forced bank | 42 complete episodes; 28 pass, 14 fail; 50,064 steps; option union 6/6, best constant 5/6 | Complementary finite-bank capability on inspected development scenes |
| Actual six-context policy comparison | Updated ridge 6/6; updated script 6/6; original learner/script and constant prior 5/6; walk 2/6 | Added teaching data exposes a useful response; learner does not beat updated script on passage |
| Passage-time tradeoff | Updated learner and script each add 0.50 s on short and long contexts | Coverage improvement has an efficiency regression; mean increase is 0.20 s on five shared successes |
| Common-pool geometry | 3,840 candidates; only 169/645 reference-selected pairs preserve executed contrast | Commanded and achieved geometry disagree; this is geometric evidence, not physical passage |
| First acquisition yield M1 | Uniform 0/3 adaptation-required tasks; target-only, executed contrast, replay each 3/3 | More challenging useful tasks than uniform at this tiny budget; target-only ties the proposed contrast |
| M1 cross-corpus recorded readout | One seed: executed 5/5, target-only 4/5, uniform 2/5; other seeds tie across arms | Preliminary data effect; no new closed-loop evaluation and no replay gain |
| Fixed-data learner comparison | Ridge and tree each 4/6 whole-context holdout proxies | Lower fitted loss does not establish broader generalization |
| Native tree comparison | Ridge 6/6 versus tree 4/6; tree 0.12 s slower on four shared successes | Current tree is not a useful replacement; keep common ridge for acquisition |
| Information reveal intervention | Reveal by 1.00 s supports 6/6 finite shared-policy capability; at 1.40 s only 5/6 | An explicit information-timing limitation in recorded tables, not new physical latency performance |
| Physical sensor pilot | Four short-passage conditions all pass; latency changes option and time | Intervention pipeline works on this context; broad robustness remains unmeasured |
| Matched option extraction | Four prior and four authored candidates each qualify 8/8 schedules; 17 total runs including neutral | Repeatable qualification with tied yield; independent incremental coverage remains pending |

The geometry retention rate is 26.2%, recomputed as 169/645. It should not be described as a 73.8% physical failure rate. The 96 M1 encounter episodes consume 114,432 steps excluding bootstrap, with every outcome known. Executed contrast and replay deliberately share scenes, so their task-yield counts are not independent discoveries.

The M1 cross-corpus folds use eight unique acquired scenes across overlapping validation sets. A fixed late sustained schedule covers all eight, so this readout does not establish a perceptive-selection advantage over every constant. Keep the later sustained constant distinct from the constant prior baseline reported in the readout receipt.

The native policy archive retains 35 complete captures and one partial verified contact failure among 36 assignments. The partial failure remains in the denominator with null successful time. These records should not be silently converted to 36 complete sensor trajectories.

Sources: [project guide](PROJECT_GUIDE.md), [expanded construction and acquisition](EXPANDED_DEVELOPMENT_STUDY.md), [decision-learning results](DECISION_LEARNING_RESULTS_20260909.md), [information teaching](INFORMATION_CONSISTENT_TEACHING.md), [sensor pilot](SENSOR_SENSITIVITY_STUDY.md), and [option extraction](OPTION_EXTENSION_STUDY.md). Underlying receipts were inspected for M1 task yield, common-pool construction, and cross-corpus readout; this brief does not claim a fresh audit of every archived trajectory.

## 5. How far are we from a solid ICRA project?

| Evidence gate | Assessment | What closes it |
| --- | --- | --- |
| Runnable integrated system | Substantial development evidence | Preserve reproducible runtime and artifact bindings |
| Precise method and full paper prose | Full working draft exists | Consolidate current method, compress for venue, audit every claim |
| Useful motion alternatives | Established locally | Independently fixed task coverage if claiming repeatable option-generation value |
| Construction mechanism | Geometric evidence plus first physical yield point | Matched learning curves against reference, target-only, and uniform construction |
| Learned decision value | Local script parity; efficiency regression | Actual policy comparison with matched finite-bank capability and strongest constants |
| Replay benefit | Not established | Same-data, same-learner comparison against uniform/coverage and cheap refreshed error |
| Generalization | Reserved results absent | Freeze using development evidence, then execute independent layouts with full denominators |
| Sensor robustness | One-context pilot | Paired multi-context perturbation evidence at a fixed final checkpoint |
| Hardware or general navigation | Not established | Separate interfaces and experiments only if those claims are pursued |

My assessment is that we have an advanced research prototype and a coherent candidate paper, but the main comparative contribution remains open. A completion percentage would hide this: one decisive negative result could require narrowing the story even after all scheduled jobs finish. Hardware is not an automatic requirement for the bounded simulation claim; it becomes necessary if the paper promises physical transfer.

The strongest concern is that target-only acquisition may be enough. The second is that a constant schedule or strong script may explain most of the attainable performance. The third is that the six development contexts are too reused to support generalization. These are questions the next experiments can answer, rather than problems that better wording will resolve.

## 6. Minimum decisive experiment plan

1. **Finish matched construction curves.** Retain five arms and three corpus seeds through the staged budgets. Report admitted tasks, solvable tasks, adaptation-required tasks, unknowns, and downstream completion against actual acquisition steps. Include rejected proposals and simulator startup costs separately. If executed contrast only improves geometry but not policy learning, report that narrower result.
2. **Run the 138-episode M8 panel.** All 15 policies, the strong script, and all seven schedules use the same six contexts and physics seed. This separates missing repertoire capability from poor selection. It is a development decision panel, not the final generalization table. Analyze construction effects paired by corpus seed; three seeds are not 18 independent layouts.
3. **Test incremental option coverage.** The 204-episode fixed-task panel asks whether additional prior samples solve tasks outside the original bank and outside equally repaired authored alternatives. Qualification alone cannot answer this. Retain every candidate and its incremental coverage, rather than selecting only the best sample after seeing outcomes.
4. **Freeze the final method and evaluate reserved layouts.** Bind exact models, schedules, scorer, sensor interface, baselines, and denominator before execution. Include all seven forced schedules on matched nominal conditions so capability and selection can be separated. The older 972-policy-episode proposal and suggested 252 nominal bank episodes are planning quantities; the final expanded comparison needs a consistent adopted manifest rather than silently reusing an incompatible plan.
5. **Run only claim-relevant ablations.** Reference versus executed geometry tests execution information; target-only versus contrast tests the negative alternative; same-scene replay controls test prioritization; controlled information timing tests continuation realizability. Fix perturbations on development data. Compare successful time on mutually successful cases and show the count beside the time difference.

Report unknown outcomes separately from known failures. With partial bank measurements, provide capability bounds. Preserve finite-bank/policy disagreements rather than clipping them away. Predefine an operationally meaningful effect before final evaluation; do not invent a required success margin or universal trial count after seeing results.

## 7. Paper structure and claim decision

The updated full manuscript already contains the following argument: overhead traversal requires executable and observable alternatives; executed envelopes propose decision-bearing constraints; complete continuations teach commitment; development experiments expose complementary capability, data-extension effects, geometry mismatch, and limitations. The new M1 table adds physical acquisition yield without claiming held-out improvement.

For the final conference paper, center the main text on the construction comparison and downstream task result. Keep the option matrix and a clear method figure. Move initialization history, exhaustive scorer details, portable-format documentation, and secondary learner investigations to supporting material where venue rules allow. Main figures should show matched acquisition curves, capability versus policy completion, paired time costs, and representative failures. Do not plot pending results.

If executed construction improves independent completion or reaches comparable performance with less measured acquisition, lead with that result. If it ties target-only, remove claims that negative contrast is necessary. If replay ties simple weighting, describe it as an evaluated optional component. If constants solve the acquired population, broaden development task diversity before the final freeze or narrow the claim to acquisition mechanics. If reserved performance is weak, retain the informative failure analysis; do not present inspected layouts as unseen tests.

The official [ICRA 2027 call](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/) lists September 15, 2026 for contributed papers and video windows August 5–September 9 and September 17–22. As of this brief, about six calendar days remain. The site labels the deadline timezone “PST”; use the submission portal for the exact cutoff. The research outcome cannot be promised on that schedule.

A practical proposed writing cadence is: September 9–10, consolidate completed development comparisons and lock the central claim; September 11–13, complete the chosen independent evidence if runtime permits; September 14, finish source-linked tables, uncertainty, limitations, and formatting; September 15, final submission checks. This is a planning recommendation, not a rollout-time forecast or an instruction to truncate assigned panels. Submission requires a coherent completed evidence package; finishing every optional extension is less important than answering the central comparison honestly.

## 8. Review scope and reproducibility

This update inspected current method prose, status and study notes, the live operational monitor, selected original JSON results, and the staged experiment manifest. It checked primary sources for the most direct positioning and current venue dates. It did not reexecute simulations, change acquisition contracts, select new models, or launch reserved evaluation. The workspace contains extensive existing modifications; this update is confined to the paper, this brief, a dated read-only receipt, and an entry link.

The method draft is the complete prose artifact; this brief is the research assessment and experiment plan. Neither asserts curriculum superiority, unseen-source transfer, hardware results, energy savings, or guaranteed ICRA acceptance.
