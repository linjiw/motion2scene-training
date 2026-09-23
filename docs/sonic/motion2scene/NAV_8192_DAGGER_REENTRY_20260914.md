# Short-cycle navigation DAgger, braking-aware sampling and a phase-rewind re-entry probe

September 14, 2026. Follow-up to the [24-task stopping study](NAV_8192_STOPPING_20260913.md), which ended with three recommendations: switch early and often in a short-cycle DAgger loop, weight braking rows, and build a re-entry continuation before collecting late switches. This report runs the first two as a matched pair of arms and tests the third as a diagnostic probe. All runs use the 8192-env teacher, the `s1-3200` motor, the 24-task panel (19 tasks feasible for teacher and motor) and seed 91260.

![Re-entry probe and DAgger cycle 1](evidence/nav-8192-dagger-reentry-20260914/dagger-reentry-summary.png)

## Re-entry probe: rewinding the reference clock recovers late states the nominal clock cannot

`navigation_reentry.ReentryProbeCallback` replays the exact nav-v1 learner prefix to the switch tick, then rewinds the reference clock to the frame whose reference pelvis is nearest the measured pelvis (never forward, so no obstacle can be skipped) and lets the frozen motor track from there. It was run on the 33 late states (50% and 75% switches) that the nominal-clock takeover had failed in the earlier sweep; one state was never reached because the learner hit a wall first.

| On the same 32 late states | Nominal clock | Phase rewind |
|---|---:|---:|
| Suffix completes the task before the original deadline | 0 | **6** |
| Suffix enters the goal tolerance at all | 1 | **11** |
| Mean longest post-switch hold (ticks) | 0.7 | 13.9 |
| Prohibited contact | 0 | 0 |

Rewinding helps exactly where the learner is *behind* the reference along its path (00413, 00770, 00801, 00976: rewinds of 18–72 frames, reference error after rewind ≤ 0.25 m). It does nothing where the learner is *beside* the path (00120, 00677, 00908: rewind 0–4 frames, lateral error 0.2–0.7 m) and it fails when the rewind is so large that the remaining deadline cannot absorb it (00265-corridor: 198 frames rewound, ends 0.16 m from the goal with no time to hold). Two rewinds of 99–146 frames reach 24–38 hold ticks but not 50.

These are privileged diagnostics. A phase rewind is a reference-phase switch, which the recovery loader rejects by contract; no probe row entered any fit. The result justifies a versioned re-entry continuation contract (rewind-only, deadline-aware, commands reconstructed from the rewound clock), and a lateral re-entry that the nearest-frame rule cannot provide.

## Short-cycle DAgger, cycle 1

Both arms start from `nav-v2-recovery` (4/19 feasible tasks). One cycle = switches at 12% and 25% of the teacher's completion tick on the 19 feasible tasks (38 attempts, shared between arms because the behaviour checkpoint is identical), refit for 3,000 updates from the current checkpoint on every supported recovery so far (fresh = this cycle at 50%), then the unassisted panel.

- Collection: **32/38 supported**, 9,123 rows; the training view grows to 94 episodes and 26,776 rows.
- Uniform arm: rows uniform within episode. Approach arm: rows whose body-frame goal lies within 1.0 m get sampling weight 4 (`approach_weight`, a new explicit option in `navigation_motor.fit`; 16,221 of 26,776 rows qualify).

| Arm (19 feasible tasks, seed 91260) | Completed | Reached goal | Holds ≥ 30 ticks | Contact | Mean final distance |
|---|---:|---:|---:|---:|---:|
| nav-v2-recovery (start) | 4 | 9 | 6 | 1 | 1.40 m |
| cycle 1, uniform | 3 | 9 | 6 | 3 | 0.99 m |
| cycle 1, approach-weighted | 4 | 10 | 6 | 1 | 0.96 m |

Completion does not move in one cycle; which tasks pass does (approach: gains 00677-corridor, 00908 both; loses 00677-clear, 00707-clear, 00976-corridor, three of them with holds of 32–46 ticks). Mean final distance improves in both arms, and the approach arm keeps contacts at one versus three for uniform.

## Short-cycle DAgger, cycle 2 (appended September 23)

Each arm now collects from its own cycle-1 checkpoint: same 12%/25% switches on the 19 feasible tasks, 34/38 supported in both arms. Each arm refits for 3,000 updates from cycle 1 on all supported recoveries so far (fresh = this cycle at 50%), with the same sampling rule it used before.

| Arm (19 feasible tasks, seed 91260) | Start | Cycle 1 | **Cycle 2** | Reached goal (c2) | Holds ≥ 30 ticks (c2) | Contact (c2) | Mean final distance (c2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 4 | 3 | **2** | 11 | 4 | 1 | 0.98 m |
| approach-weighted | 4 | 4 | **7** | 9 | 7 | 2 | 1.03 m |

Approach cycle 2 passes 00413-clear, 00677-clear, 00707-clear, 00707-corridor, 00770-corridor, 00908-clear and 00908-corridor. Two of these were near misses in cycle 1: 00677-clear (32-tick hold) and 00707-clear (46). Two are new: 00413-clear and 00770-corridor never reached the goal in cycle 1. Its contacts are 00801-corridor and 00711-corridor. Uniform cycle 2 passes only 00707-clear and 00801-corridor. Five of its attempts reach the goal and hold 15–49 ticks, but none completes the 50-tick hold. Uniform DAgger reaches the goal more often and holds less. That fits the braking hypothesis: without approach weighting, extra recoveries teach arrival but not stopping.

This is the first arm on the local stack to exceed the 4/24 → 4/19 plateau. It still rests on **one evaluation seed and one training seed**. The earlier 4/24 → 1/24 confirmation drop shows that seed noise here can be as large as this effect. Before the arms are compared, both cycle-2 checkpoints need seeds 91262 and 91264.


## Reading

- Early switches are cheap positive data (84% supported) but one cycle of them does not change the completion count; the near-miss population (holds of 30–46 ticks) is where the adapter sits.
- Braking-aware sampling is not harmful and slightly reduces contacts and overshoot in this cycle; the effect is within seed noise for completions.
- The re-entry probe is the clearest positive signal in this series: for the "stopped short" failure mode the frozen motor *can* finish the task if the reference is re-timed. Turning that into training data requires a contract change, not more collection.

## Next

1. Version a rewind-only re-entry continuation (`reference_phase_switch=True`, `rewind_frames`, deadline check) in the recovery receipt and loader, then collect re-entry recoveries at 50–75% switches; these cover the late approach/braking states the early switches never reach.
2. A lateral re-entry rule (nearest frame on the path *plus* a bounded lateral shift of the reference, or a walk-back continuation) for the "beside the path" states.
3. Two more evaluation seeds per arm before comparing arms. One seed cannot separate 3 from 4, and 2 versus 7 needs confirmation too (see cycle 2).

## Evidence

[`evidence/nav-8192-dagger-reentry-20260914/`](evidence/nav-8192-dagger-reentry-20260914/) ([file index](evidence/nav-8192-dagger-reentry-20260914/README.md)) holds:
- per-state probe outcomes (`reentry-probes.json`);
- cycle-1 recoveries, fit configs and receipts for both arms;
- per-task panel results and the figure;
- cycle-2 evidence, added September 23: the switch plan and per-arm recovery receipts, training manifests, fit configs and receipts, panel `results.txt`, and per-task `task-result.json`, `process-result.json`, stage config and launch command for both c2 panels.

Packet `workspace/nav-8192` (dagger/, collect/reentry-v1-91260/) holds shards, traces and logs. Code: `scripts/navigation_distill/{dagger_arm,dagger_cycle,run_reentry}.sh`, `navigation_motor.approach_row_weights`, `navigation_reentry.py`.
