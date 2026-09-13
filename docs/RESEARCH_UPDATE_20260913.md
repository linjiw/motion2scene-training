# Research update — September 13, 2026

This update supersedes the package's earlier pending-recovery summary. It records completed source experiments; syncing the package does not rerun native training or physics. See the [sync manifest](publication/research-sync-20260913.json) for source revision, snapshot time and copied-file SHA-256 hashes.

## Methods and research artifacts

The current pipeline constructs candidate scenes from executable motion, physically qualifies demonstrations and corrections, and trains a goal/map adapter through a frozen motor specialist. Geometry compatibility alone does not certify execution or useful scene-dependent decisions.

The motor consumes 930 native history values and 114 current-command coordinates and produces 29 joint actions at 50 Hz. The navigation adapter predicts those commands from native history, body-frame start/goal, known obstacle primitives and causal localized velocity. It receives no reference clock, clip identifier or future demonstration trajectory. This assumes known maps and ideal localized poses; it is not camera navigation.

The [latest methods package](sonic/motion2scene/methods_v2_20260913/README.md) includes editable prose, architecture figures, a [reading PDF](sonic/motion2scene/methods_v2_20260913/research_methods.pdf), and an architecture atlas. The [overview](sonic/motion2scene/overview_20260913/README.md) supplies the broader pipeline explanation. These are research drafts; proposed generality is not a measured result.

## Results and analysis

| Experiment | Measured outcome | Interpretation |
|---|---|---|
| Full-command motor recovery | 88/89 train; 10/20 development completions across three evaluation seeds | Useful motor anchor, with remaining generalization limits |
| Original bounded stopping panel | Full-command motor 8/8; original goal/map adapter 0/8 | Supported motor execution does not automatically transfer to goal control |
| Exact-motor demonstrations and supported recovery | Initial motor-data fit 2/8; selected recovery fit 4/8, then 1/8 on confirmation | Motor-state teaching helps the first panel but is unstable across evaluation seeds |
| Subsequent continuation comparison | Parent 4/8, extra old replay 3/8, fresh recovery/replay 4/8 on first seed; parent 1/8 versus fresh 2/8 on confirmation | Fresh recovery adds one confirmation completion; no broad navigation claim |
| Expanded motor/navigation replay | Registered panel running at snapshot | No completed aggregate result yet |

The latest fresh-recovery checkpoint succeeds on 00908 clear on both tested seeds where its parent fails. Both confirmation panels fail the unchanged prohibited-contact criterion on 00413 corridor. The fresh panel has two completions, one contact failure and five timeouts; the parent has one completion, one contact failure and six timeouts. Neither has a fall. Reduced contact magnitude does not count as success.

Success requires 3D pelvis goal distance ≤0.25 m and speed ≤0.10 m/s for 50 consecutive control ticks before the task deadline. Prohibited contact above 1 N or pelvis height below 0.25 m terminates the attempt. This is a one-second hold, without a separate posture/heading criterion or proof of indefinite stability.

The latest continuation study records 56 completed native attempts and 24,061 control steps: 20 motor continuations, four original-teacher controls and 32 unassisted navigation evaluations. Two new fits each use 3,000 updates. Intervened trajectories are teaching/diagnostic data, not autonomous successes. The original eight tasks span four clips and one ancestry group; these results do not establish held-out-layout reasoning or independent training-seed robustness.

Fresh recovery also changes motion exposure. A registered replay-weighting control matches the prior mixture's expected 62.5% 00908 / 12.5% each remaining training motion before attributing improvements to correction quality. The expanded panel adds five motions absent from navigation demonstrations but present in motor training. All attempted tasks, including full-command control failures, remain in its denominator.

## Research log and evidence

1. [Terminal-control audit](sonic/motion2scene/NAVIGATION_TERMINAL_CONTROL_20260913.md): separates late arrival from speed violations and drift; implements causal localization and same-state targets.
2. [Supported recovery](sonic/motion2scene/NAVIGATION_SUPPORTED_RECOVERY_20260913.md): executes exact-motor demonstrations, qualifies corrective suffixes, and compares equal-budget fitting forks. [Evidence](sonic/motion2scene/evidence/navigation-supported-recovery-20260913/README.md).
3. [Continuation study](sonic/motion2scene/NAVIGATION_CONTINUATION_STUDY_20260913.md): tests bounded continuation providers, preserves unsuccessful attempts, and reports paired confirmation panels. [Evidence and experiment log](sonic/motion2scene/evidence/navigation-continuation-20260913/README.md).
4. [Expanded replay protocol](sonic/motion2scene/NAVIGATION_EXPANDED_REPLAY_20260913.md): defines matched inherited-motor controls, new task selection, strict scoring and the motion-weighting experiment; results pending.

The experimental candidate is identified by SHA-256 `d378947413cdd167e71e3271440e9a18fbfd5a609836aa0be7d98cefe3a6e9cd`. Its raw checkpoint and large execution arrays remain at the external paths recorded in the evidence. This sync includes compact reports, receipts, scores, code and methods artifacts; it does not add that checkpoint to the package's five existing data bundles.

The next experiment should qualify coherent locomotion re-entry, approach, braking and hold from displaced clear-scene states before extending to corridor recovery. Then compare old replay, motion-weight-matched replay and qualified fresh corrections at fixed budgets. Goal/route/posture switches, independent layouts and camera observations remain later evidence requirements.
