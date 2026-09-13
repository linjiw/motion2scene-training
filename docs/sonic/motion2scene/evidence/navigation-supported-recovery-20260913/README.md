# Supported navigation recovery evidence

Compact records for the [research report](../../NAVIGATION_SUPPORTED_RECOVERY_20260913.md).
Native traces, training shards, checkpoints and complete simulator logs remain in
`/home/linjiw/research-data/m2s-nav-supported-recovery-20260913/`.
Absolute paths identify the recorded machine; they are not bundled dataset assets.

Plans, configs, checkpoint hashes, per-run commands/source hashes/scores and bound
collection manifests are included. Copied experiment drivers are archival scripts:
their parent-directory convention expects the original external packet, and they
must be adapted to unused output paths before rerunning. Do not overwrite evidence.
Training receipts verify frozen motor tensors and full-command action identity;
physical task outcomes are reported separately. Failed recovery attempts remain in
the manifest. `native-run-summary.json` includes demonstrations and privileged
interventions as well as unassisted navigation; its total is not a success rate.

The four motor confirmation controls were adaptively selected failed cases, not a
full-panel or information-matched navigation baseline. The confirmation seed is
an evaluation variation on the same eight training tasks, not new layouts.
