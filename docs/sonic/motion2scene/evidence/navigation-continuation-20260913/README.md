# Continuation study evidence

See the [research report](../../NAVIGATION_CONTINUATION_STUDY_20260913.md).
This directory contains plans, source/command records, bound manifests, scores,
checkpoint hashes, training receipts, validation, audits and an original trace figure.
Large arrays, checkpoints and simulator logs remain in
`/home/linjiw/research-data/m2s-nav-continuation-20260913/`.

The copied experiment drivers are archival scripts whose parent-directory convention
expects the external packet. Absolute paths must be adapted to existing assets and
unused outputs before rerunning. Do not overwrite recorded experiments. The figure
can be regenerated using `plot-continuations.py` in that packet from the repository root.

Completed native counts include privileged probes and original-teacher interventions,
not just unassisted navigation. One checkpoint-loading OOM is recorded separately;
the failed launch produced no task trajectory and was retried unchanged sequentially.
The all-probes loader audit includes nonselected candidates only to validate the
reconstruction contract; it is not the selected-provider training dataset.
