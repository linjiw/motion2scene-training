# Evidence: navigation DAgger and phase-rewind re-entry (nav-8192)

Compact receipts for the [DAgger and re-entry report](../../NAV_8192_DAGGER_REENTRY_20260914.md). They were copied from the packet `workspace/nav-8192` on the robotixx host. That folder is gitignored and also holds the traces, shards, checkpoints and logs, none of which are copied here.

All panels use evaluation seed 91260, the seed that produced every training label, and rollouts are bit-deterministic. The cycle-2 results are therefore **in-sample** (see the [roadmap](../../../../ROADMAP_20260923.md#2-where-each-layer-stands)). Both c2 fits use training seed 91482.

## Files

**Cross-stage tables.** Regenerate them with `scripts/navigation_distill/export_results.py --packet workspace/nav-8192 --out <this folder>`. The glob matches `eval/` and `collect/` stages only, not `dagger/`.

| File | Contents |
|---|---|
| `task-results.csv` | One row per `task-result.json` in every eval and collect stage: success, reached, hold, final distance, force, fall, steps, stop reason, actor profile and student sha. It covers 16 stages, including both c1 and both c2 panels. The c2 rows were added on September 23; the other 345 rows are unchanged. |
| `recovery-attempts.csv` | One row per collect-stage `recovery.json` (motor-demo, recovery-v1): takeover tick, supported, rows, suffix score and behaviour sha |

**Re-entry probe** (report §1).

| File | Contents |
|---|---|
| `reentry-plan.txt` | The 33 late learner states (task, switch tick) that were probed |
| `reentry-probes.json` | Per-state outcome: rewind frames, reference and nominal errors, hold, distance, contact, for both the rewound and the nominal clock |
| `dagger-reentry-summary.png` | Report figure (probe and cycle 1) |

**DAgger cycle 1** (both arms start from `nav-v2-recovery`).

| File | Contents |
|---|---|
| `dagger-c1-recoveries.csv` | Cycle-1 collection: 38 attempts at 12% and 25% switches, 32 supported. Both arms share it, because their behaviour checkpoint is identical |
| `dag-{approach,uniform}-c1-fit-config.json` | Fit configs (`dagger/<arm>/c1-fit-config.json`) |
| `dag-{approach,uniform}-c1-fit-receipt.json` | Fit receipts (`dagger/<arm>/c1-fit/receipt.json`) |
| `dag-{approach,uniform}-c1-results.txt` | Unassisted 19-task panel (`eval/dag-<arm>-c1-91260/results.txt`) |

**DAgger cycle 2** (added September 23; each arm collects from its own c1 checkpoint).

| File | Contents |
|---|---|
| `dagger-c2-plan.txt` | Switch plan, 38 attempts at 12% and 25% of the teacher's completion tick. Both arms' `dagger/<arm>/c2-plan.txt` files are identical, so only one copy is kept |
| `dag-{approach,uniform}-c2-recovery-results.txt` | Collection summary (`dagger/<arm>/c2-recovery/results.txt`): 34/38 supported in each arm |
| `dagger-c2-recovery-attempts.csv` | Per-attempt recovery receipts for both arms (stage `dag-<arm>/c2-recovery`): takeover tick, supported, rows, suffix hold and distance, behaviour sha |
| `dagger-c2-task-results.csv` | Scores of the same 76 **assisted** collection episodes (motor takeover). These are not unassisted navigation results |
| `dag-{approach,uniform}-c2-manifest.json` | Training-view manifests: 128 parent episodes and 48 unsupported attempts, each bound by SHA-256 to its receipt and trace. `dataset_manifest_sha256` in the fit config hashes this file |
| `dag-{approach,uniform}-c2-fit-config.json` | Fit configs (`dagger/<arm>/c2-fit-config.json`, identical in content to `c2-fit/config.json`): warm start from c1, 3,000 updates and recovery fraction 0.5. The approach arm adds `approach_weight` 4 within 1.0 m |
| `dag-{approach,uniform}-c2-fit-receipt.json` | Fit receipts (`dagger/<arm>/c2-fit/receipt.json`): rows (36,959 approach and 37,077 uniform), episodes, qualified recoveries, and approach rows (22,375) |
| `dag-{approach,uniform}-c2-results.txt` | Unassisted 19-task panel (`eval/dag-<arm>-c2-91260/results.txt`): approach 7/19, uniform 2/19 |
| `dag-{approach,uniform}-c2-91260/<task>/` | Per-task receipts from the panel: `task-result.json` (score, checkpoint and task sha), `process-result.json` (exit code, wall time), `config.json` (stage config with checkpoint paths and shas) and `command.json` (the exact launch, including `++seed=91260`) |

Regenerate the cycle-2 files from the repository root with the packet at `workspace/nav-8192`:

```bash
E=docs/sonic/motion2scene/evidence/nav-8192-dagger-reentry-20260914 P=workspace/nav-8192
python scripts/navigation_distill/export_results.py --packet $P --out $E \
  --copy-receipts dag-approach-c2-91260 --copy-receipts dag-uniform-c2-91260
python scripts/navigation_distill/export_results.py --packet $P --out $E --prefix dagger-c2- \
  --glob 'dagger/*/c2-recovery/*/task/task-result.json'
cp $P/dagger/dag-approach/c2-plan.txt $E/dagger-c2-plan.txt
for a in approach uniform; do
  cp $P/eval/dag-$a-c2-91260/results.txt $E/dag-$a-c2-results.txt
  cp $P/dagger/dag-$a/c2-recovery/results.txt $E/dag-$a-c2-recovery-results.txt
  cp $P/dagger/dag-$a/c2-fit-config.json $E/dag-$a-c2-fit-config.json
  cp $P/dagger/dag-$a/c2-fit/receipt.json $E/dag-$a-c2-fit-receipt.json
  cp $P/dagger/dag-$a/c2-manifest.json $E/dag-$a-c2-manifest.json
done
```

**Left in the packet:**
- traces (`task/trace.npz`) and recovery shards (`task/motor-recovery.npz`);
- checkpoints (`c*-fit/step-003000.pt`, 72 MB each). Their SHA-256 is recorded as `student_sha256` in each panel `task-result.json`. On September 23 the c2 checkpoints on disk were re-hashed and matched: approach `426f0fc526a70f0e…`, uniform `3af09f466ada1e19…`;
- fit curves (`c*-fit/metrics.jsonl`);
- native and Hydra logs, and the c1 switch plans.
