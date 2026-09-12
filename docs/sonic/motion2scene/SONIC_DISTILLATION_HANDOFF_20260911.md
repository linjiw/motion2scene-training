> Latest: [repaired-data teacher training, matched student replays, and navigation improvement plan](REPAIRED_TEACHER_NAVIGATION_PLAN_20260912.md). A new 128-environment teacher fit is running on 89 repaired clips.

> 2026-09-12 update: [100-motion experiment, scene implementation, and trajectory-role correction](BFM_SCENE_NAVIGATION_COVERAGE_20260912.md). The final student is not navigation-qualified; use the corrected video/scene receipts linked there.

# SONIC teacher and scene-navigation distillation: restart handoff

**Latest student work:** [BFM implementation and training pilot](BFM_DISTILLATION_IMPLEMENTATION_20260912.md) records the new masked foundation, enabled action-residual path, native collection and first DAgger training round. Scene/navigation training still requires qualified command control and executed scene-teacher labels.

**September 12 follow-up:** Training completed and all 195 final manifest hashes verified. A new matched evaluation and five recorded-motion videos are documented in [SONIC_TRACKING_EVALUATION_20260912.md](SONIC_TRACKING_EVALUATION_20260912.md). Final-trained development completion was 6/20 versus 8/20 for the release; do not treat the final checkpoint as a qualified BFM teacher. The September 11 snapshot below remains historical.

Prepared September 11, 2026, America/New_York. Read the adjacent `SONIC_DISTILLATION_SNAPSHOT_20260911.json` for the timestamped process, progress, configuration and hash snapshot. This handoff describes an **active** training run; it is not a completion receipt.

## Start here after reconnecting

Authoritative checkout: `/home/linjiw/groot-wbc-sonic-sim-trackb`, branch `research/cg-wbc-v0-golden-path` at handoff. The existing job launched from commit `05b15a0970af13701a61ae7bfea615bd5f3578e4` with additional source changes captured in its packet. The commit containing this document records the training helpers and distillation prototypes; it does not retroactively become the experiment's launch commit.

```bash
cd /home/linjiw/groot-wbc-sonic-sim-trackb
git log -1 --format='%H %s' -- docs/motion2scene/SONIC_DISTILLATION_HANDOFF_20260911.md
git status --short
```

There is substantial older uncommitted Motion2Scene/manuscript/viewer work outside this commit. Preserve it. Do not use `git reset --hard`, `git clean`, or a blanket stash during a restart. Python environments, checkpoints and research packets remain on this machine outside Git; committing these source files is not a backup of those assets.

**Do not launch training again just because the chat, SSH connection or editor restarted.** The monitor, trainer and auditor are detached. An operating-system reboot would stop them. Before a planned reboot, verify the final receipts below and retain the external data directories. If the host already rebooted, stale PID files do not establish that any job is alive.

## Active teacher run

Run packet: `/home/linjiw/research-data/m2s-sonic-teacher-8000-20260911`.

| Item | Recorded configuration / location |
|---|---|
| Monitor / trainer / auditor PID at handoff | 351092 / 351209 / 352570; verify current process identity |
| Training | 8,000 additional PPO iterations, 512 environments, 24 rollout steps/environment/iteration |
| Optimization | Five PPO optimization epochs/iteration, four minibatches; an iteration is not one motion epoch |
| Seed and split | Seed 91145; 100 resident uniformly sampled training motions, 20 development motions excluded |
| Physics / action rates | 200 Hz / 50 Hz, decimation 4 |
| Scope | Motion tracking, obstacles disabled, cameras disabled |
| Initialization | Completed 200-iteration checkpoint; strict actor/critic and optimizer/scheduler history; fresh simulator state |
| Checkpoints | `tracking-run-1/model_step_XXXXXX.pt` every 500 iterations; `tracking-run-1/last.pt` rolls every 100 |
| Bound | 48-hour wall cap; no automatic retry or extension |
| Main log | `attempt-1/training.log` |
| Structured progress | `tracking-run-1/progress.jsonl`, `tracking-run-1/metrics.jsonl` |
| Final auditor log | `audit.log` |
| Online W&B | https://wandb.ai/16726/hindsight-sonic-tracking/runs/8117132f |

The maximum rollout counts in `plan.json` are ceilings, not measured execution. Use the callback's observed counters; initialization physics is not measured by that callback. All motions being seen and a high iteration count do not establish complete-motion tracking quality.

Read current state without importing Isaac Lab or starting a process:

```bash
python3 - <<'PY'
import json
from pathlib import Path

packet = Path('/home/linjiw/research-data/m2s-sonic-teacher-8000-20260911')
progress = packet / 'tracking-run-1/progress.jsonl'
if progress.exists():
    raw = progress.read_bytes()
    complete_lines = raw[:raw.rfind(b'\n') + 1].splitlines()
    if complete_lines:
        row = json.loads(complete_lines[-1])
        print({k: row.get(k) for k in ('state', 'iteration', 'observed_env_transitions',
                                      'observed_env_physics_steps', 'wall_seconds')})
for name in ('attempt-1/process.json', 'monitor-process.json', 'audit-process.json'):
    path = packet / name
    if not path.exists():
        print(name, 'MISSING')
        continue
    record = json.loads(path.read_text())
    pid = record.get('pid')
    proc = Path('/proc') / str(pid)
    print(name, 'recorded PID', pid, 'process exists', proc.exists())
    if proc.exists():
        print('cwd:', (proc / 'cwd').resolve())
        print('command:', (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
for name in ('attempt-1/exit.json', 'tracking-run-1/training-receipt.json',
             'FINAL-RECEIPT.json', 'FINAL-MANIFEST.json'):
    print(name, 'present' if (packet / name).exists() else 'not present')
PY
```

Match the command and cwd to this run; a reused PID can belong to something else. If the trainer is active, continue monitoring it. Do not re-run `launch.py`, `prepare_long.py`, the old 200-iteration launcher, or `distill.py` on this packet.

## Completion and interrupted-run handling

Expected normal completion sequence:

1. `tracking-run-1/training-receipt.json` reports `state=complete`, iteration 8000, and a final checkpoint path/hash.
2. W&B finishes flushing, the trainer exits, and the monitor writes `attempt-1/exit.json` with exit code 0.
3. The detached auditor writes `motion-training-outcomes.csv`, `FINAL-RECEIPT.json` and `FINAL-MANIFEST.json`.

All three stages matter. `last.pt`, a log mentioning iteration 8000, or a W&B graph alone is insufficient. Tracking-success and navigation-passage columns remain NA because this run is not a matched qualification evaluation.

After the trainer/monitor have exited and all final files exist, verify the receipt and manifest:

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

packet = Path('/home/linjiw/research-data/m2s-sonic-teacher-8000-20260911')
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
receipt = json.loads((packet / 'FINAL-RECEIPT.json').read_text())
assert receipt['state'] == 'complete', receipt['state']
assert receipt['exit']['exit_code'] == 0
assert receipt['training']['iteration'] == 8000
checkpoint = receipt['training']['checkpoint']
assert sha(Path(checkpoint['path'])) == checkpoint['sha256']
manifest = json.loads((packet / 'FINAL-MANIFEST.json').read_text())
failures = []
for item in manifest['files']:
    path = packet / item['path']
    if not path.is_file() or sha(path) != item['sha256']:
        failures.append(item['path'])
assert not failures, failures
print('Verified completed training and', len(manifest['files']), 'file hashes')
print('Checkpoint:', checkpoint)
print('Teacher quality:', receipt['teacher_quality'])
PY
```

If the monitor has exited but the auditor is still alive, let it finish. If the auditor is absent, `attempt-1/exit.json` exists, and **none** of `motion-training-outcomes.csv`, `FINAL-RECEIPT.json`, `FINAL-MANIFEST.json` has been written, the audit-only command is:

```bash
.venv_isaaclab/bin/python -m gear_sonic.research.hindsight_training.audit_long /home/linjiw/research-data/m2s-sonic-teacher-8000-20260911
```

This command does not launch simulation. Do not run competing auditors. It creates files exclusively: if any output is partial or an audit failed, preserve it and diagnose from `audit.log`; do not delete outputs to make the command succeed. `finalize.py` is the historical 200-iteration reconciler, not the long-run finalizer; it also references the older local renderer source.

If training is interrupted, preserve every log, receipt, checkpoint and measured counter. No exact simulator-state resume is implemented by `LongTrackingTrainer`; `resume=True` is deliberately rejected. The present `prepare_long.py` is specifically for continuation from the completed 200-iteration parent, not from an arbitrary interrupted long run. A new continuation needs a new packet, verified checkpoint and a reconciled remaining budget. Work after the last persisted checkpoint still consumed execution budget; do not reset accounting to the checkpoint iteration or restart another 8,000 iterations. If counters or required artifacts are missing, mark them BLOCKED/NA before designing any retry.

## Runtime and GPU facts

Use `.venv_isaaclab/bin/python`: Python 3.11.16, PyTorch 2.7.0+cu128, Isaac Sim 5.1, Isaac Lab 0.54.2 in the inspected runtime. Isaac Lab checkout: `/home/linjiw/IsaacLab`, commit `37ddf626871758333d6ed89cf64ad702aef127d0`. Verify again after any environment change.

This host reported an NVML kernel/library mismatch while actual CUDA allocation and Isaac Lab headless physics worked. Do not interpret `nvidia-smi` failure alone as failed training or reinstall drivers during the active job. The 8,000-iteration packet's CUDA free-memory launch floor is **8 GiB**; the earlier 200-iteration packet used 2 GiB. Neither is a measured peak. Vulkan/graphics warnings do not establish camera or renderer qualification.

## Frozen assets and source provenance

| Asset | Local location / role |
|---|---|
| Dataset v1 | `/home/linjiw/research-data/m2s-hindsight-dataset-v1-20260911` |
| Completed 200-iteration parent | `/home/linjiw/research-data/m2s-sonic-teacher-student-v1-20260911` |
| Current 8,000-iteration packet | `/home/linjiw/research-data/m2s-sonic-teacher-8000-20260911` |
| BFM paper and first scene-distillation design | `/home/linjiw/research-data/m2s-scene-distillation-design-20260911` |
| Preferred frozen-foundation navigation revision | `/home/linjiw/research-data/m2s-navigation-foundation-v2-20260911` |

The adjacent snapshot binds the immutable manifests, current configuration, and available permanent checkpoints with SHA-256. `INPUT-MANIFEST.json`, `attempt-1/code-input-manifest.json`, `launched-source.tar.gz` and the original process receipt describe what the running job actually used. Do not overwrite their recorded launch commit after this Git commit. A hash created at handoff does not prove an earlier preregistration date.

Dataset facts: 120 motions (100 train / 20 development), 240 geometry-only scenes, 25,335 reference frames. All 120 three-obstacle scenes are prefixes of their five-obstacle counterparts attached to the same motion. No executed scene-teacher labels exist in that frozen dataset. Keep missing outcomes null/NA.

The active dataset web server is a separate detached process (PID 319906 at handoff), serving this frozen directory on `127.0.0.1:8766`. Its HTTP response was checked when writing the handoff. If it stopped after a reboot, first check that the port is free, then run:

```bash
nohup python3 -m http.server 8766 --bind 127.0.0.1 --directory /home/linjiw/research-data/m2s-hindsight-dataset-v1-20260911 >> /tmp/m2s-dataset-http.log 2>&1 < /dev/null &
```

From your own computer, using your existing SSH host name:

```bash
ssh -N -L 52323:127.0.0.1:8766 linjiw@YOUR_EXISTING_SSH_HOST
```

Open `http://localhost:52323/dataset/index.html`. If another tunnel already owns port 52323, reuse it or choose a different local port. Do not expose the dataset publicly just to recover the SSH view. Renderer/generator work under `scripts/research/lflh_next/` is older uncommitted work; preserve it and the frozen dataset, although it is outside this source commit's scope.

## Latest design decision and next work

Read the [preferred foundation/navigation design](../../gear_sonic/research/scene_distillation/FOUNDATION_NAVIGATION.md).

The user's chosen direction is **tracking teacher → BFM-style reusable motion foundation → downstream scene-and-goal command selection**. The teacher's motion prior is preserved; navigation learns how to use it. During downstream training the foundation prior, token adapter and SONIC decoder are frozen. The primary residual scale is zero. A latent residual is a separate ablation and differs from BFM's demonstrated action residual.

Inference uses measured robot history, start/goal and a known obstacle map; no future reference, external ground-truth route, phase, motion ID or camera. The implemented four-command navigation profile is a prototype, not full BFM control-interface fidelity or a qualified controller. A Gaussian behavior foundation checkpoint has not yet been trained here. The earlier direct scene-conditioned CVAE and route-conditioned offline student remain comparisons, not the preferred deployment path.

Next work, in order:

1. Reconcile completed training and preserve the selected checkpoint. Run only a separately locked, bounded matched tracker qualification; audit complete motions and failures rather than selecting attractive clips.
2. Prepare the BFM-style foundation-distillation and command-qualification protocol. Verify that intended commands and skills are expressible before freezing the foundation. Current code supplies architecture and losses, not this completed stage.
3. Implement and qualify scene-dependent continuation selection and same-state teacher queries. The tracking teacher itself does not choose obstacle-dependent routes. The present scene USD loader is single-environment; batched articulated scene training is not complete.
4. Collect bounded scene-teacher labels, including discriminating changed-scene/changed-goal tasks, then train the navigation command head through the frozen foundation. Evaluate whole-chain goal completion, contacts and terminal stabilization with the public prior alone.

These engineering stages do not discharge the original Motion2Scene physical-utility, manuscript, adoption or ancestry-held-out transfer gates. Keep the original A/B/C pilot unchanged and the eighteen reserved layouts unopened. No hardware, protective-stop, crawling, generalization or dataset-utility claim follows from the prototypes.

## Validation of this source commit

The preceding implementation checks passed 50 tests, including synthetic gradients through a frozen foundation, no privileged posterior dependency at inference, coordinate/shape contracts, missing-label handling, and exact frozen-prior preservation. The native CPU boundary check used an actual SONIC checkpoint and had action parity error 0.0; its foundation weights and command bounds were synthetic. This was not navigation training or a physical qualification.

```bash
.venv_isaaclab/bin/python -m pytest -q decoupled_wbc/tests/test_hindsight_training.py decoupled_wbc/tests/test_hindsight_long_tracking.py decoupled_wbc/tests/test_scene_distillation.py decoupled_wbc/tests/test_foundation_navigation.py
.venv_research/bin/python -m black --check gear_sonic/research/hindsight_training gear_sonic/research/scene_distillation decoupled_wbc/tests/test_hindsight_training.py decoupled_wbc/tests/test_hindsight_long_tracking.py decoupled_wbc/tests/test_scene_distillation.py decoupled_wbc/tests/test_foundation_navigation.py
.venv_research/bin/python -m ruff check --select E,F,I gear_sonic/research/hindsight_training gear_sonic/research/scene_distillation decoupled_wbc/tests/test_hindsight_training.py decoupled_wbc/tests/test_hindsight_long_tracking.py decoupled_wbc/tests/test_scene_distillation.py decoupled_wbc/tests/test_foundation_navigation.py
```

Historical `finalize.py` expects the local renderer source mentioned above; it is not needed for normal completion of the active long run. Large weights, motion data, paper PDF and runtime logs are intentionally referenced by local paths/manifests, not embedded in Git.
