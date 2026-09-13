import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;e=Path('docs/motion2scene/evidence/navigation-continuation-20260913');e.mkdir(parents=True,exist_ok=True)
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
for src in p.glob('*.json'):
 if '-config.json' not in src.name or src.name in ['replay-config.json','recovery-config.json']:
  shutil.copy2(src,e/src.name)
for name in ['probe.py','resume-probe.py','teacher-controls.py','study.py','audit.py','plot-continuations.py','task-outcome-audit.py','export-evidence.py']:
 shutil.copy2(p/name,e/name)
records=[]
for f in sorted(p.glob('*/task/task-result.json')):
 r=json.loads(f.read_text());stage=f.parent.parent.name;role='privileged_motor_continuation' if (f.parent/'recovery.json').exists() else ('original_teacher_intervention' if (f.parent/'teacher-continuation.json').exists() else 'unassisted_navigation');records.append(dict(stage=stage,execution_role=role,score_sha256=sha(f),**r))
 for src in [f,f.parent/'recovery.json',f.parent/'teacher-continuation.json',f.parent.parent/'command.json',f.parent.parent/'source-sha256.json',f.parent.parent/'process-result.json',p/(stage+'-config.json')]:
  if src.exists():
   dest=e/'runs'/stage/src.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
checkpoints={}
for name in ['replay-fit','recovery-fit']:
 d=p/name/'training'
 if not d.exists():continue
 for src in [d/'receipt.json',d/'config.json']:
  if src.exists():
   dest=e/name/src.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(src,dest)
 f=d/'step-003000.pt'
 if f.exists():checkpoints[name]=dict(path=str(f),sha256=sha(f))
(e/'checkpoints.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
(e/'native-run-summary.json').write_text(json.dumps(dict(completed_attempts=len(records),control_steps=sum(r['control_steps'] for r in records),records=records),indent=2)+'\n')
collection=[]
for f in p.glob('*/task/recovery.json'):collection.append(json.loads(f.read_text()))
teacher_steps=sum(max(0,r['control_steps']-json.loads((p/r['stage']/'task/teacher-continuation.json').read_text())['takeover_tick']) for r in records if r['stage'].startswith('teacher-'))
cost=dict(all_collection_attempts=len(collection),collection_physics_steps=sum(r['rows'] for r in collection),motor_target_forwards=sum(r['rows'] for r in collection),original_teacher_collection_diagnostic_forwards=sum(r['rows'] for r in collection),original_teacher_control_action_forwards=teacher_steps,all_completed_native_attempts=len(records),all_native_control_steps=sum(r['control_steps'] for r in records),infrastructure_failed_launches=1,failed_launch_task_steps=0,qualification_note='All probe variants and failed attempts count toward acquisition; only the selected-provider manifest enters the training comparison.')
(p/'acquisition-cost.json').write_text(json.dumps(cost,indent=2));shutil.copy2(p/'acquisition-cost.json',e/'acquisition-cost.json')
(e/'README.md').write_text('''# Continuation study evidence

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
''')
print(json.dumps(cost,indent=2))

for row in json.loads((p/'infrastructure-failures.json').read_text()):
 stage=row['stage'];dest=e/'failed-launches'/stage;dest.mkdir(parents=True,exist_ok=True)
 for src in [p/stage/'command.json',p/stage/'source-sha256.json',p/(stage+'-config.json')]:
  if src.exists():shutil.copy2(src,dest/src.name)
 log=(p/stage/'process.log').read_text();lines=[line for line in log.splitlines() if 'torch.OutOfMemoryError:' in line]
 (dest/'error.txt').write_text('\n'.join(lines)+'\n')
