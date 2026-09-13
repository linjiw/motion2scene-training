import json,os,subprocess,time
from pathlib import Path
from gear_sonic.research.hindsight_training.runtime import sha
p=Path(__file__).resolve().parent;repo=Path.cwd();py=str(repo/'.venv_isaaclab/bin/python');old=p.parent/'m2s-nav-motor-bridge-20260913';teacherpacket=p.parent/'m2s-bfm-motor-recovery-20260913';previous=p.parent/'m2s-nav-terminal-control-20260913';ledger=[]
def run(name,cmd):
 out=p/name;out.mkdir();(out/'command.json').write_text(json.dumps(cmd,indent=2));(out/'source-sha256.json').write_text(json.dumps({str(f):sha(f) for f in (repo/'gear_sonic/research/scene_distillation').glob('*.py')},indent=2));t=time.monotonic()
 with (out/'process.log').open('w') as f:r=subprocess.run(cmd,env=dict(os.environ,PYTHONPATH=str(repo),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT,timeout=600)
 row=dict(stage=name,exit_code=r.returncode,wall_seconds=time.monotonic()-t);ledger.append(row);(p/'progress.json').write_text(json.dumps(ledger,indent=2));print(row,flush=True)
 if r.returncode:raise RuntimeError(name)
def configfile(name,c):
 cp=p/(name+'-config.json');cp.write_text(json.dumps(c,indent=2));return cp
def fit(name,c):
 cp=configfile(name,c);run(name+'-fit',[py,'-m','gear_sonic.research.scene_distillation.navigation_motor','--config',str(cp),'--output',str(p/(name+'-fit')/'training')]);return p/(name+'-fit')/('training/step-%06d.pt'%c['updates'])
def native(name,task,checkpoint,target,**extra):
 c=json.loads((old/('nav-'+task+'-config.json')).read_text());c.update(student_checkpoint=str(checkpoint),student_sha256=sha(checkpoint),actor_profile='nav_goal_map_localization_v2',output=str(p/name/'task'),**extra);cp=configfile(name,c);cmd=json.loads((old/('nav-'+task)/'command.json').read_text());rep={'++eval_output_dir':str(p/name/'unused'),'++eval_base_dir':str(p/name/'hydra'),'++callbacks.im_eval.stage_config':str(cp),'++callbacks.im_eval._target_':target};cmd=[a.split('=',1)[0]+'='+rep[a.split('=',1)[0]] if a.split('=',1)[0] in rep else a for a in cmd];run(name,cmd)
def evaluate(label,checkpoint):
 rows=[]
 for task in task_ids:
  name=label+'-'+task;native(name,task,checkpoint,'gear_sonic.research.scene_distillation.navigation_motor_runtime.NavigationMotorCallback');rows.append(dict(stage=name,**json.loads((p/name/'task/task-result.json').read_text())))
 (p/(label+'-results.json')).write_text(json.dumps(rows,indent=2));print(label,'SUCCESS',sum(r['navigation_success'] for r in rows),'/',len(rows),flush=True)
def trace_binding(receipt):
 f=Path(receipt['path']).parent/'trace.npz';return dict(path=str(f),sha256=sha(f))
manifest=json.loads((p/'motor-manifest.json').read_text())
for parent in manifest['parents']:parent['trace']=trace_binding(parent)
mp=p/'motor-bound-manifest.json';mp.write_text(json.dumps(manifest,indent=2));c=json.loads((previous/'localized-config.json').read_text());tm=teacherpacket/'stopping-tasks-v2/manifest.json';c.update(dataset_view='motor_recovery',dataset_manifest=str(mp),dataset_manifest_sha256=sha(mp),task_manifest=str(tm),task_manifest_sha256=sha(tm),purpose='motor-state demonstrations then supported learner-prefix recovery; fixed motor',updates=6000)
task_ids=[e['task_id'] for e in json.loads(tm.read_text())['tasks']]
# Sixteen smoke updates exercise the actual runtime decoder target contract first.
fit('smoke',{**c,'updates':16})
baseck=fit('motor-data',c);evaluate('motor-data',baseck)
recovery_parents=[]
for task in task_ids:
 oldscore=json.loads((teacherpacket/('stop-v2-'+task)/'task/task-result.json').read_text());switch=oldscore['control_steps']//2;name='recovery-'+task;native(name,task,baseck,'gear_sonic.research.scene_distillation.navigation_recovery.MotorRecoveryCollectionCallback',takeover_tick=switch,motor_checkpoint=c['motor_checkpoint'],motor_sha256=c['motor_sha256']);f=p/name/'task/recovery.json';entry=dict(path=str(f),sha256=sha(f));entry['trace']=trace_binding(entry);recovery_parents.append(entry);r=json.loads(f.read_text());print('RECOVERY',task,r['supported'],r['supported_rows'],flush=True)
combined={**manifest,'parents':manifest['parents']+recovery_parents};cp=p/'combined-manifest.json';cp.write_text(json.dumps(combined,indent=2));qualified=sum(json.loads(Path(e['path']).read_text())['supported'] for e in recovery_parents);(p/'recovery-summary.json').write_text(json.dumps(dict(attempts=len(recovery_parents),qualified=qualified,learner_queries=qualified,parents=recovery_parents),indent=2))
fork={**c,'initial_navigation_checkpoint':str(baseck),'initial_navigation_sha256':sha(baseck),'updates':3000}
control=fit('replay-control',fork);evaluate('replay-control',control)
if qualified:
 dagger=fit('dagger',{**fork,'dataset_manifest':str(cp),'dataset_manifest_sha256':sha(cp),'recovery_fraction':0.5});evaluate('dagger',dagger)
else:
 (p/'dagger-blocked.json').write_text(json.dumps(dict(reason='No task-compatible motor recovery was executed successfully; no unsupported fit'),indent=2))
print('COMPLETE',flush=True)
