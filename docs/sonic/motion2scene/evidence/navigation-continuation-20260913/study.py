import json,os,subprocess,sys,time
from pathlib import Path
from gear_sonic.research.hindsight_training.runtime import sha
p=Path(__file__).resolve().parent;repo=Path.cwd();prior=p.parent/'m2s-nav-supported-recovery-20260913';plan=json.loads((p/'plan.json').read_text());checkpoint=Path(plan['behavior_checkpoint']['path']);py=str(repo/'.venv_isaaclab/bin/python');base=json.loads((prior/'dagger-config.json').read_text());tasks=[r['stage'].removeprefix('dagger-') for r in json.loads((prior/'dagger-results.json').read_text())]
def bind(f):return dict(path=str(f),sha256=sha(f))
def run(name,cmd):
 out=p/name;out.mkdir();(out/'command.json').write_text(json.dumps(cmd,indent=2));(out/'source-sha256.json').write_text(json.dumps({str(f):sha(f) for f in (repo/'gear_sonic/research/scene_distillation').glob('*.py')},indent=2));t=time.monotonic()
 with (out/'process.log').open('w') as f:r=subprocess.run(cmd,env=dict(os.environ,PYTHONPATH=str(repo),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT,timeout=600)
 (out/'process-result.json').write_text(json.dumps(dict(exit_code=r.returncode,wall_seconds=time.monotonic()-t),indent=2))
 if r.returncode:raise RuntimeError(name)
def native(name,task,ck,seed,extra=None):
 source='dagger-'+task;c=json.loads((prior/(source+'-config.json')).read_text());c.update(student_checkpoint=str(ck),student_sha256=sha(ck),output=str(p/name/'task'),**(extra or {}));cp=p/(name+'-config.json');cp.write_text(json.dumps(c,indent=2));cmd=json.loads((prior/source/'command.json').read_text());rep={'++seed':str(seed),'++eval_output_dir':str(p/name/'unused'),'++eval_base_dir':str(p/name/'hydra'),'++callbacks.im_eval.stage_config':str(cp),'++callbacks.im_eval._target_':'gear_sonic.research.scene_distillation.navigation_recovery.MotorRecoveryCollectionCallback' if extra else 'gear_sonic.research.scene_distillation.navigation_motor_runtime.NavigationMotorCallback'};cmd=[a.split('=',1)[0]+'='+rep[a.split('=',1)[0]] if a.split('=',1)[0] in rep else a for a in cmd];run(name,cmd);score=json.loads((p/name/'task/task-result.json').read_text());print(name,score['navigation_success'],score['max_hold_ticks'],flush=True);return score
if sys.argv[1]=='collect':
 probes=json.loads((p/'probe-results.json').read_text());assert len(probes)==plan['max_probes'];counts={k['kind']:sum(r['supported'] for r in probes if r['provider']['kind']==k['kind']) for k in plan['providers']};provider=max(plan['providers'],key=lambda c:counts[c['kind']]);(p/'provider-selection.json').write_text(json.dumps(dict(counts=counts,selected=provider),indent=2));new=[]
 for row in probes:
  if row['provider']['kind']==provider['kind']:
   f=p/row['stage']/'task/recovery.json';new.append({**bind(f),'trace':bind(f.parent/'trace.npz')})
 for task in tasks:
  old=json.loads((prior/('recovery-'+task+'-config.json')).read_text());name='collect-'+task;native(name,task,checkpoint,91261,dict(takeover_tick=old['takeover_tick'],motor_checkpoint=base['motor_checkpoint'],motor_sha256=base['motor_sha256'],continuation=provider));f=p/name/'task/recovery.json';new.append({**bind(f),'trace':bind(f.parent/'trace.npz')})
 old=json.loads((prior/'combined-manifest.json').read_text());mp=p/'combined-manifest.json';mp.write_text(json.dumps({**old,'parents':old['parents']+new},indent=2));qualified=[json.loads(Path(e['path']).read_text()) for e in new];(p/'collection-summary.json').write_text(json.dumps(dict(selected_provider=provider['kind'],new_attempts=len(new),new_supported=sum(r['supported'] for r in qualified),new_supported_rows=sum(r['supported_rows'] for r in qualified),new_learner_queries=sum(r['learner_query_rows'] for r in qualified),new_parents=new),indent=2));print('COLLECTION COMPLETE',flush=True)
elif sys.argv[1]=='fit':
 summary=json.loads((p/'collection-summary.json').read_text());assert summary['new_supported']>0,'No supported fresh recovery; do not run falsely labelled aggregation';c={**base,'seed':91470,'initial_navigation_checkpoint':str(checkpoint),'initial_navigation_sha256':sha(checkpoint),'fresh_recovery_behavior_sha256':sha(checkpoint),'recovery_fraction':0,'updates':3000,'purpose':'equal-update old replay versus qualified current-policy recovery under broader collection seed'};results={}
 for name in ['replay','recovery']:
  spec=c.copy()
  if name=='recovery':spec.update(dataset_manifest=str(p/'combined-manifest.json'),dataset_manifest_sha256=sha(p/'combined-manifest.json'),recovery_fraction=.5)
  cp=p/(name+'-config.json');cp.write_text(json.dumps(spec,indent=2));run(name+'-fit',[py,'-m','gear_sonic.research.scene_distillation.navigation_motor','--config',str(cp),'--output',str(p/(name+'-fit')/'training')]);ck=p/(name+'-fit')/'training/step-003000.pt';rows=[]
  for task in tasks:rows.append(dict(task=task,**native(name+'-'+task,task,ck,91260)))
  results[name]=rows;(p/(name+'-results.json')).write_text(json.dumps(rows,indent=2));print(name,'SUCCESSES',sum(r['navigation_success'] for r in rows),flush=True)
 selected=max(results,key=lambda k:sum(r['navigation_success'] for r in results[k]));ck=p/(selected+'-fit')/'training/step-003000.pt';(p/'checkpoint-selection.json').write_text(json.dumps(dict(selected=selected,checkpoint=bind(ck),seed=91262),indent=2));rows=[]
 for task in tasks:rows.append(dict(task=task,**native('confirm-'+task,task,ck,91262)))
 (p/'confirmation-results.json').write_text(json.dumps(dict(condition=selected,seed=91262,evaluations=rows,successes=sum(r['navigation_success'] for r in rows)),indent=2));print('SELECTED CONFIRMATION COMPLETE',flush=True)
 rows=[]
 for task in tasks:rows.append(dict(task=task,**native('parent-confirm-'+task,task,checkpoint,91262)))
 (p/'parent-confirmation-results.json').write_text(json.dumps(dict(condition='parent',seed=91262,evaluations=rows,successes=sum(r['navigation_success'] for r in rows)),indent=2));print('STUDY COMPLETE',flush=True)
else:raise ValueError('collect or fit required')
