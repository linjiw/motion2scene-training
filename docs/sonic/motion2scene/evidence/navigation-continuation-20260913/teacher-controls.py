import json,os,subprocess,time
from pathlib import Path
from gear_sonic.research.hindsight_training.runtime import sha
p=Path(__file__).resolve().parent;repo=Path.cwd();plan=json.loads((p/'plan.json').read_text());(p/'teacher-control-plan.json').write_text(json.dumps(dict(cases=plan['cases'],scope='same navigation prefix then original teacher, diagnostic only; does not affect provider or checkpoint selection'),indent=2));rows=[]
for case in plan['cases']:
 source=f"probe-{case['seed']}-{case['task']}-nominal";name=f"teacher-{case['seed']}-{case['task']}";out=p/name;out.mkdir();c=json.loads((p/(source+'-config.json')).read_text());c.pop('continuation');c.update(output=str(out/'task'));cp=p/(name+'-config.json');cp.write_text(json.dumps(c,indent=2));cmd=json.loads((p/source/'command.json').read_text());rep={'++eval_output_dir':str(out/'unused'),'++eval_base_dir':str(out/'hydra'),'++callbacks.im_eval.stage_config':str(cp),'++callbacks.im_eval._target_':'gear_sonic.research.scene_distillation.navigation_takeover.OriginalTeacherContinuationCallback'};cmd=[a.split('=',1)[0]+'='+rep[a.split('=',1)[0]] if a.split('=',1)[0] in rep else a for a in cmd];(out/'command.json').write_text(json.dumps(cmd,indent=2));(out/'source-sha256.json').write_text(json.dumps({str(f):sha(f) for f in (repo/'gear_sonic/research/scene_distillation').glob('*.py')},indent=2));t=time.monotonic()
 with (out/'process.log').open('w') as f:r=subprocess.run(cmd,env=dict(os.environ,PYTHONPATH=str(repo),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT,timeout=360)
 if r.returncode:raise RuntimeError(name)
 receipt=json.loads((out/'task/teacher-continuation.json').read_text());rows.append(dict(stage=name,wall_seconds=time.monotonic()-t,**receipt));(p/'teacher-results.json').write_text(json.dumps(rows,indent=2));print(name,receipt['outcomes']['timely_suffix_supported'],receipt['outcomes']['local_score']['max_hold_ticks'],flush=True)
print('COMPLETE',flush=True)
