import json
from pathlib import Path
import numpy as np
from gear_sonic.research.hindsight_training.runtime import sha
p=Path(__file__).resolve().parent;prior=p.parent/'m2s-nav-supported-recovery-20260913';rows=[]
for f in sorted(p.glob('*/task/recovery.json')):
 r=json.loads(f.read_text());name=f.parent.parent.name;c=json.loads((p/(name+'-config.json')).read_text());cmd=json.loads((p/name/'command.json').read_text());seed=int(next(a.split('=',1)[1] for a in cmd if a.startswith('++seed=')));task=json.loads(Path(r['task']['path']).read_text());task_id=task['task_id'];source=('dagger-' if seed==91260 else 'confirm-')+task_id
 with np.load(f.parent/'trace.npz') as z,np.load(prior/source/'task/trace.npz') as original,np.load(f.parent/'motor-recovery.npz') as q:
  n=min(r['takeover_tick'],len(z['speed']));parity=all(np.array_equal(z[k][:n],original[k][:n]) for k in ['root_xyz','speed','undesired_force']);assert parity,name
  k=r['takeover_tick'];d=np.linalg.norm(q['measured_root_xyz'][k:,:2]-np.asarray(task['goal_xyz'])[:2],axis=1);delta=np.linalg.norm(q['controls'][k:]-q['nominal_controls'][k:],axis=1)
  rows.append(dict(stage=name,exact_unassisted_prefix=parity,seed=seed,provider=r.get('continuation'),switch_tick=k,entry_distance_xy_m=float(d[0]) if len(d) else None,min_distance_xy_m=float(d.min()) if len(d) else None,changed_command_rows=int((delta>1e-7).sum()),max_command_change_l2=float(delta.max()) if len(delta) else 0,supported=r['supported'],supported_rows=r['supported_rows'],learner_queries=r['learner_query_rows'],rows=r['rows']))
(p/'prefix-command-audit.json').write_text(json.dumps(rows,indent=2));print('AUDITED',len(rows),'ALL PREFIXES EXACT')
controls=[]
for f in sorted(p.glob('teacher-*/task/teacher-continuation.json')):
 r=json.loads(f.read_text());name=f.parent.parent.name;c=json.loads((p/(name+'-config.json')).read_text());task=json.loads(Path(c['task_path']).read_text());seed=int(name.split('-')[1]);source=('dagger-' if seed==91260 else 'confirm-')+task['task_id'];k=r['takeover_tick']
 with np.load(f.parent/'trace.npz') as z,np.load(prior/source/'task/trace.npz') as original:
  n=min(k,len(z['speed']));parity=all(np.array_equal(z[key][:n],original[key][:n]) for key in ['root_xyz','speed','undesired_force']);assert parity,name
 controls.append(dict(stage=name,exact_unassisted_prefix=parity,**r))
(p/'teacher-prefix-audit.json').write_text(json.dumps(controls,indent=2));print('TEACHER CONTROLS',len(controls),'PREFIXES EXACT')
