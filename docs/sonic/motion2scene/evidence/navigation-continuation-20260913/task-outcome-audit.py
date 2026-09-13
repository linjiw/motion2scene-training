import json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent;rows=[]
for f in sorted(p.glob('*/task/task-result.json')):
 name=f.parent.parent.name
 if name.startswith(('probe-','collect-','teacher-')):continue
 c=json.loads((p/(name+'-config.json')).read_text());task=json.loads(Path(c['task_path']).read_text());score=json.loads(f.read_text());z=np.load(f.parent/'trace.npz');distance=np.linalg.norm(z['root_xyz']-task['goal_xyz'],axis=1);good=(distance<=task['goal_tolerance_m'])&(z['speed']<=task['terminal_speed_mps']);run=best=0;reset=None;longest_reset=None;first_hold=None
 for i,valid in enumerate(good):
  if valid:
   run+=1;best=max(best,run)
   if run>=task['hold_ticks'] and first_hold is None:first_hold=i
  else:
   if run:reset=dict(tick=i,ended_run_ticks=run,position_violation=bool(distance[i]>task['goal_tolerance_m']),speed_violation=bool(z['speed'][i]>task['terminal_speed_mps']),distance_m=float(distance[i]),speed_mps=float(z['speed'][i]))
   if run and (longest_reset is None or run>longest_reset["ended_run_ticks"]):longest_reset=reset.copy()
   run=0
 assert best==score['max_hold_ticks']
 rows.append(dict(stage=name,task=task['task_id'],success=score['navigation_success'],goal_ever_reached=score['goal_ever_reached'],max_hold_ticks=best,final_hold_ticks=run,first_hold_completion_tick=first_hold,last_hold_reset=reset,longest_hold_reset=longest_reset if longest_reset and longest_reset["ended_run_ticks"]==best else None,deadline_censored_hold=bool(run and len(good)==task['deadline_ticks']),control_steps=len(good),final_distance_m=float(distance[-1]),collision_free=score['collision_free'],fell=score['fell']))
(p/'task-outcome-audit.json').write_text(json.dumps(rows,indent=2));print('AUDITED UNASSISTED TASKS',len(rows))
