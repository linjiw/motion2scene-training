import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;out=Path('docs/motion2scene/evidence/navigation-continuation-20260913');plan=json.loads((p/'plan.json').read_text());probes=json.loads((p/'probe-results.json').read_text());colors=['#276FBF','#238A56','#A54395','#777777'];labels=['Nominal motor','Velocity feedback','Velocity + keypoints','Original teacher']
fig,axes=plt.subplots(4,2,figsize=(11,11),layout='constrained')
for i,case in enumerate(plan['cases']):
 task=case['task'];rows=[r for r in probes if r['case']==case];stages=[next(r['stage'] for r in rows if r['provider']['kind']==kind) for kind in ['nominal','goal_velocity','goal_velocity_keypoints']]+[f"teacher-{case['seed']}-{task}"]
 for j,stage in enumerate(stages):
  c=json.loads((p/(stage+'-config.json')).read_text());t=json.loads(Path(c['task_path']).read_text());z=np.load(p/stage/'task/trace.npz');k=case['switch'];time=(np.arange(k,len(z['speed']))-k+1)*.02;distance=np.linalg.norm(z['root_xyz'][k:]-t['goal_xyz'],axis=1)
  for ax,y in zip(axes[i],[distance,z['speed'][k:]]):ax.plot(time,y,color=colors[j],label=labels[j],alpha=.85,lw=1.7,linestyle='--' if j==3 else '-')
 axes[i,0].axhline(.25,color='black',ls=':',lw=1);axes[i,1].axhline(.10,color='black',ls=':',lw=1)
 axes[i,0].set_ylabel(f"{task}\nseed {case['seed']}, entry {case['switch']}\nGoal distance (m)",fontsize=9);axes[i,1].set_ylabel('Root speed (m/s)')
 for ax in axes[i]:ax.set_xlim(0,(t['deadline_ticks']-k)*.02);ax.grid(alpha=.15);ax.set_xlabel('Seconds after takeover (original deadline at right)')
axes[0,0].set_title('3D distance to goal; dotted line = 0.25 m')
axes[0,1].set_title('3D speed; dotted line = 0.10 m/s')
fig.suptitle('Executed continuations from identical navigation arrival states\nSuccess requires both limits for 50 consecutive ticks',fontsize=14)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
fig.savefig(out/'continuation-traces.png',dpi=150);fig.savefig(out/'continuation-traces.pdf');print(out/'continuation-traces.png')
