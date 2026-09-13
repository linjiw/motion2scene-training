import json
from pathlib import Path
import torch,yaml,numpy as np
from gear_sonic.research.hindsight_training.runtime import sha
from gear_sonic.research.scene_distillation.motor_runtime import load_motor
p=Path(__file__).resolve().parent;c=json.loads((p/'motor-data-config.json').read_text());cp=p.parent/'m2s-bfm-repaired-navigation-prep-20260912/teacher/config.yaml';cfg=yaml.safe_load(cp.read_text());policy=cfg['manager_env']['observations']['policy'];encoder=cfg['algo']['config']['actor']['backbone']['encoders']['g1']['inputs'];torch.set_num_threads(2);m,d,_=load_motor(c['motor_checkpoint'],c['teacher_sha256'],'cpu');rows=[]
with torch.no_grad():
 for task in ['00908-stop-corridor','00976-stop-clear','00265-stop-clear']:
  f=p/('recovery-'+task)/'task/motor-recovery.npz';rec=json.loads((f.parent/'recovery.json').read_text());z=np.load(f);i=rec['takeover_tick'];h=torch.from_numpy(z['proprio'][i:i+1].copy());controls=torch.from_numpy(z['controls'][i:i+1].copy());mask=torch.from_numpy(z['control_mask'][i:i+1].copy());original=d(m.prior_step(h,controls,mask)['tokens'],h);perturb={}
  for axis in [0,1]:
   for delta in [-.2,.2]:
    changed=controls.clone();changed[:,8:50].reshape(1,14,3)[:,:,axis]+=delta
    action=d(m.prior_step(h,changed,mask)['tokens'],h);perturb[f'keypoint_translation_axis{axis}_{delta:+.1f}m']=float((action-original).square().mean().sqrt())
  for axis in [0,1]:
   for delta in [-.2,.2]:
    changed=controls.clone();changed[:,2+axis]+=delta
    action=d(m.prior_step(h,changed,mask)['tokens'],h);perturb[f'desired_velocity_axis{axis}_{delta:+.1f}mps']=float((action-original).square().mean().sqrt())
  rows.append(dict(task=task,decision_tick=i,action_change_rms=perturb,motor_teacher_action_rms=float(((z['motor_actions'][i]-z['teacher_actions'][i])**2).mean()**.5)))
r=dict(config_binding=dict(path=str(cp),sha256=sha(cp)),actor_history_terms=[k for k in policy if not k.startswith('_') and k not in ['enable_corruption','concatenate_terms']],g1_encoder_inputs=encoder,actor_has_direct_global_position=False,actor_has_direct_linear_velocity=False,scope='static input-sensitivity diagnostic only; modified commands were not executed or admitted as expert labels',counterfactual='Horizontal rigid translation of the robot leaves original teacher actor observations and reference joint/orientation inputs unchanged; the added motor forecaster can still respond through relative keypoints.',motor_sensitivity=rows)
(p/'interface-audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
