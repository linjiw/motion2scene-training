#!/usr/bin/env bash
# Exact command executed from the repository root; refuses to overwrite its report.
# To reproduce, change only the output report path to a fresh path before running.
set -euo pipefail
.venv_isaaclab/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,datetime,sys
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'scripts/research')]
from motion2scene_response_diversity import response_summary
seed=93203
D=Path('/home/linjiw/research-data/groot-wbc');o=D/'m2s-expanded-and-capability-stages-20260910-v8'
def ref(p):return {'path':str(p.absolute()),'sha256':'sha256:'+hashlib.sha256(p.read_bytes()).hexdigest()}
def bound(r):
 p=Path(r['path']);assert ref(p)==r;return json.loads(p.read_text())
plan_ref=ref(D/'m2s-expanded-acquisition-plan-20260909-v1/plan.json');assert plan_ref['sha256']=='sha256:bf551021c82fd42d0e6b002a2941a71b1358d72a469903105a2230bdfae3a4ed';plan=bound(plan_ref)
runs=[next(r for r in plan['runs'] if r['run_id']==f'seed{seed}_{arm}') for arm in ['target_only','uniform']]
reports=[]
for run in runs:
 slot=run['rounds'][7];root=Path(slot['model_directory']).parents[1];cp=root/'controller/complete_007.json';c=bound(ref(cp));training=bound(c['training_result']);reg=bound(training['registration']);groups=bound(training['teachers']);assert c['completed_through_round']==7 and c['reserved_evaluation_started'] is False
 pre_path=Path(slot['student_directory']).parent/'preupdate.json';pre=bound(ref(pre_path));oldtrain=bound(pre['training_result']);oldgroups=bound(oldtrain['teachers']);assert groups[:-1]==oldgroups;assert reg['collections']==[g['collection'] for g in groups];assert pre['earlier_collections']==[g['collection'] for g in oldgroups]
 assert reg['l2']==10.0 and reg['weighting']=='uniform_per_phase'
 release=bound(ref(Path(slot['student_directory']).parent/'student_complete.json'));assert release['preupdate']==ref(pre_path);sr=bound(release['student']);sm=bound(sr['manifest']);assert sm['policy']==pre['model']==oldtrain['policy'];assert ref(Path(sm['policy']['path']))==sm['policy']
 tr=bound(groups[-1]['collection']);tm=bound(tr['manifest']);assert sm['scene_definition']==tm['scene_definition']==slot['scene'];assert [x['forced_option_id'] for x in tm['cells']]==slot['branch_order'];assert [x['forced_option_id'] for x in tr['rows']]==slot['branch_order'];assert len(sr['rows'])==1
 row=sr['rows'][0];interface=bound(row['sensor']);assert interface['timed_policy_sha256']==sm['policy']['sha256'];assert row['mode']=='learned' and row['task_outcome_admitted'] and row['measurement_admitted']
 tables=[{r['forced_option_id']:r['outcome']['task_outcome'] for r in bound(group['collection'])['rows']} for group in groups[1:]]
 assert ref(Path(c['model']['path']))==c['model']==training['policy']
 reports.append(dict(run_id=run['run_id'],candidate_id=slot['candidate_id'],scene=slot['scene'],completion=ref(cp),preupdate=ref(pre_path),student_collection=release['student'],teacher_collection=groups[-1]['collection'],generating_model=sm['policy'],new_model=c['model'],prior_teacher_prefix_preserved=True,learner_l2=reg['l2'],weighting=reg['weighting'],student_task_outcome=row['outcome']['task_outcome'],student_passage_time_s=row['costs']['passage_time_s'],student_selected_schedule=next((s['to'] for s in interface['switches'] if s['from']=='neutral'),'neutral'),student_physics_steps=row['physics_steps'],teacher_outcomes={r['forced_option_id']:dict(outcome=r['outcome']['task_outcome'],passage_time_s=r['costs']['passage_time_s'],steps=r['physics_steps']) for r in tr['rows']},new_encounter_physics_steps=row['physics_steps']+sum(r['physics_steps'] for r in tr['rows']),cumulative_physics_steps=c['accounting']['actual_recorded_physics_steps'],prefix_response_summary=response_summary(tables,groups[0]['option_ids'])))
report=dict(schema='motion2scene_completed_acquired_corpora_readout_v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=plan_ref,budget=7,physics_seed=seed,corpora=reports,new_physics_from_readout=0,new_fits_from_readout=0,verification_scope='Hash-bound completed collection/training receipts, actual policy/scene/branch identities and unchanged previous teacher prefixes. Uses original collection scoring; no second raw-record audit.',interpretation='Target-only and uniform acquired different scenes. These actual students belong to generating M6 policies; they do not evaluate the new M7 fits or provide a common-set constructor comparison. Passage time is crossing plus stabilization, not full adaptation/recovery completion. A passing late-schedule choice does not isolate WAIT information benefit from selecting the same complete schedule initially. Historical physical outcomes stay attached to generating policies. Finite-bank coverage does not establish perceptual realizability or held-out performance.')
out=o/f'seed{seed}_target_uniform_M7_bound_readout.json';assert not out.exists();out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(output=ref(out),corpora=[{k:r[k] for k in ['run_id','candidate_id','student_task_outcome','student_passage_time_s','student_selected_schedule','new_encounter_physics_steps','cumulative_physics_steps','prefix_response_summary','teacher_outcomes']} for r in reports])))
PY

