"""Derive exploratory diagnostics without changing experimental receipts."""
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
DATA = Path('/home/linjiw/research-data/groot-wbc')

inputs = {}

def read(p):
    path = Path(p)
    raw = path.read_bytes()
    inputs[str(path)] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
    return json.loads(raw)

def write(name, d):
    (OUT/name).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')

plan=read(DATA/'m2s-support-pilot-plan-20260910-v1/plan.json')
pool=read(DATA/'m2s-support-preserving-pools-20260910-v1/result.json')
write('proposal-cost.json',dict(actual_shared_clearance_queries=pool['actual_shared_clearance_queries'],
    fresh_proposals=pool['total_fresh_proposals'], physics_steps=pool['physical_steps'],
    per_seed=[{k:p[k] for k in ['seed','proposed_candidates','screened_candidates','excluded_before_geometry',
        'eligible_counts','support','shared_clearance_search_seconds','all_arm_queue_selection_seconds']} for p in pool['pools']],
    accounting='Shared search cost measured once; no invented per-arm allocation. Mixture eligible_counts=0 is an implementation placeholder, not zero proposal support.'))
write('selected-proposals.json',read(DATA/'m2s-support-pilot-plan-20260910-v1/selection.json'))
training=[]
for run in plan['runs']:
    encounters=[]
    for slot in run['rounds']:
        if slot['index']==0:continue
        path=Path(slot['inherited_teacher']) if slot['shared_prefix'] else Path(slot['teacher_directory'])/'result.json'
        result=read(path)
        passing=[r['cell_id'].removeprefix('forced_') for r in result['rows'] if r['outcome']['task_outcome']=='pass']
        encounters.append(dict(index=slot['index'],shared_prefix=slot['shared_prefix'],candidate_id=slot['candidate_id'],
            proposal_channel=slot['proposal_channel'], passing_schedules=passing,scored_branches=len(result['rows']),
            complete_measurement_branches=sum(r['measurement_admitted'] for r in result['rows']),
            usable_continuation_target_count=None))
    sets=[set(e['passing_schedules']) for e in encounters if e['passing_schedules']]
    options=sorted(set.union(*sets)) if sets else []
    covering=set.intersection(*sets) if sets else set()
    cover=next((k for k in range(1,8) if any(all(set(c)&s for s in sets) for c in itertools.combinations(options,k))),None) if sets else None
    fit=read(Path(run['rounds'][-1]['model_directory'])/'result.json')['fit']
    training.append(dict(run_id=run['run_id'],encounters=encounters,solvable_encounters=len(sets),
        complete_consequential_decisions=fit['complete_consequential_decisions'],
        excluded_decision_indices=fit['excluded_decision_indices'],
        phase_l2=[p['l2'] for p in fit['phase_fits']],
        covering_schedules=sorted(covering),minimum_cover_size_of_solvable_encounters=cover,
        added_solvable_encounters=sum(bool(e['passing_schedules']) for e in encounters if not e['shared_prefix']),
        target_availability_note='Passing branches do not certify causal target usability; target counts left NA pending teacher-dataset audit.'))
write('training-diagnostics.json',training)
rows=list(csv.DictReader((OUT/'assignment-outcomes.csv').open()))
diag=read(OUT/'physical-diagnostics.json')
byid={r['assignment_id']:r for r in diag}
for row in rows:
    d=byid.get(row['assignment_id'])
    row['partial_capture']=None if d is None else d['outcome']['measurement_status']=='partial'
    row['finite_horizon_before_completion']=None if d is None else any(e['kind']=='finite_captured_reference_horizon_before_completion' for e in d['outcome']['physical_events'])
with (OUT/'censoring-outcomes.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:'NA' if v is None else v for k,v in r.items()} for r in rows)
write('censoring-summary.json',{stage:dict(partial_captures_with_verified_failure=sum(r['partial_capture'] is True for r in rows if r['stage']==stage),
    horizon_events=sum(r['finite_horizon_before_completion'] is True for r in rows if r['stage']==stage),
    unresolved_technical_interruptions=sum(r['state']=='interrupted_missing_receipt' for r in rows if r['stage']==stage),
    outcome_unknown=sum(r['outcome']=='unknown' for r in rows if r['stage']==stage),
    note='Event flags overlap physical failures; do not add them as disjoint counts. No physical failure is excluded as censored.') for stage in ['acquisition','evaluation']})

proposal_rows=[]
for entry in pool['pools']:
    broad=read(entry['arms']['support_broad']['path'])
    strict=read(entry['arms']['support_strict']['path'])
    strict_ids={r['candidate_id'] for r in strict['rows']}
    for r in broad['rows']:
        proposal_rows.append(dict(seed=entry['seed'],candidate_id=r['candidate_id'],
            inside_positive_screen=r['inside_positive_screen'],inside_negative_screen=r['inside_negative_screen'],
            strict_admitted=r['candidate_id'] in strict_ids,strict_rejected=r['candidate_id'] not in strict_ids,
            broad_admitted=True))
assert len(proposal_rows)==3840
with (OUT/'proposal-eligibility.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(proposal_rows[0]));w.writeheader();w.writerows(proposal_rows)

write('diagnostic-inputs.json', list(inputs.values()))
