"""Read completed M5 bindings with the existing finite-bank response statistic."""

import datetime
import json
from pathlib import Path
import sys

ROOT = Path('/home/linjiw/groot-wbc-sonic-sim-trackb')
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/research')]
import motion2scene_response_diversity as diversity
from motion2scene_build_timed_replay import read_bound
from motion2scene_timing_diagnostic import artifact, write_new

D = Path('/home/linjiw/research-data/groot-wbc')
runroot = D / 'm2s-expanded-acquisition-20260909-v1'
out = Path(__file__).parent / 'all_M5_bound_response_tables.json'
assert not out.exists()
m4_ref = artifact(D / 'm2s-five-arm-M4-response-diversity-20260910-v1/result.json')
assert m4_ref['sha256'] == 'sha256:2a25c83e64a0bd34e3d4615a85aac90f52bf9acc4b1cfc0c08a4f09bb97c646e'
m4 = read_bound(m4_ref)
completions = sorted(runroot.glob('*/controller/complete_005.json'))
assert len(completions) == 15
reports = []
for path in completions:
    completion_ref = artifact(path)
    c = read_bound(completion_ref)
    training = read_bound(c['training_result'])
    registration = read_bound(training['registration'])
    groups = read_bound(training['teachers'])
    assert c['completed_through_round'] == 5 and c['reserved_evaluation_started'] is False
    assert len(groups) == 6 and registration['collections'] == [g['collection'] for g in groups]
    assert registration['l2'] == 10.0
    prior = next(r for r in m4['corpora'] if r['run_id'] == c['run_id'])
    assert [g['collection'] for g in groups[:-1]] == [s['teacher'] for s in prior['source_results']]
    tasks = []
    for i, g in enumerate(groups[1:], 1):
        collection = read_bound(g['collection'])
        table = {r['forced_option_id']: r['outcome']['task_outcome'] for r in collection['rows']}
        assert len(table) == len(collection['rows']) == 7
        if i <= 4:
            assert table == prior['tasks'][i-1]['outcomes']
            assert g['scene_id'] == prior['tasks'][i-1]['candidate_id']
        tasks.append(dict(round=i, scene_id=g['scene_id'], collection=g['collection'], outcomes=table))
    response = diversity.response_summary([t['outcomes'] for t in tasks], groups[0]['option_ids'])
    reports.append(dict(run_id=c['run_id'], arm=prior['arm'], seed=prior['seed'], completion=completion_ref,
        training_result=c['training_result'], teachers=training['teachers'], weighting=registration['weighting'], learner_l2=registration['l2'],
        M4_raw_audit_tables_unchanged=True, tasks=tasks, responses=response,
        recorded_captures=c['accounting']['distinct_original_recorded_captures'],
        actual_recorded_physics_steps=c['accounting']['actual_recorded_physics_steps']))
target = next(r for r in reports if r['run_id']=='seed93203_target_only')
assert target['responses']['bank_solvable_lower'] == 5
assert target['responses']['best_fixed_passages_lower'] == 4
assert target['responses']['minimum_cover_size'] == 2
witness = [target['tasks'][i-1] for i in (3,5)]
passing = [{s for s,v in w['outcomes'].items() if v=='pass'} for w in witness]
assert all(passing) and not set.intersection(*passing)
target_raw_ref=artifact(Path(__file__).parent/'seed93203_target_M5_selection_verified.json')
target_raw=read_bound(target_raw_ref)
assert target_raw['teacher_collection']==witness[1]['collection'] and target_raw['new_teacher_targets_raw_reaudited'] is True
assert target_raw['teacher_outcomes']=={
    row['forced_option_id']:dict(outcome=row['outcome']['task_outcome'],passage_time_s=row['costs']['passage_time_s'],steps=row['physics_steps'])
    for row in read_bound(witness[1]['collection'])['rows']
}
assert all(r['responses']['one_fixed_covers_every_solvable_task'] is True for r in reports if r['run_id']!='seed93203_target_only')
report=dict(schema='motion2scene_completed_checkpoint_bound_tables_v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),budget=5,
    scope='Interim acquisition mechanism readout from completed bound tables, not a new all-capture raw audit or policy evaluation.',
    implementation=[artifact(Path(__file__)),artifact(Path(diversity.__file__))],prior_M4_raw_audit=m4_ref,target_M5_raw_audit=target_raw_ref,
    corpora=reports,total_recorded_captures=sum(r['recorded_captures'] for r in reports),total_recorded_physics_steps=sum(r['actual_recorded_physics_steps'] for r in reports),
    total_postbootstrap_encounter_assignments=sum(r['responses']['assigned_tasks'] for r in reports),
    target_only_complementarity_witness=witness,corpora_without_one_passing_fixed_schedule=['seed93203_target_only'],
    new_physics_from_readout=0,new_fits_from_readout=0,
    interpretation='Target-only seed 93203 has disjoint measured passing sets at rounds 3 and 5. Both source tables have independent raw-record verification. This establishes finite-bank response complementarity in one acquired corpus, not sensor realizability or downstream policy advantage. Executed contrast and replay still each admit one fixed passing schedule per corpus. All assigned unsolvable tasks remain included. Teacher unknowns are explicit; the previously reconciled reference student unknown remains in acquisition history and measured cost. No queue, learner, baseline or reserved assignment is changed; the M8 comparison remains pending.')
assert report['total_recorded_captures']==705 and report['total_recorded_physics_steps']==839184
write_new(out,report)
print(json.dumps(dict(output=artifact(out),recorded_captures=report['total_recorded_captures'],physics_steps=report['total_recorded_physics_steps'],target_response=target['responses'])))
