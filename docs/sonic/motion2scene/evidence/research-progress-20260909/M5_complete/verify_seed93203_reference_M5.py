"""Verify the final M5 reference encounter using original raw auditors."""

import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path('/home/linjiw/groot-wbc-sonic-sim-trackb')
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/research')]
import motion2scene_build_timed_replay as replay
from motion2scene_expanded_acquisition import priority_records
from motion2scene_timing_diagnostic import artifact, checked, write_new
from motion2scene_train_timed_schedules import audit_collection
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import load_verified_registry
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_curriculum import DEFAULT_RULE
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import load_schedule_policy, choose_schedule_option

D = Path('/home/linjiw/research-data/groot-wbc')
run = D / 'm2s-expanded-acquisition-20260909-v1/seed93203_reference_contrast'
out = Path(__file__).parent / 'seed93203_reference_M5_verified.json'
assert not out.exists()
completion = json.loads((run / 'controller/complete_005.json').read_text())
training = replay.read_bound(completion['training_result'])
registration = replay.read_bound(training['registration'])
stored = replay.read_bound(training['teachers'])
registry_ref = registration['registry']
bank = load_verified_registry(Path(registry_ref['path']), registry_ref['sha256'])
assert registration['l2'] == 10.0 and registration['weighting'] == 'uniform_per_phase'
preupdate = json.loads((run / 'round_005/preupdate.json').read_text())
old_training = replay.read_bound(preupdate['training_result'])
old_groups = replay.read_bound(old_training['teachers'])
assert stored[:-1] == old_groups
assert registration['collections'] == [g['collection'] for g in stored]
assert preupdate['earlier_collections'] == [g['collection'] for g in old_groups]
teacher_path = checked(Path(stored[-1]['collection']['path']), stored[-1]['collection']['sha256'])
group = audit_collection(teacher_path, bank, registry_ref)
assert group == stored[-1]
print('New seven-branch teacher raw audit matches stored targets', flush=True)
students, captures, unknown = replay.audit_students(run / 'round_005/student/result.json', bank, registry_ref)
assert len(students) == 1 and not unknown
student = students[0]
assert student['manifest']['policy'] == preupdate['model'] == old_training['policy']
assert student['source_admitted'] and student['task_outcome_admitted']
student_result = replay.read_bound(student['collection'])
row = student_result['rows'][0]
interface = replay.read_bound(row['sensor'])
model_ref = student['manifest']['policy']
model = load_schedule_policy(Path(model_ref['path']), model_ref['sha256'], bank)
records = priority_records(bank, [group], [student], DEFAULT_RULE)
phases = []
for target, gap in zip(group['targets'], records, strict=True):
    tick = target['phase_tick']
    packet = next(p for p in interface['observations'] if p['tick'] == tick)
    actual = student['phase_records'].get(tick, {})
    active = bank.option_ids.index(packet['active_before'])
    chosen, values = choose_schedule_option('learned', interface['feature_names'], packet['features'], packet['legal_mask'], active, tick, None, bank, policy=model)
    assert chosen == packet['selected_option_id']
    if values is None:
        assert packet['policy_values'] is None
    else:
        np.testing.assert_allclose(values, packet['policy_values'], rtol=0, atol=1e-12)
    phases.append(dict(
        tick=tick, preaction_capture_elapsed_s=packet['capture_elapsed_s'],
        active_before=packet['active_before'], selected_option_id=chosen,
        recorded_history_sha256=actual.get('recorded_history_sha256'),
        raw_matched_neutral_history=actual.get('recorded_history_sha256') == target.get('recorded_history_sha256'),
        causal_features_equal=actual.get('features') == target.get('features'),
        legal_mask_equal=actual.get('legal_mask') == target.get('legal_mask'),
        recorded_choice_reconstructed=True,
        legal_policy_values=None if values is None else {name: value for name, value, legal in zip(bank.option_ids, values, packet['legal_mask'], strict=True) if legal},
        complete_legal_action_table=target['complete_legal_action_table'],
        admitted_continuation_counts=target['admitted_continuation_counts'],
        expected_continuation_counts=target['expected_continuation_counts'],
        teacher_action=target['teacher_action'],
        teacher_neutral_prefix_passing_actions=[bank.option_ids[i] for i, passed in enumerate(target['pass_labels']) if passed and target['legal_mask'][i]],
        teacher_neutral_prefix_passing_continuations=[bank.option_ids[target['continuation_option_indices'][i]] for i, passed in enumerate(target['pass_labels']) if passed and target['legal_mask'][i]],
        historical_physical_gap=gap['gap'], gap_reasons=gap['reasons'], deadline=gap['deadline'],
    ))
assert phases[0]['raw_matched_neutral_history'] and phases[0]['causal_features_equal'] and phases[0]['legal_mask_equal']
teacher_result = replay.read_bound(group['collection'])
prefix=[]
for g in stored[1:]:
    collection = replay.read_bound(g['collection'])
    assert len(collection['rows']) == len(bank.option_ids)
    prefix.append(dict(scene_id=g['scene_id'], collection=g['collection'], passing_schedules=[r['forced_option_id'] for r in collection['rows'] if r['outcome']['task_outcome']=='pass'], unknown_schedules=[r['forced_option_id'] for r in collection['rows'] if r['outcome']['task_outcome']=='unknown']))
solvable=[set(p['passing_schedules']) for p in prefix if p['passing_schedules']]
common=sorted(set.intersection(*solvable)) if solvable else []
report=dict(
    schema='motion2scene_acquired_selection_verification_v1', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), run_id='seed93203_reference_contrast',
    completion=artifact(run / 'controller/complete_005.json'), preupdate=artifact(run / 'round_005/preupdate.json'),
    teacher_collection=group['collection'], student_collection=student['collection'], generating_model=model_ref, new_model=training['policy'], registry=registry_ref,
    implementation=[artifact(Path(__file__)), artifact(Path(replay.__file__)), artifact(ROOT / 'scripts/research/motion2scene_expanded_acquisition.py')],
    scene_id=group['scene_id'], physics_seed=group['physics_seed'], original_teacher_prefix_preserved=True, new_teacher_targets_raw_reaudited=True, student_raw_reaudited=True,
    learner_l2=registration['l2'], weighting=registration['weighting'], student_task_outcome=row['outcome']['task_outcome'], student_physical_events=row['outcome']['physical_events'],
    switches=interface['switches'], phases=phases, gap_rule=DEFAULT_RULE,
    teacher_outcomes={r['forced_option_id']:dict(outcome=r['outcome']['task_outcome'],passage_time_s=r['costs']['passage_time_s'],steps=r['physics_steps']) for r in teacher_result['rows']},
    encounter_physics_steps=group['physics_steps']+sum(c['physics_steps'] for c in captures), cumulative_physics_steps=completion['accounting']['actual_recorded_physics_steps'],
    postbootstrap_prefix=prefix, bank_solvable_encounters=len(solvable), assigned_encounters=len(prefix), common_passing_fixed_schedules_on_solvable_encounters=common,
    new_physics_from_verification=0, new_fits_from_verification=0,
    interpretation='Actual generating M4 policy on one acquired development encounter. Neutral teacher branches at later phases are not alternatives available after the student has already committed. Historical gaps attach to the generating M4 policy; uniform reference fitting weights are unchanged. Surface visibility does not establish semantic distinguishability. Prefix capability includes every assigned post-bootstrap task and retains unknowns; it is not held-out or post-update policy performance.'
)
write_new(out,report)
print(json.dumps(dict(output=artifact(out),teacher_outcomes=report['teacher_outcomes'],phases=[{k:p[k] for k in ['tick','selected_option_id','raw_matched_neutral_history','historical_physical_gap','gap_reasons']} for p in phases],common_passing_fixed_schedules=common,bank_solvable=report['bank_solvable_encounters'],assigned=report['assigned_encounters'])))
