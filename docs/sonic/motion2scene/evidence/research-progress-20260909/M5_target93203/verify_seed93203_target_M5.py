"""Bounded raw-record verification of one acquired selection failure; no new fit/physics."""

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
run = D / 'm2s-expanded-acquisition-20260909-v1/seed93203_target_only'
out = Path(__file__).parent / 'seed93203_target_M5_selection_verified.json'
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
assert student['source_admitted'] and student['task_outcome_admitted'] and not student['passed']
assert len(captures) == 1 and captures[0]['physics_steps'] == 1192
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
    actual = student['phase_records'][tick]
    matched = actual['recorded_history_sha256'] == target['recorded_history_sha256']
    assert matched and actual['features'] == target['features'] and actual['legal_mask'] == target['legal_mask']
    chosen, values = choose_schedule_option('learned', interface['feature_names'], packet['features'], packet['legal_mask'], 0, tick, None, bank, policy=model)
    assert chosen == packet['selected_option_id']
    np.testing.assert_allclose(values, packet['policy_values'], rtol=0, atol=1e-12)
    phases.append(dict(
        tick=tick, preaction_capture_elapsed_s=packet['capture_elapsed_s'],
        selected_option_id=chosen, recorded_history_sha256=actual['recorded_history_sha256'],
        raw_matched_neutral_history=True, causal_features_equal=True, legal_mask_equal=True,
        recorded_values_reconstructed=True,
        legal_policy_values={name: value for name, value, legal in zip(bank.option_ids, values, packet['legal_mask'], strict=True) if legal},
        complete_legal_action_table=target['complete_legal_action_table'],
        admitted_continuation_counts=target['admitted_continuation_counts'],
        expected_continuation_counts=target['expected_continuation_counts'],
        teacher_action=target['teacher_action'],
        passing_immediate_actions=[bank.option_ids[i] for i, passed in enumerate(target['pass_labels']) if passed and target['legal_mask'][i]],
        passing_continuations=[bank.option_ids[target['continuation_option_indices'][i]] for i, passed in enumerate(target['pass_labels']) if passed and target['legal_mask'][i]],
        historical_physical_gap=gap['gap'], gap_reasons=gap['reasons'], deadline=gap['deadline'],
    ))
teacher_result = replay.read_bound(group['collection'])
report = dict(
    schema='motion2scene_acquired_selection_verification_v1',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), run_id='seed93203_target_only',
    completion=artifact(run / 'controller/complete_005.json'),
    preupdate=artifact(run / 'round_005/preupdate.json'),
    teacher_collection=group['collection'], student_collection=student['collection'],
    generating_model=model_ref, new_model=training['policy'], registry=registry_ref,
    implementation=[artifact(Path(__file__)), artifact(Path(replay.__file__)), artifact(ROOT / 'scripts/research/motion2scene_expanded_acquisition.py')],
    scene_id=group['scene_id'], physics_seed=group['physics_seed'],
    original_teacher_prefix_preserved=True, new_teacher_targets_raw_reaudited=True,
    student_raw_reaudited=True, learner_l2=registration['l2'], weighting=registration['weighting'],
    student_task_outcome=row['outcome']['task_outcome'], student_physical_events=row['outcome']['physical_events'],
    student_selected_schedule=next(s['to'] for s in interface['switches'] if s['from']=='neutral'),
    phases=phases, gap_rule=DEFAULT_RULE,
    teacher_outcomes={r['forced_option_id']: dict(outcome=r['outcome']['task_outcome'], passage_time_s=r['costs']['passage_time_s'], steps=r['physics_steps']) for r in teacher_result['rows']},
    encounter_physics_steps=group['physics_steps']+captures[0]['physics_steps'],
    cumulative_physics_steps=completion['accounting']['actual_recorded_physics_steps'],
    new_physics_from_verification=0, new_fits_from_verification=0,
    interpretation='The generating M4 student missed the last passing prior entry at tick 50 on this acquired development condition. This is not an evaluation of the new M5 model or a held-out result. Historical physical gaps attach to the generating M4 policy; uniform target-only fitting weights are unchanged. Surface visibility does not establish semantic distinguishability.'
)
assert report['encounter_physics_steps'] == 9536
write_new(out, report)
print(json.dumps(dict(output=artifact(out), phases=[{k:p[k] for k in ['tick','selected_option_id','passing_immediate_actions','historical_physical_gap','gap_reasons']} for p in phases])))
