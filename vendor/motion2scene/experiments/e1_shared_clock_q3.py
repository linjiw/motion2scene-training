"""Prepare and audit the registered three-carrier shared-clock Q3 pilot."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from e1_route_heldout import ROOT, checked, sha, write_new

from motion2scene.motion.paired_semantics import (
    PairedSemanticPolicy,
    assess_controller_retention,
    assess_paired_reduction,
)
from motion2scene.motion.route_retention import assess_route_retention, compare_routes

DATA = Path('/home/linjiw/research-data/groot-wbc/cg-wbc-v2-shared-seed-confirmatory/e1_shared_clock_duck_v1')
SOURCE = Path('/home/linjiw/groot-wbc-sonic-sim-trackb')
MANIFEST = ROOT / 'experiments/registrations/E1_SHARED_CLOCK_Q3_V1_MANIFEST.json'
PREDICTIONS = ROOT / 'experiments/registrations/E1_SHARED_CLOCK_Q3_V1_PREDICTIONS.md'
VALIDATION = Path('/home/linjiw/research-data/groot-wbc/cg-wbc-route-retention-heldout-v1/q3/relative_retention_result_v2.json')
STATIONS = np.linspace(0, 1, 101)


def prepare() -> None:
    validation = json.loads(VALIDATION.read_text())
    if not validation['analysis_complete'] or validation['prediction_4_met'] is not True:
        raise ValueError('complete held-out route validation must meet its registered gate')
    candidates = json.loads((DATA / 'candidates.json').read_text())
    template = json.loads((ROOT / 'experiments/registrations/E1_CONTROLLED_DUCK_Q3_V1_MANIFEST.json').read_text())
    cells = []
    for ladder in candidates['ladders']:
        if ladder['generation_seed'] not in (41001, 41002, 41003):
            continue
        if not ladder['reference_ladder_candidate']:
            raise ValueError('registered pilot carrier failed reference eligibility')
        neutral_id = f"m2s_clock_{ladder['ladder_group_id']}__neutral"
        for level in ladder['levels']:
            motion = checked(Path(level['sonic_motion']), level['sonic_motion_sha256'])
            provenance = motion.with_suffix('.pkl.manifest.json')
            csv = checked(Path(level['csv']), level['csv_sha256'])
            write_new(provenance, {
                'schema_version': 'motion2scene_sonic_conversion_provenance_v1',
                'scene_start_xyz': [0.0, 0.0, 0.0], 'scene_yaw': 0.0,
                'canonicalize_horizontal_origin': True, 'source_fps': 30.0,
                'input': {'path': str(csv), 'sha256': sha(csv)},
                'output': {'path': str(motion), 'sha256': sha(motion)},
                'candidate_manifest_sha256': sha(DATA / 'candidates.json'),
            })
            cell = copy.deepcopy(template['cells'][0])
            cell_id = f"m2s_clock_{ladder['ladder_group_id']}__{level['label']}"
            cell.update({
                'cell_id': cell_id, 'base_carrier_id': ladder['base_carrier_id'],
                'ladder_group_id': ladder['ladder_group_id'], 'ladder_level': level['ladder_level'],
                'body_mode': 'walk' if level['label'] == 'neutral' else 'controlled_crouch',
                'runtime_seed': 7800, 'hydra_overrides': ['++seed=7800'],
                'task_prompt': 'A humanoid walks straight at a steady pace',
                'motion': {'path': str(motion), 'sha256': sha(motion),
                           'conversion_provenance': str(provenance),
                           'conversion_provenance_sha256': sha(provenance),
                           'scene_start_xyz': [0.0, 0.0, 0.0]},
                'output': str(DATA / 'q3_v1/rollouts' / cell_id),
            })
            if level['label'] != 'neutral':
                cell['depends_on_acceptance_of'] = neutral_id
            cells.append(cell)
    if len(cells) != 9:
        raise ValueError('expected nine cells')
    template.update({
        'experiment': 'M2S-E1-shared-clock-q3-v1',
        'purpose': 'Nine-run shared-clock neutral/d055/d085 acquisition pilot',
        'cells': cells,
        'registered_predictions': {'path': str(PREDICTIONS), 'sha256': sha(PREDICTIONS)},
        'eligibility': {
            'reference_gate': {'path': str(DATA / 'candidates.json'), 'sha256': sha(DATA / 'candidates.json')},
            'rule': 'first three registered shared-clock carrier IDs, all reference eligible',
            'heldout_route_validation': {'path': str(VALIDATION), 'sha256': sha(VALIDATION)},
        },
    })
    template['execution_policy']['cost_ceiling'].update({'rollouts': 9, 'gpu_hours_contended': 0.9375})
    template['execution_policy']['timing_override'] = 'Fresh shared-clock Q3 pilot registered before launch, serial trajectory-only at seed 7800.'
    write_new(MANIFEST, template)
    print('prepared nine shared-clock Q3 cells')


def analyze() -> None:
    from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints
    from gear_sonic.dataset_generation.hallucination.motion_envelope import extract_envelope
    from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

    manifest = json.loads(MANIFEST.read_text())
    run_path = DATA / 'q3_v1/run_record.json'
    run = json.loads(run_path.read_text())
    if run['status'] != 'completed' or run['manifest_sha256'] != sha(MANIFEST):
        raise ValueError('analysis requires the complete registered batch')
    candidates = json.loads(checked(DATA / 'candidates.json', manifest['eligibility']['reference_gate']['sha256']).read_text())
    reference = {
        (ladder['ladder_group_id'], level['label']): level
        for ladder in candidates['ladders'] for level in ladder['levels']
    }
    rows, envelopes = [], {}
    for cell in manifest['cells']:
        record = run['cells'][cell['cell_id']]
        label = cell['cell_id'].rsplit('__', 1)[-1]
        key = (cell['ladder_group_id'], label)
        row = {'cell_id': cell['cell_id'], 'ladder_group_id': key[0], 'label': label,
               'status': record['status'], 'tracker_survived': None, 'route_retained': None,
               'q3_controller_retained': False}
        if record['status'] == 'completed':
            scientific = record['scientific']
            artifact = scientific['artifacts']
            path = checked(Path(artifact['trajectory']), artifact['trajectory_sha256'])
            with path.open('rb') as handle:
                payload, _ = best_evaluable_payload(pickle.load(handle))
            ref = reference[key]
            qpos = np.loadtxt(checked(Path(ref['csv']), ref['csv_sha256']), delimiter=',')
            route = compare_routes(qpos[:, :2], np.asarray(payload['root_pos_w'])[:, :2],
                                   expected_route='straight', reference_fps=30,
                                   achieved_fps=float(payload['fps']))
            decision = assess_route_retention(route)
            envelopes[key] = extract_envelope(extract_keypoints(payload), cell['cell_id'], fractions=STATIONS).up_m
            row.update({'tracker_survived': scientific['outcome'] == 'accepted',
                        'rejection_reasons': scientific['rejection_reasons'],
                        'route_retained': decision.retained, 'route_decision': decision.to_dict(),
                        'route_metrics': route.to_dict(), 'trajectory_sha256': sha(path)})
        rows.append(row)
    by_key = {(row['ladder_group_id'], row['label']): row for row in rows}
    policy = PairedSemanticPolicy(minimum_effect=0.05)
    for key, target in envelopes.items():
        if key[1] == 'neutral':
            continue
        neutral_key = (key[0], 'neutral')
        if neutral_key not in envelopes:
            continue
        ref = assess_paired_reduction(STATIONS, np.asarray(reference[key]['whole_body_top_m']),
                                     np.asarray(reference[neutral_key]['whole_body_top_m']),
                                     route_valid=True, policy=policy)
        achieved = assess_paired_reduction(STATIONS, target, envelopes[neutral_key],
                                          route_valid=by_key[key]['route_retained'] and by_key[neutral_key]['route_retained'],
                                          policy=policy)
        retained = assess_controller_retention(ref, achieved, policy=policy)
        by_key[key].update({'reference_semantic': ref.to_dict(), 'achieved_semantic': achieved.to_dict(),
                           'controller_retention': retained.to_dict(),
                           'q3_controller_retained': by_key[key]['tracker_survived'] and by_key[neutral_key]['tracker_survived'] and retained.semantic_status.value == 'controller_retained'})
    groups = sorted({row['ladder_group_id'] for row in rows})
    prefixes = [group for group in groups if by_key[(group, 'neutral')]['tracker_survived']
                and by_key[(group, 'neutral')]['route_retained']
                and all(by_key[(group, label)]['q3_controller_retained'] for label in ('d055', 'd085'))]
    result = {'schema_version': 'motion2scene_shared_clock_q3_result_v1',
              'manifest_sha256': sha(MANIFEST), 'run_record_sha256': sha(run_path),
              'driver_sha256': sha(Path(__file__)), 'registered_cells': 9,
              'q3_three_level_retained_prefixes': prefixes, 'q4_admitted_ladders': 0,
              'rows': rows}
    write_new(DATA / 'q3_v1/retention_v1.json', result)
    print('Three-level retained prefixes:', prefixes)
    for row in rows:
        print(row['cell_id'], row['tracker_survived'], row['route_retained'], row['q3_controller_retained'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'analyze'))
    args = parser.parse_args()
    sys.path.insert(0, str(SOURCE))
    {'prepare': prepare, 'analyze': analyze}[args.mode]()
