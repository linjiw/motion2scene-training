"""Read-only reconciliation of the existing pilot; never launches physics."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
DATA = Path('/home/linjiw/research-data/groot-wbc')
PLAN = DATA / 'm2s-support-pilot-plan-20260910-v1/plan.json'
PANEL = DATA / 'm2s-support-validation-panel-20260910-v1'
cache, bindings, problems = {}, {}, []
current_document = None


def digest(p):
    s = p.stat()
    key = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if key not in cache:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        cache[key] = h.hexdigest()
    return cache[key]


def bind(p, expected=None):
    p = Path(p)
    if not p.is_file():
        problems.append(dict(path=str(p), issue='missing', expected=expected))
        return
    actual = digest(p)
    bindings[str(p)] = dict(path=str(p), sha256=actual, bytes=p.stat().st_size)
    if expected and actual != expected.removeprefix('sha256:'):
        problems.append(dict(path=str(p), issue='hash_mismatch', expected=expected, actual=actual, referenced_by=current_document))


def refs(d):
    if isinstance(d, dict):
        if isinstance(d.get('path'), str) and isinstance(d.get('sha256'), str):
            bind(d['path'], d['sha256'])
        for k, v in d.items():
            if 'reserved' not in k and k not in ('domain_lock', 'generation_domain'):
                refs(v)
    elif isinstance(d, list):
        for v in d:
            refs(v)


def read(p):
    global current_document
    current_document = str(p)
    bind(p)
    d = json.loads(Path(p).read_text())
    refs(d)
    return d


def write(name, d):
    (OUT / name).write_text(json.dumps(d, indent=2, sort_keys=True) + '\n')


def csvfile(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: 'NA' if v is None else v for k, v in r.items()})


plan, prepared = read(PLAN), read(PANEL / 'prepared.json')
study = read(PANEL / 'study.json')
sample = read(Path(study['validation_sample']['path']))
rows, attempts, diagnostics = [], [], []


def outcome_row(meta, folder, cell_id, result):
    row = dict(meta, cell_id=cell_id, outcome=None, classification=None,
               passage=None, physics_steps=None, wall_seconds=None, exit_status=None,
               selected_schedule=None, passage_time_s=None, measurement_admitted=None,
               state='unrun', result_path=None, contact_dt_s=None)
    manifest = read(folder / 'manifest.json')
    cell = next(c for c in manifest['cells'] if c['cell_id'] == cell_id)
    dest = Path(cell['output'])
    ap = dest / 'attempt.json'
    if ap.exists():
        a = read(ap)
        assert a['command'] == cell['command']
        row.update(wall_seconds=a.get('wall_seconds'), exit_status=a.get('exit_status'), state='attempted_unscored')
        attempts.append(dict(assignment_id=meta['assignment_id'], path=str(ap), **a))
    elif dest.exists():
        row['state'] = 'interrupted_missing_receipt'
        for f in dest.rglob('*'):
            if f.is_file():
                bind(f)
    if result is not None:
        assert result['manifest']['path'] == str(folder / 'manifest.json')
        rr = next(x for x in result['rows'] if x['cell_id'] == cell_id)
        o = rr['outcome']
        row.update(outcome=o['task_outcome'], classification=o['classification'],
                   passage=(rr.get('passage') or {}).get('pass'), physics_steps=rr['physics_steps'],
                   selected_schedule=(rr.get('schedule_audit') or {}).get('chosen_option_id'),
                   passage_time_s=(rr.get('costs') or {}).get('passage_time_s'),
                   measurement_admitted=rr['measurement_admitted'], state='scored',
                   result_path=str(folder / 'result.json'))
        import numpy as np
        for key, dtkey in [('physics_beam_contacts', 'physics_dt'), ('all_body_contacts', 'physics_dt_s')]:
            ref = rr.get(key) or (rr.get('raw_artifacts') or {}).get(key)
            if ref:
                with np.load(ref['path']) as z:
                    dt = float(z[dtkey])
                    assert abs(dt - .005) < 1e-8
                    assert len(z['physics_steps']) == rr['physics_steps']
                    row['contact_dt_s'] = dt
        diagnostics.append(dict(assignment_id=meta['assignment_id'], outcome=o,
                                passage=rr.get('passage'), contact_audit=rr.get('contact_audit')))
    rows.append(row)


prefix_checks = []
for seed in plan['acquisition_seeds']:
    runs = [r for r in plan['runs'] if r['seed'] == seed]
    prefixes = [[s for s in r['rounds'] if s['shared_prefix']] for r in runs]
    assert prefixes[0] == prefixes[1] == prefixes[2]
    for slot in prefixes[0]:
        for key in ('inherited_teacher', 'inherited_student', 'inherited_model'):
            if slot.get(key):
                read(Path(slot[key]))
    prefix_checks.append(dict(seed=seed, byte_identical_serialized_prefix_across_arms=True,
                              prefix_sha256=hashlib.sha256(json.dumps(prefixes[0], sort_keys=True).encode()).hexdigest(),
                              m4_result=prefixes[0][-1]['inherited_model']))
for run in plan['runs']:
    for slot in run['rounds']:
        if slot['shared_prefix']:
            continue
        read(Path(slot['model_directory']) / 'result.json')
        for role in ('student', 'teacher'):
            folder = Path(slot[role + '_directory'])
            m = read(folder / 'manifest.json')
            result = read(folder / 'result.json') if (folder / 'result.json').exists() else None
            for cell in m['cells']:
                meta = dict(stage='acquisition', assignment_id=f"{run['run_id']}/round_{slot['index']:03d}/{role}/{cell['cell_id']}",
                            policy_id=run['run_id'], arm=run['arm'], corpus_seed=run['seed'],
                            physics_seed=cell['runtime_seed'], scene_id=m['scene_definition']['path'],
                            candidate_id=slot['candidate_id'], proposal_channel=slot['proposal_channel'], role=role)
                outcome_row(meta, folder, cell['cell_id'], result)
assert len(rows) == 288
for a in prepared['assignments']:
    folder = Path(a['collection']['path']).parent
    m = read(folder / 'manifest.json')
    result = read(folder / 'result.json') if (folder / 'result.json').exists() else None
    meta = dict(stage='evaluation', assignment_id=a['assignment_id'], policy_id=a['policy_id'],
                arm=next((x['arm'] for x in study['models'] if x['policy_id'] == a['policy_id']), None),
                corpus_seed=next((x['seed'] for x in study['models'] if x['policy_id'] == a['policy_id']), None),
                physics_seed=a['physics_seed'], scene_id=a['scene_id'], candidate_id=None,
                proposal_channel=None, role=a['mode'])
    outcome_row(meta, folder, m['cells'][0]['cell_id'], result)
assert len(rows) == 488
csvfile('assignment-outcomes.csv', rows)
write('attempts.json', attempts)
write('physical-diagnostics.json', diagnostics)
write('prefix-verification.json', prefix_checks)
aggregate = []
for stage, policy in sorted({(r['stage'], r['policy_id']) for r in rows}):
    own = [r for r in rows if (r['stage'], r['policy_id']) == (stage, policy)]
    aggregate.append(dict(stage=stage, policy_id=policy, assigned=len(own),
                          scored=sum(r['state'] == 'scored' for r in own),
                          passes=sum(r['outcome'] == 'pass' for r in own),
                          physical_failures=sum(r['outcome'] == 'failure' for r in own),
                          unknown=sum(r['outcome'] == 'unknown' for r in own),
                          interrupted=sum(r['state'] == 'interrupted_missing_receipt' for r in own),
                          unrun=sum(r['state'] == 'unrun' for r in own),
                          steps_known=sum(r['physics_steps'] or 0 for r in own),
                          wall_seconds_known=sum(r['wall_seconds'] or 0 for r in own),
                          nonzero_exit_receipts=sum(r['exit_status'] not in (None, 0) for r in own),
                          schedule_counts=json.dumps(dict(Counter(r['selected_schedule'] for r in own if r['selected_schedule'])),sort_keys=True)))
csvfile('aggregate-outcomes.csv', aggregate)
evalrows = [r for r in rows if r['stage'] == 'evaluation']
pairs = []
for left, right in [('support_broad','support_strict'),('support_mixture','support_strict'),('support_mixture','support_broad')]:
    for seed in plan['acquisition_seeds']:
        maps = [{r['scene_id']: r for r in evalrows if r['arm']==arm and r['corpus_seed']==seed} for arm in (left,right)]
        wins=losses=ties=missing=0
        for context in maps[0]:
            a,b = maps[0][context]['outcome'],maps[1][context]['outcome']
            if a not in ('pass','failure') or b not in ('pass','failure'):
                missing+=1
            elif a==b: ties+=1
            elif a=='pass': wins+=1
            else: losses+=1
        pairs.append(dict(left=left,right=right,seed=seed,paired_wins=wins,paired_losses=losses,paired_ties=ties,missing_pairs=missing,
                          observed_pair_difference=(wins-losses)/(10-missing) if missing<10 else None,
                          full_panel_lower=(wins-losses-missing)/10,full_panel_upper=(wins-losses+missing)/10,
                          interval_method='deterministic missing-outcome identification bounds, not confidence interval',
                          confidence_interval=None,p_value=None,permutation_space=None,p_resolution=None,
                          reason='Incomplete panel; no documented exchangeability or multiplicity scheme. Shared 10 contexts and 3 matched corpus seeds, one physics seed; executions are not independent replicates.',
                          independent_unit_count=None,analysis_status='exploratory audit after 171 validation outcomes accessible'))
write('paired-comparisons.json',pairs)
write('validation-bank.json',[dict(scene_id=s,
    observed_branches=sum(r['role']=='forced' and r['state']=='scored' for r in evalrows if r['scene_id']==s),
    passing_schedules=[r['policy_id'] for r in evalrows if r['scene_id']==s and r['role']=='forced' and r['outcome']=='pass'],
    script_outcome=next(r['outcome'] for r in evalrows if r['scene_id']==s and r['policy_id']=='script')) for s in sorted({r['scene_id'] for r in evalrows})])
# Bind the development evidence used in the revised manuscript separately.
dev = read(DATA / 'm2s-M8-native-development-statistics-20260910-v1/result.json')
devpanel = read(DATA / 'm2s-M8-native-development-panel-20260909-v1/prepared.json')
devresults = []
for assignment in devpanel['assignments']:
    folder = Path(assignment['collection']['path']).parent
    result = read(folder / 'result.json')
    devresults.append(dict(assignment_id=assignment['assignment_id'], policy_id=assignment['policy_id'],
        scene_id=assignment['scene_id'], outcome=result['rows'][0]['outcome']['task_outcome'],
        selected_schedule=(result['rows'][0].get('schedule_audit') or {}).get('chosen_option_id')))
write('development-verified-rows.json',devresults)
calibration = read(DATA / 'm2s-envelope-predictor-calibration-20260910-v1/result.json')
write('calibration-record.json',calibration)
# Keep current-source mismatches visible; a matching archive is not a repair.
unique = {}
for problem in problems:
    key = (problem['path'],problem['issue'],problem.get('expected'))
    if key not in unique:
        unique[key] = dict(problem, reference_count=0)
    unique[key]['reference_count'] += 1
for problem in unique.values():
    expected = (problem.get('expected') or '').removeprefix('sha256:')
    problem['verified_matching_archives'] = [b['path'] for b in bindings.values()
        if b['sha256'] == expected and 'snapshot' in b['path']]
write('source-drift.json',list(unique.values()))
write('verification.json',dict(bindings_verified=len(bindings),problems=problems,raw_contact_dt_s=.005,
    note='Hashes verified now do not establish registration time. Scorer receipts audited, not independently rerun physics. Missing attempt cost is unknown, not zero.',
    acquisition_assigned=288,evaluation_assigned=200,physics_launched_by_audit=0))
write('input-artifact-manifest.json',list(bindings.values()))
print(json.dumps(dict(rows=len(rows),bindings=len(bindings),problems=len(problems),states=dict(Counter(r['state'] for r in rows)),outcomes=dict(Counter(str(r['outcome']) for r in rows))),indent=2))
