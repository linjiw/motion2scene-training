"""CPU schema validation on recorded M2 data; never writes acquisition slots."""
import json
from pathlib import Path
import sys

ROOT = Path('/home/linjiw/groot-wbc-sonic-sim-trackb')
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/research')]
from motion2scene_checkpoint_controls import load_checkpoint
from motion2scene_decision_study import read_checked
from motion2scene_expanded_acquisition import ExpandedController
from motion2scene_export_acquisition_curve_dataset import audit_expanded_replay
from motion2scene_timing_diagnostic import artifact

DATA = Path('/home/linjiw/research-data/groot-wbc')
OUT = Path(__file__).parent / 'expanded_format_validation'
OUT.mkdir(exist_ok=False)
plan_ref = artifact(DATA / 'm2s-primary-acquisition-plan-tie-proposed-v5/plan.json')
plan = read_checked(plan_ref)
adoption = read_checked(artifact(DATA / 'm2s-primary-acquisition-adoption-v3/adoption.json'))
runtime = read_checked(adoption['runtime_freeze'])
rule = read_checked(runtime['replay_rule'])
run_id = 'seed93201_observation_curriculum'
declaration = dict(scope='CPU format validation of the registered expanded fitter on recorded M2 data; '
    'not M8 acquisition, not an online trajectory and not a policy performance experiment',
    original_plan=plan_ref, run_id=run_id, native_fitter=artifact(ROOT / 'scripts/research/motion2scene_expanded_acquisition.py'),
    replay_rule=runtime['replay_rule'], new_physics_steps=0)
(OUT / 'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
controller = object.__new__(ExpandedController)
controller.context = dict(plan=plan, runtime=runtime, expanded_plan_ref=artifact(OUT / 'declaration.json'))
controller.expanded_run = dict(arm='observation_curriculum')
print(json.dumps(dict(status='reauditing_recorded_M2_prefix', run_id=run_id)), flush=True)
bank, groups, students, _ = load_checkpoint(plan, run_id, 2)
models = []
for checkpoint in range(3):
    folder = OUT / f'model_{checkpoint:03d}'
    result = controller.fit_expanded(groups[:checkpoint+1], students[:checkpoint+1], folder)
    sidecar_ref = artifact(folder / 'replay.json')
    audit_expanded_replay(read_checked(sidecar_ref), bank, groups[:checkpoint+1], students[:checkpoint+1], rule)
    (folder / 'replay_verification.json').write_text(json.dumps(dict(original_sidecar=sidecar_ref,
        replay_rule=runtime['replay_rule'], teacher_collections=[g['collection'] for g in groups[:checkpoint+1]],
        historical_outcomes_independently_reaudited=True, new_physics_steps=0), indent=2)+'\n')
    models.append(dict(checkpoint=checkpoint, result=artifact(folder / 'result.json'),
        files={p.name: artifact(p) for p in sorted(folder.iterdir()) if p.is_file()}))
    print(json.dumps(dict(status='native_expanded_format_fit_complete', checkpoint=checkpoint)), flush=True)
release = DATA / 'm2s-acquisition-M2-portable-20260909-v1'
index = json.loads((release / 'release.json').read_text())
corpus = next(c for c in index['corpora'] if c['run_id'] == run_id)
originals = [json.loads((release / m['files']['result.json']['path']).read_text()) for m in corpus['models']]
report = dict(scope=declaration['scope'], models=models, originals=originals,
    student_sources=[s['collection'] for s in students[1:]], source_release=artifact(release / 'release.json'),
    corpus=corpus, plan=controller.context['expanded_plan_ref'], replay_rule=runtime['replay_rule'],
    native_fits=3, new_physics_steps=0)
(OUT / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
