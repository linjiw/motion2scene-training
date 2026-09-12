"""Reconstruct native expanded-format fits with the bundled NumPy-only reader."""
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BASE = Path(__file__).parent
TOOLKIT = BASE / 'toolkit_readonly_v3'
sys.path[:0] = [str(TOOLKIT), str(TOOLKIT / 'scripts/research')]
from motion2scene_acquisition_curve_dataset_baseline import inspect_package, read_json, reconstruct_model, verify_model_provenance

release = Path('/home/linjiw/research-data/groot-wbc/m2s-acquisition-M2-portable-20260909-v1')
native = read_json(BASE / 'expanded_format_validation/result.json')
corpus = native['corpus']
bank, targets, inspection = inspect_package(release / corpus['dataset'], corpus['dataset_manifest_sha256'])
models = []
for item in native['models']:
    files = {k: Path(v['path']) for k,v in item['files'].items()}
    collections = read_json(files['registration.json'])['collections']
    verify_model_provenance(files, item['result'], collections, item['checkpoint'],
        'observation_curriculum', native['plan'], native['replay_rule'])
    _, result = reconstruct_model(bank, targets, files, native['originals'], native['student_sources'])
    models.append(dict(checkpoint=item['checkpoint'], **result))
report = dict(scope=native['scope'], native_fits=3, portable_fits=len(models), models=models,
    inspection=inspection, new_physics_steps=0)
(BASE / 'expanded_format_reconstruction.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(portable_fits=len(models), new_physics_steps=0)))
