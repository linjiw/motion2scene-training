"""Reconstruct all three frozen versions of the first completed portable corpus."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'toolkit'),str(ROOT/'toolkit/scripts/research')]
import motion2scene_acquisition_dataset_baseline as b
import numpy as np
release=ROOT.parent/'m2s-acquisition-M2-portable-20260909-v1'
corpus=b.read_json(release/'receipts/seed93201_uniform.json')
package=release/corpus['dataset']
bank,targets,inspection=b.inspect_package(package,corpus['dataset_manifest_sha256'])
inspection['student_policy_bindings']=b.verify_student_models(package,[b.read_json(b.bound(release,m['files']['result.json'])) for m in corpus['models']])
reports=[]
for model in corpus['models']:
    files={k:b.bound(release,v) for k,v in model['files'].items()}
    result,registration=b.read_json(files['result.json']),b.read_json(files['registration.json'])
    rows=b.prefix_targets(targets,b.read_json(files['teachers.json']),registration['collections'])
    rows,weights=b.archived_weights(registration,result,rows)
    fit,report=b.fit_timed_schedule_policy(bank,np.asarray([r['features'] for r in rows]),b.expected_feature_names(7),np.asarray([r['phase_tick'] for r in rows]),np.asarray([r['pass_labels'] for r in rows],bool),np.asarray([r['passage_time_s'] for r in rows],float),np.asarray([r['admitted'] for r in rows],bool),np.asarray([r['legal_mask'] for r in rows],bool),sample_weights=weights,l2=registration['l2'],allow_measured_tie_initialization=True)
    comparison=b.compare_models(fit,b.arrays(files['policy.npz']),targets)
    reports.append(dict(checkpoint=model['checkpoint'],comparison=comparison))
output=dict(inspection=inspection,reconstructed_fits=reports,new_physics_steps=0,guard='Python-level original-path and runtime-import restrictions; not OS isolation')
(ROOT/'smoke_result.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
print(json.dumps(output))
