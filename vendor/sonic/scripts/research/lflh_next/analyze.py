"""Post-hoc engineering analysis; does not fit models or execute physics.

The original contrast field used a negative LOWER bound as a penetration proxy.
That is insufficient. Retain original receipts and compute sampled penetration
witnesses separately. Even these concern the capsule model, not native physics.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.lflh_next.benchmark import BANK, bind, read
from scripts.research.lflh_next.geometry import box_sdf
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import load_reset_capture
from gear_sonic.dataset_generation.swept_volume import CollisionCapsule, body_capsules_world


def main(out):
    started = time.monotonic()
    torch.set_num_threads(2)
    design = json.loads((out/'design.json').read_text())
    for ref in design['inputs']:
        bind(ref['path'], ref['sha256'])
    bank = json.loads(BANK.read_text())
    geometry = read(read(bank['manifest'])['geometry'])
    shapes = {}
    for shape in geometry['shapes']:
        shapes.setdefault(shape['owner'], []).append(CollisionCapsule(tuple(shape['start']), tuple(shape['end']), shape['radius']))
    radius = []
    for row in bank['rows']:
        ref = read(row['evidence'])['artifacts']['trajectory']
        p = load_reset_capture(bind(ref['path'], ref['sha256']))
        a, _, r, _ = body_capsules_world(p['body_pos_w'], p['body_quat_w'], p['body_names'], capsules=shapes)
        radius.append(torch.tensor(np.broadcast_to(r[None, :, None], (len(a),len(r),design['config']['axial_samples'])).copy()).flatten())
    radius = torch.stack(radius)
    data = torch.load(out/'bank.pt', weights_only=True)
    records, summaries = [], []
    for fit in json.loads((out/'results.json').read_text())['fits']:
        receipt = json.loads((out/f"{fit['architecture']}-{fit['seed']}"/'receipt.json').read_text())
        for condition, rows in [('conditioned', receipt['rows']), ('input_rotated', receipt['shuffled_rows'])]:
            for row in rows:
                c = torch.tensor([row['x_m'],0.,row['underside_m']+.1])
                sdf = box_sdf(data['dense_points'],c,torch.tensor([.15,1.,.1]),torch.tensor(0.))
                upper = (sdf-radius).amin(-1)
                lower = (sdf-data['dense_radius']).amin(-1)
                i = row['target']; alt = torch.cat((upper[:i],upper[i+1:]))
                records.append(dict(architecture=fit['architecture'],seed=fit['seed'],condition=condition,**row,
                    legacy_contrast_is_valid_witness=False,
                    capsule_model_contrast=bool(lower[i]>=.01 and alt.min()<=-.01),
                    sampled_target_penetration=bool(upper[i]<0),
                    sampled_min_target_gap_m=float(upper[i])))
        rows = [r for r in records if r['architecture']==fit['architecture'] and r['seed']==fit['seed'] and r['condition']=='conditioned']
        summaries.append(dict(architecture=fit['architecture'],seed=fit['seed'],draws=len(rows),
            parameters=fit['parameters'],train_seconds=fit['train_seconds'],
            reconstruction_count=sum(r['reconstructed'] for r in rows),
            capsule_clear_count=sum(r['bounded_clear_at_recorded_frames'] for r in rows),
            capsule_contrast_count=sum(r['capsule_model_contrast'] for r in rows),
            sampled_target_penetration_count=sum(r['sampled_target_penetration'] for r in rows),
            sparse_false_clear_count=sum(r['sparse_false_clear'] for r in rows)))
    for name, rows in [('draws.csv',records),('architecture-results.csv',summaries)]:
        with (out/name).open('x') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'posthoc-analysis.json').open('x') as f:
        json.dump(dict(status='exploratory correction after scores accessible',new_physics_steps=0,
            reason='A negative lower bound is not a penetration witness. Legacy contrast retained only as a proxy.',
            wall_seconds=time.monotonic()-started,results=summaries),f,indent=2)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]))
