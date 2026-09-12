"""Bounded CPU-only optimization diagnostic on a previously executed seven-motion bank.

No physical outcome training, pilot validation, protected layouts or simulator imports.
This is an in-bank engineering benchmark, not an LfLH reproduction or transfer test.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import load_reset_capture
from gear_sonic.dataset_generation.swept_volume import CollisionCapsule, body_capsules_world
from scripts.research.lflh_next.geometry import box_sdf, capsule_samples, soft_lower_min

BANK = Path('/home/linjiw/research-data/groot-wbc/m2s-prior-splice-seven-schedule-analysis-v2/environment_result.json')
CONFIG = dict(seeds=[101, 102], architectures=['unconditional', 'mlp', 'conv', 'transformer'],
              updates=120, learning_rate=0.002, training_frames=16, axial_samples=5,
              draws_per_target=8, threads=2, wall_cap_seconds=600,
              beam_x_bounds=[1.0, 4.5], beam_underside_bounds=[0.7, 1.65],
              beam_half_extents=[0.15, 1.0, 0.1], margin_m=0.01,
              distance_temperature_m=0.002, choice_temperature_m=0.025,
              clearance_weight=100.0, kl_weight=0.001, log_sigma_bounds=[-5., 1.])
BINDINGS = {}


def bind(path, expected=None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected.removeprefix('sha256:'):
        raise ValueError(f'input hash mismatch: {path}')
    BINDINGS[str(path)] = dict(path=str(path), sha256=digest, bytes=path.stat().st_size)
    return path


def read(ref):
    return json.loads(bind(ref['path'], ref.get('sha256')).read_text())


def write(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    source = json.loads(bind(BANK).read_text())
    manifest = read(source['manifest'])
    geometry = read(manifest['geometry'])
    shapes = {}
    for shape in geometry['shapes']:
        shapes.setdefault(shape['owner'], []).append(CollisionCapsule(tuple(shape['start']), tuple(shape['end']), shape['radius']))
    profiles, dense, train, names = [], [], [], []
    for row in source['rows']:
        if not row['qualified']:
            raise ValueError('unqualified bank: refuse rather than drop a motion')
        evidence = read(row['evidence'])
        ref = evidence['artifacts']['trajectory']
        payload = load_reset_capture(bind(ref['path'], ref['sha256']))
        a, b, r, _ = body_capsules_world(payload['body_pos_w'], payload['body_quat_w'], payload['body_names'], capsules=shapes)
        if len(a) != 298:
            raise ValueError('unexpected bank capture length')
        a, b, r = [torch.tensor(x, dtype=torch.float32) for x in (a, b, r)]
        idx = torch.linspace(0, len(a)-1, CONFIG['training_frames']).round().long()
        lo = torch.minimum(a, b)-r[None, :, None]
        hi = torch.maximum(a, b)+r[None, :, None]
        profiles.append(torch.cat((lo.amin(1), hi.amax(1)), -1)[idx])
        for dest, aa, bb in ((dense, a, b), (train, a[idx], b[idx])):
            p, rad, cover = capsule_samples(aa, bb, r, CONFIG['axial_samples'])
            dest.append((p.reshape(-1, 3), (rad+cover).reshape(-1)))
        names.append(row['cell_id'])
    for path in (Path(__file__), Path(__file__).with_name('geometry.py'),
                 ROOT/'gear_sonic/dataset_generation/swept_volume.py',
                 ROOT/'gear_sonic/dataset_generation/hallucination/motion2scene_reset_capture.py',
                 ROOT/'gear_sonic/dataset_generation/trajectory_segments.py',
                 ROOT/'gear_sonic/dataset_generation/trajectory_validation.py'):
        bind(path)
    tensors = dict(profiles=torch.stack(profiles), train_points=torch.stack([x[0] for x in train]),
                   train_radius=torch.stack([x[1] for x in train]),
                   dense_points=torch.stack([x[0] for x in dense]), dense_radius=torch.stack([x[1] for x in dense]))
    torch.save(tensors, out/'bank.pt')
    bind(out/'bank.pt')
    write(out/'design.json', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), config=CONFIG,
          names=names, source=source['manifest'], inputs=list(BINDINGS.values()),
          outcome_access='Earlier development and partial pilot results already accessible; new offline design is exploratory.',
          scope='Seven motions from one existing qualified bank. Fit and score on the same bank, fresh latent draws only. No source-ancestry holdout or population inference.',
          labels='Desired motion index; no new physical outcome labels; decoder is geometric clearance ranking, not an optimal dynamics planner.',
          forbidden=['pilot acquisition/evaluation contexts', 'reserved layouts', 'new physics', 'GPU use', 'checkpoint selection by scoring results'],
          ceiling='8 fits x 120 updates; 600 s training/evaluation wall cap. Do not extend if unfavorable.'))
    print('Prepared hash-bound bank and dated exploratory design before model training', flush=True)


class Encoder(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == 'unconditional':
            self.raw = nn.Parameter(torch.zeros(4))
        elif kind == 'mlp':
            self.network = nn.Sequential(nn.Flatten(), nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 4))
        elif kind == 'conv':
            self.network = nn.Sequential(nn.Conv1d(6, 32, 3, padding=1), nn.SiLU(), nn.Conv1d(32, 32, 3, padding=1), nn.SiLU(), nn.Flatten(), nn.Linear(512, 4))
        else:
            self.project = nn.Linear(6, 32)
            self.position = nn.Parameter(torch.randn(1, 16, 32)*0.01)
            layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0., batch_first=True)
            self.network = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            self.head = nn.Linear(32, 4)

    def forward(self, x):
        if self.kind == 'unconditional':
            raw = self.raw.expand(len(x), -1)
        elif self.kind == 'conv':
            raw = self.network(x.transpose(1, 2))
        elif self.kind == 'transformer':
            raw = self.head(self.network(self.project(x)+self.position).mean(1))
        else:
            raw = self.network(x)
        return raw[:, :2], raw[:, 2:].clamp(*CONFIG['log_sigma_bounds'])


def gaps(z, points, radius, *, smooth):
    # z: batch,2. All candidates see exactly the same proposed beam for a row.
    unit = z.sigmoid()
    x = 1.+3.5*unit[:, 0]
    bottom = 0.7+0.95*unit[:, 1]
    centre = torch.stack((x, x*0, bottom+0.1), -1)[:, None, None, :]
    signed = box_sdf(points[None], centre, torch.tensor(CONFIG['beam_half_extents']), torch.tensor(0.))-radius[None]
    if smooth:
        return soft_lower_min(signed, CONFIG['distance_temperature_m'])
    return signed.amin(-1)


def score(model, profile, data, seed, shuffle=False):
    rng = torch.Generator().manual_seed(seed)
    x = profile.roll(1, dims=0) if shuffle else profile
    mean, log_sigma = model(x)
    draws = torch.randn(CONFIG['draws_per_target'], len(x), 2, generator=rng)
    rows = []
    for j in range(CONFIG['draws_per_target']):
        z = mean + log_sigma.exp()*draws[j]
        dense = gaps(z, data['dense_points'], data['dense_radius'], smooth=False)
        sparse = gaps(z, data['train_points'], data['train_radius'], smooth=False)
        for i in range(len(x)):
            other = torch.cat((dense[i, :i], dense[i, i+1:]))
            rows.append(dict(target=i, draw=j, x_m=float(1+3.5*z[i, 0].sigmoid()),
                             underside_m=float(0.7+0.95*z[i, 1].sigmoid()),
                             reconstructed=bool(dense[i].argmax()==i),
                             bounded_clear_at_recorded_frames=bool(dense[i, i]>=CONFIG['margin_m']),
                             contrast=bool(dense[i, i]>=CONFIG['margin_m'] and other.min()<=-CONFIG['margin_m']),
                             sparse_false_clear=bool(sparse[i, i]>=CONFIG['margin_m'] and dense[i, i]<CONFIG['margin_m']),
                             dense_lower_gap_m=float(dense[i, i]), sparse_lower_gap_m=float(sparse[i, i])))
    return rows


def run(out):
    design = json.loads((out/'design.json').read_text())
    if design['config'] != CONFIG:
        raise ValueError('design drift')
    for ref in design['inputs']:
        bind(ref['path'], ref['sha256'])
    if (out/'started.json').exists():
        raise ValueError('retain every attempt; no automatic resume or overwrite')
    torch.set_num_threads(CONFIG['threads'])
    torch.use_deterministic_algorithms(True)
    data = torch.load(out/'bank.pt', weights_only=True, map_location='cpu')
    profile = data['profiles']
    # Scales fitted only from this explicitly in-bank training set; saved for reuse.
    mu, scale = profile.mean((0,1)), profile.std((0,1)).clamp_min(0.05)
    profile = (profile-mu)/scale
    start = time.monotonic()
    write(out/'started.json', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
          runtime=platform.platform(), device='cpu', threads=CONFIG['threads'], physics_steps=0,
          normalization_mean=mu.tolist(), normalization_scale=scale.tolist()))
    results = []
    for seed in CONFIG['seeds']:
        for kind in CONFIG['architectures']:
            folder = out/f'{kind}-{seed}'
            folder.mkdir(exist_ok=False)
            torch.manual_seed(seed)
            model = Encoder(kind)
            opt = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
            rng = torch.Generator().manual_seed(seed+1000)
            updates, history = 0, []
            fit_start = time.monotonic()
            state = 'complete'
            for step in range(CONFIG['updates']):
                if time.monotonic()-start >= CONFIG['wall_cap_seconds']:
                    state = 'wall_cap'; break
                mean, log_sigma = model(profile)
                z = mean+log_sigma.exp()*torch.randn(mean.shape, generator=rng)
                gap = gaps(z, data['train_points'], data['train_radius'], smooth=True)
                ce = nn.functional.cross_entropy(gap/CONFIG['choice_temperature_m'], torch.arange(7))
                penalty = (CONFIG['margin_m']-gap.diag()).clamp_min(0).square().mean()
                kl = (mean.square()+torch.exp(2*log_sigma)-1-2*log_sigma).mean()/2
                loss = ce+CONFIG['clearance_weight']*penalty+CONFIG['kl_weight']*kl
                if not torch.isfinite(loss):
                    state = 'nonfinite_loss'; break
                opt.zero_grad(); loss.backward(); opt.step(); updates += 1
                if step % 20 == 0 or step == CONFIG['updates']-1:
                    history.append(dict(update=updates, loss=float(loss.detach()), ce=float(ce.detach()), clearance_penalty=float(penalty.detach())))
            train_seconds = time.monotonic()-fit_start
            torch.save(dict(state_dict=model.state_dict(), kind=kind, config=CONFIG, mean=mu, scale=scale), folder/'model.pt')
            eval_start = time.monotonic()
            with torch.no_grad():
                model.eval()
                scored = score(model, profile, data, seed+2000) if state == 'complete' else []
                shuffled = score(model, profile, data, seed+2000, shuffle=True) if state == 'complete' else []
            record = dict(architecture=kind, seed=seed, updates=updates, state=state,
                          parameters=sum(p.numel() for p in model.parameters()), train_seconds=train_seconds,
                          evaluation_seconds=time.monotonic()-eval_start, history=history, rows=scored, shuffled_rows=shuffled)
            write(folder/'receipt.json', record)
            summary = {k:v for k,v in record.items() if k not in ('rows','history','shuffled_rows')}
            for metric in ('reconstructed','bounded_clear_at_recorded_frames','contrast','sparse_false_clear'):
                summary[metric] = sum(r[metric] for r in scored)/len(scored) if scored else None
                summary['shuffled_'+metric] = sum(r[metric] for r in shuffled)/len(shuffled) if shuffled else None
            summary['draws_scored'] = len(scored)
            results.append(summary)
            print(json.dumps(summary), flush=True)
            if state != 'complete' or time.monotonic()-start>=CONFIG['wall_cap_seconds']:
                write(out/'results.json', dict(state='stopped', fits=results, wall_seconds=time.monotonic()-start, physics_steps=0)); return
    write(out/'results.json', dict(state='complete', fits=results, wall_seconds=time.monotonic()-start, physics_steps=0,
          inference='Descriptive in-bank engineering scores. No physical passage labels, p-values or source transfer.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else run)(args.out)
