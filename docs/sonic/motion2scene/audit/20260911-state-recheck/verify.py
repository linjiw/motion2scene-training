"""Recheck existing audit bindings without modifying the experiment or opening reserved artifacts."""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / '20260911-pilot-boundary'
report = {'utc': datetime.now(timezone.utc).isoformat(), 'physics_launched': 0,
          'runtime': platform.platform(), 'python': platform.python_version(),
          'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'manifests': {}}
for name in ('audit-manifest.json', 'input-artifact-manifest.json'):
    source = OLD / name
    data = json.loads(source.read_text())
    rows = data['artifacts'] if isinstance(data, dict) else data
    result = {'manifest_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'matched': 0, 'bytes_hashed': 0, 'mismatches': [], 'missing': [], 'reserved_skipped': []}
    for row in rows:
        p = Path(row['path'])
        if 'reserved' in str(p).lower() or 'layout-b' in str(p).lower():
            result['reserved_skipped'].append(str(p))
            continue
        if not p.is_file():
            result['missing'].append(str(p))
            continue
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
                result['bytes_hashed'] += len(chunk)
        if h.hexdigest() == row['sha256']:
            result['matched'] += 1
        else:
            result['mismatches'].append({'path': str(p), 'expected': row['sha256'], 'actual': h.hexdigest()})
    report['manifests'][name] = result
    print(name, result['matched'], 'matched', len(result['mismatches']), 'changed', len(result['missing']), 'missing', flush=True)
panel = Path('/home/linjiw/research-data/groot-wbc/m2s-support-validation-panel-20260910-v1')
report['evaluation_results_present'] = [i for i in range(200) if (panel / f'episodes/episode_{i:03d}/result.json').exists()]
report['episode_171_files'] = sorted(str(p.relative_to(panel)) for p in (panel / 'episodes/episode_171').rglob('*') if p.is_file())
report['pilot_processes'] = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        args = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        if ('python' in args.split(' ')[0] and any(s in args for s in ('motion2scene_support_acquisition.py', 'motion2scene_support_validation_panel.py', 'sonic_rollout_isaaclab.py'))):
            report['pilot_processes'].append({'pid': int(p.name), 'command': args})
    except (OSError, IndexError):
        pass
(OUT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print('Evaluation receipts:', len(report['evaluation_results_present']), 'active pilot processes:', len(report['pilot_processes']))
