"""Verify actual extraction of the already byte-audited compact release."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

BASE = Path(__file__).resolve().parent
receipt = json.loads((BASE / 'package_result.json').read_text())
archive = Path(receipt['archive']['path'])
root = BASE / 'relocated'
root.mkdir(exist_ok=False)
start = time.monotonic()
subprocess.run(['tar', '-xzf', str(archive), '-C', str(root)], check=True)
rows = json.loads((BASE / 'archive_members.json').read_text())
expected = {row['path'] for row in rows}
actual = {str(p.relative_to(root)) for p in root.rglob('*')}
if actual != expected:
    raise ValueError('extracted member paths differ')
cached = {}
files = 0
for row in rows:
    path = root / row['path']
    if path.is_symlink():
        raise ValueError('unexpected symlink')
    if row['kind'] == 'directory':
        if not path.is_dir():
            raise ValueError('missing directory')
        continue
    stat = path.stat()
    if not path.is_file() or stat.st_size != row['size_bytes']:
        raise ValueError('extracted size/type mismatch')
    identity = (stat.st_dev, stat.st_ino)
    if identity not in cached:
        with path.open('rb') as f:
            cached[identity] = hashlib.file_digest(f, 'sha256').hexdigest()
    if cached[identity] != row['sha256']:
        raise ValueError('extracted file hash mismatch: ' + row['path'])
    files += 1
result = dict(schema='motion2scene_acquisition_archive_extraction_v1',status='complete',archive=receipt['archive'],root=str(root),members=len(rows),file_paths=files,unique_file_inodes=len(cached),all_extracted_bytes_match=True,wall_seconds=time.monotonic()-start,new_physics_steps=0,new_fits=0)
with (BASE / 'extraction_result.json').open('x') as f:
    json.dump(result,f,indent=2,sort_keys=True)
    f.write('\n')
print(json.dumps(result))
