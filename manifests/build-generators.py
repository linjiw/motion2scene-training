import json,tarfile,hashlib
from pathlib import Path
r=Path('/home/linjiw/motion2scene-training');d=Path('/home/linjiw/research-data');entries=[]
p=r/'bundles/learned-scene-models.tar.gz'
with tarfile.open(p,'w:gz',compresslevel=1) as tar:
 for directory in sorted(d.glob('m2s-*')):
  if any(x in directory.name for x in ['teacher','bfm','qualification','repaired']):continue
  for f in sorted(directory.rglob('*.pt')):
   if f.is_symlink():continue
   rel=str(f.relative_to(d));tar.add(f,arcname=rel);entries.append({'path':rel,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size})
(r/'manifests/learned-scene-models.json').write_text(json.dumps({'files':entries},indent=2)+'\n')
c=json.loads((r/'manifests/catalog.json').read_text());c['bundles'].append({'name':'learned-scene-models','archive':'bundles/learned-scene-models.tar.gz','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'files':len(entries),'description':'Learned scene/event/shape generators, probabilities and tensor labels; experimental qualification status preserved','manifest':'manifests/learned-scene-models.json'});(r/'manifests/catalog.json').write_text(json.dumps(c,indent=2)+'\n');print(len(entries),p.stat().st_size)
