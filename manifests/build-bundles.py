from pathlib import Path
import tarfile,hashlib,json,time
root=Path('/home/linjiw/motion2scene-training'); data=Path('/home/linjiw/research-data');catalog=[]
def bundle(name,files,description):
 path=root/'bundles'/f'{name}.tar.gz';entries=[]
 with tarfile.open(path,'w:gz',compresslevel=1) as tar:
  for src,rel in files:
   if not src.is_file() or src.is_symlink():continue
   digest=hashlib.sha256(src.read_bytes()).hexdigest();tar.add(src,arcname=rel,recursive=False);entries.append({'path':rel,'sha256':digest,'bytes':src.stat().st_size})
 manifest=root/'manifests'/f'{name}.json';manifest.write_text(json.dumps({'files':entries},indent=2)+'\n')
 digest=hashlib.sha256(path.read_bytes()).hexdigest();catalog.append({'name':name,'archive':str(path.relative_to(root)),'sha256':digest,'bytes':path.stat().st_size,'files':len(entries),'description':description,'manifest':str(manifest.relative_to(root))});print(name,len(entries),path.stat().st_size,flush=True)
bundle('repaired-motions',[(f,str(f.relative_to(data))) for f in (data/'m2s-sonic-repaired-v1_1-20260912').rglob('*')], '89 screened train clips, 20 development clips, all 100 candidates, repair frames and ledger')
excluded={'m2s-sonic-repaired-teacher-128env-20260912','m2s-sonic-repaired-v1_1-20260912'}
files=[]
for directory in sorted(data.glob('m2s-*')):
 if directory.name in excluded:continue
 for f in sorted(directory.rglob('*')):
  if f.is_symlink() or not f.is_file():continue
  if any(x in {'wandb','__pycache__'} for x in f.parts) or f.suffix in {'.pt','.pth','.ckpt','.gz','.zip','.log'}:continue
  files.append((f,str(f.relative_to(data))))
bundle('research-datasets-scenes',files,'Motion2Scene data, scene/obstacle tasks, BFM labels, evaluations and historical manifests; excludes duplicate model checkpoints, archives and logs')
bundle('robot-assets',[(f,'vendor/sonic/'+str(f.relative_to(Path('/home/linjiw/groot-wbc-sonic-sim-trackb')))) for f in Path('/home/linjiw/groot-wbc-sonic-sim-trackb/gear_sonic/data').rglob('*')],'SONIC robot models and local assets; original asset notices apply')
models=[('m2s-sonic-teacher-student-v1-20260911/inputs/sonic_release.pt','models/sonic_release.pt'),('m2s-sonic-teacher-8000-20260911/tracking-run-1/model_step_008000.pt','models/previous_teacher.pt'),('m2s-bfm-100motions-20260912/fit-2/step-003000.pt','models/best_student.pt')]
bundle('checkpoints',[(data/src,dst) for src,dst in models],'Released initialization, previous completed teacher, and best measured BFM student; current live teacher excluded')
(root/'manifests/catalog.json').write_text(json.dumps({'schema':'motion2scene_portable_bundles_v1','bundles':catalog},indent=2)+'\n')
