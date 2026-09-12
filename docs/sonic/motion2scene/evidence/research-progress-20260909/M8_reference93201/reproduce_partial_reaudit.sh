#!/usr/bin/env bash
# Exact command executed from the repository root; refuses to overwrite its report.
# To reproduce, change only the output report path to a fresh path before running.
set -euo pipefail
.venv_isaaclab/bin/python - <<'PY'
from pathlib import Path
import datetime,json,sys
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'scripts/research')]
import motion2scene_collect_timed_schedules as collector
from motion2scene_timing_diagnostic import artifact
from motion2scene_build_timed_replay import read_bound
D=Path('/home/linjiw/research-data/groot-wbc');o=D/'m2s-expanded-and-capability-stages-20260910-v8';root=D/'m2s-expanded-acquisition-20260909-v1/seed93201_reference_contrast'
out=o/'seed93201_reference_M8_partial_reaudit.json';assert not out.exists()
assessments=sorted((root/'controller/attempts').glob('round008_*/assessment.json'));assert len(assessments)==8
reports=[]
for f in assessments:
 saved=read_bound(artifact(f));row=saved['row']
 if row['outcome']['measurement_status']!='partial':continue
 manifest=read_bound(saved['manifest']);scene=read_bound(manifest['scene_definition']);attempt=read_bound(row['attempt'])
 assert saved['attempt']==row['attempt'];assert manifest['split']=='development'
 cell=next(c for c in manifest['cells'] if c['cell_id']==row['cell_id'])
 assert cell['runtime_seed']==93201 and row['mode']=='forced' and attempt['exit_status']!=0
 for r in row['raw_artifacts'].values():assert artifact(Path(r['path']))==r
 bank=collector.load_verified_registry(manifest['registry']['path'],manifest['registry']['sha256'])
 collector.validate_collection_context(cell,manifest,bank,scene,actual=True)
 fresh=collector.analyze_incomplete(cell,manifest,bank,scene,attempt,'nonzero_process_exit')
 assert fresh=={k:v for k,v in row.items() if k!='attempt'}
 assert fresh['outcome']['task_outcome']=='failure' and fresh['contact_audit']['complete_synchronized_streams']
 log=Path(cell['output'])/'rollout.log'
 reports.append(dict(assessment=artifact(f),manifest=saved['manifest'],attempt=row['attempt'],raw_artifacts=row['raw_artifacts'],recomputed_row_equals_original=True,measured_physics_steps=fresh['physics_steps'],outcome=fresh['outcome'],contact_audit=fresh['contact_audit'],log=artifact(log),terminal_error_lines=[l for l in log.read_text().splitlines() if l.startswith(('ValueError:','RuntimeError:'))][-3:]))
assert sorted(r['measured_physics_steps'] for r in reports)==[628,644,644]
report=dict(schema='motion2scene_bounded_partial_failure_reaudit_v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_id='seed93201_reference_contrast',budget=8,partial_captures=reports,implementation=[artifact(Path(collector.__file__)),artifact(Path(collector.classify_timed_attempt.__code__.co_filename)),artifact(Path(collector.audit_environment_contacts.__code__.co_filename))],new_physical_executions=0,new_fits=0,interpretation='Independent recomputation from the three hash-checked raw partial recordings, sensors and synchronized contact streams exactly reproduces their known task failures and recorded costs. Capture termination and measured physical events remain distinct. Later uncaptured fall, passage or recovery outcomes remain unknown. No retries, imputed full duration, unrelated raw re-audits, fitting or native changes.')
out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(output=artifact(out),partial_captures=[dict(steps=r['measured_physics_steps'],events=r['outcome']['physical_events'],errors=r['terminal_error_lines']) for r in reports])))
PY

