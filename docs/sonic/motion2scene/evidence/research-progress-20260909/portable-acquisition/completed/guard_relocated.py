"""Python-level trusted-reader isolation; not an operating-system sandbox."""
import os
from pathlib import Path
import runpy
import site
import sys
ROOT = Path('/home/linjiw/groot-wbc-sonic-sim-trackb')
DATA = Path('/home/linjiw/research-data/groot-wbc')
ALLOWED = [Path(__file__).resolve().parent, Path(__file__).resolve().parent / 'relocated', *map(Path, site.getsitepackages())]
IMMUTABLE = [Path(__file__).resolve().parent / 'relocated', Path(__file__).resolve().parent / 'relocated']
BLOCKED = {'torch', 'isaaclab', 'omni', 'pxr', 'joblib', 'mujoco'}
def guard(event, args):
    if event == 'import' and args[0].split('.')[0] in BLOCKED:
        raise PermissionError('Forbidden runtime import: ' + args[0])
    if event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename'}:
        for item in args[:2]:
            if isinstance(item, (str, bytes, os.PathLike)) and any(Path(os.fsdecode(item)).resolve().is_relative_to(p) for p in IMMUTABLE):
                raise PermissionError('Mutation of read-only release: ' + os.fsdecode(item))
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        value = Path(os.fsdecode(args[0])).resolve()
        mode = args[1] or ''
        flags = args[2] if len(args) > 2 else 0
        if (any(c in mode for c in 'wax+') or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)) and any(value.is_relative_to(p) for p in IMMUTABLE):
            raise PermissionError('Write to read-only release: ' + str(value))
        if (value.is_relative_to(ROOT) or value.is_relative_to(DATA)) and not any(value.is_relative_to(p) for p in ALLOWED):
            raise PermissionError('Forbidden original path: ' + str(value))
sys.addaudithook(guard)
for forbidden in [ROOT / 'README.md', DATA / 'm2s-acquisition-M2-portable-20260909-v1/release.json', DATA / 'm2s-primary-acquisition-plan-tie-proposed-v5/plan.json']:
    try:
        forbidden.read_bytes()
    except PermissionError:
        pass
    else:
        raise RuntimeError('Original-path guard is inactive')
script = Path(sys.argv[1]).resolve()
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name='__main__')
