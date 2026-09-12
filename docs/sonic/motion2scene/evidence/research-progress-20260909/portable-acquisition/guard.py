"""Python-level trusted-reader isolation; not an operating-system sandbox."""
import os
from pathlib import Path
import runpy
import sys
ROOT = Path('/home/linjiw/groot-wbc-sonic-sim-trackb')
DATA = Path('/home/linjiw/research-data/groot-wbc')
ALLOWED = [Path(__file__).resolve().parent, DATA / 'm2s-acquisition-M2-portable-20260909-v1']
BLOCKED = {'torch', 'isaaclab', 'omni', 'pxr', 'joblib', 'mujoco'}
def guard(event, args):
    if event == 'import' and args[0].split('.')[0] in BLOCKED:
        raise PermissionError('Forbidden runtime import: ' + args[0])
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        value = Path(os.fsdecode(args[0])).resolve()
        if (value.is_relative_to(ROOT) or value.is_relative_to(DATA)) and not any(value.is_relative_to(p) for p in ALLOWED):
            raise PermissionError('Forbidden original path: ' + str(value))
sys.addaudithook(guard)
script = Path(sys.argv[1]).resolve()
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name='__main__')
