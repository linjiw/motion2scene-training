#!/usr/bin/env bash
# Fresh Linux/Blackwell environment; never modifies an existing virtual environment.
set -euo pipefail
KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-$KIT_ROOT/.venv_native}"
LAB_DIR="${2:-$KIT_ROOT/external/IsaacLab}"
if [[ -e "$ENV_DIR" ]]; then
  echo "Refusing existing environment: $ENV_DIR. Choose a new directory." >&2
  exit 1
fi
command -v uv >/dev/null || { echo 'Install uv first (python -m pip install uv).' >&2; exit 1; }
# Legacy sdists (e.g. Isaac Lab's flatdict==4.0.1) import pkg_resources, removed in setuptools 81.
export UV_BUILD_CONSTRAINT="$KIT_ROOT/requirements/build-constraints.txt"
if [[ ! -d "$LAB_DIR" ]]; then
  git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git "$LAB_DIR"
fi
[[ "$(git -C "$LAB_DIR" rev-parse HEAD)" == "$(git -C "$LAB_DIR" rev-parse v2.3.2^{commit})" ]] || {
  echo 'Isaac Lab checkout must be v2.3.2; refusing another revision.' >&2; exit 1;
}
uv venv --python 3.11 "$ENV_DIR"
NATIVE_PY="$ENV_DIR/bin/python"
uv pip install --python "$NATIVE_PY" 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
for package in isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks; do
  uv pip install --python "$NATIVE_PY" -e "$LAB_DIR/source/$package"
done
uv pip install --python "$NATIVE_PY" -c "$KIT_ROOT/requirements/native-constraints.txt" \
  -e "$KIT_ROOT/vendor/sonic/gear_sonic[training]" \
  mujoco pin pyarrow imageio imageio-ffmpeg opencv-python-headless matplotlib pandas pytest rich
# Enforce the CUDA build AFTER every dependency-changing install.
uv pip install --python "$NATIVE_PY" --index-url https://download.pytorch.org/whl/cu128 \
  --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
  'torch==2.7.0' 'torchvision==0.22.0' 'torchaudio==2.7.0'
uv pip install --python "$NATIVE_PY" --no-deps -e "$KIT_ROOT"
"$NATIVE_PY" - <<'PY'
import torch
assert torch.__version__ == '2.7.0+cu128', torch.__version__
assert torch.cuda.is_available(), 'CUDA unavailable'
x = torch.ones(128, 128, device='cuda')
assert float((x @ x)[0, 0]) == 128
print('Final CUDA build and kernel probe passed')
PY
uv pip check --python "$NATIVE_PY" > "$ENV_DIR/dependency-check.txt" 2>&1 || true
cat "$ENV_DIR/dependency-check.txt"
echo 'Setup completed; inspect dependency-check.txt. See docs/SETUP.md for known metadata conflicts.'
echo 'Run m2s doctor --profile teacher --cuda using this environment after unpacking data.'
echo 'A successful install/probe does not certify a full Isaac training launch.'
