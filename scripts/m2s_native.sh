#!/usr/bin/env bash
# Run the native m2s CLI with a clean, user-owned process environment.
# Usage: scripts/m2s_native.sh doctor --profile teacher --cuda
set -euo pipefail
KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NATIVE_ENV="${M2S_NATIVE_ENV:-$KIT_ROOT/.venv_native}"
# Host Python paths (e.g. a sourced ROS distribution) shadow venv packages and break pytest.
unset PYTHONPATH
# Isaac Lab logs under tempfile.gettempdir()/isaaclab; on shared hosts /tmp/isaaclab may belong
# to another user. TMPDIR must exist, or Python silently falls back to /tmp.
export TMPDIR="${M2S_TMPDIR:-$HOME/.cache/m2s/tmp}"
# Keep robot URDF-to-USD conversions separate from other Isaac Lab checkouts.
export ISAACLAB_USD_CACHE_DIR="${ISAACLAB_USD_CACHE_DIR:-$HOME/.cache/m2s/isaaclab-usd}"
mkdir -p "$TMPDIR" "$ISAACLAB_USD_CACHE_DIR"
exec "$NATIVE_ENV/bin/m2s" "$@"
