#!/usr/bin/env bash
# Run one scene-task mode over a list of task JSONs, sequentially.
# Usage: run_stage.sh <mode> <stage_dir> <seed> <student_or_-> [--takeover-tick N] [--motor CKPT] -- task.json...
# Each task lands in <stage_dir>/<task_id>/{config.json,command.json,native.log,task/task-result.json}.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=${M2S_KIT:-}
if [ -z "$KIT" ]; then d=$HERE; while [ "$d" != "/" ]; do [ -f "$d/pyproject.toml" ] && [ -d "$d/vendor/sonic" ] && KIT=$d && break; d=$(dirname "$d"); done; fi
[ -n "$KIT" ] || { echo "set M2S_KIT to the checkout root" >&2; exit 2; }
export NAV_PACKET=${NAV_PACKET:-$KIT/workspace/nav-8192}
PY="env -u PYTHONPATH PYTHONPATH=$KIT/vendor/sonic TRL_EXPERIMENTAL_SILENCE=1 CUDA_VISIBLE_DEVICES= $KIT/.venv_native/bin/python"
MODE=${1:?mode}; STAGE=${2:?stage dir}; SEED=${3:?seed}; STUDENT=${4:?student or -}
shift 4
EXTRA=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do EXTRA+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
mkdir -p "$STAGE"
for TASK in "$@"; do
  ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_id"])' "$TASK")
  OUT=$STAGE/$ID
  if [ -f "$OUT/task/task-result.json" ]; then echo "skip $ID"; continue; fi
  rm -rf "$OUT"; mkdir -p "$OUT"
  ARGS=(--mode "$MODE" --task "$TASK" --output "$OUT/task" --config "$OUT/config.json" "${EXTRA[@]}")
  [ "$STUDENT" != "-" ] && ARGS+=(--student "$STUDENT")
  TARGET=$($PY "$HERE/stage_config.py" "${ARGS[@]}" 2>/dev/null | tail -1)
  echo "=== $(date -u +%FT%TZ) $MODE $ID" >> "$STAGE/stage.log"
  bash "$HERE/scene_native.sh" "$OUT" "$TASK" "$SEED" "$TARGET" "$OUT/config.json" >> "$STAGE/stage.log" 2>&1
  code=$?
  if [ -f "$OUT/task/task-result.json" ]; then
    python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(sys.argv[2], "success" if s["navigation_success"] else "FAIL", "hold", s["max_hold_ticks"], "reach", s["goal_ever_reached"], "force", round(s["max_undesired_force_n"],2), "fell", s["fell"], "steps", s["control_steps"], "stop", s["stop_reason"])' "$OUT/task/task-result.json" "$ID" | tee -a "$STAGE/results.txt"
  else
    echo "$ID PROCESS_FAILED exit=$code" | tee -a "$STAGE/results.txt"
  fi
done
echo "STAGE COMPLETE $(date -u +%FT%TZ)" >> "$STAGE/stage.log"
