#!/usr/bin/env bash
# Learner-prefix motor recoveries from a plan file of "task.json takeover_tick" lines.
# Usage: run_recovery.sh <stage_dir> <seed> <navigation_ckpt> <motor_ckpt> <plan.txt>
# Each attempt lands in <stage_dir>/<task_id>-t<tick>/task/{recovery.json,trace.npz,motor-recovery.npz}.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=${M2S_KIT:-}
if [ -z "$KIT" ]; then d=$HERE; while [ "$d" != "/" ]; do [ -f "$d/pyproject.toml" ] && [ -d "$d/vendor/sonic" ] && KIT=$d && break; d=$(dirname "$d"); done; fi
[ -n "$KIT" ] || { echo "set M2S_KIT to the checkout root" >&2; exit 2; }
export NAV_PACKET=${NAV_PACKET:-$KIT/workspace/nav-8192}
PY="env -u PYTHONPATH PYTHONPATH=$KIT/vendor/sonic TRL_EXPERIMENTAL_SILENCE=1 CUDA_VISIBLE_DEVICES= $KIT/.venv_native/bin/python"
STAGE=${1:?stage}; SEED=${2:?seed}; NAV=${3:?navigation ckpt}; MOTOR=${4:?motor ckpt}; PLAN=${5:?plan}
mkdir -p "$STAGE"
while read -r TASK TICK; do
  [ -z "$TASK" ] && continue
  ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_id"])' "$TASK")-t$TICK
  OUT=$STAGE/$ID
  if [ -f "$OUT/task/recovery.json" ]; then echo "skip $ID"; continue; fi
  rm -rf "$OUT"; mkdir -p "$OUT"
  TARGET=$($PY "$HERE/stage_config.py" --mode recovery --task "$TASK" --output "$OUT/task" --config "$OUT/config.json" \
             --student "$NAV" --motor "$MOTOR" --takeover-tick "$TICK" 2>/dev/null | tail -1)
  echo "=== $(date -u +%FT%TZ) recovery $ID" >> "$STAGE/stage.log"
  bash "$HERE/scene_native.sh" "$OUT" "$TASK" "$SEED" "$TARGET" "$OUT/config.json" >> "$STAGE/stage.log" 2>&1
  code=$?
  if [ -f "$OUT/task/recovery.json" ]; then
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); s=r["suffix_score"] or {}; print(sys.argv[2], "SUPPORTED" if r["supported"] else "unsupported", "rows", r["supported_rows"], "hold", s.get("max_hold_ticks"), "dist", round(s.get("final_goal_distance_m",-1),2), "force", round(s.get("max_undesired_force_n",-1),2))' "$OUT/task/recovery.json" "$ID" | tee -a "$STAGE/results.txt"
  else
    echo "$ID PROCESS_FAILED exit=$code" | tee -a "$STAGE/results.txt"
  fi
done < "$PLAN"
echo "STAGE COMPLETE $(date -u +%FT%TZ)" >> "$STAGE/stage.log"
