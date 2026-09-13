#!/usr/bin/env bash
# One native Isaac Lab launch through eval_agent_trl.py for a distillation packet ($DISTILL_PACKET).
# Usage: native.sh <out_dir> <train|development> <num_envs> <seed> <callback_target> [extra ++overrides...]
# Mirrors scripts/teacher_review/run_eval.sh (env -u PYTHONPATH, user-owned TMPDIR/USD cache).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=${M2S_KIT:-$(cd "$HERE/../.." && pwd)}
PACKET=${DISTILL_PACKET:-$KIT/workspace/distill-8192}
PY=$KIT/.venv_native/bin/python
OUT=${1:?out}; SPLIT=${2:?split}; N=${3:?num_envs}; SEED=${4:?seed}; TARGET=${5:?callback}
shift 5
case "$SPLIT" in
  train) MOTIONS=89 ;;
  development) MOTIONS=20 ;;
  *) echo "unknown split $SPLIT" >&2; exit 2 ;;
esac
[ -e "$OUT/metrics" ] && { echo "refusing to overwrite $OUT/metrics" >&2; exit 3; }
if [ -z "${M2S_ALLOW_CONCURRENT:-}" ] && pgrep -u "$(id -u)" -f "eval_agent_trl.py|train_agent_trl.py" >/dev/null; then
  echo "another Isaac process is running; launch sequentially" >&2; exit 4
fi
mkdir -p "$OUT" "$HOME/.cache/m2s/tmp" "$HOME/.cache/m2s/isaaclab-usd"
ARGS=(
  gear_sonic/eval_agent_trl.py
  "checkpoint=$PACKET/teacher/model_step_000500.pt"
  "++headless=true"
  "++num_envs=$N"
  "++seed=$SEED"
  "++use_wandb=false"
  "++use_encoder=g1"
  "++eval_output_dir=$OUT/metrics"
  "++eval_base_dir=$OUT/hydra"
  "++eval_callbacks=[im_eval]"
  "++run_eval_loop=false"
  "++trainer.schedule_dict=null"
  "++manager_env.config.terrain_type=plane"
  "++manager_env.config.render_results=false"
  "++manager_env.config.render_ego=false"
  "++manager_env.commands.motion.debug_vis=false"
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=$PACKET/motions/$SPLIT"
  "++manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load=$MOTIONS"
  "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true"
  "++manager_env.commands.motion.motion_lib_cfg.multi_thread=false"
  "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=false"
  "++manager_env.commands.motion.cat_upper_body_poses=false"
  "++manager_env.commands.motion.freeze_frame_aug=false"
  "++callbacks.im_eval._target_=$TARGET"
  "++callbacks.im_eval.max_eval_steps=null"
  "$@"
)
"$PY" -c 'import json,sys; print(json.dumps(sys.argv[1:], indent=2))' "$PY" "${ARGS[@]}" > "$OUT/command.json"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader > "$OUT/gpu-before.csv" 2>&1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$OUT/gpu-before.csv" 2>&1
( while true; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
      | sed "s/^/$(date +%s),/" >> "$OUT/gpu-apps.csv"
    sleep 5
  done ) &
SAMPLER=$!
START=$(date +%s)
cd "$KIT/vendor/sonic" || exit 2
env -u PYTHONPATH \
  TMPDIR="$HOME/.cache/m2s/tmp" \
  ISAACLAB_USD_CACHE_DIR="$HOME/.cache/m2s/isaaclab-usd" \
  PYTHONPATH="$KIT/vendor/sonic" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  timeout "${M2S_TIMEOUT:-9000}" "$PY" "${ARGS[@]}" > "$OUT/native.log" 2>&1 &
CHILD=$!
echo "$CHILD" > "$OUT/pid"
wait "$CHILD"
CODE=$?
END=$(date +%s)
kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
printf '{"exit_code": %d, "wall_seconds": %d, "timed_out": %s}\n' "$CODE" "$((END - START))" \
  "$([ "$CODE" -eq 124 ] && echo true || echo false)" > "$OUT/receipt.json"
echo "exit=$CODE wall_s=$((END - START)) out=$OUT"
exit "$CODE"
