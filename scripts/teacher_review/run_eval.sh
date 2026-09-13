#!/usr/bin/env bash
# One native Isaac Lab evaluation for the teacher-8192-500 review.
# Usage: run_eval.sh <release|trained500|previous8000> <development|train> [callback_target] [out_suffix]
# Mirrors workspace/m2s-repaired-teacher-step6200-comparison-20260912/current/command.json
# with the scripts/m2s_native.sh environment (unset PYTHONPATH, user-owned TMPDIR and USD cache).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=${M2S_KIT:-$(cd "$HERE/../.." && pwd)}
REVIEW=${REVIEW:-$KIT/workspace/teacher-8192-500-review}
PY=$KIT/.venv_native/bin/python
ARM=${1:?arm}; SPLIT=${2:?split}
TARGET=${3:-gear_sonic.research.hindsight_training.pose_capture.PoseCaptureQualificationCallback}
SUFFIX=${4:-}
case "$ARM" in
  release) CKPT=model_step_041550.pt ;;
  trained500) CKPT=model_step_000500.pt ;;
  previous8000) CKPT=model_step_008000.pt ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
case "$SPLIT" in
  development) N=20 ;;
  train) N=89 ;;
  *) echo "unknown split $SPLIT" >&2; exit 2 ;;
esac
STAGE=$REVIEW/eval/$ARM
OUT=$STAGE/$SPLIT$SUFFIX
[ -f "$STAGE/config.yaml" ] && [ -e "$STAGE/$CKPT" ] || { echo "not staged: $STAGE" >&2; exit 2; }
[ -e "$OUT/metrics" ] && { echo "refusing to overwrite $OUT/metrics" >&2; exit 3; }
if pgrep -u "$(id -u)" -f "eval_agent_trl.py|train_agent_trl.py" >/dev/null; then
  echo "another Isaac eval/train process is running; run launches sequentially" >&2; exit 4
fi
mkdir -p "$OUT" "$HOME/.cache/m2s/tmp" "$HOME/.cache/m2s/isaaclab-usd"

ARGS=(
  gear_sonic/eval_agent_trl.py
  "checkpoint=$STAGE/$CKPT"
  "++headless=true"
  "++num_envs=$N"
  "++seed=91231"
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
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=$KIT/workspace/teacher-8192-500/motions/$SPLIT"
  "++manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load=$N"
  "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true"
  "++manager_env.commands.motion.motion_lib_cfg.multi_thread=false"
  "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=false"
  "++manager_env.commands.motion.cat_upper_body_poses=false"
  "++manager_env.commands.motion.freeze_frame_aug=false"
  "++callbacks.im_eval._target_=$TARGET"
  "++callbacks.im_eval.max_eval_steps=null"
)
"$PY" -c 'import json,sys; print(json.dumps(sys.argv[1:], indent=2))' "$PY" "${ARGS[@]}" > "$OUT/command.json"

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$OUT/gpu-before.csv" 2>&1
( while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader >> "$OUT/gpu.csv" 2>/dev/null
    sleep 2
  done ) &
SAMPLER=$!
START=$(date +%s.%N)
cd "$KIT/vendor/sonic" || exit 2
env -u PYTHONPATH \
  TMPDIR="$HOME/.cache/m2s/tmp" \
  ISAACLAB_USD_CACHE_DIR="$HOME/.cache/m2s/isaaclab-usd" \
  PYTHONPATH="$KIT/vendor/sonic" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  timeout 2400 "$PY" "${ARGS[@]}" > "$OUT/evaluation.log" 2>&1
CODE=$?
END=$(date +%s.%N)
kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
"$PY" - "$OUT" "$CODE" "$START" "$END" "$STAGE/$CKPT" <<'PYEOF'
import json, sys, os
out, code, start, end, ckpt = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
peak = baseline = None
try:
    baseline = int(open(f"{out}/gpu-before.csv").read().split(",")[1].split()[0])
    peak = max(int(l.split(",")[1].split()[0]) for l in open(f"{out}/gpu.csv") if l.strip())
except Exception:
    pass
json.dump({"exit_code": code, "timed_out": code == 124, "wall_seconds": end - start,
           "checkpoint": ckpt, "checkpoint_target": os.path.realpath(ckpt),
           "gpu_total_mib_before": baseline, "gpu_total_mib_peak": peak},
          open(f"{out}/receipt.json", "w"), indent=2)
PYEOF
echo "exit=$CODE wall_s=$(awk "BEGIN{print $END - $START}") out=$OUT"
[ "$CODE" -eq 0 ] || exit "$CODE"
env -u PYTHONPATH "$PY" "$HERE/check_run.py" --review "$REVIEW" "$ARM" "$SPLIT" --suffix "$SUFFIX"
