#!/usr/bin/env bash
# One single-environment scene-task launch through eval_agent_trl.py.
# Usage: scene_native.sh <out_dir> <task.json> <seed> <callback_target> <stage_config.json>
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=${M2S_KIT:-}
if [ -z "$KIT" ]; then d=$HERE; while [ "$d" != "/" ]; do [ -f "$d/pyproject.toml" ] && [ -d "$d/vendor/sonic" ] && KIT=$d && break; d=$(dirname "$d"); done; fi
[ -n "$KIT" ] || { echo "set M2S_KIT to the checkout root" >&2; exit 2; }
PACKET=${NAV_PACKET:-$KIT/workspace/nav-8192}
PY=$KIT/.venv_native/bin/python
OUT=${1:?out}; TASK=${2:?task}; SEED=${3:?seed}; TARGET=${4:?callback}; STAGE=${5:?stage config}
[ -e "$OUT/task" ] && { echo "refusing to overwrite $OUT/task" >&2; exit 3; }
TEACHER=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_checkpoint"])' "$PACKET/ids.json")
SCENE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["scene_usd_path"])' "$TASK")
MOTIONS=$(dirname "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["native_motion"]["path"])' "$TASK")")
mkdir -p "$OUT" "$HOME/.cache/m2s/tmp" "$HOME/.cache/m2s/isaaclab-usd"
ARGS=(
  gear_sonic/eval_agent_trl.py
  "checkpoint=$TEACHER"
  "++headless=true"
  "++num_envs=1"
  "++seed=$SEED"
  "++use_wandb=false"
  "++use_encoder=g1"
  "++eval_output_dir=$OUT/unused_metrics"
  "++eval_base_dir=$OUT/hydra"
  "++eval_callbacks=[im_eval]"
  "++run_eval_loop=false"
  "++trainer.schedule_dict=null"
  "++manager_env.config.terrain_type=scene_usd"
  "++manager_env.config.render_results=false"
  "++manager_env.config.render_ego=false"
  "++manager_env.commands.motion.debug_vis=false"
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTIONS"
  "++manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load=1"
  "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true"
  "++manager_env.commands.motion.motion_lib_cfg.multi_thread=false"
  "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=false"
  "++manager_env.commands.motion.cat_upper_body_poses=false"
  "++manager_env.commands.motion.freeze_frame_aug=false"
  "++callbacks.im_eval._target_=$TARGET"
  "++manager_env._target_=gear_sonic.research.scene_distillation.scene_env.SceneQualificationEnvCfg"
  "++manager_env.config.scene_usd_path=$SCENE"
  "++manager_env.config.navigation_task_path=$TASK"
  "++callbacks.im_eval.stage_config=$STAGE"
)
"$PY" -c 'import json,sys; print(json.dumps(sys.argv[1:], indent=2))' "$PY" "${ARGS[@]}" > "$OUT/command.json"
START=$(date +%s)
cd "$KIT/vendor/sonic" || exit 2
env -u PYTHONPATH \
  TMPDIR="$HOME/.cache/m2s/tmp" \
  ISAACLAB_USD_CACHE_DIR="$HOME/.cache/m2s/isaaclab-usd" \
  PYTHONPATH="$KIT/vendor/sonic" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  timeout -s KILL "${M2S_TIMEOUT:-1500}" "$PY" "${ARGS[@]}" > "$OUT/native.log" 2>&1
CODE=$?
END=$(date +%s)
printf '{"exit_code": %d, "wall_seconds": %d}\n' "$CODE" "$((END - START))" > "$OUT/process-result.json"
echo "exit=$CODE wall_s=$((END - START)) out=$OUT"
exit "$CODE"
