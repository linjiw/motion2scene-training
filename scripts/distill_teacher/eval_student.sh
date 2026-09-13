#!/usr/bin/env bash
# Unassisted nominal-start motor evaluation. Usage: eval_student.sh <checkpoint|teacher> <train|development> <seed> <out>
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=${M2S_KIT:-$(cd "$HERE/../.." && pwd)}
PACKET=${DISTILL_PACKET:-$KIT/workspace/distill-8192}
CKPT=${1:?checkpoint}; SPLIT=${2:?split}; SEED=${3:?seed}; OUT=${4:?out}
case "$SPLIT" in train) N=89 ;; development) N=20 ;; *) exit 2 ;; esac
SHA=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_sha256"])' "$PACKET/ids.json")
if [ "$CKPT" = teacher ]; then
  exec "$HERE/native.sh" "$OUT" "$SPLIT" "$N" "$SEED" gear_sonic.research.hindsight_training.qualify.TrackingQualificationCallback
fi
exec "$HERE/native.sh" "$OUT" "$SPLIT" "$N" "$SEED" \
  gear_sonic.research.scene_distillation.motor_runtime.MotorEvaluationCallback \
  "++callbacks.im_eval.student_checkpoint=$CKPT" "++callbacks.im_eval.teacher_sha256=$SHA"
