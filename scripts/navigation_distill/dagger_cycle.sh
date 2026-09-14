#!/usr/bin/env bash
# One short-cycle navigation DAgger round for one arm.
# Usage: dagger_cycle.sh <arm> <cycle> <current_nav_ckpt> <fractions "0.12 0.25"> [extra nav_fit_config args...]
# 1. early-switch recoveries driven by the current adapter on the teacher-feasible tasks
# 2. refit from the current adapter on every supported recovery so far (fresh = this cycle) + motor demos
# 3. unassisted panel on the 19 teacher-feasible tasks at seed 91260
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${NAV_PACKET:?set NAV_PACKET}"
export NAV_PACKET=$PWD M2S_TIMEOUT=3600
K=${M2S_KIT:?set M2S_KIT}
PY="env -u PYTHONPATH PYTHONPATH=$K/vendor/sonic TRL_EXPERIMENTAL_SILENCE=1 $K/.venv_native/bin/python"
MOTOR=${NAV_MOTOR:?set NAV_MOTOR}
ARM=${1:?arm}; C=${2:?cycle}; NAV=${3:?current nav ckpt}; FRACS=${4:?fractions}; shift 4
D=dagger/$ARM; mkdir -p $D
SHA=$(sha256sum "$NAV" | cut -d' ' -f1)
echo "=== cycle $C start $(date -u +%FT%TZ) nav=$NAV" >> $D/cycle.log
python3 "$HERE"/plan_recovery.py --tasks tasks/manifest.json --reference collect/teacher-91260 --fractions $FRACS --out $D/c$C-plan.txt >> $D/cycle.log
"$HERE"/run_recovery.sh $PWD/$D/c$C-recovery 91260 "$NAV" $MOTOR $D/c$C-plan.txt >> $D/cycle.log 2>&1
STAGES="collect/motor-demo-91260 collect/recovery-v1-91260"
for k in $(seq 1 $C); do STAGES="$STAGES $D/c$k-recovery"; done
$PY "$HERE"/aggregate_recovery.py --stages $STAGES --output $D/c$C-manifest.json >> $D/cycle.log 2>&1
$PY "$HERE"/nav_fit_config.py --motor $MOTOR --manifest $D/c$C-manifest.json --view motor_recovery --task-manifest tasks/manifest.json \
  --warm-start "$NAV" --recovery-fraction 0.5 --fresh-behavior $SHA --updates 3000 --seed $((91480 + C)) \
  --purpose "$ARM cycle $C: early-switch DAgger, fresh recoveries from $SHA" "$@" --out $D/c$C-fit-config.json >> $D/cycle.log 2>&1
( cd $K/vendor/sonic && $PY -m gear_sonic.research.scene_distillation.navigation_motor --config $PWD/$D/c$C-fit-config.json --output $PWD/$D/c$C-fit > $PWD/$D/c$C-fit.log 2>&1 ) || { echo "FIT FAILED cycle $C" >> $D/cycle.log; exit 1; }
NEXT=$PWD/$D/c$C-fit/step-003000.pt
T=$(python3 -c "
import json
ok={f.split('/')[-3] for f in __import__('glob').glob('collect/teacher-91260/*/task/task-result.json') if json.load(open(f))['navigation_success']}
print(' '.join(t['path'] for t in json.load(open('tasks/manifest.json'))['tasks'] if t['task_id'] in ok))")
"$HERE"/run_stage.sh nav $PWD/eval/$ARM-c$C-91260 91260 $NEXT -- $T >> $D/cycle.log 2>&1
echo "=== cycle $C done $(date -u +%FT%TZ) next=$NEXT result=$(grep -c success eval/$ARM-c$C-91260/results.txt)/19" >> $D/cycle.log
echo "$NEXT"
