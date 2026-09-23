#!/usr/bin/env bash
# nav8192-confirm-v1 Stage 1: held-out-seed confirmation of the short-cycle DAgger arms.
# Usage: launch.sh A|B|probe   (run each inside its own tmux session)
# Registration: experiments/nav8192-confirm-v1/registration.json (committed before launch).
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export M2S_KIT=$KIT M2S_TIMEOUT=${M2S_TIMEOUT:-3600}
RUN="$KIT/scripts/navigation_distill/run_stage.sh"
W=$KIT/workspace/nav-8192
OUT=$KIT/workspace/confirm-v1/eval
mkdir -p "$OUT"
FEAS=$(ls -d "$W"/eval/dag-approach-c2-91260/0* | xargs -n1 basename)
task_paths() { # map task ids (e.g. 00908-stop-clear) to task JSON paths
  for id in "$@"; do m=${id%%-stop-*}; v=${id##*-stop-}; echo "$W/tasks/$m/$v.json"; done
}
FEAS_T=$(task_paths $FEAS)
ALL_T=$(python3 -c "import json;print(' '.join(t['path'] for t in json.load(open('$W/tasks/manifest.json'))['tasks']))")
APP=$W/dagger/dag-approach/c2-fit/step-003000.pt
UNI=$W/dagger/dag-uniform/c2-fit/step-003000.pt
REC=$W/fit/nav-v2-recovery/step-003000.pt
nav() { NAV_PACKET=$W "$RUN" nav "$OUT/$1-$2" "$2" "$3" -- $FEAS_T; }
teach() { NAV_PACKET=$W "$RUN" teacher "$OUT/teacher8192-$1" "$1" - -- $ALL_T; }
release() { NAV_PACKET=$KIT/workspace/nav-release "$RUN" teacher "$OUT/release-$1" "$1" - -- $ALL_T; }
probe() {
  local PT; PT=$(task_paths 00908-stop-clear 00677-stop-corridor)
  NAV_PACKET=$W "$RUN" nav "$OUT/probe-approach-c2-91260" 91260 "$APP" -- $PT
  local NOM="++eval_remove_events=[physics_material,add_joint_default_pos,base_com,randomize_rigid_body_mass,push_robot,compliance_force_push]"
  for sd in 92601 92602; do M2S_EXTRA_ARGS="$NOM" NAV_PACKET=$W "$RUN" nav "$OUT/probe-approach-c2-nominal-$sd" $sd "$APP" -- $PT; done
}
case "${1:?A|B|probe|dry}" in
  A)
    for sd in 92601 92602 92603; do nav approach-c2 $sd "$APP"; nav uniform-c2 $sd "$UNI"; done
    nav recovery 92603 "$REC" ;;
  B)
    nav recovery 92601 "$REC"; teach 92601; release 92601
    nav recovery 92602 "$REC"; teach 92602; teach 92603; probe ;;
  probe) probe ;;
  dry) echo "$FEAS_T" | wc -w; echo "$ALL_T" | wc -w; for t in $FEAS_T $ALL_T; do [ -f "$t" ] || echo "missing $t"; done; exit 0 ;;
esac
echo "CHAIN $1 COMPLETE $(date -u +%FT%TZ)" >> "$OUT/chains.log"
