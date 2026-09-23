#!/usr/bin/env bash
# Phase 0.4 goal/map-use ablation (docs/ROADMAP_20260923.md §8) for one navigation checkpoint
# at one physics seed, on the 19 teacher-feasible tasks registered in
# experiments/nav8192-confirm-v1/registration.json (task sha256s are checked).
#
# Usage: run_goal_ablation.sh [--dry] <nav_ckpt> <seed> [out_root]
#   rot90 arm: run_stage.sh nav --goal-rotation-deg 90 -> <out_root>/goal-rot90-<seed>
#   zero arm:  run_stage.sh nav --zero-obstacles       -> <out_root>/zero-obstacles-<seed>
#   then analyze_goal_ablation.py on each panel (CPU) -> <panel>/goal-ablation.json
# Env: ARMS="rot90 zero" (default both, in that order); NAV_PACKET (default workspace/nav-8192);
#   BASELINE=<unablated panel, same checkpoint and seed> (default: the matching
#   workspace/confirm-v1 panel when the checkpoint is a registered arm and that panel exists);
#   M2S_TIMEOUT (default 3600 s per task); M2S_ALLOW_CONCURRENT=1 to launch next to >=2 other
#   Isaac evaluations; M2S_PYTHON for the CPU steps (default .venv_native; Isaac launches
#   always use run_stage.sh's). Default out_root: workspace/phase0/0.4-goal-ablation/<sha12>.
# --dry validates the checkpoint, tasks and stage configs in a temp dir and launches nothing.
# This launches Isaac Sim (one GPU process per task, ~4 GB each). Scoring always uses the
# ORIGINAL goal; the ablation changes only what the adapter observes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=${M2S_KIT:-}
if [ -z "$KIT" ]; then d=$HERE; while [ "$d" != "/" ]; do [ -f "$d/pyproject.toml" ] && [ -d "$d/vendor/sonic" ] && KIT=$d && break; d=$(dirname "$d"); done; fi
[ -n "$KIT" ] || { echo "set M2S_KIT to the checkout root" >&2; exit 2; }
export M2S_KIT=$KIT M2S_TIMEOUT=${M2S_TIMEOUT:-3600}
export NAV_PACKET=${NAV_PACKET:-$KIT/workspace/nav-8192}
PY="env -u PYTHONPATH PYTHONPATH=$KIT/vendor/sonic TRL_EXPERIMENTAL_SILENCE=1 CUDA_VISIBLE_DEVICES= ${M2S_PYTHON:-$KIT/.venv_native/bin/python}"
DRY=0; [ "${1:-}" = "--dry" ] && { DRY=1; shift; }
NAV=$(realpath "${1:?nav checkpoint}"); SEED=${2:?seed}
[ -f "$NAV" ] || { echo "missing checkpoint $NAV" >&2; exit 2; }
SHA=$(sha256sum "$NAV" | cut -d' ' -f1)
OUT=${3:-$KIT/workspace/phase0/0.4-goal-ablation/${SHA:0:12}}
REG=$KIT/experiments/nav8192-confirm-v1/registration.json
ARMS=${ARMS:-rot90 zero}

# Feasible task paths from the registration; refuse any task whose bytes changed.
TASKS=$(python3 - "$REG" "$NAV_PACKET" <<'PY'
import hashlib, json, sys
reg, packet = json.load(open(sys.argv[1])), sys.argv[2]
paths = []
for task_id in reg["tasks"]["feasible_19"]:
    motion, variant = task_id.split("-stop-")
    path = f"{packet}/tasks/{motion}/{variant}.json"
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != reg["tasks"]["task_sha256"][task_id]:
        sys.exit(f"task changed since registration: {path}")
    paths.append(path)
print(" ".join(paths))
PY
) || exit 2
# The registered arm (if any) with this checkpoint, for the default unablated baseline.
ARM=$(python3 -c 'import json,sys; print(next((k for k,v in json.load(open(sys.argv[1]))["arms"].items() if v["sha256"]==sys.argv[2]), ""))' "$REG" "$SHA")
if [ -z "${BASELINE:-}" ]; then
  case "$ARM" in
    dag-approach-c2) B=approach-c2 ;; dag-uniform-c2) B=uniform-c2 ;; nav-v2-recovery) B=recovery ;; *) B= ;;
  esac
  [ -n "$B" ] && [ -d "$KIT/workspace/confirm-v1/eval/$B-$SEED" ] && BASELINE=$KIT/workspace/confirm-v1/eval/$B-$SEED
fi
BASELINE=${BASELINE:-}

flags() { case "$1" in rot90) echo "--goal-rotation-deg 90" ;; zero) echo "--zero-obstacles" ;; *) return 1 ;; esac; }
panel() { case "$1" in rot90) echo "$OUT/goal-rot90-$SEED" ;; zero) echo "$OUT/zero-obstacles-$SEED" ;; esac; }

# Preflight: every arm's stage config must carry its flag (run_stage.sh hides config errors).
CHECK=$(mktemp -d)
for A in $ARMS; do
  F=$(flags "$A") || { echo "unknown arm $A (use rot90, zero)" >&2; rm -rf "$CHECK"; exit 2; }
  # shellcheck disable=SC2086
  $PY "$HERE/stage_config.py" --mode nav --task "${TASKS%% *}" --output "$CHECK/$A/task" \
    --config "$CHECK/$A.json" --student "$NAV" $F > /dev/null || { rm -rf "$CHECK"; exit 2; }
  python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c.get("goal_rotation_deg")==90.0 or c.get("zero_obstacles") is True, c' "$CHECK/$A.json" || { rm -rf "$CHECK"; exit 2; }
done
rm -rf "$CHECK"

echo "checkpoint $NAV sha256 $SHA (registered arm: ${ARM:-none})"
echo "seed $SEED; arms: $ARMS; tasks: $(echo "$TASKS" | wc -w); out $OUT"
echo "baseline for pairing: ${BASELINE:-none}"
for A in $ARMS; do echo "  $A: $(flags "$A") -> $(panel "$A")"; done
[ "$DRY" = 1 ] && { echo "dry run: nothing launched"; exit 0; }

# Count the Isaac python processes only (not their `timeout` wrappers or shells).
RUNNING=$(pgrep -fc '^[^ ]*/python[0-9.]* gear_sonic/eval_agent_trl[.]py' || true)
if [ "${RUNNING:-0}" -ge 2 ] && [ "${M2S_ALLOW_CONCURRENT:-0}" != 1 ]; then
  echo "$RUNNING Isaac evaluations already running; set M2S_ALLOW_CONCURRENT=1 to add another" >&2
  exit 3
fi
mkdir -p "$OUT"
python3 - "$OUT/ablation-plan-$SEED.json" "$NAV" "$SHA" "$SEED" "$ARM" "$BASELINE" "$ARMS" \
  "$(git -C "$KIT" rev-parse HEAD 2>/dev/null)" $TASKS <<'PY'
import json, sys, time
out, nav, sha, seed, arm, baseline, arms, commit, *tasks = sys.argv[1:]
json.dump(dict(item="phase0-0.4-goal-map-ablation", checkpoint=nav, sha256=sha, seed=int(seed),
               registered_arm=arm or None, baseline=baseline or None, arms=arms.split(),
               harness_commit=commit or None, tasks=tasks,
               rule="adapter ignores the goal if final XY stays within 0.5 m of the reference "
                    "endpoint in >=70% of tasks (goal rotated +90 deg about the start)",
               launched=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
          open(out, "w"), indent=2)
PY
for A in $ARMS; do
  P=$(panel "$A")
  # shellcheck disable=SC2086
  "$HERE/run_stage.sh" nav "$P" "$SEED" "$NAV" $(flags "$A") -- $TASKS
  $PY "$HERE/analyze_goal_ablation.py" --panel "$P" ${BASELINE:+--baseline "$BASELINE"} \
    --output "$P/goal-ablation.json" | tee "$P/goal-ablation.txt"
done
echo "GOAL ABLATION COMPLETE $(date -u +%FT%TZ)" >> "$OUT/chains.log"
