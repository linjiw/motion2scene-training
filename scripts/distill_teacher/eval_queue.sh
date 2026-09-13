#!/usr/bin/env bash
# Sequential evaluation queue; lines "checkpoint split seed out" from a queue file, appended over time.
# Skips entries whose receipt exists. Runs alongside online training (M2S_ALLOW_CONCURRENT=1).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=${M2S_KIT:-$(cd "$HERE/../.." && pwd)}
PACKET=${DISTILL_PACKET:-$KIT/workspace/distill-8192}
cd "$PACKET"
QUEUE=${1:?queue}
export M2S_ALLOW_CONCURRENT=1
while true; do
  ran=0
  while read -r ckpt split seed out; do
    [ -z "$ckpt" ] && continue
    [ -f "$out/receipt.json" ] && continue
    echo "=== $(date -u +%FT%TZ) eval $ckpt $split $seed" >> eval/queue.log
    "$HERE/eval_student.sh" "$ckpt" "$split" "$seed" "$out" >> eval/queue.log 2>&1
    python3 "$HERE/score.py" "$out" >> eval/queue.log
    ran=1
    break
  done < "$QUEUE"
  if [ $ran -eq 0 ]; then
    grep -q "^STOP" "$QUEUE" && { echo "QUEUE STOPPED" >> eval/queue.log; exit 0; }
    sleep 30
  fi
done
