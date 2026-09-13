#!/usr/bin/env bash
# Queue train (queue-a) and development (queue-b) seed-91260 evaluations for each new online checkpoint.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=${M2S_KIT:-$(cd "$HERE/../.." && pwd)}
PACKET=${DISTILL_PACKET:-$KIT/workspace/distill-8192}
cd "$PACKET"
STAGE=${1:?stage dir}; TAG=${2:?tag}
W=$PWD
while true; do
  for ck in $(ls $STAGE/training/step-*.pt 2>/dev/null); do
    step=$(basename $ck .pt | sed 's/step-0*//')
    grep -q "$ck train" eval/queue-a.txt || echo "$ck train 91260 $W/eval/$TAG-$step-train-91260" >> eval/queue-a.txt
    grep -q "$ck development" eval/queue-b.txt || echo "$ck development 91260 $W/eval/$TAG-$step-dev-91260" >> eval/queue-b.txt
  done
  [ -f $STAGE/training/receipt.json ] && { sleep 5; for ck in $(ls $STAGE/training/step-*.pt); do :; done; echo "watch done" ; exit 0; }
  sleep 20
done
