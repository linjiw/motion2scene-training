#!/usr/bin/env bash
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=${M2S_KIT:-$(cd "$HERE/../.." && pwd)}
PACKET=${DISTILL_PACKET:-$KIT/workspace/distill-8192}
cd "$PACKET"
for s in "$@"; do
  echo "=== $(date -u +%FT%TZ) start seed $s" >> collect/sequence.log
  "$HERE/native.sh" $PWD/collect/seed-$s train 89 $s gear_sonic.research.scene_distillation.collect.PrefixFoundationCollectionCallback ++callbacks.im_eval.collection_lock=$PWD/collect/collection-lock.json >> collect/sequence.log 2>&1
  code=$?
  echo "=== $(date -u +%FT%TZ) done seed $s exit=$code" >> collect/sequence.log
  [ $code -eq 0 ] || { echo "SEQUENCE STOPPED" >> collect/sequence.log; exit $code; }
done
echo "SEQUENCE COMPLETE" >> collect/sequence.log
