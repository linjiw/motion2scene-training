#!/usr/bin/env bash
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Run N DAgger cycles for one arm, chaining checkpoints. Usage: dagger_arm.sh <arm> <start_ckpt> <cycles> [fit args...]
cd "${NAV_PACKET:?set NAV_PACKET}"
ARM=$1; NAV=$2; N=$3; shift 3
for c in $(seq 1 $N); do
  NAV=$("$HERE"/dagger_cycle.sh $ARM $c "$NAV" "0.12 0.25" "$@" | tail -1) || exit 1
done
echo "ARM COMPLETE $(date -u +%FT%TZ)" >> dagger/$ARM/cycle.log
