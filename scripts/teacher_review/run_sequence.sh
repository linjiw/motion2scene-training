#!/usr/bin/env bash
# Run review evaluations sequentially; stop at the first failed launch or check.
# Usage: run_sequence.sh previous8000/development trained500/development ...
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=${M2S_KIT:-$(cd "$HERE/../.." && pwd)}
R=${REVIEW:-$KIT/workspace/teacher-8192-500-review}
export REVIEW=$R
for pair in "$@"; do
  arm=${pair%/*}; split=${pair#*/}
  echo "=== $(date -u +%FT%TZ) start $arm $split" >> $R/sequence.log
  bash $HERE/run_eval.sh "$arm" "$split" >> $R/sequence.log 2>&1
  code=$?
  echo "=== $(date -u +%FT%TZ) done $arm $split exit=$code" >> $R/sequence.log
  [ $code -eq 0 ] || { echo "SEQUENCE STOPPED" >> $R/sequence.log; exit $code; }
done
echo "SEQUENCE COMPLETE" >> $R/sequence.log
