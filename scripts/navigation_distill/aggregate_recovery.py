"""Bind MotorRecoveryCollectionCallback receipts (supported or not) into a recovery manifest.

Unsupported attempts are listed for cost accounting but excluded from parents, since the
loader rejects manifests whose parents contain no supported rows only at the very end.
"""

import argparse
import json
from pathlib import Path

from gear_sonic.research.hindsight_training.runtime import sha, write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stages", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    parents, unsupported = [], []
    for stage in a.stages:
        for receipt in sorted(stage.glob("*/task/recovery.json")):
            r = json.loads(receipt.read_text())
            trace = receipt.parent / "trace.npz"
            entry = dict(path=str(receipt.resolve()), sha256=sha(receipt),
                         trace=dict(path=str(trace.resolve()), sha256=sha(trace)))
            if r["supported"]:
                parents.append(entry)
            else:
                unsupported.append(dict(entry, takeover_tick=r["takeover_tick"], rows=r["rows"]))
    write_new(a.output, dict(schema="motor_navigation_recovery_manifest_v1", parents=parents,
                             unsupported_attempts=unsupported))
    print(json.dumps(dict(supported=len(parents), unsupported=len(unsupported))))


if __name__ == "__main__":
    main()
