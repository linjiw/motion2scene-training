"""Plot all eight held-out routes in each reference's chord frame."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
from e1_route_heldout import checked

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--run-record', type=Path, required=True)
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_repo))
    from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

    result = json.loads(args.result.read_text())
    if not result['analysis_complete']:
        raise ValueError('requires complete result')
    run = json.loads(checked(args.run_record, result['run_record_sha256']).read_text())
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    for row, ax in zip(result['rows'], axes.flat, strict=True):
        reference = np.loadtxt(checked(Path(row['csv']), row['csv_sha256']), delimiter=',')[:, :2]
        artifact = run['cells'][row['cell_id']]['scientific']['artifacts']
        trajectory = checked(Path(artifact['trajectory']), artifact['trajectory_sha256'])
        with trajectory.open('rb') as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
        achieved = np.asarray(payload['root_pos_w'])[:, :2]
        reference -= reference[0].copy()
        achieved = achieved - achieved[0]
        tangent = reference[-1] / np.linalg.norm(reference[-1])
        frame = np.column_stack([tangent, [-tangent[1], tangent[0]]])
        reference = reference @ frame
        achieved = achieved @ frame
        ax.plot(*reference.T, color='#4169a1', label='Reference', linewidth=1.6)
        ax.plot(*achieved.T, color='#d77d20', label='Achieved', linewidth=1.1)
        ax.scatter(*achieved[-1], color='#d77d20', s=16)
        track = 'pass' if row['tracker_survived'] else 'fail'
        route = 'pass' if row['route_retained'] else 'fail'
        ax.set_title(f"Seed {row['generation_seed']} | tracker {track}, route {route}", fontsize=9)
        ax.set_xlabel('Reference-chord progress (m)')
        ax.grid(alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel('Lateral position (m; exaggerated scale)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Fresh neutral controls: 7/8 tracker survivors; 5/7 retain the registered route')
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix('.png'), dpi=160)
    fig.savefig(args.out.with_suffix('.svg'))
    plt.close(fig)


if __name__ == '__main__':
    main()
