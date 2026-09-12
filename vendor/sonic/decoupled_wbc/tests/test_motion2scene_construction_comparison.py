from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_construction_comparison import selected_support


def test_proposal_requires_its_chosen_positive_and_complete_offset_support():
    rows = [
        dict(
            candidate_id=str(i),
            stratum="short",
            positive_option_id="adapt",
            negative_option_id="walk",
        )
        for i in range(3)
    ]
    result = selected_support(
        rows,
        ["0", "1", "2"],
        ["walk", "adapt"],
        np.array([[0.02, 0.02], [0.02, 0.02], [0.02, 0.02]]),
        np.array([[81, 81], [81, 1], [81, 81]]),
        np.array([[-0.02, 0.02], [-0.02, 0.02], [0.02, 0.02]]),
        0.01,
        81,
    )
    assert result["chosen_positive_clear"] == 2
    assert result["chosen_negative_intersects"] == 2
    assert result["chosen_contrast_retained"] == 1
