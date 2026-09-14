import numpy as np
import pytest

from gear_sonic.research.scene_distillation.navigation_reentry import rewind_frame


def test_rewind_frame_only_rewinds_and_matches_nearest_reference():
    path = np.stack([np.linspace(0, 5, 101), np.zeros(101)], 1)
    # Robot stopped short at x=2.0 while the clock is at frame 80 (x=4.0): rewind to x=2.0.
    assert rewind_frame(path, [2.0, 0.3], 80) == 40
    # Robot ahead of the clock cannot pull the reference forward.
    assert rewind_frame(path, [4.9, 0.0], 40) == 40
    # A minimum rewind keeps the target strictly behind the nominal frame.
    assert rewind_frame(path, [4.0, 0.0], 80, min_rewind=10) == 70
    assert rewind_frame(path, [0.0, 0.0], 0) == 0
    with pytest.raises(ValueError):
        rewind_frame(path[:, :1], [0.0, 0.0], 5)
