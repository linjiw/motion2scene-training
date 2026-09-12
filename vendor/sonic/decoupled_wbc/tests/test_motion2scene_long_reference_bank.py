import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_long_reference_bank import (
    audit_finite_root_route,
)


@pytest.mark.parametrize("source_frames,target_frames", [(120, 199), (180, 299)])
def test_registered_route_uses_source_clock(source_frames, target_frames):
    raw = np.column_stack(
        [np.arange(source_frames) / 30, np.zeros(source_frames), np.ones(source_frames)]
    )
    root = np.column_stack(
        [np.arange(target_frames) / 50, np.zeros(target_frames), np.ones(target_frames)]
    )
    assert audit_finite_root_route(root, raw, 30, target_frames) < 1e-12
    root[:, 0] += 0.01
    with pytest.raises(ValueError, match="differs from source"):
        audit_finite_root_route(root, raw, 30, target_frames)


def test_endpoint_padding_and_wrong_registered_budget_rejected():
    raw = np.zeros((180, 3))
    with pytest.raises(ValueError, match="exclude the source endpoint"):
        audit_finite_root_route(np.zeros((300, 3)), raw, 30, 300)
    with pytest.raises(ValueError, match="invalid registered"):
        audit_finite_root_route(np.zeros((199, 3)), raw, 30, 299)
