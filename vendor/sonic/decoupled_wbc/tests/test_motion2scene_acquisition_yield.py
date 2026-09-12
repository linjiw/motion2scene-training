"""A passage contrast is not necessarily an adaptation requirement."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_acquisition_yield import teaching_yield


def test_all_passing_is_not_a_passage_contrast():
    result = teaching_yield(dict(neutral="pass", adapt="pass"))
    assert result["bank_solvable"] and not result["passage_contrast"]
    assert not result["adaptation_required"]


def test_failed_adaptation_with_passing_walking_does_not_require_adaptation():
    result = teaching_yield(dict(neutral="pass", adapt="failure"))
    assert result["passage_contrast"] and not result["adaptation_required"]


def test_failed_walking_with_passing_adaptation_is_required():
    result = teaching_yield(dict(neutral="failure", adapt="pass"))
    assert result["passage_contrast"] and result["adaptation_required"]


def test_all_failed_and_unknown_have_different_interpretations():
    assert not teaching_yield(dict(neutral="failure", adapt="failure"))["bank_solvable"]
    with pytest.raises(ValueError, match="physically known"):
        teaching_yield(dict(neutral="unknown", adapt="pass"))
