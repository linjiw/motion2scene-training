from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_expanded_plan import balanced_extension


def row(index, stratum):
    return dict(candidate_id=str(index), stratum=stratum)


def test_extending_preserves_prefixes_and_matched_composition():
    prefix = [
        row(i, s)
        for i, s in enumerate(("short", "sustained", "early_constraint", "short_then_short"))
    ]
    a = prefix + [
        row(4, "short"),
        row(5, "short_then_sustained"),
        row(6, "sustained"),
        row(7, "early_constraint"),
    ]
    b = prefix + list(reversed(a[4:]))
    result = balanced_extension({"a": a, "b": b}, {"a": prefix, "b": prefix}, 8)
    assert result["obtained"] == 8 and not result["shortfall"]
    assert result["selected"]["a"][:4] == prefix
    assert [r["stratum"] for r in result["selected"]["a"]] == [
        r["stratum"] for r in result["selected"]["b"]
    ]
    assert result["selected"]["a"][4]["stratum"] == "short_then_sustained"


def test_shortfall_is_retained_without_duplicates_or_cross_arm_refill():
    prefix = [row(i, "short") for i in range(4)]
    result = balanced_extension(
        {"a": prefix + [row(4, "short")], "b": prefix}, {"a": prefix, "b": prefix}, 8
    )
    assert result["obtained"] == 4 and result["shortfall"] == 4


def test_mismatched_prior_strata_cannot_be_called_controlled_continuation():
    a = [row(i, "short") for i in range(4)]
    b = [row(i, "sustained") for i in range(4)]
    with pytest.raises(ValueError, match="matched-stratum"):
        balanced_extension({"a": a, "b": b}, {"a": a, "b": b}, 8)
