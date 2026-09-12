import pytest
import torch

from scripts.research.lflh_next.qualification.split_safe_learning import Generator, eligible_rows


def receipt(name, role):
    return {
        "climb_g1_stem": name,
        "split": role,
        "expected_g1_sha256": "local",
        "source_sha256": "source",
        "hf_path": f"g1/{name}.npy",
    }


def test_reserved_files_cannot_enter_fit_or_development_score():
    ledger = [
        {"path": "/tmp/a.npz", "original_split": "train"},
        {"path": "/tmp/b.npz", "original_split": "test"},
    ]
    rows = eligible_rows(ledger, [receipt("a", "evaluation"), receipt("b", "training")])
    assert not rows[0]["eligible"] and rows[0]["new_role"] == "excluded_reserved"
    assert rows[1]["eligible"] and rows[1]["new_role"] == "development_score"


def test_absent_identity_and_duplicate_receipt_fail_closed():
    ledger = [{"path": "/tmp/a.npz", "original_split": "train"}]
    with pytest.raises(ValueError):
        eligible_rows(ledger, [])
    with pytest.raises(ValueError):
        eligible_rows(ledger, [receipt("a", "training"), receipt("a", "training")])


def test_unknown_source_role_is_not_admitted():
    rows = eligible_rows(
        [{"path": "/tmp/a.npz", "original_split": "train"}], [receipt("a", "unknown")]
    )
    assert not rows[0]["eligible"]


def test_unconditional_control_has_no_motion_dependence():
    model = Generator(False).eval()
    logits = model(torch.randn(3, 272), torch.randn(225, 6))
    torch.testing.assert_close(logits[0], logits[2])
    torch.testing.assert_close(logits.softmax(-1).sum(-1), torch.ones(3))
