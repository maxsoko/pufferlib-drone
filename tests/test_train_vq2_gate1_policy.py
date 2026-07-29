from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_vq2_gate1_policy.sh"


def test_vq2_gate1_full_policy_reuses_native_contract_and_removes_mask():
    text = SCRIPT.read_text()

    assert "train_vq2_gate1_phase_adapter.sh" in text
    assert "--tag vq2_n286_gate1_policy" in text
    assert "--train.train-encoder-feature-start -1" in text
    assert "--train.train-encoder-feature-end -1" in text
    assert "14550" not in text
    assert "5600" not in text
    assert "31000" not in text
