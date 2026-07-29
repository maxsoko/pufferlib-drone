from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_policy_callable_load.py"


def test_policy_load_check_is_inference_only():
    text = SCRIPT.read_text()

    assert "policy_callable_checkpoint" in text
    assert "infer([0.0] * 32)" in text
    assert "14550" not in text
    assert "5600" not in text
    assert "31000" not in text
