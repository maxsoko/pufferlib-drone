from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_vq2_gate2_candidate.sh"


def test_gate2_evaluator_is_native_only_and_uses_measured_layout():
    text = SCRIPT.read_text()

    assert "eval_drone_race_checkpoint.py" in text
    assert "--env.num-gates 2" in text
    assert "--env.gate0-x 10.78" in text
    assert "--env.gate1-x 25.13" in text
    assert "--env.gate1-y 8.96" in text
    assert "--env.gate1-z 1.65" in text
    assert "--env.observable-gate-index-denominator 6" in text
    assert "--env.observable-gate-phase-onehot 1" in text
    assert "--env.gate-position-randomize-from-index 0" in text
    assert "--checkpoint-layout-precision-bytes 4" in text
    assert "14550" not in text
    assert "5600" not in text
    assert "31000" not in text
