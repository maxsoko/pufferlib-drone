from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_vq2_gate2_phase_reset_candidate.sh"


def test_gate2_phase_reset_evaluator_uses_measured_transition_only():
    text = SCRIPT.read_text()

    assert "eval_drone_race_checkpoint.py" in text
    assert "--env.start-gate-index 1" in text
    assert "--env.start-elapsed-time 3.25" in text
    assert "--env.start-vx 4.677" in text
    assert "--env.start-qw 0.998520" in text
    assert "--env.gate1-x 14.74" in text
    assert "--env.gate1-y 8.70" in text
    assert "--env.gate1-z 1.095" in text
    assert "--env.observable-gate-index-denominator 6" in text
    assert "--checkpoint-layout-precision-bytes 4" in text
    assert '"$@"' in text
    assert "14550" not in text
    assert "5600" not in text
    assert "31000" not in text
