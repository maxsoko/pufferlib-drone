from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "screen_vq2_gate1_policy.sh"


def test_vq2_gate1_screen_is_native_exact_and_command_free():
    text = SCRIPT.read_text()

    assert "eval_drone_race_checkpoint.py" in text
    assert "--eval-episodes 128" in text
    assert "--require-exact-episodes" in text
    assert "--env.gate0-x 10.78" in text
    assert "--env.gate0-y 0.084" in text
    assert "--env.gate0-z 0.56" in text
    assert "--env.sitl-gate-obs-sample-interval-steps 4" in text
    assert "14550" not in text
    assert "5600" not in text
    assert "31000" not in text
