from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_vq2_gate1_candidate.sh"


def test_vq2_gate1_candidate_eval_has_fixed_and_perturbed_native_contracts():
    text = SCRIPT.read_text()

    assert "eval_drone_race_checkpoint.py" in text
    assert "PERTURBED" in text
    assert "--require-exact-episodes" in text
    assert "--env.gate0-x 10.78" in text
    assert "--env.gate0-y 0.084" in text
    assert "--env.gate0-z 0.56" in text
    assert "--env.gate-position-jitter-z 0.20" in text
    assert "--env.sitl-plant-domain-randomize 1" in text
    assert "--checkpoint-layout-precision-bytes 4" in text
    assert "14550" not in text
    assert "5600" not in text
    assert "31000" not in text
