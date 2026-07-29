from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_vq2_gate1_aperture_stage.sh"


def test_vq2_gate1_aperture_stage_is_exact_native_policy_training():
    text = SCRIPT.read_text()

    assert "pufferlib.pufferl train" in text
    assert 'GATE_RADIUS="${GATE_RADIUS:-1.60}"' in text
    assert "--env.gate0-x 10.78" in text
    assert "--env.gate0-y 0.084" in text
    assert "--env.gate0-z 0.56" in text
    assert "--env.gate-position-domain-randomize 0" in text
    assert "--env.sitl-plant-domain-randomize 0" in text
    assert "--train.train-encoder-feature-start -1" in text
    assert "--train.train-encoder-feature-end -1" in text
    assert "14550" not in text
    assert "5600" not in text
    assert "31000" not in text
