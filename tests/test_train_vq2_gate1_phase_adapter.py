from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_vq2_gate1_phase_adapter.sh"


def test_vq2_gate1_training_is_native_only_and_feature_bounded():
    text = SCRIPT.read_text()

    assert "pufferlib.pufferl train" in text
    assert "--env.num-gates 1" in text
    assert "--env.gate0-x 10.78" in text
    assert "--env.gate0-y 0.084" in text
    assert "--env.gate0-z 0.56" in text
    assert "--env.sitl-gate-obs-sample-interval-steps 4" in text
    assert "--train.train-encoder-feature-start 24" in text
    assert "--train.train-encoder-feature-end 24" in text
    assert "14550" not in text
    assert "5600" not in text
    assert "31000" not in text
