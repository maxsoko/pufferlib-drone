from pathlib import Path


SCRIPT = Path("scripts/train_full_policy_six_gate_anchored_aperture.sh")


def test_n108_script_uses_exact_anchor_target_mix_and_full_network():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "0000000001048576.bin" in source
    assert "--env.gate-radius 0.95" in source
    assert "--env.gate0-radius 0.75" in source
    assert "--env.gate3-radius 0.875" in source
    assert "--env.gate5-radius 0.75" in source
    assert "--env.gate-radius-profile-mix 1" in source
    assert "--env.gate-radius-profile-mix-probability 0.5" in source
    assert "--env.gate-radius-profile-mix-target 0.75" in source
    assert "--env.gate-position-domain-randomize-probability 0.5" in source
    assert "--env.observable-gate-phase-onehot 1" in source
    assert "--train.learning-rate 1e-7" in source
    assert "--train.train-encoder-feature-start -1" in source
    assert "--train.train-encoder-feature-end -1" in source
