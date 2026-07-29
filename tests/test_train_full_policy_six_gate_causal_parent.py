from pathlib import Path


SCRIPT = Path("scripts/train_full_policy_six_gate_causal_parent.sh")


def test_n109_script_uses_causal_parent_and_only_registered_lr_rung():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "0000000000327680.bin" in source
    assert "--env.gate-radius-profile-mix 1" in source
    assert "--env.gate-radius-profile-mix-probability 0.5" in source
    assert "--env.gate-position-domain-randomize-probability 0.5" in source
    assert "--env.observable-gate-phase-onehot 1" in source
    assert "--train.learning-rate 1e-6" in source
    assert "--train.min-lr-ratio 0.1" in source
    assert "--train.train-encoder-feature-start -1" in source
    assert "--train.train-encoder-feature-end -1" in source
