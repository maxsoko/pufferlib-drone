from pathlib import Path


SCRIPT = Path("scripts/train_full_policy_six_gate_early_phase_adapter.sh")


def test_n107_script_freezes_everything_except_gate2_gate3_phase_columns():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "0000000001048576.bin" in source
    assert "--env.gate-radius 0.75" in source
    assert "--env.gate-position-domain-randomize 0" in source
    assert "--env.observable-gate-phase-onehot 1" in source
    assert "--train.train-encoder-feature-start 25" in source
    assert "--train.train-encoder-feature-end 26" in source
    assert "--train.total-timesteps \"$TIMESTEPS\"" in source
