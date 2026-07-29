from pathlib import Path


SCRIPT = Path("scripts/train_full_policy_six_gate_live_prefix_late.sh")


def test_n112_script_preserves_live_prefix_and_trains_only_late_phase_columns():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "v6c_latest_obsmatch_r135_dropout_e10.bin" in source
    assert "--source-precision-bytes 2 --target-precision-bytes 4" in source
    assert "--env.observable-gate-phase-onehot 1" in source
    assert "--train.total-timesteps \"$TIMESTEPS\"" in source
    assert "--train.learning-rate 1e-3" in source
    assert "--train.min-lr-ratio 0.1" in source
    assert "--train.train-encoder-feature-start 27" in source
    assert "--train.train-encoder-feature-end 29" in source
