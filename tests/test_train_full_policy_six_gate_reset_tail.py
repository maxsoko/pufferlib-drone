from pathlib import Path


SCRIPT = Path("scripts/train_full_policy_six_gate_reset_tail.sh")


def test_n114_trains_a_separate_whole_network_tail_from_fixed_gate4_start():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "ppo_n104_phase_adapter" in source
    assert "--env.start-gate-index 3" in source
    assert "--env.mixed-start-curriculum 0" in source
    assert "--env.gate-radius 2" in source
    assert "--env.gate-position-randomize-from-index 4" in source
    assert "--env.observable-gate-phase-onehot 1" in source
    assert "--train.total-timesteps \"$TIMESTEPS\"" in source
    assert "--train.learning-rate 3e-4" in source
    assert "--train.min-lr-ratio 0.1" in source
    assert "train-encoder-feature" not in source
    assert "FlightSim" not in source
