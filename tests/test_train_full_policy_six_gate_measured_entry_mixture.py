from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_full_policy_six_gate_measured_entry_mixture.sh"


def test_measured_entry_training_is_one_policy_and_mixes_full_starts():
    text = SCRIPT.read_text()
    assert "n209_measured_entry_mixture" in text
    assert "--env.start-gate-index 3" in text
    assert "--env.mixed-start-curriculum 1" in text
    assert "--env.segment-start-probability 0.50" in text
    assert "--env.observable-gate-progress 1" in text
    assert "--env.observable-gate-index-denominator 6" in text
    assert "train-encoder-feature-start" not in text
    assert "gate4_checkpoint" not in text


def test_measured_entry_training_randomizes_every_empirical_state_channel():
    text = SCRIPT.read_text()
    for option in (
        "--env.start-elapsed-time-jitter",
        "--env.start-x-jitter",
        "--env.start-y-jitter",
        "--env.start-z-jitter",
        "--env.start-vx-jitter",
        "--env.start-vy-jitter",
        "--env.start-vz-jitter",
        "--env.start-roll-jitter-rad",
        "--env.start-pitch-jitter-rad",
        "--env.start-yaw-jitter-rad",
        "--env.start-wx-jitter",
        "--env.start-wy-jitter",
        "--env.start-wz-jitter",
    ):
        assert option in text
