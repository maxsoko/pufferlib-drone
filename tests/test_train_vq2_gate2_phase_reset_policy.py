from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_vq2_gate2_phase_reset_policy.sh"


def test_gate2_phase_reset_ppo_is_native_pufferlib_only():
    text = SCRIPT.read_text()

    assert "pufferlib.pufferl train" in text
    assert "--env.start-gate-index 1" in text
    assert "--env.start-elapsed-time 3.25" in text
    assert "--env.start-vx 4.677" in text
    assert "--env.gate1-x 14.74" in text
    assert "--env.gate1-y 8.70" in text
    assert "--env.gate1-z 1.095" in text
    assert "--env.teacher-action-blend" in text
    assert "--env.w-action-teacher" in text
    assert "--env.teacher-pitch-speed-control 1" in text
    assert "--env.teacher-pitch-speed-target-m-s 1.30" in text
    assert "--env.teacher-thrust-world-frame 1" in text
    assert "--env.teacher-yaw-control 1" in text
    assert "TEACHER_BLEND=\"${TEACHER_BLEND:-1.0}\"" in text
    assert "--env.observable-gate-index-denominator 6" in text
    assert "--train.train-encoder-feature-start -1" in text
    assert "14550" not in text
    assert "5600" not in text
    assert "31000" not in text
