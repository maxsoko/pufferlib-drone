from pathlib import Path


SCRIPT = Path("scripts/train_full_policy_six_gate_backward_tail_gate6.sh")
STAGE_B = Path("scripts/train_full_policy_six_gate_backward_tail_gate5.sh")
MEASURED_GATE5 = Path("scripts/train_full_policy_six_gate_measured_gate5_terminal.sh")
APERTURE_GATE5 = Path("scripts/train_full_policy_six_gate_measured_gate5_aperture.sh")


def test_n115a_bootstraps_final_gate_with_a_bounded_whole_network_stage():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "ppo_n104_phase_adapter" in source
    assert "--env.start-gate-index 5" in source
    assert "--env.mixed-start-curriculum 0" in source
    assert "--env.gate-position-randomize-from-index 5" in source
    assert "--env.observable-gate-phase-onehot 1" in source
    assert "--env.w-action-teacher 5" in source
    assert "--env.teacher-pitch-from-gate-index 5" in source
    assert "--train.total-timesteps \"$TIMESTEPS\"" in source
    assert "--train.learning-rate 1e-3" in source
    assert "--train.ent-coef 0.001" in source
    assert "train-encoder-feature" not in source


def test_n115b_moves_reset_to_gate5_and_uses_the_promoted_gate6_parent():
    source = STAGE_B.read_text(encoding="utf-8")
    assert "0000000000294912.bin" in source
    assert "--env.start-gate-index 4" in source
    assert "--env.gate-position-randomize-from-index 4" in source
    assert "--env.teacher-pitch-from-gate-index 4" in source
    assert "--train.learning-rate 5e-4" in source
    assert "--train.total-timesteps \"$TIMESTEPS\"" in source
    assert "train-encoder-feature" not in source


def test_n117_uses_measured_n112_entry_and_makes_gate5_terminal():
    source = MEASURED_GATE5.read_text(encoding="utf-8")
    assert "--env.num-gates 5" in source
    assert "--env.start-gate-index 4" in source
    assert "--env.start-vx 4.1875" in source
    assert "--env.start-vy -0.2708830" in source
    assert "--env.start-qw 0.999819994" in source
    assert "--env.teacher-roll-until-gate-index 5" in source
    assert "--train.total-timesteps \"$TIMESTEPS\"" in source


def test_n118_uses_fixed_target_and_randomizes_only_the_teaching_aperture():
    source = APERTURE_GATE5.read_text(encoding="utf-8")
    assert "0000000000294912.bin" in source
    assert "--env.gate-radius-randomize 1" in source
    assert "--env.gate-radius-min 2 --env.gate-radius-max 4" in source
    assert "--env.gate-radius-randomize-gate-index 4" in source
    assert "--env.gate-position-domain-randomize 0" in source
    assert "--train.learning-rate 5e-4" in source
