import math
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import policy_callable_final_gate_handoff as handoff


def observation(*, phase=1.0, visible=True, forward_rate=-7.0, down=2.0, yaw_error=0.2):
    values = [0.0] * 23
    values[0] = math.tanh(forward_rate / 5.0)
    values[6] = 1.0
    values[10] = 1.0 if visible else 0.0
    values[11] = math.tanh(30.0 / 10.0)
    values[12] = math.tanh(8.0 / 5.0)
    values[13] = math.tanh(down / 5.0)
    values[14] = yaw_error / (math.pi / 4.0)
    values[17] = 0.8
    values[22] = phase
    return values


def test_gate_index_uses_phase_denominator():
    cfg = handoff.FinalGateHandoffConfig(phase_denominator=3)
    assert handoff.gate_index_from_observation(observation(phase=0.0), cfg) == 0
    assert handoff.gate_index_from_observation(observation(phase=2 / 3), cfg) == 2
    assert handoff.gate_index_from_observation(observation(phase=1.0), cfg) == 3

    official_cfg = handoff.FinalGateHandoffConfig(phase_denominator=6)
    for gate_index in range(7):
        assert handoff.gate_index_from_observation(
            observation(phase=gate_index / 6), official_cfg
        ) == gate_index


def test_post_prefix_state_resets_on_each_official_gate_transition():
    handoff._POST_PREFIX_GATE_INDEX = None
    handoff._FINAL_GATE_PHASE_STARTED_S = 10.0
    handoff._FINAL_GATE_DESCENT_PULSE_COUNT = 2
    handoff._FINAL_GATE_REANCHOR_COUNT = 3
    handoff._LAST_FINAL_GATE_VECTOR = (4.0, 0.5, 1.0)

    assert handoff._start_post_prefix_gate(3)
    assert handoff._POST_PREFIX_GATE_INDEX == 3
    assert handoff._FINAL_GATE_PHASE_STARTED_S is None
    assert handoff._FINAL_GATE_DESCENT_PULSE_COUNT == 0


def test_six_gate_phase_keeps_recurrent_checkpoint_input_legacy_compatible(
    monkeypatch,
):
    captured = {}

    class FakeModel:
        input_dim = 23

        def infer(self, values):
            captured["observation"] = list(values)
            return [0.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(handoff, "_resolve_model", lambda: FakeModel())
    monkeypatch.setenv("PUFFER_POLICY_RACE_PHASE_DENOMINATOR", "6")
    handoff._clear_post_prefix_gate_state()
    handoff._POST_PREFIX_GATE_INDEX = None
    handoff._LAST_YAW_ACTION = 0.25

    handoff.infer(observation(phase=3 / 6, visible=False))

    assert captured["observation"][22] == pytest.approx(0.25)
    assert handoff._POST_PREFIX_GATE_INDEX == 3
    assert handoff._FINAL_GATE_REANCHOR_COUNT == 0
    assert handoff._LAST_FINAL_GATE_VECTOR is None

    handoff._FINAL_GATE_PHASE_STARTED_S = 20.0
    handoff._FINAL_GATE_DESCENT_PULSE_COUNT = 1
    assert not handoff._start_post_prefix_gate(3)
    assert handoff._FINAL_GATE_PHASE_STARTED_S == pytest.approx(20.0)
    assert handoff._FINAL_GATE_DESCENT_PULSE_COUNT == 1

    assert handoff._start_post_prefix_gate(4)
    assert handoff._POST_PREFIX_GATE_INDEX == 4
    assert handoff._FINAL_GATE_PHASE_STARTED_S is None
    assert handoff._FINAL_GATE_DESCENT_PULSE_COUNT == 0


def test_final_gate_anchor_prevents_incremental_alias_walk():
    assert not handoff.final_gate_pose_is_associated(
        [14.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
        8.0,
        anchor_vector=[0.0, 0.0, 0.0],
        max_anchor_drift_m=12.0,
    )


def test_final_gate_lateral_servo_levels_on_dropout():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_lateral_roll_kp=0.04,
        final_gate_max_lateral_roll_rad=0.15,
    )
    visible = observation(phase=1.0, down=3.0, yaw_error=0.2)
    visible[12] = math.tanh(3.0 / 5.0)
    action = handoff.final_gate_action(visible, [0.0] * 4, cfg)
    assert action[1] == pytest.approx(-0.24)

    hidden = observation(phase=1.0, visible=False)
    held = handoff.final_gate_action(hidden, [0.0] * 4, cfg, fallback_action=action)
    assert held[1] == 0.0
    assert handoff.final_gate_pose_is_associated(
        [14.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
        8.0,
        anchor_vector=[0.0, 0.0, 0.0],
        max_anchor_drift_m=12.0,
        allow_reseed=True,
    )


def test_final_gate_pulse_can_hold_only_bounded_saved_lateral_during_dropout():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_lateral_roll_kp=0.02,
        final_gate_max_lateral_roll_rad=0.10,
        final_gate_hold_pulse_lateral=True,
    )
    action = [0.0, 0.0, -0.4, 0.2]
    held = handoff.final_gate_pulse_lateral_hold_action(
        action,
        (18.0, 7.0, 8.0),
        None,
        10.0,
        cfg,
    )
    assert held == pytest.approx([0.0, -0.2, -0.4, 0.2])

    current_wins = handoff.final_gate_pulse_lateral_hold_action(
        action,
        (18.0, 7.0, 8.0),
        (17.0, 2.0, 7.0),
        10.0,
        cfg,
    )
    assert current_wins == pytest.approx(action)

    expired = handoff.final_gate_pulse_lateral_hold_action(
        action,
        (18.0, 7.0, 8.0),
        None,
        None,
        cfg,
    )
    assert expired == pytest.approx(action)

    disabled = handoff.final_gate_pulse_lateral_hold_action(
        action,
        (18.0, 7.0, 8.0),
        None,
        10.0,
        handoff.FinalGateHandoffConfig(),
    )
    assert disabled == pytest.approx(action)


def test_final_gate_pulse_pitch_levels_only_while_enabled_and_active():
    action = [-0.24, -0.1, -0.4, 0.2]
    disabled = handoff.final_gate_pulse_pitch_action(
        action,
        10.0,
        handoff.FinalGateHandoffConfig(),
    )
    assert disabled == pytest.approx(action)

    cfg = handoff.FinalGateHandoffConfig(
        final_gate_level_pitch_during_descent_pulse=True,
    )
    active = handoff.final_gate_pulse_pitch_action(action, 10.0, cfg)
    assert active == pytest.approx([0.0, -0.1, -0.4, 0.2])
    expired = handoff.final_gate_pulse_pitch_action(action, None, cfg)
    assert expired == pytest.approx(action)


def test_gate3_early_bank_activates_only_for_fast_far_left_entry():
    cfg = handoff.FinalGateHandoffConfig(
        gate3_correction_index=2,
        gate3_close_forward_m=15.0,
        gate3_early_bank_forward_m=20.0,
        gate3_early_bank_min_closing_rate_m_s=7.0,
        gate3_left_trigger_m=-0.3,
        gate3_roll_floor_rad=0.45,
    )
    fast_far = observation(phase=2 / 3, forward_rate=-7.2)
    fast_far[11] = math.tanh(18.0 / 10.0)
    fast_far[12] = math.tanh(-3.0 / 5.0)
    assert handoff.gate3_close_correction_action(
        fast_far, [0.2, 0.1, -0.2, 0.01], cfg
    ) == pytest.approx([0.2, 0.9, -0.2, 0.01])

    slow_far = list(fast_far)
    slow_far[0] = math.tanh(-6.8 / 5.0)
    assert handoff.gate3_close_correction_action(
        slow_far, [0.2, 0.1, -0.2, 0.01], cfg
    ) == pytest.approx([0.2, 0.1, -0.2, 0.01])

    fast_far_centered = list(fast_far)
    fast_far_centered[12] = math.tanh(-0.2 / 5.0)
    assert handoff.gate3_close_correction_action(
        fast_far_centered, [0.2, 0.1, -0.2, 0.01], cfg
    ) == pytest.approx([0.2, 0.1, -0.2, 0.01])

    close = list(slow_far)
    close[11] = math.tanh(14.0 / 10.0)
    assert handoff.gate3_close_correction_action(
        close, [0.2, 0.1, -0.2, 0.01], cfg
    ) == pytest.approx([0.2, 0.9, -0.2, 0.01])


def test_gate3_early_bank_requires_a_threshold_when_enabled(monkeypatch):
    monkeypatch.setenv("PUFFER_GATE3_EARLY_BANK_FORWARD_M", "20")
    monkeypatch.setenv("PUFFER_GATE3_EARLY_BANK_MIN_CLOSING_RATE_M_S", "0")

    with pytest.raises(ValueError, match="speed threshold must be positive"):
        handoff.load_config()


def test_gate3_severe_projected_miss_adapts_bank_without_changing_other_actions():
    cfg = handoff.FinalGateHandoffConfig(
        gate3_correction_index=2,
        gate3_close_forward_m=15.0,
        gate3_early_bank_forward_m=20.0,
        gate3_early_bank_min_closing_rate_m_s=6.5,
        gate3_severe_bank_forward_m=22.0,
        gate3_severe_bank_min_closing_rate_m_s=5.0,
        gate3_severe_intercept_target_right_m=-3.5,
        gate3_severe_roll_floor_rad=0.5,
        gate3_left_trigger_m=-0.3,
        gate3_roll_floor_rad=0.45,
        gate3_intercept_plane_forward_m=2.0,
        gate3_intercept_target_right_m=-0.8,
        gate3_counter_roll_rad=-0.5,
    )
    severe = observation(phase=2 / 3, forward_rate=-5.2)
    severe[1] = math.tanh(0.05 / 3.0)
    severe[11] = math.tanh(21.35 / 10.0)
    severe[12] = math.tanh(-4.34 / 5.0)
    assert handoff.gate3_close_correction_action(
        severe, [0.42, 0.4, -0.6, 0.01], cfg
    ) == pytest.approx([0.42, 1.0, -0.6, 0.01])

    midpoint = list(severe)
    midpoint[1] = 0.0
    midpoint[11] = math.tanh(12.0 / 10.0)
    midpoint[12] = math.tanh(-2.15 / 5.0)
    assert handoff.gate3_close_correction_action(
        midpoint, [0.42, 0.4, -0.6, 0.01], cfg
    ) == pytest.approx([0.42, 0.95, -0.6, 0.01])

    recovered_projection = list(severe)
    recovered_projection[1] = math.tanh(0.7 / 3.0)
    recovered_projection[12] = math.tanh(-2.2 / 5.0)
    assert handoff.gate3_close_correction_action(
        recovered_projection, [0.42, 0.4, -0.6, 0.01], cfg
    ) == pytest.approx([0.42, 0.4, -0.6, 0.01])


def test_gate3_severe_bank_configuration_is_all_or_nothing(monkeypatch):
    monkeypatch.setenv("PUFFER_GATE3_SEVERE_BANK_FORWARD_M", "22")

    with pytest.raises(ValueError, match="requires range, speed, and roll"):
        handoff.load_config()


def test_gate3_close_correction_holds_roll_floor_only_for_left_miss():
    cfg = handoff.FinalGateHandoffConfig(
        gate3_correction_index=2,
        gate3_close_forward_m=8.0,
        gate3_left_trigger_m=-0.8,
        gate3_roll_floor_rad=0.22,
    )
    close_left = observation(phase=2 / 3)
    close_left[11] = math.tanh(5.0 / 10.0)
    close_left[12] = math.tanh(-1.4 / 5.0)
    action = handoff.gate3_close_correction_action(
        close_left, [0.1, 0.05, -0.2, 0.01], cfg
    )
    assert action == pytest.approx([0.1, 0.44, -0.2, 0.01])

    close_centered = list(close_left)
    close_centered[12] = math.tanh(-0.6 / 5.0)
    assert handoff.gate3_close_correction_action(
        close_centered, [0.1, 0.05, -0.2, 0.01], cfg
    ) == pytest.approx([0.1, 0.05, -0.2, 0.01])


def test_gate3_close_correction_can_brake_while_left_error_is_active():
    cfg = handoff.FinalGateHandoffConfig(
        gate3_correction_index=2,
        gate3_close_forward_m=15.0,
        gate3_left_trigger_m=-0.5,
        gate3_roll_floor_rad=0.35,
        gate3_brake_pitch_rad=-0.12,
    )
    close_left = observation(phase=2 / 3)
    close_left[11] = math.tanh(9.0 / 10.0)
    close_left[12] = math.tanh(-2.5 / 5.0)

    action = handoff.gate3_close_correction_action(
        close_left, [0.4, 0.2, -0.1, 0.01], cfg
    )

    assert action == pytest.approx([-0.24, 0.7, -0.1, 0.01])


def test_gate3_close_brake_activates_only_above_closing_rate_threshold():
    cfg = handoff.FinalGateHandoffConfig(
        gate3_correction_index=2,
        gate3_close_forward_m=15.0,
        gate3_left_trigger_m=-0.3,
        gate3_roll_floor_rad=0.45,
        gate3_brake_pitch_rad=-0.12,
        gate3_brake_min_closing_rate_m_s=9.7,
    )
    close_left = observation(phase=2 / 3, forward_rate=-9.6)
    close_left[11] = math.tanh(9.0 / 10.0)
    close_left[12] = math.tanh(-2.0 / 5.0)
    unbraked = handoff.gate3_close_correction_action(
        close_left, [0.4, 0.2, -0.1, 0.01], cfg
    )
    assert unbraked == pytest.approx([0.4, 0.9, -0.1, 0.01])

    close_left[0] = math.tanh(-10.1 / 5.0)
    braked = handoff.gate3_close_correction_action(
        close_left, [0.4, 0.2, -0.1, 0.01], cfg
    )
    assert braked == pytest.approx([-0.24, 0.9, -0.1, 0.01])


def test_gate3_speed_brake_hysteresis_latches_until_release_speed():
    cfg = handoff.FinalGateHandoffConfig(
        gate3_correction_index=2,
        gate3_close_forward_m=15.0,
        gate3_brake_pitch_rad=-0.12,
        gate3_brake_min_closing_rate_m_s=9.0,
        gate3_brake_release_closing_rate_m_s=8.5,
    )
    sample = observation(phase=2 / 3, forward_rate=-9.1)
    sample[11] = math.tanh(12.0 / 10.0)
    assert handoff.gate3_speed_brake_latched(sample, cfg, False)

    sample[0] = math.tanh(-8.7 / 5.0)
    assert handoff.gate3_speed_brake_latched(sample, cfg, True)
    assert not handoff.gate3_speed_brake_latched(sample, cfg, False)

    sample[0] = math.tanh(-8.4 / 5.0)
    assert not handoff.gate3_speed_brake_latched(sample, cfg, True)


def test_gate3_rate_correction_counter_banks_before_the_plane():
    cfg = handoff.FinalGateHandoffConfig(
        gate3_correction_index=2,
        gate3_close_forward_m=15.0,
        gate3_roll_floor_rad=0.35,
        gate3_brake_pitch_rad=-0.2,
        gate3_lateral_kp=0.25,
        gate3_lateral_kd=0.12,
    )
    close_left = observation(phase=2 / 3)
    close_left[1] = math.tanh(1.0 / 3.0)
    close_left[11] = math.tanh(8.0 / 10.0)
    close_left[12] = math.tanh(-2.0 / 5.0)
    bank = handoff.gate3_close_correction_action(
        close_left, [0.4, 0.2, -0.1, 0.01], cfg
    )
    assert bank == pytest.approx([-0.4, 0.7, -0.1, 0.01])

    counter_bank = list(close_left)
    counter_bank[1] = math.tanh(3.0 / 3.0)
    counter_bank[11] = math.tanh(4.0 / 10.0)
    counter_bank[12] = math.tanh(-0.8 / 5.0)
    action = handoff.gate3_close_correction_action(
        counter_bank, [0.4, 0.2, -0.1, 0.01], cfg
    )
    assert action == pytest.approx([-0.4, -0.32, -0.1, 0.01])


def test_gate3_bang_bang_intercept_banks_then_counter_banks_before_plane():
    cfg = handoff.FinalGateHandoffConfig(
        gate3_correction_index=2,
        gate3_close_forward_m=15.0,
        gate3_left_trigger_m=-0.3,
        gate3_roll_floor_rad=0.45,
        gate3_intercept_plane_forward_m=2.0,
        gate3_intercept_target_right_m=0.0,
        gate3_counter_roll_rad=-0.2,
    )
    close_left = observation(phase=2 / 3, forward_rate=-8.0)
    close_left[1] = math.tanh(0.5 / 3.0)
    close_left[11] = math.tanh(13.0 / 10.0)
    close_left[12] = math.tanh(-3.5 / 5.0)
    bank = handoff.gate3_close_correction_action(
        close_left, [0.4, 0.2, -0.1, 0.01], cfg
    )
    assert bank == pytest.approx([0.4, 0.9, -0.1, 0.01])

    predicted_center = list(close_left)
    predicted_center[1] = math.tanh(2.8 / 3.0)
    predicted_center[11] = math.tanh(6.5 / 10.0)
    predicted_center[12] = math.tanh(-1.45 / 5.0)
    counter_bank = handoff.gate3_close_correction_action(
        predicted_center, [0.4, 0.7, -0.1, 0.01], cfg
    )
    assert counter_bank == pytest.approx([0.4, -0.4, -0.1, 0.01])


def test_gate3_counter_latch_ignores_far_noise_then_holds_counter_bank():
    cfg = handoff.FinalGateHandoffConfig(
        gate3_correction_index=2,
        gate3_close_forward_m=20.0,
        gate3_left_trigger_m=-0.3,
        gate3_roll_floor_rad=0.5,
        gate3_intercept_plane_forward_m=2.0,
        gate3_intercept_target_right_m=-0.9,
        gate3_counter_roll_rad=-0.5,
        gate3_counter_latch_forward_m=6.0,
        gate3_centered_approach_pitch_rad=0.06,
        gate3_latched_level_roll_threshold_rad=0.1,
    )
    noisy_far = observation(phase=2 / 3, forward_rate=-9.0)
    noisy_far[1] = math.tanh(2.6 / 3.0)
    noisy_far[11] = math.tanh(11.4 / 10.0)
    noisy_far[12] = math.tanh(-2.55 / 5.0)
    assert not handoff.gate3_counter_latch_ready(noisy_far, cfg)

    close_ready = list(noisy_far)
    close_ready[0] = math.tanh(-6.6 / 5.0)
    close_ready[1] = math.tanh(2.37 / 3.0)
    close_ready[11] = math.tanh(4.31 / 10.0)
    close_ready[12] = math.tanh(-1.68 / 5.0)
    close_ready[6] = math.cos(0.15)
    close_ready[7] = math.sin(0.15)
    assert handoff.gate3_counter_latch_ready(close_ready, cfg)
    action = handoff.gate3_close_correction_action(
        close_ready,
        [0.4, 0.8, -0.1, 0.01],
        cfg,
        force_counter_bank=True,
    )
    assert action == pytest.approx([0.12, -1.0, -0.1, 0.01])

    close_ready[6] = 1.0
    close_ready[7] = 0.0
    level_action = handoff.gate3_close_correction_action(
        close_ready,
        [0.4, 0.8, -0.1, 0.01],
        cfg,
        force_counter_bank=True,
    )
    assert level_action == pytest.approx([0.12, 0.0, -0.1, 0.01])


def test_gate3_close_hover_compensates_for_observable_tilt():
    cfg = handoff.FinalGateHandoffConfig(
        gate3_correction_index=2,
        gate3_close_forward_m=20.0,
        gate3_left_trigger_m=-0.3,
        gate3_roll_floor_rad=0.5,
        gate3_tilt_compensated_hover_thrust=0.27,
    )
    tilted = observation(phase=2 / 3)
    tilted[6] = math.cos(0.25)
    tilted[7] = math.sin(0.25)
    tilted[11] = math.tanh(10.0 / 10.0)
    tilted[12] = math.tanh(-2.0 / 5.0)
    action = handoff.gate3_close_correction_action(
        tilted, [0.2, 0.2, -1.0, 0.01], cfg
    )
    assert action[2] > 0.0
    assert handoff._encode_thrust(0.27 / math.cos(0.5), cfg) == pytest.approx(
        action[2]
    )


def test_gate3_search_brakes_before_a_close_pose_is_available():
    cfg = handoff.FinalGateHandoffConfig(
        gate3_correction_index=2,
        gate3_close_forward_m=15.0,
        gate3_search_brake_pitch_rad=-0.12,
    )
    hidden = observation(phase=2 / 3, visible=False)
    hidden_action = handoff.gate3_close_correction_action(
        hidden, [0.4, 0.2, -0.1, 0.01], cfg
    )
    assert hidden_action == pytest.approx([-0.24, 0.2, -0.1, 0.01])

    far = observation(phase=2 / 3)
    far[11] = math.tanh(25.0 / 10.0)
    far_action = handoff.gate3_close_correction_action(
        far, [0.4, 0.2, -0.1, 0.01], cfg
    )
    assert far_action == pytest.approx([-0.24, 0.2, -0.1, 0.01])


def test_gate3_raw_subphase_levels_search_and_approaches_when_centered():
    cfg = handoff.FinalGateHandoffConfig(
        gate3_correction_index=2,
        gate3_close_forward_m=15.0,
        gate3_left_trigger_m=-0.3,
        gate3_search_brake_pitch_rad=-0.12,
        gate3_level_roll_during_search=True,
        gate3_centered_approach_pitch_rad=0.06,
    )
    hidden = observation(phase=2 / 3, visible=False)
    hidden_action = handoff.gate3_close_correction_action(
        hidden, [0.4, 0.5, -0.1, 0.01], cfg
    )
    assert hidden_action == pytest.approx([-0.24, 0.0, -0.1, 0.01])

    centered = observation(phase=2 / 3)
    centered[11] = math.tanh(7.5 / 10.0)
    centered[12] = math.tanh(-0.15 / 5.0)
    centered_action = handoff.gate3_close_correction_action(
        centered, [-0.8, 0.7, -0.1, 0.01], cfg
    )
    assert centered_action == pytest.approx([0.12, 0.0, -0.1, 0.01])


def test_final_gate_brakes_levels_and_points_at_visible_gate():
    cfg = handoff.FinalGateHandoffConfig()
    action = handoff.final_gate_action(observation(), [0.4, -0.8, 0.1, 0.0], cfg)
    assert action[0] < 0.0
    assert action[1] == 0.0
    assert action[2] < 0.0
    assert action[3] == pytest.approx(-0.2 / math.pi)


def test_final_gate_does_not_accelerate_until_yaw_centered():
    cfg = handoff.FinalGateHandoffConfig()
    uncentered = handoff.final_gate_action(
        observation(forward_rate=0.0, yaw_error=0.3), [0.8, 0.0, 0.0, 0.0], cfg
    )
    centered = handoff.final_gate_action(
        observation(forward_rate=0.0, down=0.5, yaw_error=0.01),
        [0.8, 0.0, 0.0, 0.0],
        cfg,
    )
    assert uncentered[0] < 0.0
    assert centered[0] > 0.0


def test_final_gate_does_not_accelerate_until_vertically_centered():
    cfg = handoff.FinalGateHandoffConfig(
        center_before_accel_yaw_error_rad=0.2,
        center_before_accel_down_error_m=1.5,
    )
    action = handoff.final_gate_action(
        observation(forward_rate=0.0, down=3.0, yaw_error=0.1),
        [0.8, 0.0, 0.0, 0.0],
        cfg,
    )
    assert action[0] < 0.0


def test_dropout_holds_last_final_action():
    cfg = handoff.FinalGateHandoffConfig()
    fallback = [-0.2, 0.0, -0.1, 0.05]
    assert handoff.final_gate_action(
        observation(visible=False), [0.9, 0.9, 0.9, 0.9], cfg, fallback_action=fallback
    ) == fallback


def test_final_gate_dropout_yaw_freeze_uses_current_observable_attitude():
    current_yaw_rad = -1.72
    hidden = observation(visible=False)
    hidden[6] = math.cos(current_yaw_rad / 2.0)
    hidden[9] = math.sin(current_yaw_rad / 2.0)
    stale = [-0.24, 0.0, -0.1, -2.51 / math.pi]

    disabled = handoff.final_gate_dropout_yaw_action(
        stale,
        hidden,
        None,
        handoff.FinalGateHandoffConfig(),
    )
    assert disabled == pytest.approx(stale)

    cfg = handoff.FinalGateHandoffConfig(
        final_gate_freeze_yaw_on_dropout=True,
    )
    frozen = handoff.final_gate_dropout_yaw_action(stale, hidden, None, cfg)
    assert frozen[:3] == pytest.approx(stale[:3])
    assert frozen[3] == pytest.approx(current_yaw_rad / math.pi)

    associated = handoff.final_gate_dropout_yaw_action(
        stale,
        hidden,
        (18.0, 2.0, 8.0),
        cfg,
    )
    assert associated == pytest.approx(stale)


def test_delayed_final_gate_recovery_waits_then_uses_last_associated_height():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_descent_delay_s=3.0,
        final_gate_descent_thrust=0.22,
        final_gate_recovery_pitch_rad=0.06,
    )
    hover_action = [-0.24, 0.0, handoff._encode_thrust(0.27, cfg), -0.1]
    gate_below = (25.0, 4.0, 12.0)

    assert handoff.delayed_final_gate_recovery_action(
        hover_action, gate_below, 2.9, cfg
    ) == pytest.approx(hover_action)
    descending = handoff.delayed_final_gate_recovery_action(
        hover_action, gate_below, 3.0, cfg
    )
    assert descending[0] == pytest.approx(0.12)
    assert descending[2] == pytest.approx(handoff._encode_thrust(0.22, cfg))

    vertically_centered = (21.0, 2.0, 1.0)
    assert handoff.delayed_final_gate_recovery_action(
        hover_action, vertically_centered, 4.0, cfg
    ) == pytest.approx(hover_action)


def test_delayed_final_gate_recovery_can_require_a_near_associated_gate():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_descent_delay_s=3.0,
        final_gate_descent_max_forward_m=25.0,
        final_gate_descent_thrust=0.22,
    )
    hover_action = [-0.24, 0.0, handoff._encode_thrust(0.27, cfg), -0.1]

    stale_descent_action = list(hover_action)
    stale_descent_action[2] = handoff._encode_thrust(0.22, cfg)
    for unavailable_or_far in (None, (30.0, 4.0, 12.0)):
        assert handoff.delayed_final_gate_recovery_action(
            stale_descent_action, unavailable_or_far, 4.0, cfg
        ) == pytest.approx(hover_action)
    descending = handoff.delayed_final_gate_recovery_action(
        hover_action, (25.0, 4.0, 12.0), 4.0, cfg
    )
    assert descending[2] == pytest.approx(handoff._encode_thrust(0.22, cfg))


def test_final_gate_hover_approach_is_delayed_bounded_and_fail_safe():
    cfg = handoff.FinalGateHandoffConfig(
        max_pitch_rad=0.12,
        center_before_accel_down_error_m=1.5,
        hover_thrust=0.27,
        final_gate_hover_approach_pitch_rad=0.03,
        final_gate_hover_approach_max_forward_m=30.0,
        final_gate_descent_delay_s=3.0,
        final_gate_descent_max_forward_m=10.0,
    )
    braking = [-0.24, 0.1, handoff._encode_thrust(0.22, cfg), -0.1]
    expected = [0.06, 0.1, handoff._encode_thrust(0.27, cfg), -0.1]

    assert handoff.final_gate_hover_approach_action(
        braking, (20.0, 2.0, 8.0), 3.0, cfg
    ) == pytest.approx(expected)
    for vector, elapsed_s in (
        ((20.0, 2.0, 8.0), 2.99),
        ((30.1, 2.0, 8.0), 3.0),
        ((10.0, 2.0, 8.0), 3.0),
        ((20.0, 2.0, 1.5), 3.0),
    ):
        assert handoff.final_gate_hover_approach_action(
            braking, vector, elapsed_s, cfg
        ) == pytest.approx(braking)

    stale_forward = [0.06, 0.1, handoff._encode_thrust(0.22, cfg), -0.1]
    fail_safe = handoff.final_gate_hover_approach_action(
        stale_forward, None, 4.0, cfg
    )
    assert fail_safe[0] == pytest.approx(-0.24)
    assert fail_safe[2] == pytest.approx(handoff._encode_thrust(0.27, cfg))


def test_final_gate_descent_pulse_bridges_one_dropout_then_expires():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_descent_delay_s=3.0,
        final_gate_descent_max_forward_m=25.0,
        final_gate_descent_pulse_s=2.0,
        final_gate_descent_thrust=0.22,
    )
    trigger = (24.0, -3.0, 12.0)
    count = 0
    last_end_s = None

    vector, started_s, saved, count, last_end_s = handoff.final_gate_descent_pulse_vector(
        trigger, 2.9, 10.0, None, None, count, last_end_s, cfg
    )
    assert vector is None
    assert started_s is None
    assert saved is None

    vector, started_s, saved, count, last_end_s = handoff.final_gate_descent_pulse_vector(
        trigger, 3.0, 10.1, started_s, saved, count, last_end_s, cfg
    )
    assert vector == pytest.approx(trigger)
    assert started_s == pytest.approx(10.1)
    assert count == 1

    vector, started_s, saved, count, last_end_s = handoff.final_gate_descent_pulse_vector(
        None, 4.0, 11.1, started_s, saved, count, last_end_s, cfg
    )
    assert vector == pytest.approx(trigger)

    vector, started_s, saved, count, last_end_s = handoff.final_gate_descent_pulse_vector(
        trigger, 5.0, 12.1, started_s, saved, count, last_end_s, cfg
    )
    assert vector is None
    assert started_s is None
    assert last_end_s == pytest.approx(12.1)

    vector, _, _, _, _ = handoff.final_gate_descent_pulse_vector(
        trigger, 6.0, 13.1, started_s, saved, count, last_end_s, cfg
    )
    assert vector is None


def test_final_gate_descent_pulses_require_cooldown_and_respect_count():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_descent_delay_s=3.0,
        final_gate_descent_max_forward_m=25.0,
        final_gate_descent_pulse_s=2.0,
        final_gate_descent_pulse_max_count=2,
        final_gate_descent_pulse_cooldown_s=1.0,
        final_gate_descent_thrust=0.22,
    )
    trigger = (24.0, 0.0, 8.0)
    state = (None, None, 0, None)

    vector, *state = handoff.final_gate_descent_pulse_vector(
        trigger, 3.0, 10.0, *state, cfg
    )
    assert vector == pytest.approx(trigger)
    assert state[2] == 1

    vector, *state = handoff.final_gate_descent_pulse_vector(
        trigger, 5.0, 12.0, *state, cfg
    )
    assert vector is None

    vector, *state = handoff.final_gate_descent_pulse_vector(
        trigger, 5.9, 12.9, *state, cfg
    )
    assert vector is None

    vector, *state = handoff.final_gate_descent_pulse_vector(
        trigger, 6.0, 13.0, *state, cfg
    )
    assert vector == pytest.approx(trigger)
    assert state[2] == 2

    vector, *state = handoff.final_gate_descent_pulse_vector(
        trigger, 9.0, 16.0, *state, cfg
    )
    assert vector is None
    assert state[2] == 2


def test_final_gate_descent_pulse_ends_early_on_fresh_vertical_alignment():
    cfg = handoff.FinalGateHandoffConfig(
        center_before_accel_down_error_m=1.5,
        final_gate_descent_delay_s=3.0,
        final_gate_descent_max_forward_m=25.0,
        final_gate_descent_pulse_s=2.0,
        final_gate_descent_thrust=0.22,
    )
    state = (None, None, 0, None)
    vector, *state = handoff.final_gate_descent_pulse_vector(
        (20.0, 0.0, 3.0), 3.0, 10.0, *state, cfg
    )
    assert vector == pytest.approx((20.0, 0.0, 3.0))

    vector, *state = handoff.final_gate_descent_pulse_vector(
        (18.0, 0.0, 1.5), 4.0, 11.0, *state, cfg
    )
    assert vector is None
    assert state[0] is None
    assert state[2] == 1
    assert state[3] == pytest.approx(11.0)


def test_later_final_gate_descent_pulse_can_be_shorter():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_descent_delay_s=3.0,
        final_gate_descent_max_forward_m=25.0,
        final_gate_descent_pulse_s=2.0,
        final_gate_descent_later_pulse_s=1.0,
        final_gate_descent_pulse_max_count=2,
        final_gate_descent_pulse_cooldown_s=1.0,
        final_gate_descent_thrust=0.22,
    )
    trigger = (20.0, 0.0, 4.0)
    state = (None, None, 0, None)
    vector, *state = handoff.final_gate_descent_pulse_vector(
        trigger, 3.0, 10.0, *state, cfg
    )
    assert vector == pytest.approx(trigger)
    vector, *state = handoff.final_gate_descent_pulse_vector(
        trigger, 5.0, 12.0, *state, cfg
    )
    assert vector is None
    vector, *state = handoff.final_gate_descent_pulse_vector(
        trigger, 6.0, 13.0, *state, cfg
    )
    assert vector == pytest.approx(trigger)
    vector, *state = handoff.final_gate_descent_pulse_vector(
        trigger, 7.0, 14.0, *state, cfg
    )
    assert vector is None
    assert state[2] == 2
    assert state[3] == pytest.approx(14.0)


def test_later_final_gate_descent_pulse_can_require_a_closer_range():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_descent_delay_s=3.0,
        final_gate_descent_max_forward_m=25.0,
        final_gate_descent_later_max_forward_m=10.0,
        final_gate_descent_pulse_s=2.0,
        final_gate_descent_later_pulse_s=1.0,
        final_gate_descent_pulse_max_count=2,
        final_gate_descent_pulse_cooldown_s=1.0,
        final_gate_descent_thrust=0.22,
    )
    state = (None, None, 0, None)
    vector, *state = handoff.final_gate_descent_pulse_vector(
        (20.0, 0.0, 5.0), 3.0, 10.0, *state, cfg
    )
    assert vector is not None
    vector, *state = handoff.final_gate_descent_pulse_vector(
        (20.0, 0.0, 5.0), 5.0, 12.0, *state, cfg
    )
    assert vector is None
    vector, *state = handoff.final_gate_descent_pulse_vector(
        (20.0, 0.0, 5.0), 6.0, 13.0, *state, cfg
    )
    assert vector is None
    assert state[2] == 1
    vector, *state = handoff.final_gate_descent_pulse_vector(
        (9.0, 0.0, 3.0), 6.1, 13.1, *state, cfg
    )
    assert vector == pytest.approx((9.0, 0.0, 3.0))
    assert state[2] == 2


def test_final_gate_descent_pulse_can_require_lateral_admission():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_descent_delay_s=3.0,
        final_gate_descent_max_forward_m=20.75,
        final_gate_descent_max_abs_right_m=4.5,
        final_gate_descent_pulse_s=0.5,
        final_gate_descent_thrust=0.22,
    )
    state = (None, None, 0, None)

    for outside in ((18.0, 4.51, 8.0), (18.0, -4.51, 8.0)):
        vector, *state = handoff.final_gate_descent_pulse_vector(
            outside, 3.0, 10.0, *state, cfg
        )
        assert vector is None
        assert state[2] == 0

    boundary = (18.0, -4.5, 8.0)
    vector, *state = handoff.final_gate_descent_pulse_vector(
        boundary, 3.0, 10.1, *state, cfg
    )
    assert vector == pytest.approx(boundary)
    assert state[2] == 1


def test_final_gate_descent_pulse_lateral_admission_is_disabled_by_default():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_descent_delay_s=3.0,
        final_gate_descent_max_forward_m=20.75,
        final_gate_descent_pulse_s=0.5,
        final_gate_descent_thrust=0.22,
    )
    trigger = (18.0, 20.0, 8.0)

    vector, _, _, count, _ = handoff.final_gate_descent_pulse_vector(
        trigger, 3.0, 10.0, None, None, 0, None, cfg
    )
    assert vector == pytest.approx(trigger)
    assert count == 1


def test_final_gate_descent_pulse_requires_fail_safe_range(monkeypatch):
    monkeypatch.setenv("PUFFER_FINAL_GATE_DESCENT_DELAY_S", "3")
    monkeypatch.setenv("PUFFER_FINAL_GATE_DESCENT_PULSE_S", "2")
    monkeypatch.setenv("PUFFER_FINAL_GATE_DESCENT_THRUST", "0.22")

    with pytest.raises(ValueError, match="requires a positive range"):
        handoff.load_config()


def test_final_gate_raw_pose_continuity_rejects_later_gate_alias():
    previous = (27.4, 5.4, 12.1)
    active_gate_return = (22.3, 5.9, 11.0)
    later_gate_alias = (38.0, -7.4, 13.1)

    assert handoff.final_gate_pose_is_associated(
        active_gate_return, previous, 8.0
    )
    assert not handoff.final_gate_pose_is_associated(
        later_gate_alias, previous, 8.0
    )


def test_final_gate_association_grace_allows_handoff_reseed_only_when_requested():
    transition_pose = (27.4, 5.7, 1.4)
    active_gate_pose = (33.1, 9.6, 7.2)

    assert not handoff.final_gate_pose_is_associated(
        active_gate_pose, transition_pose, 8.0
    )
    assert handoff.final_gate_pose_is_associated(
        active_gate_pose,
        transition_pose,
        8.0,
        allow_reseed=True,
    )


def test_final_gate_reanchor_requires_a_persistent_close_coherent_cluster():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_max_pose_jump_m=8.0,
        final_gate_max_anchor_drift_m=12.0,
        final_gate_reanchor_persistence_s=0.75,
        final_gate_descent_max_forward_m=25.0,
    )
    started_s = None
    cluster = None
    for now_s, vector in (
        (10.0, (40.0, -11.0, 12.0)),
        (10.25, (35.0, -10.0, 12.0)),
        (10.5, (29.0, -8.0, 12.5)),
        (10.75, (26.0, -7.0, 12.8)),
    ):
        reanchor, started_s, cluster = handoff.final_gate_rejected_cluster_state(
            vector, now_s, started_s, cluster, cfg
        )
        assert not reanchor

    reanchor, started_s, cluster = handoff.final_gate_rejected_cluster_state(
        (24.0, -6.0, 13.0), 11.0, started_s, cluster, cfg
    )
    assert reanchor
    assert started_s == pytest.approx(10.0)
    assert cluster == pytest.approx((24.0, -6.0, 13.0))


def test_final_gate_reanchor_resets_persistence_on_a_large_jump():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_max_pose_jump_m=8.0,
        final_gate_max_anchor_drift_m=12.0,
        final_gate_reanchor_persistence_s=0.75,
        final_gate_descent_max_forward_m=25.0,
    )
    _, started_s, cluster = handoff.final_gate_rejected_cluster_state(
        (30.0, -8.0, 12.0), 10.0, None, None, cfg
    )
    reanchor, started_s, cluster = handoff.final_gate_rejected_cluster_state(
        (20.0, 4.0, 2.0), 11.0, started_s, cluster, cfg
    )
    assert not reanchor
    assert started_s == pytest.approx(11.0)
    assert cluster == pytest.approx((20.0, 4.0, 2.0))


def test_final_gate_reanchor_can_use_a_range_distinct_from_descent():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_max_pose_jump_m=8.0,
        final_gate_max_anchor_drift_m=12.0,
        final_gate_reanchor_persistence_s=0.75,
        final_gate_reanchor_max_forward_m=30.0,
        final_gate_descent_max_forward_m=25.0,
    )
    _, started_s, cluster = handoff.final_gate_rejected_cluster_state(
        (34.0, -8.0, 12.0), 10.0, None, None, cfg
    )
    reanchor, _, _ = handoff.final_gate_rejected_cluster_state(
        (29.0, -6.0, 12.5), 10.75, started_s, cluster, cfg
    )
    assert reanchor


def test_close_final_gate_reanchor_is_delayed_centered_and_fast():
    cfg = handoff.FinalGateHandoffConfig(
        center_before_accel_down_error_m=1.5,
        final_gate_max_pose_jump_m=8.0,
        final_gate_reanchor_persistence_s=0.75,
        final_gate_descent_delay_s=3.0,
        final_gate_descent_max_forward_m=17.0,
        final_gate_close_reanchor_persistence_s=0.125,
        final_gate_close_reanchor_max_right_m=2.0,
    )
    close = (16.6, -0.4, 9.4)
    assert handoff.final_gate_close_reanchor_candidate(close, 3.0, cfg)
    for rejected, elapsed_s in (
        (close, 2.99),
        ((17.1, -0.4, 9.4), 3.0),
        ((16.6, -2.1, 9.4), 3.0),
        ((16.6, -0.4, 1.5), 3.0),
        (None, 3.0),
    ):
        assert not handoff.final_gate_close_reanchor_candidate(
            rejected, elapsed_s, cfg
        )

    reanchor, started_s, cluster = handoff.final_gate_rejected_cluster_state(
        close,
        10.0,
        None,
        None,
        cfg,
        persistence_s=cfg.final_gate_close_reanchor_persistence_s,
        max_forward_m=cfg.final_gate_descent_max_forward_m,
    )
    assert not reanchor
    reanchor, _, _ = handoff.final_gate_rejected_cluster_state(
        (16.3, -0.3, 9.7),
        10.125,
        started_s,
        cluster,
        cfg,
        persistence_s=cfg.final_gate_close_reanchor_persistence_s,
        max_forward_m=cfg.final_gate_descent_max_forward_m,
    )
    assert reanchor


def test_close_final_gate_reanchor_count_defaults_to_one_and_is_configurable(
    monkeypatch,
):
    assert handoff.FinalGateHandoffConfig().final_gate_close_reanchor_max_count == 1

    monkeypatch.setenv("PUFFER_FINAL_GATE_CLOSE_REANCHOR_MAX_COUNT", "2")
    assert handoff.load_config().final_gate_close_reanchor_max_count == 2


def test_crossing_final_gate_reanchor_is_delayed_tight_and_coherent():
    cfg = handoff.FinalGateHandoffConfig(
        final_gate_max_pose_jump_m=8.0,
        final_gate_descent_delay_s=3.0,
        final_gate_crossing_reanchor_persistence_s=0.125,
        final_gate_crossing_reanchor_max_forward_m=5.0,
        final_gate_crossing_reanchor_max_right_m=1.0,
        final_gate_crossing_reanchor_max_abs_down_m=1.5,
    )
    first = (2.98, -0.49, 0.84)
    second = (3.62, 0.15, 1.12)
    assert handoff.final_gate_crossing_reanchor_candidate(first, 3.0, cfg)
    assert handoff.final_gate_crossing_reanchor_candidate(second, 3.125, cfg)
    for rejected, elapsed_s in (
        (first, 2.99),
        ((5.01, 0.0, 0.0), 3.0),
        ((3.0, 1.01, 0.0), 3.0),
        ((3.0, 0.0, -1.51), 3.0),
        ((3.0, 0.0, 1.51), 3.0),
        (None, 3.0),
    ):
        assert not handoff.final_gate_crossing_reanchor_candidate(
            rejected, elapsed_s, cfg
        )

    reanchor, started_s, cluster = handoff.final_gate_rejected_cluster_state(
        first,
        20.453,
        None,
        None,
        cfg,
        persistence_s=cfg.final_gate_crossing_reanchor_persistence_s,
        max_forward_m=cfg.final_gate_crossing_reanchor_max_forward_m,
    )
    assert not reanchor
    reanchor, _, _ = handoff.final_gate_rejected_cluster_state(
        second,
        20.578,
        started_s,
        cluster,
        cfg,
        persistence_s=cfg.final_gate_crossing_reanchor_persistence_s,
        max_forward_m=cfg.final_gate_crossing_reanchor_max_forward_m,
    )
    assert reanchor


def test_crossing_final_gate_reanchor_defaults_disabled_and_requires_all_bounds(
    monkeypatch,
):
    cfg = handoff.FinalGateHandoffConfig()
    assert not handoff.final_gate_crossing_reanchor_candidate(
        (3.0, 0.0, 0.0), 10.0, cfg
    )

    monkeypatch.setenv("PUFFER_FINAL_GATE_CROSSING_REANCHOR_PERSISTENCE_S", "0.125")
    with pytest.raises(ValueError, match="requires all bounds"):
        handoff.load_config()

    monkeypatch.setenv("PUFFER_FINAL_GATE_CLOSE_REANCHOR_MAX_COUNT", "0")
    with pytest.raises(ValueError, match="close final-gate re-anchor count"):
        handoff.load_config()
