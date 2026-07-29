import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts import policy_callable_six_gate_hybrid as hybrid


def _observation(gate: int) -> np.ndarray:
    values = np.zeros(32, dtype=np.float32)
    values[10] = 1.0
    values[23] = gate / 6.0
    if gate < 6:
        values[24 + gate] = 1.0
    values[17] = 0.95
    values[30] = 0.03125
    values[18] = 0.125
    values[22] = -0.375
    values[31] = 0.4375
    return values


def test_legacy_prefix_bridge_restores_confidence_time_and_preserves_last_yaw():
    values = _observation(1)

    legacy = hybrid._legacy_prefix_observation(values)

    assert legacy.shape == (23,)
    assert legacy[17] == pytest.approx(0.03125)
    assert legacy[18] == pytest.approx(0.4375)
    assert legacy[22] == pytest.approx(-0.375)
    assert legacy[:17] == pytest.approx(values[:17])
    assert legacy[19:22] == pytest.approx(values[19:22])


def test_tail_bridge_clears_prefix_only_reserved_slot():
    values = _observation(4)
    values[31] = 0.2

    current = hybrid._tail_observation(values)

    assert current[:30] == pytest.approx(values[:30])
    assert current[30:32] == pytest.approx((0.0, 0.0))


def test_prefix_owns_gates_zero_through_two(monkeypatch):
    seen = []

    class Prefix:
        def infer(self, observation):
            seen.append(tuple(observation))
            return [0.1, 0.2, 0.3, 0.4]

    monkeypatch.setattr(hybrid, "_resolve_prefix", lambda: Prefix())
    monkeypatch.setattr(
        hybrid.tail,
        "_promoted_gate3_intercept",
        lambda _observation, actions: actions,
    )

    assert hybrid.infer(_observation(0)) == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert hybrid.infer(_observation(2)) == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert len(seen) == 2


def test_gate3_live_path_is_exact_n160_promoted_then_close_severe(monkeypatch):
    calls = []

    class Prefix:
        def infer(self, _observation):
            return [0.1, 0.2, 0.3, 0.4]

    def promoted(_observation, actions):
        calls.append("promoted")
        actions[1] = -0.4
        return actions

    def close_severe(_observation, actions):
        calls.append("close_severe")
        actions[1] = 0.25
        return actions

    def retired(*_args, **_kwargs):
        raise AssertionError("retired Gate-3 helper entered the live path")

    monkeypatch.setattr(hybrid, "_resolve_prefix", lambda: Prefix())
    monkeypatch.setattr(hybrid.tail, "_promoted_gate3_intercept", promoted)
    monkeypatch.setattr(hybrid, "_gate3_close_severe_boost", close_severe)
    for name in (
        "_gate3_terminal_margin_boost",
        "_gate3_projected_vertical_floor",
        "_gate3_terminal_level",
        "_gate3_late_positive_residual_roll_floor",
        "_gate3_minimum_energy_lateral_controller",
    ):
        monkeypatch.setattr(hybrid, name, retired)

    assert hybrid.infer(_observation(2)) == pytest.approx((0.1, 0.25, 0.3, 0.4))
    assert calls == ["promoted", "close_severe"]


def test_gate1_late_residual_roll_floor_retains_positive_miss_correction():
    values = _observation(0)
    values[11] = np.tanh(3.764 / 10.0)
    values[12] = np.tanh(0.953 / 5.0)
    values[0] = np.tanh(-6.467 / 5.0)
    values[1] = np.tanh(0.477 / 3.0)

    governed = hybrid._gate1_late_residual_roll_floor(
        values,
        [0.2361, -0.2733, -0.1394, 0.000153],
    )

    assert governed == pytest.approx((0.2361, -0.7, -0.1394, 0.000153))


def test_gate1_late_residual_roll_floor_preserves_bounds_and_stronger_roll():
    actions = [0.2, -0.3, -0.1, 0.001]
    safe = _observation(0)
    safe[11] = np.tanh(4.0 / 10.0)
    safe[12] = np.tanh(0.2 / 5.0)
    safe[0] = np.tanh(-7.0 / 5.0)
    safe[1] = np.tanh(0.1 / 3.0)
    assert hybrid._gate1_late_residual_roll_floor(safe, actions.copy()) == actions

    opposite = safe.copy()
    opposite[12] = np.tanh(-1.0 / 5.0)
    assert hybrid._gate1_late_residual_roll_floor(opposite, actions.copy()) == actions

    far = safe.copy()
    far[11] = np.tanh(6.01 / 10.0)
    far[12] = np.tanh(1.2 / 5.0)
    assert hybrid._gate1_late_residual_roll_floor(far, actions.copy()) == actions

    hidden = safe.copy()
    hidden[10] = 0.0
    hidden[12] = np.tanh(1.2 / 5.0)
    assert hybrid._gate1_late_residual_roll_floor(hidden, actions.copy()) == actions

    unsafe = safe.copy()
    unsafe[12] = np.tanh(1.0 / 5.0)
    stronger = [0.2, -0.8, -0.1, 0.001]
    assert hybrid._gate1_late_residual_roll_floor(unsafe, stronger.copy()) == stronger


def test_tail_advances_at_gate_three_but_safe_handoff_owns_output(monkeypatch):
    seen = []
    monkeypatch.setattr(
        hybrid.tail,
        "infer",
        lambda observation: seen.append(tuple(observation)) or [0.0] * 4,
    )
    hybrid._clear_gate4_state()

    assert hybrid.infer(_observation(3)) == pytest.approx((-0.24, 0.0, 0.0, 0.0))
    assert seen[0][30:32] == (0.0, 0.0)


def _gate4_observation(
    forward_m: float,
    right_m: float,
    down_m: float,
    *,
    yaw_error_rad: float = 0.0,
) -> np.ndarray:
    values = _observation(3)
    values[11] = np.tanh(forward_m / 10.0)
    values[12] = np.tanh(right_m / 5.0)
    values[13] = np.tanh(down_m / 5.0)
    values[14] = yaw_error_rad / (np.pi / 4.0)
    values[6] = 1.0
    return values


def test_gate4_safe_handoff_rejects_far_lateral_initial_anchor():
    hybrid._clear_gate4_state()
    values = _gate4_observation(28.235, 11.382, 8.294)

    action = hybrid._gate4_safe_handoff(
        values,
        [1.0, -0.946, -1.0, 0.3],
        now_s=100.0,
    )

    assert action == pytest.approx((-0.24, 0.0, 0.0, 0.0), abs=1e-6)
    assert hybrid._GATE4_ANCHOR_VECTOR is None


def test_gate4_safe_handoff_yaws_only_toward_persistent_rejected_pose():
    hybrid._clear_gate4_state()
    values = _gate4_observation(
        28.235,
        11.382,
        8.294,
        yaw_error_rad=0.6,
    )
    hybrid._gate4_safe_handoff(
        values,
        [1.0, -0.946, -1.0, 0.3],
        now_s=100.0,
    )

    action = hybrid._gate4_safe_handoff(
        values,
        [1.0, -0.946, -1.0, 0.3],
        now_s=101.1,
    )

    assert action == pytest.approx(
        (
            -0.24,
            0.0,
            0.0,
            -hybrid.GATE4_ACQUISITION_MAX_YAW_STEP_RAD / np.pi,
        ),
        abs=1e-6,
    )
    assert hybrid._GATE4_ANCHOR_VECTOR is None


def test_gate4_safe_handoff_starts_vertical_control_after_association_grace():
    hybrid._clear_gate4_state()
    hybrid._gate4_safe_handoff(
        _gate4_observation(30.0, 11.0, 12.0),
        [1.0, -1.0, -1.0, 0.0],
        now_s=10.0,
    )

    action = hybrid._gate4_safe_handoff(
        _gate4_observation(25.4, 0.55, 13.8),
        [1.0, -1.0, -1.0, 0.0],
        now_s=12.0,
    )

    assert action == pytest.approx(
        (-0.24, -0.022, hybrid._encode_gate4_thrust(0.22), 0.0),
        abs=1e-6,
    )
    assert hybrid._GATE4_ANCHOR_VECTOR == pytest.approx((25.4, 0.55, 13.8))


def test_gate4_safe_handoff_requires_coherent_close_reanchor():
    hybrid._clear_gate4_state()
    hybrid._gate4_safe_handoff(
        _gate4_observation(28.0, 3.0, 8.0),
        [1.0, -1.0, -1.0, 0.0],
        now_s=10.0,
    )
    first_rejected = hybrid._gate4_safe_handoff(
        _gate4_observation(19.0, 2.0, 3.0),
        [1.0, -1.0, -1.0, 0.0],
        now_s=14.0,
    )
    still_waiting = hybrid._gate4_safe_handoff(
        _gate4_observation(18.5, 1.9, 2.9),
        [1.0, -1.0, -1.0, 0.0],
        now_s=14.124,
    )
    reanchored = hybrid._gate4_safe_handoff(
        _gate4_observation(18.0, 1.8, 2.8),
        [1.0, -1.0, -1.0, 0.0],
        now_s=14.126,
    )

    assert first_rejected == pytest.approx((-0.24, 0.0, 0.0, 0.0))
    assert still_waiting == pytest.approx((-0.24, 0.0, 0.0, 0.0))
    assert reanchored == pytest.approx(
        (-0.24, -0.072, hybrid._encode_gate4_thrust(0.22), 0.0),
        abs=1e-6,
    )
    assert hybrid._GATE4_REANCHOR_COUNT == 1


def test_gate4_safe_handoff_recenters_close_aperture_without_saturation():
    hybrid._clear_gate4_state()
    values = _gate4_observation(4.909, 3.620, 1.350)
    hybrid._gate4_safe_handoff(values, [1.0, -1.0, -1.0, 0.0], now_s=20.0)

    action = hybrid._gate4_safe_handoff(
        values,
        [1.0, -0.943, -0.943, 0.0],
        now_s=24.0,
    )

    assert action == pytest.approx(
        (
            hybrid.GATE4_ALIGNED_FORWARD_PITCH_RAD / 0.5,
            -hybrid.GATE4_LATERAL_ROLL_KP * 3.620 / 0.5,
            hybrid._encode_gate4_thrust(0.243),
            0.0,
        ),
        abs=1e-6,
    )


def test_gate4_safe_handoff_brakes_hovers_and_freezes_yaw_on_dropout():
    hybrid._clear_gate4_state()
    values = _gate4_observation(10.0, 1.0, 1.0)
    yaw_rad = 0.4
    values[6] = np.cos(yaw_rad / 2.0)
    values[9] = np.sin(yaw_rad / 2.0)
    values[10] = 0.0

    action = hybrid._gate4_safe_handoff(
        values,
        [0.8, -0.9, -1.0, -0.7],
        now_s=30.0,
    )

    assert action == pytest.approx((-0.24, 0.0, 0.0, yaw_rad / np.pi), abs=1e-6)


def test_gate4_safe_handoff_carries_last_visible_yaw_target_for_half_second():
    hybrid._clear_gate4_state()
    visible = _gate4_observation(
        10.0,
        1.0,
        1.0,
        yaw_error_rad=0.2,
    )
    hybrid._gate4_safe_handoff(
        visible,
        [0.8, -0.9, -1.0, -0.7],
        now_s=40.0,
    )
    dropout = visible.copy()
    dropout[10] = 0.0
    dropout[6:10] = 0.0
    current_yaw_rad = -0.05
    dropout[6] = np.cos(current_yaw_rad / 2.0)
    dropout[9] = np.sin(current_yaw_rad / 2.0)

    carried = hybrid._gate4_safe_handoff(
        dropout,
        [0.8, -0.9, -1.0, -0.7],
        now_s=40.2,
    )
    expired = hybrid._gate4_safe_handoff(
        dropout,
        [0.8, -0.9, -1.0, -0.7],
        now_s=40.501,
    )

    assert carried == pytest.approx((-0.24, 0.0, 0.0, -0.2 / np.pi), abs=1e-6)
    assert expired == pytest.approx(
        (-0.24, 0.0, 0.0, current_yaw_rad / np.pi),
        abs=1e-6,
    )


def test_gate3_close_severe_boost_changes_only_roll(monkeypatch):
    values = _observation(2)
    # Observable legacy scaling: forward=3 m, right=-1.2 m, closing=8 m/s.
    values[11] = np.tanh(3.0 / 10.0)
    values[12] = np.tanh(-1.2 / 5.0)
    values[0] = np.tanh(-8.0 / 5.0)
    values[1] = 0.0
    actions = [0.2, 0.91, -0.3, 0.004]

    boosted = hybrid._gate3_close_severe_boost(values, actions.copy())

    assert boosted == pytest.approx((0.2, 1.0, -0.3, 0.004))


def test_gate3_close_severe_boost_uses_original_four_metre_counter_bound():
    values = _observation(2)
    values[11] = np.tanh(3.95 / 10.0)
    values[12] = np.tanh(-0.4 / 5.0)
    values[0] = np.tanh(-8.0 / 5.0)
    actions = [0.2, -1.0, -0.3, 0.004]

    suppressed = hybrid._gate3_close_severe_boost(values, actions.copy())
    values[11] = np.tanh(4.01 / 10.0)
    outside = hybrid._gate3_close_severe_boost(values, actions.copy())

    assert suppressed == pytest.approx((0.2, 0.0, -0.3, 0.004))
    assert outside == actions


def test_gate3_close_severe_boost_preserves_right_side_counter_mode():
    values = _observation(2)
    values[11] = np.tanh(3.0 / 10.0)
    values[12] = np.tanh(0.4 / 5.0)
    values[0] = np.tanh(-8.0 / 5.0)
    actions = [0.2, -1.0, -0.3, 0.004]

    assert hybrid._gate3_close_severe_boost(values, actions.copy()) == actions


def test_gate3_close_severe_boost_does_not_suppress_counter_outside_window():
    values = _observation(2)
    values[11] = np.tanh(7.1 / 10.0)
    values[12] = np.tanh(-0.4 / 5.0)
    values[0] = np.tanh(-8.0 / 5.0)
    actions = [0.2, -1.0, -0.3, 0.004]

    assert hybrid._gate3_close_severe_boost(values, actions.copy()) == actions


def test_gate3_terminal_margin_boost_bridges_to_safe_level_threshold():
    values = _observation(2)
    values[11] = np.tanh(5.7 / 10.0)
    values[12] = np.tanh(-1.8 / 5.0)
    values[0] = np.tanh(-8.0 / 5.0)
    values[1] = np.tanh(2.4 / 3.0)
    actions = [0.2, 0.0, -0.3, 0.004]

    governed = hybrid._gate3_terminal_margin_boost(values, actions.copy())

    assert governed == pytest.approx((0.2, 0.3, -0.3, 0.004))


def test_gate3_terminal_margin_boost_preserves_safe_and_stronger_roll():
    values = _observation(2)
    values[11] = np.tanh(3.0 / 10.0)
    values[12] = np.tanh(-0.6 / 5.0)
    values[0] = np.tanh(-8.0 / 5.0)
    values[1] = np.tanh(2.4 / 3.0)
    safe = [0.2, 0.0, -0.3, 0.004]
    assert hybrid._gate3_terminal_margin_boost(values, safe.copy()) == safe

    values[12] = np.tanh(-1.4 / 5.0)
    stronger = [0.2, 0.9, -0.3, 0.004]
    assert hybrid._gate3_terminal_margin_boost(values, stronger.copy()) == stronger


def test_gate3_minimum_energy_controller_steers_clean_reference_rightward():
    hybrid._clear_gate3_state()
    values = _observation(2)
    values[11] = np.tanh(6.418 / 10.0)
    values[12] = np.tanh(-2.224 / 5.0)
    values[0] = np.tanh(-7.58 / 5.0)
    values[1] = np.tanh(2.10 / 3.0)
    values[30] = 0.2

    committed = hybrid._gate3_minimum_energy_lateral_controller(
        values,
        [0.2, 0.0, -0.3, 0.004],
    )
    assert committed == pytest.approx((0.2, 0.5, -0.3, 0.004))

    values[11] = np.tanh(4.154 / 10.0)
    held = hybrid._gate3_minimum_energy_lateral_controller(
        values, [0.21, -1.0, -0.2, 0.003]
    )
    assert held == pytest.approx((0.21, 0.5, -0.2, 0.003))

    values[11] = np.tanh(3.072 / 10.0)
    leveled = hybrid._gate3_minimum_energy_lateral_controller(
        values, [0.22, 1.0, -0.1, 0.002]
    )
    assert leveled == pytest.approx((0.22, 0.0, -0.1, 0.002))


def test_gate3_minimum_energy_controller_commits_candidate043_leftward():
    hybrid._clear_gate3_state()
    values = _observation(2)
    values[11] = np.tanh(5.65 / 10.0)
    values[12] = np.tanh(-1.681 / 5.0)
    values[0] = np.tanh(-8.31 / 5.0)
    values[1] = np.tanh(2.56 / 3.0)
    values[30] = 0.2

    committed = hybrid._gate3_minimum_energy_lateral_controller(
        values,
        [0.24, 0.0, 0.12, 0.0],
    )
    assert committed == pytest.approx((0.24, -0.5, 0.12, 0.0))

    values[11] = np.tanh(5.14 / 10.0)
    held = hybrid._gate3_minimum_energy_lateral_controller(
        values, [0.25, 1.0, 0.11, 0.001]
    )
    assert held == pytest.approx((0.25, -0.5, 0.11, 0.001))

    values[11] = np.tanh(3.53 / 10.0)
    leveled = hybrid._gate3_minimum_energy_lateral_controller(
        values, [0.26, -1.0, 0.10, 0.002]
    )
    assert leveled == pytest.approx((0.26, 0.0, 0.10, 0.002))

    values[11] = np.tanh(20.0 / 10.0)
    still_level = hybrid._gate3_minimum_energy_lateral_controller(
        values, [0.27, 1.0, 0.09, 0.003]
    )
    assert still_level == pytest.approx((0.27, 0.0, 0.09, 0.003))


def test_gate3_minimum_energy_controller_can_return_unsaturated_solution():
    hybrid._clear_gate3_state()
    values = _observation(2)
    values[11] = np.tanh(5.0 / 10.0)
    values[12] = np.tanh(-1.727 / 5.0)
    values[0] = np.tanh(-8.0 / 5.0)
    values[1] = np.tanh(2.0 / 3.0)
    values[30] = 0.2

    governed = hybrid._gate3_minimum_energy_lateral_controller(
        values,
        [0.218, -1.0, 0.215, 0.0],
    )

    assert governed[0] == pytest.approx(0.218)
    assert governed[2:] == pytest.approx((0.215, 0.0))
    assert 0.0 < governed[1] < 0.5


def test_gate3_minimum_energy_controller_preserves_untrusted_and_outside_window():
    hybrid._clear_gate3_state()
    values = _observation(2)
    values[11] = np.tanh(4.12 / 10.0)
    values[12] = np.tanh(-1.4 / 5.0)
    values[0] = np.tanh(-8.24 / 5.0)
    values[1] = np.tanh(2.10 / 3.0)
    actions = [0.2, 0.4, -0.3, 0.004]

    low_confidence = values.copy()
    low_confidence[30] = 0.09
    assert hybrid._gate3_minimum_energy_lateral_controller(
        low_confidence, actions.copy()
    ) == actions

    hybrid._clear_gate3_state()
    hidden = values.copy()
    hidden[10] = 0.0
    assert hybrid._gate3_minimum_energy_lateral_controller(
        hidden, actions.copy()
    ) == actions

    hybrid._clear_gate3_state()
    far = values.copy()
    far[11] = np.tanh(6.51 / 10.0)
    assert hybrid._gate3_minimum_energy_lateral_controller(
        far, actions.copy()
    ) == actions

    hybrid._clear_gate3_state()
    receding = values.copy()
    receding[0] = np.tanh(0.1 / 5.0)
    assert hybrid._gate3_minimum_energy_lateral_controller(
        receding, actions.copy()
    ) == actions


def test_gate3_minimum_energy_controller_reset_clears_commit_latches():
    hybrid._clear_gate3_state()
    values = _observation(2)
    values[11] = np.tanh(5.65 / 10.0)
    values[12] = np.tanh(-1.681 / 5.0)
    values[0] = np.tanh(-8.31 / 5.0)
    values[1] = np.tanh(2.56 / 3.0)
    values[30] = 0.2
    hybrid._gate3_minimum_energy_lateral_controller(
        values, [0.2, 0.0, -0.3, 0.004]
    )

    hybrid._clear_gate3_state()
    values[30] = 0.09
    actions = [0.2, 0.4, -0.3, 0.004]
    assert hybrid._gate3_minimum_energy_lateral_controller(
        values, actions.copy()
    ) == actions


def test_gate3_late_positive_residual_roll_floor_counters_trustworthy_overshoot():
    values = _observation(2)
    values[11] = np.tanh(3.623 / 10.0)
    values[12] = np.tanh(0.039 / 5.0)
    values[0] = np.tanh(-7.334 / 5.0)
    values[1] = np.tanh(2.564 / 3.0)
    values[30] = 0.20625

    governed = hybrid._gate3_late_positive_residual_roll_floor(
        values,
        [0.2, 0.0, -0.3, 0.004],
    )

    assert governed == pytest.approx((0.2, -1.0, -0.3, 0.004))


def test_gate3_late_positive_residual_roll_floor_preserves_bounded_cases():
    values = _observation(2)
    values[11] = np.tanh(2.434 / 10.0)
    values[12] = np.tanh(1.103 / 5.0)
    values[0] = np.tanh(-7.8 / 5.0)
    values[1] = np.tanh(3.071 / 3.0)
    values[30] = 0.546875
    actions = [0.2, 0.0, -0.3, 0.004]

    stale = values.copy()
    stale[30] = 0.01875
    assert hybrid._gate3_late_positive_residual_roll_floor(
        stale, actions.copy()
    ) == actions

    left = values.copy()
    left[12] = np.tanh(-0.1 / 5.0)
    assert hybrid._gate3_late_positive_residual_roll_floor(
        left, actions.copy()
    ) == actions

    slow = values.copy()
    slow[1] = np.tanh(1.99 / 3.0)
    assert hybrid._gate3_late_positive_residual_roll_floor(
        slow, actions.copy()
    ) == actions

    far = values.copy()
    far[11] = np.tanh(4.01 / 10.0)
    assert hybrid._gate3_late_positive_residual_roll_floor(
        far, actions.copy()
    ) == actions

    stronger = [0.2, -1.0, -0.3, 0.004]
    assert hybrid._gate3_late_positive_residual_roll_floor(
        values, stronger.copy()
    ) == stronger


@pytest.mark.parametrize(
    ("forward_m", "right_m", "forward_rate_m_s", "right_rate_m_s", "expected_roll"),
    (
        (2.046908, 0.821962, -9.42, -0.53, -0.3076772),
        (2.601626, -0.252033, -9.78, -1.53, 0.6867198),
    ),
)
def test_gate2_projected_aperture_governor_brakes_both_edge_strikes(
    forward_m, right_m, forward_rate_m_s, right_rate_m_s, expected_roll
):
    hybrid._clear_gate2_state()
    values = _observation(1)
    values[11] = np.tanh(forward_m / 10.0)
    values[12] = np.tanh(right_m / 5.0)
    values[0] = np.tanh(forward_rate_m_s / 5.0)
    values[1] = np.tanh(right_rate_m_s / 3.0)
    actions = [0.34, -0.09, 0.05, 0.001]

    governed = hybrid._gate2_projected_aperture_governor(values, actions.copy())

    assert governed == pytest.approx(
        (hybrid.GATE2_BRAKE_PITCH_NORM, expected_roll, 0.05, 0.001),
        abs=1e-5,
    )


def test_gate2_projected_aperture_governor_leaves_safe_projection_unchanged():
    hybrid._clear_gate2_state()
    values = _observation(1)
    values[11] = np.tanh(4.0 / 10.0)
    values[12] = np.tanh(0.4 / 5.0)
    values[0] = np.tanh(-8.0 / 5.0)
    values[1] = np.tanh(-0.4 / 3.0)
    actions = [0.3, -0.1, 0.05, 0.001]

    assert hybrid._gate2_projected_aperture_governor(values, actions.copy()) == actions


def test_gate2_projected_aperture_governor_is_bounded_to_visible_window():
    hybrid._clear_gate2_state()
    actions = [0.3, -0.1, 0.05, 0.001]
    far = _observation(1)
    far[11] = np.tanh(12.1 / 10.0)
    far[12] = np.tanh(2.0 / 5.0)
    assert hybrid._gate2_projected_aperture_governor(far, actions.copy()) == actions

    hidden = _observation(1)
    hidden[10] = 0.0
    hidden[11] = np.tanh(3.0 / 10.0)
    hidden[12] = np.tanh(2.0 / 5.0)
    assert hybrid._gate2_projected_aperture_governor(hidden, actions.copy()) == actions


def test_gate2_late_residual_roll_floor_retains_positive_miss_correction():
    values = _observation(1)
    values[11] = np.tanh(3.157895 / 10.0)
    values[12] = np.tanh(0.927632 / 5.0)
    values[0] = np.tanh(-9.105545 / 5.0)
    values[1] = np.tanh(-0.381434 / 3.0)

    governed = hybrid._gate2_late_residual_roll_floor(
        values,
        [-0.2, -0.423077, 0.302485, -0.00036],
    )

    assert governed == pytest.approx((-0.2, -0.5, 0.302485, -0.00036))


def test_gate2_late_residual_roll_floor_preserves_bounds_and_stronger_roll():
    actions = [-0.2, -0.3, 0.2, 0.001]
    safe = _observation(1)
    safe[11] = np.tanh(3.0 / 10.0)
    safe[12] = np.tanh(0.6 / 5.0)
    safe[0] = np.tanh(-9.0 / 5.0)
    safe[1] = np.tanh(-0.4 / 3.0)
    assert hybrid._gate2_late_residual_roll_floor(safe, actions.copy()) == actions

    opposite = safe.copy()
    opposite[12] = np.tanh(-1.0 / 5.0)
    assert hybrid._gate2_late_residual_roll_floor(opposite, actions.copy()) == actions

    far = safe.copy()
    far[11] = np.tanh(4.01 / 10.0)
    far[12] = np.tanh(1.2 / 5.0)
    assert hybrid._gate2_late_residual_roll_floor(far, actions.copy()) == actions

    unsafe = safe.copy()
    unsafe[12] = np.tanh(1.0 / 5.0)
    stronger = [-0.2, -0.7, 0.2, 0.001]
    assert hybrid._gate2_late_residual_roll_floor(unsafe, stronger.copy()) == stronger


def test_gate2_projected_vertical_floor_is_moderate_and_stateless():
    hybrid._clear_gate2_state()
    values = _observation(1)
    values[11] = np.tanh(12.0 / 10.0)
    values[13] = np.tanh(0.2625 / 5.0)
    values[0] = np.tanh(-7.0 / 5.0)
    values[2] = np.tanh(-1.781 / 3.0)

    learned = [-0.2, -0.08, 0.084, 0.001]
    governed = hybrid._gate2_projected_vertical_floor(
        values,
        learned.copy(),
    )

    assert not hybrid.GATE2_VERTICAL_FLOOR_RETIRED
    assert hybrid.GATE2_VERTICAL_FLOOR_STATELESS
    assert not hybrid._GATE2_VERTICAL_FLOOR_LATCHED
    assert governed == pytest.approx((-0.2, -0.08, 0.3, 0.001))

    stronger = hybrid._gate2_projected_vertical_floor(
        values,
        [0.3, 0.2, 0.75, -0.002],
    )
    assert stronger == pytest.approx((0.3, 0.2, 0.75, -0.002))


def test_gate2_projected_vertical_floor_rejects_safe_hidden_and_far_samples():
    actions = [-0.2, -0.08, 0.084, 0.001]
    for visible, forward_m, projected_family in (
        (1.0, 12.0, "safe"),
        (0.0, 12.0, "severe"),
        (1.0, 12.1, "severe"),
    ):
        hybrid._clear_gate2_state()
        values = _observation(1)
        values[10] = visible
        values[11] = np.tanh(forward_m / 10.0)
        values[13] = np.tanh((0.0 if projected_family == "safe" else -1.0) / 5.0)
        values[0] = np.tanh(-7.0 / 5.0)
        values[2] = np.tanh((0.0 if projected_family == "safe" else -2.0) / 3.0)
        assert hybrid._gate2_projected_vertical_floor(values, actions.copy()) == actions
        assert not hybrid._GATE2_VERTICAL_FLOOR_LATCHED


def test_gate2_early_window_brakes_without_changing_roll_until_close():
    hybrid._clear_gate2_state()
    actions = [0.38, -0.22, 0.09, 0.001]
    early = _observation(1)
    early[11] = np.tanh(11.847 / 10.0)
    early[12] = np.tanh(1.539 / 5.0)
    early[0] = np.tanh(-8.0 / 5.0)
    early[1] = np.tanh(-0.473 / 3.0)

    admitted = hybrid._gate2_projected_aperture_governor(
        early, actions.copy()
    )

    assert hybrid._GATE2_GOVERNOR_LATCHED
    assert admitted == pytest.approx(
        (hybrid.GATE2_BRAKE_PITCH_NORM, -0.22, 0.09, 0.001)
    )

    close = _observation(1)
    close[11] = np.tanh(4.0 / 10.0)
    close[12] = np.tanh(0.8 / 5.0)
    close[0] = np.tanh(-7.0 / 5.0)
    close[1] = np.tanh(-0.7 / 3.0)
    retained = hybrid._gate2_projected_aperture_governor(
        close, [0.4, -0.05, 0.09, 0.001]
    )
    assert retained[0] == pytest.approx(hybrid.GATE2_BRAKE_PITCH_NORM)
    assert retained[1] == pytest.approx(-0.235, abs=1e-5)
    assert retained[2:] == pytest.approx((0.09, 0.001))


def test_gate2_governor_latches_after_measured_point_three_crossing(monkeypatch):
    hybrid._clear_gate2_state()
    first = _observation(1)
    first[11] = np.tanh(4.916 / 10.0)
    first[12] = np.tanh(0.685 / 5.0)
    first[0] = np.tanh(-9.757 / 5.0)
    first[1] = np.tanh(-0.701 / 3.0)
    actions = [0.268, -0.013, 0.302, 0.0]

    admitted = hybrid._gate2_projected_aperture_governor(first, actions.copy())
    assert hybrid._GATE2_GOVERNOR_LATCHED
    assert admitted[0] == pytest.approx(hybrid.GATE2_BRAKE_PITCH_NORM)
    assert admitted[1] == pytest.approx(-0.16565, abs=1e-5)

    later_safe = _observation(1)
    later_safe[11] = np.tanh(2.1 / 10.0)
    later_safe[12] = np.tanh(0.57 / 5.0)
    later_safe[0] = np.tanh(-9.0 / 5.0)
    later_safe[1] = np.tanh(-0.64 / 3.0)
    retained = hybrid._gate2_projected_aperture_governor(
        later_safe, [0.4, 0.01, 0.04, 0.0]
    )
    assert retained[0] == pytest.approx(hybrid.GATE2_BRAKE_PITCH_NORM)
    assert retained[1] != pytest.approx(0.01)

    class Prefix:
        def infer(self, _observation):
            return [0.0] * 4

    monkeypatch.setattr(hybrid, "_resolve_prefix", lambda: Prefix())
    monkeypatch.setattr(
        hybrid.tail,
        "_promoted_gate3_intercept",
        lambda _observation, actions: actions,
    )
    hybrid.infer(_observation(2))
    assert not hybrid._GATE2_GOVERNOR_LATCHED
    assert not hybrid._GATE2_VERTICAL_FLOOR_LATCHED


def test_gate3_terminal_level_preserves_pitch_thrust_and_yaw():
    hybrid._clear_gate3_state()
    values = _observation(2)
    values[11] = np.tanh(5.38 / 10.0)
    values[12] = np.tanh(-1.35 / 5.0)
    values[13] = np.tanh(-0.09 / 5.0)
    values[0] = np.tanh(-7.9 / 5.0)
    values[1] = np.tanh(2.27 / 3.0)
    values[2] = np.tanh(-1.69 / 3.0)

    governed = hybrid._gate3_terminal_level(
        values,
        [0.24, -1.0, 0.06, -0.001],
    )

    assert hybrid._GATE3_TERMINAL_LEVEL_LATCHED
    assert governed == pytest.approx((0.24, 0.0, 0.06, -0.001))

    hidden = _observation(2)
    hidden[10] = 0.0
    retained = hybrid._gate3_terminal_level(
        hidden,
        [0.3, 0.8, -0.2, 0.002],
    )
    assert retained == pytest.approx((0.3, 0.0, -0.2, 0.002))


def test_gate3_projected_lateral_miss_does_not_change_any_action():
    hybrid._clear_gate3_state()
    values = _observation(2)
    values[11] = np.tanh(11.0 / 10.0)
    values[12] = np.tanh(-3.38 / 5.0)
    values[13] = np.tanh(0.0)
    values[0] = np.tanh(-8.0 / 5.0)
    values[1] = np.tanh(2.4 / 3.0)
    values[2] = np.tanh(0.0)
    actions = [0.24, -1.0, 0.10, -0.001]

    governed = hybrid._gate3_projected_vertical_floor(values, actions.copy())

    assert not hybrid._GATE3_VERTICAL_FLOOR_LATCHED
    assert governed == pytest.approx(actions)

    safe_close = _observation(2)
    safe_close[11] = np.tanh(4.0 / 10.0)
    safe_close[12] = np.tanh(-0.4 / 5.0)
    safe_close[13] = np.tanh(0.2 / 5.0)
    safe_close[0] = np.tanh(-6.0 / 5.0)
    retained = hybrid._gate3_projected_vertical_floor(
        safe_close,
        [0.3, 0.7, 0.2, 0.002],
    )
    assert retained == pytest.approx((0.3, 0.7, 0.2, 0.002))


def test_gate3_early_negative_vertical_miss_floors_only_low_thrust():
    hybrid._clear_gate3_state()
    values = _observation(2)
    values[11] = np.tanh(9.751 / 10.0)
    values[12] = np.tanh(-2.5 / 5.0)
    values[13] = np.tanh(1.0 / 5.0)
    values[0] = np.tanh(-7.5 / 5.0)
    values[1] = np.tanh(1.5 / 3.0)
    values[2] = np.tanh(-2.5 / 3.0)

    floored = hybrid._gate3_projected_vertical_floor(
        values,
        [0.3, -1.0, -0.210, -0.001],
    )

    assert hybrid._GATE3_VERTICAL_FLOOR_LATCHED
    assert floored == pytest.approx((0.3, -1.0, 0.0, -0.001))

    stronger = hybrid._gate3_projected_vertical_floor(
        values,
        [0.3, 0.8, 0.4, 0.002],
    )
    assert stronger == pytest.approx((0.3, 0.8, 0.4, 0.002))


def test_gate3_vertical_floor_rejects_safe_hidden_and_far_samples():
    actions = [0.24, -0.7, 0.1, -0.001]
    for visible, forward_m, right_m, down_m in (
        (1.0, 8.0, -0.2, 0.1),
        (0.0, 8.0, -3.0, 2.0),
        (1.0, 12.1, -3.0, 2.0),
    ):
        hybrid._clear_gate3_state()
        values = _observation(2)
        values[10] = visible
        values[11] = np.tanh(forward_m / 10.0)
        values[12] = np.tanh(right_m / 5.0)
        values[13] = np.tanh(down_m / 5.0)
        values[0] = np.tanh(-8.0 / 5.0)
        assert (
            hybrid._gate3_projected_vertical_floor(values, actions.copy())
            == actions
        )
        assert not hybrid._GATE3_VERTICAL_FLOOR_LATCHED


@pytest.mark.parametrize(
    ("forward_m", "right_m", "down_m"),
    (
        (6.1, 0.0, 0.0),
        (5.0, 1.5, 0.0),
        (5.0, 0.0, 2.0),
    ),
)
def test_gate3_terminal_level_rejects_unsafe_admission(
    forward_m, right_m, down_m
):
    hybrid._clear_gate3_state()
    values = _observation(2)
    values[11] = np.tanh(forward_m / 10.0)
    values[12] = np.tanh(right_m / 5.0)
    values[13] = np.tanh(down_m / 5.0)
    values[0] = np.tanh(-7.0 / 5.0)
    actions = [0.24, -1.0, 0.06, -0.001]

    assert hybrid._gate3_terminal_level(values, actions.copy()) == actions
    assert not hybrid._GATE3_TERMINAL_LEVEL_LATCHED
