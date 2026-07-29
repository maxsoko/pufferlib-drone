from scripts.sweep_vq2_native_oracle_pd import (
    ROLL_RATE_GAINS,
    SPEEDS_M_S,
    selection_key,
)


def test_surface_is_exactly_the_preregistered_four_by_four_grid() -> None:
    assert SPEEDS_M_S == (2.5, 3.0, 3.5, 4.0)
    assert ROLL_RATE_GAINS == (0.1, 0.3, 0.5, 0.7)


def test_selection_prioritizes_success_then_safety_then_progress() -> None:
    base = {
        "success_rate": 0.0,
        "crash_rate": 0.0,
        "out_of_order_rate": 0.0,
        "mean_gates_passed": 2.0,
        "missed_gate_rate": 1.0,
        "terminal_crossing_radial_m": 1.0,
        "completion_time_s": 0.0,
        "target_speed_m_s": 3.0,
        "roll_rate_per_m_s": 0.3,
    }
    success = {**base, "success_rate": 0.1, "mean_gates_passed": 1.0}
    crash = {**success, "crash_rate": 0.1, "mean_gates_passed": 5.0}
    progress = {**base, "mean_gates_passed": 3.0}
    assert selection_key(success) > selection_key(progress)
    assert selection_key(success) > selection_key(crash)
    assert selection_key(progress) > selection_key(base)
