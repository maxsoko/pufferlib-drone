from scripts.sweep_vq2_native_oracle_spline_speed import (
    EPISODE_SECONDS,
    SPEEDS_M_S,
    selection_key,
    summarize,
)


def test_spline_speed_surface_matches_preregistration() -> None:
    assert SPEEDS_M_S == (1.5, 2.0, 2.5, 3.0, 3.5)
    assert EPISODE_SECONDS == 60.0


def test_summarize_does_not_require_direct_controller_fields() -> None:
    report = {
        "episodes": 16,
        "wall_time_seconds": 1.0,
        "fixed_environment": {"teacher_pitch_speed_target_m_s": 2.0},
        "metrics": {"env/success_rate": 0.5, "env/gates_passed": 4.0},
    }
    cell = summarize(report)
    assert cell["successes"] == 8
    assert cell["target_speed_m_s"] == 2.0


def test_selection_prioritizes_solve_then_safety() -> None:
    base = {
        "success_rate": 0.5,
        "crash_rate": 0.0,
        "out_of_order_rate": 0.0,
        "mean_gates_passed": 4.0,
        "missed_gate_rate": 0.5,
        "terminal_crossing_radial_m": 0.2,
        "completion_time_s": 30.0,
        "target_speed_m_s": 2.0,
    }
    more_success = {**base, "success_rate": 0.6, "crash_rate": 0.2}
    unsafe_tie = {**base, "crash_rate": 0.1}
    assert selection_key(more_success) > selection_key(base)
    assert selection_key(base) > selection_key(unsafe_tie)
