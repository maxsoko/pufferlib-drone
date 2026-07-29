from scripts.sweep_vq2_native_oracle_true_velocity import (
    ROLL_RATE_GAINS,
    TARGET_SPEED_M_S,
)


def test_true_velocity_surface_matches_preregistration() -> None:
    assert TARGET_SPEED_M_S == 3.5
    assert ROLL_RATE_GAINS == (0.1, 0.3, 0.5, 0.7)
