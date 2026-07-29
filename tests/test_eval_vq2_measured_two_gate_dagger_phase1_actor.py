from scripts.eval_vq2_measured_two_gate_dagger_phase1_actor import (
    CHECKPOINT_SHA256,
    SEED,
    TAG,
    configure_screen,
    screen,
)


def test_sf060_configures_fresh_phase1_dagger_screen() -> None:
    configure_screen()
    assert SEED == screen.SEED == 42060
    assert screen.TAG == TAG
    assert screen.CHECKPOINT_SHA256 == CHECKPOINT_SHA256
    assert screen.AGENTS == screen.EPISODES == 512
