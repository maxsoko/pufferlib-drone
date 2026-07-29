from scripts.eval_vq2_measured_two_gate_dagger_second_actor import (
    CHECKPOINT_SHA256,
    SEED,
    TAG,
    configure_screen,
    screen,
)


def test_sf057_configures_fresh_second_dagger_screen() -> None:
    configure_screen()
    assert SEED == screen.SEED == 42057
    assert screen.TAG == TAG
    assert screen.CHECKPOINT_SHA256 == CHECKPOINT_SHA256
    assert screen.AGENTS == screen.EPISODES == 512
