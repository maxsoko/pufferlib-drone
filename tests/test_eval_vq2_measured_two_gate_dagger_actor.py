from scripts.eval_vq2_measured_two_gate_dagger_actor import (
    CHECKPOINT_SHA256,
    SEED,
    TAG,
    configure_screen,
    screen,
)


def test_sf054_configures_fresh_dagger_screen() -> None:
    configure_screen()
    assert SEED == 42054
    assert screen.SEED == SEED
    assert screen.TAG == TAG
    assert screen.CHECKPOINT_SHA256 == CHECKPOINT_SHA256
    assert screen.AGENTS == screen.EPISODES == 512
    assert screen.STEP_LIMIT == 2048
