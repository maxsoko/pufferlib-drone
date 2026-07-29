from scripts.eval_vq2_measured_two_gate_aggregate_dagger_actor import (
    CHECKPOINT_SHA256,
    SEED,
    TAG,
    configure_screen,
    screen,
)


def test_sf067_configures_fresh_aggregate_dagger_screen() -> None:
    configure_screen()
    assert SEED == screen.SEED == 42067
    assert screen.TAG == TAG
    assert screen.CHECKPOINT_SHA256 == CHECKPOINT_SHA256
    assert screen.AGENTS == screen.EPISODES == 512
