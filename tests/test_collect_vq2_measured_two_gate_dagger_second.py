from scripts.collect_vq2_measured_two_gate_dagger_second import (
    CHECKPOINT_SHA256,
    SEED,
    TAG,
    configure_iteration,
    iteration,
)


def test_sf055_configures_fresh_second_iteration() -> None:
    configure_iteration()
    assert SEED == 42055
    assert iteration.SEED == SEED
    assert iteration.TAG == TAG
    assert iteration.CHECKPOINT_SHA256 == CHECKPOINT_SHA256
    assert iteration.AGENTS == iteration.EPISODES == 64
    assert iteration.COLLECTION_STEP_LIMIT == 768
