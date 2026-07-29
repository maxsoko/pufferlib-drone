from __future__ import annotations

from scripts.train_vq2_public_phase_full import CONFIG, dataset_factory


def test_sf045_opens_encoder_without_changing_actor_contract() -> None:
    assert CONFIG.train_encoder
    assert CONFIG.learning_rate == 2e-5
    assert CONFIG.phase_zero_loss_weight == 2.0


def test_sf045_dataset_is_the_bounded_public_prefix() -> None:
    dataset = dataset_factory()
    assert dataset.time_steps == 768
    assert dataset.metadata["bounded_prefix_dataset"] is True
    assert dataset.metadata["observation"]["public_status_values_per_record"] == 1

