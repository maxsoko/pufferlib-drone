from scripts.train_vq2_measured_two_gate_full import CAUSAL_HORIZON, CONFIG, dataset_factory


def test_measured_fit_stops_before_gate2_completion_phase() -> None:
    dataset = dataset_factory()
    assert CAUSAL_HORIZON == 1472
    assert dataset.time_steps == 1472
    assert CONFIG.validation_agents == 8
    assert CONFIG.train_encoder

