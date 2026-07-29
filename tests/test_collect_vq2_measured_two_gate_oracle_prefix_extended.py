from scripts.collect_vq2_measured_two_gate_oracle_prefix_extended import PREFIX_STEPS, SEED


def test_sf048_changes_only_bound_and_seed() -> None:
    assert PREFIX_STEPS == 1600
    assert SEED == 42048

