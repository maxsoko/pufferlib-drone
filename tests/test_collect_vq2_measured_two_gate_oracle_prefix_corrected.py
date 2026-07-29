from scripts.collect_vq2_measured_two_gate_oracle_prefix import MAX_EXECUTED_LABEL_ERROR


def test_measured_oracle_uses_admitted_sf016_parity_bound() -> None:
    assert MAX_EXECUTED_LABEL_ERROR == 5e-5

