import policy_callable_n219_unified_tail as policy


def test_n219_tail_reuses_n210_topology_with_new_immutable_hash_pin():
    assert policy.EXPECTED_N219_CHECKPOINT_SHA256 == (
        "20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec"
    )
    assert (
        policy.base.EXPECTED_N209_CHECKPOINT_SHA256
        == policy.EXPECTED_N219_CHECKPOINT_SHA256
    )


def test_n219_snapshot_declares_predictor_adapted_unassisted_tail(monkeypatch):
    monkeypatch.setenv("PUFFER_GATE3_COUNTER_ENTRY_M", "-1.2")
    parameters = policy.controller_snapshot()["n219_unified_tail"]["parameters"]
    assert parameters["tail_checkpoint"] == "n219_predictor_step_32768"
    assert parameters["gate_motion_predict_dropout"] is True
    assert parameters["teacher_intervention_at_deployment"] == 0.0
    assert parameters["prefix_gate_indices"] == [0, 1, 2]
    assert parameters["n209_tail_gate_indices"] == [3, 4, 5]
