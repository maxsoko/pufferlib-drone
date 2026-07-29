import policy_callable_n214_unified_tail as policy


def test_n214_tail_reuses_n210_topology_with_new_immutable_hash_pin():
    assert policy.EXPECTED_N214_CHECKPOINT_SHA256 == (
        "999ab9061ca04f7d4b66d33991afd7fa14d1ca0049d9398a66f361419a15fa75"
    )
    assert (
        policy.base.EXPECTED_N209_CHECKPOINT_SHA256
        == policy.EXPECTED_N214_CHECKPOINT_SHA256
    )


def test_n215_snapshot_declares_unassisted_promoted_tail(monkeypatch):
    monkeypatch.setenv("PUFFER_GATE3_COUNTER_ENTRY_M", "-1.2")
    parameters = policy.controller_snapshot()["n215_unified_tail"]["parameters"]
    assert parameters["tail_checkpoint"] == "n214_step_65536"
    assert parameters["teacher_intervention_at_deployment"] == 0.0
    assert parameters["prefix_gate_indices"] == [0, 1, 2]
    assert parameters["n209_tail_gate_indices"] == [3, 4, 5]
