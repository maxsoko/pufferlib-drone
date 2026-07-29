from types import SimpleNamespace

from scripts.eval_vq2_frozen_representation_test import _action_gates


def test_test_action_gates_require_both_partitions() -> None:
    args = SimpleNamespace(maximum_action_rmse=0.001, maximum_action_max_abs=0.01)
    test = {"rmse": 0.0009, "max_abs": 0.009}
    dense = {"rmse": 0.0008, "max_abs": 0.008}
    assert all(_action_gates(test, dense, args).values())
    dense["rmse"] = 0.0011
    assert not _action_gates(test, dense, args)["dense_rmse_bounded"]
