from types import SimpleNamespace

from scripts.calibrate_vq2_frozen_representation_actor import (
    _action_gates,
    _actor_candidate_key,
    _probe_candidate_key,
)


def test_actor_selection_prefers_passing_lower_drift() -> None:
    args = SimpleNamespace(maximum_action_rmse=0.001, maximum_action_max_abs=0.01)
    failing = {
        "step": 50,
        "successful_drift": {"rmse": 0.0011, "max_abs": 0.005},
        "dense_drift": {"rmse": 0.0005, "max_abs": 0.005},
    }
    failing["gates"] = _action_gates(
        failing["successful_drift"], failing["dense_drift"], args
    )
    passing = {
        "step": 100,
        "successful_drift": {"rmse": 0.0007, "max_abs": 0.004},
        "dense_drift": {"rmse": 0.0008, "max_abs": 0.006},
    }
    passing["gates"] = _action_gates(
        passing["successful_drift"], passing["dense_drift"], args
    )
    assert _actor_candidate_key(passing, args) > _actor_candidate_key(failing, args)


def test_probe_selection_prefers_admitted_candidate() -> None:
    weak = {
        "step": 0,
        "metrics": {
            "correlation": 0.95,
            "near_plane_mae_m": 0.6,
            "direction_accuracy": 0.8,
            "mae_m": 0.4,
        },
        "gates": {"near": False},
    }
    admitted = {
        "step": 50,
        "metrics": {
            "correlation": 0.9,
            "near_plane_mae_m": 0.4,
            "direction_accuracy": 0.8,
            "mae_m": 0.4,
        },
        "gates": {"near": True},
    }
    assert _probe_candidate_key(admitted, 675) > _probe_candidate_key(weak, 675)
