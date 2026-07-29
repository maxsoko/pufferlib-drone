import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
module = importlib.import_module("policy_callable_n221_observable_course")


def observation_for_gate(gate: int, *, forward_rate: float = 0.0) -> np.ndarray:
    observation = np.zeros(32, dtype=np.float32)
    observation[0] = np.tanh(forward_rate * 0.2)
    observation[6] = 1.0
    observation[10] = 1.0
    observation[11] = np.tanh(20.0 * 0.1)
    observation[17] = 1.0
    observation[23] = gate / 6.0
    return observation


def test_controller_owns_pitch_roll_and_thrust_from_gate_one(monkeypatch):
    source_action = [0.9, 0.8, 0.7, 0.4]
    monkeypatch.setattr(module.n220.n219, "infer", lambda observation: list(source_action))
    monkeypatch.setattr(module.n220.n219, "reset", lambda: None)
    module.reset()

    action = module.infer(observation_for_gate(0))

    assert action[0] == pytest.approx(-0.144)
    assert action[1:3] != source_action[1:3]
    assert action[3] == source_action[3]
    assert all(-1.0 <= value <= 1.0 for value in action)


def test_pitch_controller_reduces_forward_acceleration_at_target_speed(monkeypatch):
    monkeypatch.setattr(module.n220.n219, "infer", lambda observation: [0.0] * 4)
    monkeypatch.setattr(module.n220.n219, "reset", lambda: None)
    module.reset()

    action = module.infer(
        observation_for_gate(0, forward_rate=-module.TARGET_FORWARD_SPEED_M_S)
    )

    assert action[0] == pytest.approx(0.0, abs=1e-6)


def test_snapshot_declares_unified_observable_contract(monkeypatch):
    monkeypatch.setenv("PUFFER_GATE3_COUNTER_ENTRY_M", "-1.2")
    parameters = module.controller_snapshot()["n221_observable_course"]["parameters"]
    assert parameters["observable_controller_gate_indices"] == list(range(6))
    assert parameters["policy_owned_actions"] == ["yaw"]
    assert parameters["controller_owned_actions"] == ["pitch", "roll", "thrust"]
    assert parameters["target_forward_speed_m_s"] == pytest.approx(4.00)
    assert parameters["runtime_privileged_state"] is False
