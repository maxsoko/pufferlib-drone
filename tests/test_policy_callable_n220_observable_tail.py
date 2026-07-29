import importlib
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
module = importlib.import_module("policy_callable_n220_observable_tail")


def observation_for_gate(gate: int) -> np.ndarray:
    observation = np.zeros(32, dtype=np.float32)
    observation[6] = 1.0
    observation[10] = 1.0
    observation[11] = np.tanh(20.0 * 0.1)
    observation[23] = gate / 6.0
    return observation


def test_prefix_action_is_unchanged_and_controller_activates_only_on_gate4(monkeypatch):
    source_action = [0.1, 0.2, 0.3, 0.4]
    monkeypatch.setattr(module.n219, "infer", lambda observation: list(source_action))
    monkeypatch.setattr(module.n219, "reset", lambda: None)
    module.reset()

    assert module.infer(observation_for_gate(2)) == source_action
    tail_action = module.infer(observation_for_gate(3))

    assert tail_action[0] == source_action[0]
    assert tail_action[3] == source_action[3]
    assert tail_action[1:3] != source_action[1:3]
    assert all(-1.0 <= value <= 1.0 for value in tail_action)


def test_reset_clears_observable_controller_history(monkeypatch):
    monkeypatch.setattr(module.n219, "infer", lambda observation: [0.0] * 4)
    monkeypatch.setattr(module.n219, "reset", lambda: None)
    module.reset()
    module.infer(observation_for_gate(3))
    assert module._CONTROLLER.pose_valid

    module.reset()

    assert not module._CONTROLLER.pose_valid
    assert module._CONTROLLER.previous_gate == -1
    assert not module._CONTROLLER_ACTIVE


def test_snapshot_declares_observation_only_hybrid_contract(monkeypatch):
    monkeypatch.setenv("PUFFER_GATE3_COUNTER_ENTRY_M", "-1.2")
    snapshot = module.controller_snapshot()["n220_observable_tail"]["parameters"]

    assert snapshot["observable_controller_gate_indices"] == [3, 4, 5]
    assert snapshot["policy_owned_actions"] == ["pitch", "yaw"]
    assert snapshot["controller_owned_actions"] == ["roll", "thrust"]
    assert snapshot["runtime_privileged_state"] is False
    assert snapshot["gate_roll_bias"][3:] == [-0.4, 0.1, -0.12]
