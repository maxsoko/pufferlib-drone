import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "policy_callable_gate_pid.py"
SPEC = importlib.util.spec_from_file_location("policy_callable_gate_pid", MODULE_PATH)
policy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policy
SPEC.loader.exec_module(policy)


def test_infer_zero_when_gate_not_visible():
    obs = [0.0] * policy.OBSERVATION_SIZE
    obs[policy.OBS_GATE_VISIBLE] = -1.0
    assert policy.infer(obs) == [0.0, 0.0, 0.0, 0.0]


def test_infer_returns_clamped_action():
    obs = [0.0] * policy.OBSERVATION_SIZE
    obs[policy.OBS_GATE_VISIBLE] = 1.0
    obs[policy.OBS_GATE_FORWARD] = 0.8
    obs[policy.OBS_GATE_RIGHT] = 0.4
    obs[policy.OBS_GATE_DOWN] = -0.2
    obs[policy.OBS_GATE_YAW_ERROR] = 0.5
    obs[policy.OBS_GATE_PITCH_ERROR] = -0.3
    obs[policy.OBS_GATE_APPARENT_SIZE] = 0.1
    obs[policy.OBS_GATE_CONFIDENCE] = 0.7
    action = policy.infer(obs)
    assert len(action) == policy.ACTION_SIZE
    assert all(-1.0 <= value <= 1.0 for value in action)
    assert action[0] > 0.0
