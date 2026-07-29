import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "policy_callable_six_gate_composite",
    SCRIPTS / "policy_callable_six_gate_composite.py",
)
policy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policy
SPEC.loader.exec_module(policy)


class FakePolicy:
    def __init__(self, action):
        self.action = list(action)
        self.resets = 0

    def reset_state(self):
        self.resets += 1

    def infer(self, _observation):
        return list(self.action)


def observation(gate, *, forward=20.0, right=2.0, down=0.0, visible=True):
    values = np.zeros(32, dtype=np.float32)
    values[6] = 1.0
    values[10] = float(visible)
    values[11] = np.tanh(forward * 0.1)
    values[12] = np.tanh(right * 0.2)
    values[13] = np.tanh(down * 0.2)
    values[23] = gate / 6.0
    if gate < 6:
        values[24 + gate] = 1.0
    return values


@pytest.fixture
def fake_state(monkeypatch):
    state = policy._CompositeState(
        FakePolicy([0.0, 0.0, 0.0, 0.0]),
        FakePolicy([0.0, 0.25, 0.0, 0.0]),
        FakePolicy([-0.2, 0.1, 0.2, 0.3]),
    )
    monkeypatch.setattr(policy, "_resolve", lambda: state)
    return state


def test_phase_policies_reset_on_gates_four_through_six(fake_state):
    policy.infer(observation(2))
    policy.infer(observation(3))
    policy.infer(observation(4))
    policy.infer(observation(5, visible=False))

    assert fake_state.gate4.resets == 1
    assert fake_state.gate5.resets == 1
    assert fake_state.gate6.resets == 1


def test_gate_five_pitch_and_observable_counter_roll_latch(fake_state):
    far = policy.infer(observation(4, forward=12.0))
    near = policy.infer(observation(4, forward=9.0))
    dropout = policy.infer(observation(4, forward=12.0, visible=False))

    assert far == pytest.approx([0.1, 0.25, 0.0, 0.0])
    assert near[1] == pytest.approx(-0.5)
    assert dropout[1] == pytest.approx(-0.5)


def test_final_visible_controller_is_coordinate_free_and_directional(fake_state):
    right = policy.infer(observation(5, forward=8.0, right=3.0, down=1.0))
    assert right[0] == pytest.approx(-0.06825, abs=1e-6)
    assert right[1] == -1.0
    assert right[3] == 0.0
    assert -1.0 <= right[2] <= 1.0

    fake_state.previous_gate = -1
    fake_state.gate6_roll_direction = 0.0
    left = policy.infer(observation(5, forward=8.0, right=-3.0, down=1.0))
    assert left[1] == 1.0


def test_final_dropout_uses_recurrent_fallback_biases(fake_state):
    output = policy.infer(observation(5, visible=False))
    assert output == pytest.approx([-0.05, 0.1, 0.1, 0.3])


def test_progress_must_be_quantized(fake_state):
    values = observation(3)
    values[23] = 0.57
    with pytest.raises(ValueError, match="invalid six-gate progress"):
        policy.infer(values)


def _intercept_observation(
    *,
    gate: int = 2,
    forward: float,
    right: float,
    forward_rate: float,
    right_rate: float,
) -> np.ndarray:
    values = observation(gate, forward=forward, right=right)
    values[0] = np.tanh(forward_rate / 5.0)
    values[1] = np.tanh(right_rate / 3.0)
    return values


def test_promoted_gate3_intercept_replays_frozen_roll_modes():
    severe = _intercept_observation(
        forward=20.0, right=-4.0, forward_rate=-6.0, right_rate=0.0
    )
    early = _intercept_observation(
        forward=18.0, right=-1.0, forward_rate=-4.5, right_rate=0.0
    )
    close = _intercept_observation(
        forward=7.0, right=-2.0, forward_rate=-5.0, right_rate=0.2
    )
    counter = _intercept_observation(
        forward=5.0, right=-0.5, forward_rate=-5.0, right_rate=0.5
    )

    assert policy._promoted_gate3_intercept(severe, [0.0] * 4)[1] == pytest.approx(1.0)
    assert policy._promoted_gate3_intercept(early, [0.0] * 4)[1] == pytest.approx(0.9)
    assert policy._promoted_gate3_intercept(close, [0.0] * 4)[1] == pytest.approx(
        0.937037037
    )
    assert policy._promoted_gate3_intercept(counter, [0.0] * 4)[1] == pytest.approx(-1.0)


def test_gate1_positive_intercept_excludes_gate3_counter_bank():
    severe = _intercept_observation(
        gate=0, forward=20.0, right=-4.0, forward_rate=-6.0, right_rate=0.0
    )
    close = _intercept_observation(
        gate=0, forward=7.0, right=-2.0, forward_rate=-5.0, right_rate=0.2
    )
    counter = _intercept_observation(
        gate=0, forward=5.0, right=-0.5, forward_rate=-5.0, right_rate=0.5
    )
    base = [0.2, 0.3, -0.1, 0.4]

    severe_action = policy._promoted_gate1_positive_intercept(severe, list(base))
    close_action = policy._promoted_gate1_positive_intercept(close, list(base))
    counter_action = policy._promoted_gate1_positive_intercept(counter, list(base))

    assert severe_action == pytest.approx([0.2, 1.0, -0.1, 0.4])
    assert close_action[1] > base[1]
    assert close_action[0::2] == pytest.approx(base[0::2])
    assert close_action[3] == pytest.approx(base[3])
    assert counter_action == pytest.approx(base)


def test_gate1_positive_intercept_requires_severe_direct_offset():
    moderate_early = _intercept_observation(
        gate=0,
        forward=15.709,
        right=-0.466,
        forward_rate=-4.078,
        right_rate=-0.145,
    )
    base = [0.2, 0.22, -0.1, 0.4]

    assert policy._promoted_gate3_roll_command(moderate_early)[0] == "early"
    assert policy._promoted_gate1_positive_intercept(
        moderate_early, list(base)
    ) == pytest.approx(base)


def test_infer_activates_positive_floor_only_at_gate_one(fake_state):
    severe_gate1 = _intercept_observation(
        gate=0, forward=20.0, right=-4.0, forward_rate=-6.0, right_rate=0.0
    )
    counter_gate1 = _intercept_observation(
        gate=0, forward=5.0, right=-0.5, forward_rate=-5.0, right_rate=0.5
    )
    severe_gate2 = _intercept_observation(
        gate=1, forward=20.0, right=-4.0, forward_rate=-6.0, right_rate=0.0
    )
    counter_gate3 = _intercept_observation(
        gate=2, forward=5.0, right=-0.5, forward_rate=-5.0, right_rate=0.5
    )

    assert policy.infer(severe_gate1)[1] == pytest.approx(1.0)
    assert policy.infer(counter_gate1)[1] == pytest.approx(0.0)
    assert policy.infer(severe_gate2)[1] == pytest.approx(0.0)
    assert policy.infer(counter_gate3)[1] == pytest.approx(-1.0)
