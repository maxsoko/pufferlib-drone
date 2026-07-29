import math
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_native_gate3_controller import (
    ControllerParameters,
    _controller_roll,
    _initialize_controller_state,
    _rotate_by_quaternion,
)


class _Scenario:
    forward_rate_m_s = -8.0
    right_rate_m_s = 0.0
    initial_action = (0.0, 0.0, 0.0, 0.0)


def _observation(forward_m: float, right_m: float) -> torch.Tensor:
    values = torch.zeros((1, 32), dtype=torch.float32)
    values[:, 10] = 1.0
    values[:, 11] = math.tanh(forward_m / 10.0)
    values[:, 12] = math.tanh(right_m / 5.0)
    return values


def test_identity_quaternion_rotation_is_identity():
    assert _rotate_by_quaternion((1.0, -2.0, 3.0), (1.0, 0.0, 0.0, 0.0)) == (
        1.0,
        -2.0,
        3.0,
    )


def test_anchored_path_starts_with_positive_bank_for_gate_left():
    observation = _observation(10.0, -2.0)
    state = _initialize_controller_state(observation, _Scenario())
    roll = _controller_roll(
        observation,
        state,
        ControllerParameters(
            path_power=2.0,
            terminal_right_m=0.0,
            position_gain=1.0,
            rate_gain=0.7,
            max_roll_norm=1.0,
            max_roll_slew_norm_s=100.0,
        ),
        dt_s=1.0 / 60.0,
    )
    assert float(roll[0]) > 0.0


def test_roll_slew_limit_is_enforced():
    observation = _observation(10.0, -2.0)
    state = _initialize_controller_state(observation, _Scenario())
    roll = _controller_roll(
        observation,
        state,
        ControllerParameters(
            rate_gain=1.0,
            max_roll_norm=1.0,
            max_roll_slew_norm_s=0.6,
        ),
        dt_s=1.0 / 60.0,
    )
    assert abs(float(roll[0])) <= 0.010001
