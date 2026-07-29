import math
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_n227_gate4_terminal_pd as policy


def _norm(value: float, scale: float) -> float:
    return math.tanh(value / scale)


def _observation(*, gate=3, visible=1.0, forward=20.0, right=4.0, down=2.0,
                 forward_rate=-8.0, right_rate=0.5, down_rate=-0.5):
    obs = [0.0] * 32
    obs[0] = _norm(forward_rate, 5.0)
    obs[1] = _norm(right_rate, 3.0)
    obs[2] = _norm(down_rate, 3.0)
    obs[6] = 1.0
    obs[10] = visible
    obs[11] = _norm(forward, 10.0)
    obs[12] = _norm(right, 5.0)
    obs[13] = _norm(down, 5.0)
    obs[23] = gate / 6.0
    return obs


def test_gate4_terminal_pd_preserves_pitch_and_yaw():
    controller = policy.Gate4TerminalPlaneController()
    out = controller.apply(
        _observation(),
        [0.2, 0.7, -0.8, -0.1],
    )
    # Horizon 2.5 s: terminal right 5.25 m, terminal down 0.75 m.
    assert out == pytest.approx([0.2, -1.0, -0.075, -0.1], abs=1e-5)
    assert controller.active_samples == 1
    assert controller.changed_samples == 1


def test_gate4_terminal_pd_brakes_a_converging_lateral_path():
    controller = policy.Gate4TerminalPlaneController()
    out = controller.apply(
        _observation(forward=8.0, right=3.0, right_rate=-4.0),
        [0.0, -1.0, 0.0, 0.0],
    )
    # The observed motion crosses center by the terminal plane, so counter-roll.
    assert out[1] > 0.0


@pytest.mark.parametrize(
    "observation",
    [
        _observation(gate=2),
        _observation(gate=4),
        _observation(visible=0.0),
        _observation(forward=-1.0),
    ],
)
def test_gate4_terminal_pd_is_invariant_outside_visible_approach(observation):
    controller = policy.Gate4TerminalPlaneController()
    base = [0.1, -0.2, 0.3, -0.4]
    assert controller.apply(observation, base) == pytest.approx(base)
    assert controller.active_samples == 0


def test_gate4_terminal_pd_reset_clears_diagnostics():
    controller = policy.Gate4TerminalPlaneController()
    controller.apply(_observation(), [0.0, 0.0, 0.0, 0.0])
    controller.reset()
    assert controller.snapshot()["active_samples"] == 0
    assert controller.snapshot()["last_terminal_right_m"] is None
