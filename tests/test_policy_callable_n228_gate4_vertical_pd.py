import math
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_n228_gate4_vertical_pd as policy


def _norm(value: float, scale: float) -> float:
    return math.tanh(value / scale)


def _observation(*, gate=3, visible=1.0, forward=10.0, right=2.0, down=4.0,
                 right_rate=-3.0, down_rate=-0.5):
    obs = [0.0] * 32
    obs[1] = _norm(right_rate, 3.0)
    obs[2] = _norm(down_rate, 3.0)
    obs[6] = 1.0
    obs[10] = visible
    obs[11] = _norm(forward, 10.0)
    obs[12] = _norm(right, 5.0)
    obs[13] = _norm(down, 5.0)
    obs[23] = gate / 6.0
    return obs


def test_vertical_pd_preserves_n227_roll_before_plane():
    controller = policy.Gate4VerticalPdController()
    out = controller.apply(_observation(), [0.2, -0.4, 0.3, -0.1])
    assert out == pytest.approx([0.2, -0.4, -0.55, -0.1], abs=1e-5)


def test_vertical_pd_replaces_post_plane_saturated_fallback():
    controller = policy.Gate4VerticalPdController()
    out = controller.apply(
        _observation(forward=-2.0),
        [0.2, 1.0, -1.0, -0.1],
    )
    assert out == pytest.approx([0.2, -0.12, -0.55, -0.1], abs=1e-5)
    assert controller.post_plane_samples == 1


def test_vertical_pd_limits_a_definitive_miss():
    controller = policy.Gate4VerticalPdController()
    out = controller.apply(
        _observation(forward=-6.0),
        [0.2, 1.0, -1.0, -0.1],
    )
    assert out == pytest.approx([0.2, 0.0, 0.0, -0.1])
    assert controller.miss_limited_samples == 1


@pytest.mark.parametrize(
    "observation",
    [_observation(gate=2), _observation(gate=4), _observation(visible=0.0)],
)
def test_vertical_pd_is_invariant_outside_visible_gate4(observation):
    controller = policy.Gate4VerticalPdController()
    base = [0.1, -0.2, 0.3, -0.4]
    assert controller.apply(observation, base) == pytest.approx(base)
    assert controller.active_samples == 0
