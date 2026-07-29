from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.build_policy_trace_anchor_dataset import (
    expand_gate_phase_onehot_observation,
    interpolate_recorded_action,
    observable_feedback_action,
)


def _observation() -> np.ndarray:
    return np.zeros(23, dtype=np.float32)


def test_observable_feedback_action_tracks_a_stationary_gate() -> None:
    action = observable_feedback_action(_observation())

    assert action == pytest.approx([-0.126, 0.0, 0.0, 0.0])


def test_observable_feedback_action_has_bounded_corrective_axes() -> None:
    observation = _observation()
    observation[0] = -0.99
    observation[1] = 0.99
    observation[2] = 0.99
    observation[12] = 0.99
    observation[13] = 0.99

    action = observable_feedback_action(observation)

    assert action == pytest.approx([0.144, -0.2, -0.8, 0.0])


def test_observable_feedback_action_rejects_wrong_observation_shape() -> None:
    with pytest.raises(ValueError, match="observation has shape"):
        observable_feedback_action(np.zeros(22, dtype=np.float32))


def test_observable_feedback_action_scales_and_clamps_targets() -> None:
    observation = _observation()
    observation[12] = -0.99
    action = observable_feedback_action(
        observation,
        pitch_scale=2.0,
        roll_scale=8.0,
        thrust_scale=1.0,
        thrust_offset=-0.3,
    )

    assert action == pytest.approx([-0.252, 1.0, -0.3, 0.0])


def test_expand_legacy_live_trace_to_six_gate_phase_abi() -> None:
    observation = np.arange(23, dtype=np.float32)

    expanded = expand_gate_phase_onehot_observation(
        observation, official_gate_index=2, denominator=6
    )

    assert expanded.shape == (32,)
    assert expanded[:23] == pytest.approx(observation)
    assert expanded[23] == pytest.approx(2.0 / 6.0)
    assert expanded[24:30] == pytest.approx([0, 0, 1, 0, 0, 0])
    assert expanded[30:] == pytest.approx([0, 0])


def test_recorded_action_interpolation_does_not_smear_gate_transition() -> None:
    current = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    following = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)

    same_phase = interpolate_recorded_action(
        current, following, fraction=0.25, same_phase=True
    )
    transition = interpolate_recorded_action(
        current, following, fraction=0.25, same_phase=False
    )

    assert same_phase == pytest.approx([0.3, 0.35, 0.4, 0.45])
    assert transition == pytest.approx(current)
