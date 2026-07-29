from __future__ import annotations

import numpy as np
import pytest

from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE
from pufferlib.vq2_oracle import alignment_oracle_action


def _observation() -> np.ndarray:
    observation = np.zeros((1, ENV_OBS_SIZE), dtype=np.float32)
    privilege = observation[:, LEGAL_OBS_SIZE:]
    privilege[:, 12] = 1.0  # identity quaternion
    return observation


def test_aligned_stationary_query_has_expected_joint_action() -> None:
    action = alignment_oracle_action(_observation())
    assert action.shape == (1, 4)
    assert action.dtype == np.float32
    assert action[0, 0] == pytest.approx(-0.8)
    assert action[0, 1] == pytest.approx(0.0)
    assert action[0, 2] == pytest.approx(
        (8.69 / 32.81 - 0.27) / 0.09, abs=1e-6
    )
    assert action[0, 3] == 0.0


def test_query_uses_privilege_but_legal_values_never_change_label() -> None:
    base = _observation()
    privilege_changed = base.copy()
    # tanh(relative_world_y / 10) under identity orientation.
    privilege_changed[0, LEGAL_OBS_SIZE + 4] = np.tanh(2.0 / 10.0)
    legal_changed = base.copy()
    legal_changed[:, :LEGAL_OBS_SIZE] = 999.0
    assert not np.array_equal(
        alignment_oracle_action(base), alignment_oracle_action(privilege_changed)
    )
    assert np.array_equal(
        alignment_oracle_action(base), alignment_oracle_action(legal_changed)
    )


def test_query_rejects_legal_only_or_nonfinite_native_input() -> None:
    with pytest.raises(ValueError, match="4152"):
        alignment_oracle_action(np.zeros((1, LEGAL_OBS_SIZE), dtype=np.float32))
    invalid = _observation()
    invalid[0, LEGAL_OBS_SIZE] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        alignment_oracle_action(invalid)
