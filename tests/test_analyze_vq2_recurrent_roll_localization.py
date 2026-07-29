from __future__ import annotations

import numpy as np

from scripts.analyze_vq2_recurrent_roll_localization import (
    ACTION_WEIGHTS,
    BINS,
    CAPS,
    VALIDATION_AGENTS,
    action_summary,
    moment_summary,
)


def test_action_summary_uses_fixed_four_channel_weights() -> None:
    result = action_summary(
        np.asarray([2.0, 4.0, 8.0, 6.0]),
        np.asarray([1.0, 2.0, 4.0, 3.0]),
        2,
    )
    assert ACTION_WEIGHTS.tolist() == [1.0, 1.0, 4.0, 1.0]
    assert result["mse"] == [1.0, 2.0, 4.0, 3.0]
    assert result["mae"] == [0.5, 1.0, 2.0, 1.5]
    assert result["weighted_mse"] == 22.0 / 7.0


def test_moment_summary_reports_mean_rms_and_std() -> None:
    result = moment_summary(4.0, 10.0, 2)
    assert result["mean"] == 2.0
    assert result["rms"] == np.sqrt(5.0)
    assert result["std"] == 1.0


def test_localization_contract_is_complete_and_validation_only() -> None:
    assert BINS[0][0] == 0
    assert BINS[-1][1] == 512
    assert all(left[1] == right[0] for left, right in zip(BINS, BINS[1:]))
    assert CAPS[-1] == 512
    assert VALIDATION_AGENTS.tolist() == list(range(448, 512))

