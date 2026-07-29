from __future__ import annotations

import numpy as np

from scripts.analyze_vq2_dagger_causal_horizon import summarize_horizon


def test_horizon_summary_uses_fixed_channel_weights() -> None:
    result = summarize_horizon(np.asarray([2.0, 4.0, 8.0, 6.0]), 2)
    assert result["count"] == 2
    assert result["mse"] == [1.0, 2.0, 4.0, 3.0]
    assert result["weighted_mse"] == 22.0 / 7.0
