from __future__ import annotations

import numpy as np
import pytest

from scripts.train_vq2_recurrent_dagger_causal_prefix import (
    CAUSAL_HORIZON_STEPS,
    CONFIG,
    DAGGER_SOURCE_PATTERN,
    CausalHorizonDataset,
)


class FakeDataset:
    root = None
    report = {}
    metadata = {}
    agents = 3
    time_steps = 1000
    lengths = np.asarray([100, 500, 900])

    def chunk(self, agent_indices, start, end, *, device):
        return agent_indices, start, end, device


def test_causal_view_caps_only_time_and_lengths() -> None:
    view = CausalHorizonDataset(FakeDataset())
    assert view.time_steps == CAUSAL_HORIZON_STEPS == 384
    assert view.lengths.tolist() == [100, 384, 384]
    assert view.chunk([0], 0, 64, device="cpu") == ([0], 0, 64, "cpu")
    with pytest.raises(ValueError, match="horizon"):
        view.chunk([0], 320, 448, device="cpu")


def test_causal_fit_keeps_small_model_and_record_ratio() -> None:
    assert DAGGER_SOURCE_PATTERN == ("broad", "next", "next")
    assert CONFIG.seed == 42028
    assert CONFIG.epochs == 4
    assert CONFIG.learning_rate == 2e-5
    assert CONFIG.action_weights == (1.0, 1.0, 4.0, 1.0)
