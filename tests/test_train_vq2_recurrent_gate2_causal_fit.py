from __future__ import annotations

import numpy as np
import pytest

import scripts.train_vq2_recurrent_dagger_aggregate as aggregate
from scripts.train_vq2_recurrent_gate2_causal_fit import (
    CAUSAL_HORIZON_STEPS,
    CONFIG,
    DAGGER_SOURCE_PATTERN,
    HORIZON_REPORT_SHA256,
    NEXT_REPORT_SHA256,
    dataset_factory,
)


class FakeDataset:
    def __init__(self, root, **kwargs):
        self.root = root
        self.report = {}
        self.metadata = {}
        self.agents = 2
        self.time_steps = 2048
        self.lengths = np.asarray([500, 1200])

    def chunk(self, agent_indices, start, end, *, device):
        return agent_indices, start, end, device


def test_sf037_contract_keeps_small_actor_and_causal_gate2_window() -> None:
    assert CAUSAL_HORIZON_STEPS == 768
    assert DAGGER_SOURCE_PATTERN == ("broad", "next", "next")
    assert CONFIG.action_weights == (1.0, 1.0, 4.0, 1.0)
    assert CONFIG.loss_action_weights == (1.0, 4.0, 4.0, 1.0)
    assert CONFIG.prefer_admitted is True
    assert CONFIG.epochs == 6
    assert CONFIG.learning_rate == 5e-5


def test_sf037_source_locks_diagnosis_and_gate2_data() -> None:
    assert HORIZON_REPORT_SHA256.startswith("e81557cb")
    assert NEXT_REPORT_SHA256.startswith("d07db36e")


def test_sf037_factory_caps_only_the_next_dataset(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.train_vq2_recurrent_gate2_causal_fit._ORIGINAL_DATASET",
        FakeDataset,
    )
    old_next = aggregate.NEXT_DATASET
    try:
        aggregate.NEXT_DATASET = pytest.importorskip(
            "scripts.train_vq2_recurrent_gate2_causal_fit"
        ).NEXT_DATASET
        view = dataset_factory(aggregate.NEXT_DATASET)
        assert view.time_steps == 768
        assert view.lengths.tolist() == [500, 768]
    finally:
        aggregate.NEXT_DATASET = old_next

