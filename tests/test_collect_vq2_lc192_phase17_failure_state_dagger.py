from __future__ import annotations

import torch

from pufferlib.vq2_recurrent import RecurrentActorOutput
import scripts.collect_vq2_lc192_phase17_failure_state_dagger as lc192


def test_lc192_source_lock() -> None:
    assert lc192.verify_inputs()["frozen_non_phase16_state_exact"]


def test_lc192_feature_slice() -> None:
    state = torch.ones(1, 512, 320)
    pre = torch.zeros(512, 4)
    result = RecurrentActorOutput(pre, pre, pre)
    hidden, saved = lc192.phase17_feature_components(None, result, state, torch.tensor([0, 511]))
    assert hidden.shape == (2, 256)
    assert saved.shape == (2, 4)


def test_lc192_configuration_roundtrip() -> None:
    original = lc192.prior.adapter_feature_components
    originals = lc192.configure()
    try:
        assert lc192.prior.adapter_feature_components is lc192.phase17_feature_components
    finally:
        lc192.restore(originals)
    assert lc192.prior.adapter_feature_components is original
