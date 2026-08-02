from __future__ import annotations

import torch

import scripts.collect_vq2_lc215_raw18_24_action_sequence as target


def test_lc215_source_lock() -> None:
    parent = target.verify_inputs()
    assert parent["model"]["sequence_length"] == 2_329


def test_lc215_feature_scope() -> None:
    recurrent = torch.randn(1, target.TOTAL_AGENTS, 321)
    result = type("Result", (), {
        "pre_tanh_mean": torch.randn(target.TOTAL_AGENTS, 4)
    })()
    index = torch.tensor([0, 300, 511])
    hidden, base = target.sequence_feature_components(
        None, result, recurrent, index
    )
    assert hidden.shape == (3, 256)
    assert base.shape == (3, 4)
    assert torch.equal(hidden, recurrent[0, index, :256])


def test_lc215_contract() -> None:
    assert target.PHASE_MIN == 18
    assert target.PHASE_MAX_EXCLUSIVE == 24
    assert target.TARGET_RAW_INDEX == 24
    assert target.TOTAL_AGENTS == 512
    assert target.MAX_STEPS == 45_000
