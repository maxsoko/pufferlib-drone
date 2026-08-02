from __future__ import annotations

import torch

import scripts.build_vq2_lc216_all24_action_sequence as target


def test_lc216_sequence_contract() -> None:
    assert target.PREFIX_LENGTH == 2_329
    assert target.STATUS_GAP_STEPS == 7
    assert target.CONTINUATION_LENGTH == 7_256
    assert target.SEQUENCE_LENGTH == 9_592
    assert sum(target.EXPECTED_PHASE_COUNTS) == target.CONTINUATION_LENGTH


def test_lc216_combine_inserts_status_gap() -> None:
    prefix = torch.arange(target.PREFIX_LENGTH * 4, dtype=torch.float32).reshape(-1, 4)
    continuation = torch.arange(
        target.CONTINUATION_LENGTH * 4, dtype=torch.float32
    ).reshape(-1, 4)
    combined = target.combine_sequences(prefix, continuation)
    assert combined.shape == (target.SEQUENCE_LENGTH, 4)
    assert torch.equal(combined[:target.PREFIX_LENGTH], prefix)
    assert torch.equal(
        combined[target.PREFIX_LENGTH:target.PREFIX_LENGTH + target.STATUS_GAP_STEPS],
        prefix[-1:].repeat(target.STATUS_GAP_STEPS, 1),
    )
    assert torch.equal(
        combined[target.PREFIX_LENGTH + target.STATUS_GAP_STEPS:], continuation
    )
