from __future__ import annotations

import torch

import scripts.eval_vq2_lc110_phase8_success_action_bias as lc110


def test_lc110_source_lock_and_measured_bias_contract() -> None:
    lc110.verify_inputs()
    assert lc110.TARGET_PHASE == 8
    assert lc110.TARGET_RAW_INDEX == 9
    assert lc110.GROUP_SIZE == 128
    assert lc110.PAIR_SIZE == 256
    assert lc110.BIASES[-1][1] == (-0.010, 0.050, -0.025, 0.0)


def test_lc110_pairwise_execution_indices() -> None:
    for group in range(len(lc110.BIASES)):
        indices = lc110.pair_indices(group, device=torch.device("cpu"))
        assert indices.shape == (256,)
        assert torch.equal(indices[:128], torch.arange(128))
        assert torch.equal(
            indices[128:], torch.arange(group * 128, (group + 1) * 128)
        )
