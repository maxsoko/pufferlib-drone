from __future__ import annotations

from types import SimpleNamespace

import torch

import scripts.eval_vq2_lc064_phase2_residual_direction as lc064


def test_lc064_context_matches_candidate_checkpoint_residuals() -> None:
    payload = lc064.verify_inputs()
    context = lc064.build_candidate_context(payload, device=torch.device("cpu"))
    hidden = torch.randn(1, lc064.base.TOTAL_AGENTS, 256)
    pre = torch.randn(lc064.base.TOTAL_AGENTS, 4)
    held = torch.full((lc064.base.TOTAL_AGENTS, 1), 2.0 / lc064.base.OFFICIAL_PROGRESS_SCALE)
    actual = lc064.apply_candidate_actions(
        SimpleNamespace(pre_tanh_mean=pre), hidden, held, context
    )
    for index in (0, 1, len(lc064.SPECS) - 1):
        state = lc064.candidate_state_for_index(payload["model_state"], index)
        row = index * lc064.GROUP_SIZE
        feature = torch.tanh(
            state[lc064.RESIDUAL_NAMES[0]][lc064.TARGET_PHASE] @ hidden[0, row]
            + state[lc064.RESIDUAL_NAMES[1]][lc064.TARGET_PHASE]
        )
        residual = (
            state[lc064.RESIDUAL_NAMES[2]][lc064.TARGET_PHASE] @ feature
            + state[lc064.RESIDUAL_NAMES[3]][lc064.TARGET_PHASE]
        )
        parent = payload["model_state"]
        parent_feature = torch.tanh(
            parent[lc064.RESIDUAL_NAMES[0]][lc064.TARGET_PHASE] @ hidden[0, row]
            + parent[lc064.RESIDUAL_NAMES[1]][lc064.TARGET_PHASE]
        )
        parent_residual = (
            parent[lc064.RESIDUAL_NAMES[2]][lc064.TARGET_PHASE] @ parent_feature
            + parent[lc064.RESIDUAL_NAMES[3]][lc064.TARGET_PHASE]
        )
        expected = torch.tanh(pre[row] + residual - parent_residual)
        assert torch.allclose(actual[row], expected, atol=2e-6, rtol=0.0)


def test_lc064_is_one_fast_256_episode_vector() -> None:
    assert lc064.GROUP_SIZE * len(lc064.SPECS) == 256
    assert lc064.SPECS[0][1:] == (0.0, ())
