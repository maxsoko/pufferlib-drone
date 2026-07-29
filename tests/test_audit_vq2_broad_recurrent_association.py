import pytest
import torch

from pufferlib.vq2_dreamer import RSSMState
from scripts.audit_vq2_broad_recurrent_association import _masked_state


def test_masked_state_resets_inactive_agents() -> None:
    candidate = RSSMState(
        torch.ones(3, 2),
        torch.ones(3, 1, 2),
        torch.ones(3, 1, 2),
    )
    zero = RSSMState(
        torch.zeros(3, 2),
        torch.zeros(3, 1, 2),
        torch.zeros(3, 1, 2),
    )
    masked = _masked_state(candidate, zero, torch.tensor([True, False, True]))
    assert masked.deterministic[:, 0].tolist() == [1.0, 0.0, 1.0]
    assert masked.stochastic[:, 0, 0].tolist() == [1.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        _masked_state(candidate, zero, torch.tensor([True]))
