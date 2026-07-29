from __future__ import annotations

import pytest
import torch

from scripts.interpolate_vq2_phase_checkpoints import interpolate_state_dict


def test_interpolation_endpoints_and_midpoint_are_exact():
    left = {"x": torch.tensor([0.0, 2.0]), "n": torch.tensor([3])}
    right = {"x": torch.tensor([2.0, 6.0]), "n": torch.tensor([3])}
    assert torch.equal(interpolate_state_dict(left, right, 0.0)["x"], left["x"])
    assert torch.equal(interpolate_state_dict(left, right, 1.0)["x"], right["x"])
    assert torch.equal(
        interpolate_state_dict(left, right, 0.5)["x"], torch.tensor([1.0, 4.0])
    )


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_interpolation_rejects_invalid_alpha(alpha: float):
    with pytest.raises(ValueError, match="finite and in"):
        interpolate_state_dict({"x": torch.zeros(1)}, {"x": torch.ones(1)}, alpha)


def test_interpolation_rejects_contract_changes():
    with pytest.raises(ValueError, match="state dictionaries differ"):
        interpolate_state_dict({"x": torch.zeros(1)}, {"y": torch.ones(1)}, 0.5)
    with pytest.raises(ValueError, match="non-floating tensor differs"):
        interpolate_state_dict(
            {"x": torch.tensor([1])}, {"x": torch.tensor([2])}, 0.5
        )
