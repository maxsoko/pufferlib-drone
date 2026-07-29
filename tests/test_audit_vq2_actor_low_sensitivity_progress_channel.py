import numpy as np
import pytest
import torch

from scripts.audit_vq2_actor_low_sensitivity_progress_channel import (
    _canonical_actor_tangent_null_direction,
    _canonical_low_sensitivity_direction,
    _parse_scales,
    _weighted_correlation,
)


def test_parse_scales_requires_increasing_unique_values() -> None:
    assert _parse_scales("0.1,0.2") == (0.1, 0.2)
    with pytest.raises(ValueError):
        _parse_scales("0.2,0.1")


def test_canonical_direction_uses_smallest_deterministic_singular_vector() -> None:
    weight = torch.diag(torch.tensor([2.0, 0.25]))
    direction, metrics = _canonical_low_sensitivity_direction(weight, 2)
    assert torch.allclose(direction, torch.tensor([0.0, 1.0]))
    assert metrics["minimum_singular_value"] == pytest.approx(0.25)
    assert metrics["minimum_singular_gap_ratio"] == pytest.approx(8.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_canonical_direction_accepts_cuda_actor_weight() -> None:
    weight = torch.diag(torch.tensor([2.0, 0.25], device="cuda"))
    direction, metrics = _canonical_low_sensitivity_direction(weight, 2)
    assert direction.device.type == "cpu"
    assert torch.allclose(direction, torch.tensor([0.0, 1.0]))
    assert metrics["actor_first_layer_residual_norm"] == pytest.approx(0.25)


def test_actor_tangent_null_direction_satisfies_both_constraints() -> None:
    generator = torch.Generator().manual_seed(9)
    weight = torch.randn((3, 6), generator=generator)
    direction, metrics = _canonical_actor_tangent_null_direction(
        weight, deterministic_size=2, stochastic_groups=1, stochastic_classes=4
    )
    assert torch.linalg.vector_norm(weight @ direction) < 1e-5
    assert direction[2:].sum().abs() < 1e-5
    assert metrics["stochastic_direction_norm"] > 0.0
    assert metrics["direction_norm"] == pytest.approx(1.0)


def test_weighted_correlation_recovers_perfect_ordering() -> None:
    value = np.asarray([-2.0, -1.0, 1.0, 2.0])
    weight = np.asarray([1.0, 2.0, 2.0, 1.0])
    assert _weighted_correlation(value, 3.0 * value + 1.0, weight) == pytest.approx(1.0)
