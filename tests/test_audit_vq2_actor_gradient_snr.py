from __future__ import annotations

import pytest
import torch

from scripts.audit_vq2_actor_gradient_snr import (
    _summary,
    _virtual_adam_first_step,
)


def test_virtual_adam_first_step_matches_bias_corrected_first_update() -> None:
    parameter = torch.tensor([1.0, -2.0, 3.0])
    gradient = torch.tensor([0.5, -0.25, 0.0])
    actual = _virtual_adam_first_step(
        parameter, gradient, learning_rate=4e-5, epsilon=1e-8
    )
    expected = parameter - 4e-5 * gradient / (gradient.abs() + 1e-8)
    assert torch.equal(actual, expected)
    assert actual[0] < parameter[0]
    assert actual[1] > parameter[1]
    assert actual[2] == parameter[2]


def test_virtual_adam_first_step_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shapes"):
        _virtual_adam_first_step(
            torch.zeros(2), torch.zeros(3), learning_rate=4e-5
        )
    with pytest.raises(ValueError, match="positive"):
        _virtual_adam_first_step(
            torch.zeros(2), torch.zeros(2), learning_rate=0.0
        )
    with pytest.raises(RuntimeError, match="non-finite"):
        _virtual_adam_first_step(
            torch.zeros(2), torch.tensor([0.0, float("nan")]), learning_rate=4e-5
        )


def test_summary_reports_population_statistics() -> None:
    result = _summary([-2.0, -1.0, 1.0, 2.0])
    assert result["mean"] == 0.0
    assert result["median"] == 0.0
    assert result["min"] == -2.0
    assert result["max"] == 2.0
    assert result["std"] == pytest.approx(2.5**0.5)


def test_summary_rejects_empty_or_nonfinite_values() -> None:
    with pytest.raises(RuntimeError, match="empty or non-finite"):
        _summary([])
    with pytest.raises(RuntimeError, match="empty or non-finite"):
        _summary([1.0, float("inf")])
