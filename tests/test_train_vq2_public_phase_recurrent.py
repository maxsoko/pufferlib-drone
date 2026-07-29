from __future__ import annotations

import torch

from scripts.train_vq2_public_phase_recurrent import (
    numerical_admission,
    phase_masks,
)


def test_phase_masks_select_only_zero_and_gate2() -> None:
    observation = torch.zeros(1, 4, 4119)
    observation[0, :, -1] = torch.tensor([0.0, 1 / 6, 2 / 6, 1 / 6])
    valid = torch.tensor([[True, True, True, False]])
    phase_zero, gate2 = phase_masks(observation, valid)
    assert phase_zero.tolist() == [[True, False, False, False]]
    assert gate2.tolist() == [[False, True, False, False]]


def test_recurrent_admission_checks_both_partitions() -> None:
    good = {
        "phase_zero": {"weighted_mse": 0.029, "mse": [0.01] * 4},
        "gate2": {"weighted_mse": 0.019, "mse": [0.02] * 4},
    }
    assert numerical_admission(good)
    bad = {name: dict(value) for name, value in good.items()}
    bad["gate2"] = {"weighted_mse": 0.021, "mse": [0.02] * 4}
    assert not numerical_admission(bad)

