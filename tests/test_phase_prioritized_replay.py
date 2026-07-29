import pytest
import torch
from pathlib import Path

from pufferlib.torch_pufferl import _phase_priority_multipliers


def test_phase_priority_is_disabled_by_default():
    observations = torch.zeros((2, 4, 24))
    terminals = torch.zeros((2, 4))
    weights = _phase_priority_multipliers(observations, terminals, -1, 0.0, 4.0)
    torch.testing.assert_close(weights, torch.ones(2))


def test_phase_priority_uses_peak_observable_progress_and_caps_weight():
    observations = torch.zeros((3, 4, 24))
    observations[0, :, 23] = 1.0 / 6.0
    observations[1, :, 23] = 4.0 / 6.0
    observations[2, :, 23] = 1.0
    terminals = torch.zeros((3, 4))
    weights = _phase_priority_multipliers(observations, terminals, 23, 6.0, 4.0)
    torch.testing.assert_close(weights, torch.tensor([2.0, 4.0, 4.0]))


def test_phase_priority_ignores_new_episode_after_terminal_boundary():
    observations = torch.zeros((1, 4, 24))
    observations[0, 0, 23] = 2.0 / 6.0
    observations[0, 2:, 23] = 5.0 / 6.0
    terminals = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    weights = _phase_priority_multipliers(observations, terminals, 23, 3.0, 4.0)
    torch.testing.assert_close(weights, torch.tensor([2.0]))


def test_phase_priority_rejects_out_of_range_observation_index():
    with pytest.raises(ValueError, match="exceeds observation size"):
        _phase_priority_multipliers(
            torch.zeros((1, 4, 23)), torch.zeros((1, 4)), 23, 3.0, 4.0)


def test_cuda_encoder_column_mask_freezes_every_unselected_parameter():
    source = (Path(__file__).resolve().parents[1] / "src" / "pufferlib.cu").read_text()
    kernel_start = source.index("__global__ void mask_encoder_feature_gradients")
    kernel_end = source.index("void train_impl", kernel_start)
    kernel = source[kernel_start:kernel_end]
    assert "index < encoder_parameters" in kernel
    assert "feature >= first_feature && feature <= last_feature" in kernel
    assert "if (!selected) gradients[index] = from_float(0.0f);" in kernel

    backward = source.index("policy_backward(", source.index("void train_impl"))
    mask = source.index("mask_encoder_feature_gradients<<<", backward)
    update = source.index("muon_step(", mask)
    assert backward < mask < update
