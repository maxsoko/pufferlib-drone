from __future__ import annotations

import pytest
import torch

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE
from scripts.eval_vq2_visual_counterfactual import (
    DeltaAccumulator,
    visual_variant,
)


def test_blank_and_flip_preserve_nonvisual_observation() -> None:
    legal = torch.arange(2 * LEGAL_OBS_SIZE, dtype=torch.float32).reshape(
        2, LEGAL_OBS_SIZE
    )
    blank = visual_variant(legal, "blank")
    flipped = visual_variant(legal, "horizontal_flip")

    assert torch.count_nonzero(blank[:, :MASK_SIZE]) == 0
    assert torch.equal(blank[:, MASK_SIZE:], legal[:, MASK_SIZE:])
    assert torch.equal(flipped[:, MASK_SIZE:], legal[:, MASK_SIZE:])
    image = legal[:, :MASK_SIZE].reshape(2, 64, 64)
    assert torch.equal(flipped[:, :MASK_SIZE].reshape(2, 64, 64), image.flip(-1))
    assert torch.equal(legal[:, MASK_SIZE:], blank[:, MASK_SIZE:])


def test_visual_variant_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="expected legal observation width"):
        visual_variant(torch.zeros(2, LEGAL_OBS_SIZE - 1), "blank")
    with pytest.raises(ValueError, match="unsupported visual intervention"):
        visual_variant(torch.zeros(2, LEGAL_OBS_SIZE), "rotate")


def test_delta_accumulator_reports_exact_threshold_and_channels() -> None:
    normal = torch.zeros(2, 4)
    changed = torch.tensor(
        [[0.0, 0.5, 0.0, 0.0], [1e-7, 0.0, -0.25, 0.0]]
    )
    accumulator = DeltaAccumulator(threshold=1e-6)
    accumulator.update(normal, changed)
    report = accumulator.report()

    assert report["samples"] == 2
    assert report["exact_changed_samples"] == 2
    assert report["threshold_changed_samples"] == 2
    assert report["mean_absolute_action_delta"] == pytest.approx(
        (0.5 + 0.25 + 1e-7) / 8
    )
    assert report["max_absolute_action_delta"] == pytest.approx(0.5)
    assert report["channel_1_mean_absolute_delta"] == pytest.approx(0.25)
    assert report["channel_2_max_absolute_delta"] == pytest.approx(0.25)
