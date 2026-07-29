from __future__ import annotations

import pytest
import torch

from pufferlib.vq2_informed import PRIVILEGED_NAMES, PRIVILEGED_SIZE
from scripts.eval_vq2_informed_dreamer import DecoderErrorAccumulator


def test_decoder_error_accumulator_reports_per_target_and_aggregate() -> None:
    accumulator = DecoderErrorAccumulator()
    target = torch.zeros(2, PRIVILEGED_SIZE)
    prediction = torch.zeros_like(target)
    prediction[0, 0] = 1.0
    prediction[1, 0] = -1.0
    accumulator.update(prediction, target)
    report = accumulator.report()
    first = PRIVILEGED_NAMES[0]
    assert report["decoder_samples"] == 2
    assert report[f"decoder_{first}_mean_absolute_error"] == pytest.approx(1.0)
    assert report[f"decoder_{first}_root_mean_square_error"] == pytest.approx(1.0)
    assert report["decoder_mean_absolute_error"] == pytest.approx(
        1.0 / PRIVILEGED_SIZE
    )
    assert report["decoder_root_mean_square_error"] == pytest.approx(
        (1.0 / PRIVILEGED_SIZE) ** 0.5
    )


def test_decoder_error_accumulator_rejects_wrong_schema() -> None:
    accumulator = DecoderErrorAccumulator()
    with pytest.raises(ValueError, match="privileged ABI"):
        accumulator.update(
            torch.zeros(1, PRIVILEGED_SIZE - 1),
            torch.zeros(1, PRIVILEGED_SIZE - 1),
        )
