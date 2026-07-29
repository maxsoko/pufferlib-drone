from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pufferlib.vq2_informed import MASK_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from scripts.train_vq2_public_phase_adapter import (
    PublicPhaseDataset,
    numerical_admission,
    reconstruct_phase_batch,
)


def test_reconstruct_phase_batch_keeps_public_value_last() -> None:
    mask = np.zeros((2, 3, MASK_SIZE), dtype=np.uint8)
    mask[0, 0, 0] = 255
    tail = np.zeros((2, 3, PHASE_LEGAL_OBS_SIZE - MASK_SIZE), dtype=np.float32)
    tail[1, 2, -1] = np.float32(1.0 / 6.0)
    observation = reconstruct_phase_batch(mask, tail, device=torch.device("cpu"))
    assert observation.shape == (3, 2, PHASE_LEGAL_OBS_SIZE)
    assert observation[0, 0, 0].item() == 1.0
    assert observation[2, 1, -1].item() == pytest.approx(1.0 / 6.0)


def test_public_phase_dataset_rejects_privileged_storage(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    report = {"admitted": True}
    metadata = {
        "observation": {
            "stored_legal_width": PHASE_LEGAL_OBS_SIZE,
            "legacy_legal_width": PHASE_LEGAL_OBS_SIZE - 1,
            "legal_tail_width": PHASE_LEGAL_OBS_SIZE - MASK_SIZE,
            "public_status_values_per_record": 1,
            "stored_training_only_privileged_values_per_record": 1,
        },
        "files": {},
    }
    (root / "report.json").write_text(json.dumps(report))
    (root / "metadata.json").write_text(json.dumps(metadata))
    from scripts.collect_vq2_oracle_bc_dataset import sha256_path

    with pytest.raises(RuntimeError, match="storage boundary"):
        PublicPhaseDataset(
            root,
            verify_hashes=False,
            expected_report_sha256=sha256_path(root / "report.json"),
            expected_metadata_sha256=sha256_path(root / "metadata.json"),
        )


def test_numerical_admission_requires_all_channels_and_exact_phase_zero() -> None:
    passing = {"weighted_mse": 0.019, "mse": [0.01, 0.02, 0.03, 0.04]}
    assert numerical_admission(passing, True)
    assert not numerical_admission(passing, False)
    assert not numerical_admission(
        {"weighted_mse": 0.019, "mse": [0.01, 0.051, 0.03, 0.04]}, True
    )

