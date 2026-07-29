import subprocess
import sys
from pathlib import Path

import pytest

import numpy as np

from scripts.audit_vq2_partition_reference_batching import (
    _event_max_abs,
    _fixed_batch_event_chunks,
    _parse_batch_sizes,
)


ROOT = Path(__file__).resolve().parents[1]


def test_parse_batch_sizes_is_ordered_and_fail_closed() -> None:
    assert _parse_batch_sizes("4,64,336") == (4, 64, 336)
    with pytest.raises(ValueError, match="unique positive"):
        _parse_batch_sizes("4,4")
    with pytest.raises(ValueError, match="unique positive"):
        _parse_batch_sizes("0,4")


def test_direct_cli_help_resolves_repository_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_vq2_partition_reference_batching.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--batch-sizes" in result.stdout
    assert "test-dataset" not in result.stdout


def test_fixed_batch_event_chunks_pad_only_the_final_batch() -> None:
    chunks = _fixed_batch_event_chunks(np.arange(5), 4)
    assert [(chunk.tolist(), valid) for chunk, valid in chunks] == [
        ([0, 1, 2, 3], 4),
        ([4, 0, 1, 2], 1),
    ]


def test_event_max_abs_flattens_only_feature_axes() -> None:
    error = np.asarray([[1.0, 3.0], [4.0, 2.0]])
    assert _event_max_abs(error).tolist() == [3.0, 4.0]
    error_3d = np.asarray([[[1.0, 5.0]], [[4.0, 2.0]]])
    assert _event_max_abs(error_3d).tolist() == [5.0, 4.0]
