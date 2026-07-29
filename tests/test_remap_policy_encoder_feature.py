from pathlib import Path

import numpy as np

from scripts.policy_callable_checkpoint import CheckpointPolicy, _align8
from scripts.remap_policy_encoder_feature import remap


def _checkpoint(path: Path, *, input_dim: int = 4, hidden_dim: int = 8) -> np.ndarray:
    counts = (
        hidden_dim * input_dim,
        5 * hidden_dim,
        4,
        *([3 * hidden_dim * hidden_dim] * 3),
    )
    total = 0
    for count in counts:
        total = _align8(total + count)
    values = np.arange(total, dtype=np.float32) / np.float32(100.0)
    values.tofile(path)
    return values


def test_remap_changes_only_source_and_target_encoder_columns(tmp_path):
    source_path = tmp_path / "source.bin"
    output_path = tmp_path / "output.bin"
    original = _checkpoint(source_path)
    source_encoder = original[:32].reshape(8, 4).copy()

    remap(
        source_path,
        output_path,
        source_index=3,
        target_index=1,
        transfer_scale=1.5,
        source_retain_scale=0.0,
        input_dim=4,
        hidden_dim=8,
        num_actions=4,
    )

    migrated = np.fromfile(output_path, dtype=np.float32)
    encoder = migrated[:32].reshape(8, 4)
    np.testing.assert_array_equal(encoder[:, 0], source_encoder[:, 0])
    np.testing.assert_array_equal(encoder[:, 2], source_encoder[:, 2])
    np.testing.assert_allclose(
        encoder[:, 1], source_encoder[:, 1] + 1.5 * source_encoder[:, 3]
    )
    np.testing.assert_array_equal(encoder[:, 3], 0.0)
    np.testing.assert_array_equal(migrated[32:], original[32:])
    CheckpointPolicy.load(str(output_path), input_dim=4, hidden_dim=8)
