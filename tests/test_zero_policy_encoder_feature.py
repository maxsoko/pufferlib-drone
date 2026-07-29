from pathlib import Path

import numpy as np

from scripts.convert_policy_checkpoint_layout import pack_layout, tensor_counts
from scripts.policy_callable_checkpoint import CheckpointPolicy
from scripts.zero_policy_encoder_feature import zero_encoder_features


def _checkpoint(path: Path, *, input_dim: int = 6, hidden_dim: int = 8) -> np.ndarray:
    counts = tensor_counts(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        num_actions=4,
    )
    tensors = []
    offset = 1
    for count in counts:
        tensors.append(
            np.arange(offset, offset + count, dtype=np.float32) / np.float32(100.0)
        )
        offset += count
    serialized = pack_layout(tensors, counts, precision_bytes=4)
    serialized.tofile(path)
    return serialized


def test_zero_changes_only_selected_fp32_encoder_columns(tmp_path):
    source_path = tmp_path / "source.bin"
    output_path = tmp_path / "output.bin"
    original = _checkpoint(source_path)
    source_encoder = original[:48].reshape(8, 6).copy()

    before, after = zero_encoder_features(
        source_path,
        output_path,
        feature_indices=[5, 2, 5],
        input_dim=6,
        hidden_dim=8,
        layout_precision_bytes=4,
    )

    zeroed = np.fromfile(output_path, dtype=np.float32)
    encoder = zeroed[:48].reshape(8, 6)
    np.testing.assert_array_equal(encoder[:, 0], source_encoder[:, 0])
    np.testing.assert_array_equal(encoder[:, 1], source_encoder[:, 1])
    np.testing.assert_array_equal(encoder[:, 3], source_encoder[:, 3])
    np.testing.assert_array_equal(encoder[:, 4], source_encoder[:, 4])
    np.testing.assert_array_equal(encoder[:, 2], np.zeros(8, dtype=np.float32))
    np.testing.assert_array_equal(encoder[:, 5], np.zeros(8, dtype=np.float32))
    np.testing.assert_array_equal(zeroed[48:], original[48:])
    assert before[2] > 0.0 and before[5] > 0.0
    assert after == {2: 0.0, 5: 0.0}
    CheckpointPolicy.load(
        str(output_path),
        input_dim=6,
        hidden_dim=8,
        layout_precision_bytes=4,
    )
