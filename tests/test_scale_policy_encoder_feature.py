from pathlib import Path

import numpy as np
import pytest

from scripts.convert_policy_checkpoint_layout import pack_layout, tensor_counts
from scripts.policy_callable_checkpoint import CheckpointPolicy
from scripts.scale_policy_encoder_feature import scale_encoder_feature


def _checkpoint(
    path: Path, *, input_dim: int = 4, hidden_dim: int = 8
) -> np.ndarray:
    counts = tensor_counts(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        num_actions=4,
    )
    tensors = []
    offset = 0
    for count in counts:
        tensors.append(
            np.arange(offset, offset + count, dtype=np.float32) / np.float32(100.0)
        )
        offset += count
    serialized = pack_layout(tensors, counts, precision_bytes=4)
    serialized.tofile(path)
    return serialized


def test_scale_changes_only_selected_fp32_encoder_column(tmp_path):
    source_path = tmp_path / "source.bin"
    output_path = tmp_path / "output.bin"
    original = _checkpoint(source_path)
    source_encoder = original[:32].reshape(8, 4).copy()

    before_norm, after_norm = scale_encoder_feature(
        source_path,
        output_path,
        feature_index=2,
        scale=1.5,
        input_dim=4,
        hidden_dim=8,
        layout_precision_bytes=4,
    )

    scaled = np.fromfile(output_path, dtype=np.float32)
    encoder = scaled[:32].reshape(8, 4)
    np.testing.assert_array_equal(encoder[:, 0], source_encoder[:, 0])
    np.testing.assert_array_equal(encoder[:, 1], source_encoder[:, 1])
    np.testing.assert_allclose(encoder[:, 2], 1.5 * source_encoder[:, 2])
    np.testing.assert_array_equal(encoder[:, 3], source_encoder[:, 3])
    np.testing.assert_array_equal(scaled[32:], original[32:])
    assert after_norm == pytest.approx(1.5 * before_norm)
    CheckpointPolicy.load(
        str(output_path),
        input_dim=4,
        hidden_dim=8,
        layout_precision_bytes=4,
    )


@pytest.mark.parametrize("feature_index", (-1, 4))
def test_scale_rejects_feature_outside_encoder(tmp_path, feature_index):
    source_path = tmp_path / "source.bin"
    _checkpoint(source_path)
    with pytest.raises(ValueError, match="outside"):
        scale_encoder_feature(
            source_path,
            tmp_path / "output.bin",
            feature_index=feature_index,
            scale=1.5,
            input_dim=4,
            hidden_dim=8,
            layout_precision_bytes=4,
        )
