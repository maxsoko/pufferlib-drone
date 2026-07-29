import numpy as np

from scripts.convert_policy_checkpoint_layout import (
    pack_layout,
    tensor_counts,
    unpack_layout,
)
from scripts.resize_policy_checkpoint_input import expand_input


def test_expand_input_preserves_old_encoder_and_zeros_new_columns():
    dimensions = {"hidden_dim": 8, "num_layers": 1, "num_actions": 2}
    source_counts = tensor_counts(input_dim=3, **dimensions)
    source_tensors = [
        np.arange(count, dtype=np.float32) + 1000 * index
        for index, count in enumerate(source_counts)
    ]
    source = pack_layout(source_tensors, source_counts, precision_bytes=2)

    expanded = expand_input(
        source,
        source_input_dim=3,
        target_input_dim=5,
        source_precision_bytes=2,
        target_precision_bytes=4,
        **dimensions,
    )
    target_counts = tensor_counts(input_dim=5, **dimensions)
    actual = unpack_layout(expanded, target_counts, precision_bytes=4)
    expected_source = unpack_layout(source, source_counts, precision_bytes=2)

    encoder = actual[0].reshape(8, 5)
    np.testing.assert_array_equal(
        encoder[:, :3], expected_source[0].reshape(8, 3)
    )
    np.testing.assert_array_equal(encoder[:, 3:], 0.0)
    for expected, result in zip(expected_source[1:], actual[1:], strict=True):
        np.testing.assert_array_equal(result, expected)


def test_expand_input_rejects_shrinking():
    with np.testing.assert_raises_regex(ValueError, "target_input_dim"):
        expand_input(
            np.zeros(1, dtype=np.float32),
            source_input_dim=3,
            target_input_dim=2,
            hidden_dim=8,
            num_layers=1,
            num_actions=2,
            source_precision_bytes=2,
            target_precision_bytes=4,
        )
