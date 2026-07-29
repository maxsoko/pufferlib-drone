import numpy as np

from scripts.convert_policy_checkpoint_layout import (
    convert_layout,
    pack_layout,
    tensor_counts,
    unpack_layout,
)


def test_repack_bf16_checkpoint_for_fp32_native_arena():
    dimensions = {
        "input_dim": 1,
        "hidden_dim": 8,
        "num_layers": 1,
        "num_actions": 2,
    }
    counts = tensor_counts(**dimensions)
    tensors = [
        np.arange(count, dtype=np.float32) + np.float32(1000 * index)
        for index, count in enumerate(counts)
    ]
    # Native serialization truncates the aligned tail. Model the values that
    # can actually survive a BF16 save/load before comparing layouts.
    bf16_serialized = pack_layout(tensors, counts, precision_bytes=2)
    bf16_tensors = unpack_layout(bf16_serialized, counts, precision_bytes=2)

    fp32_serialized = convert_layout(
        bf16_serialized,
        source_precision_bytes=2,
        target_precision_bytes=4,
        **dimensions,
    )
    fp32_tensors = unpack_layout(fp32_serialized, counts, precision_bytes=4)

    for expected, actual in zip(bf16_tensors, fp32_tensors, strict=True):
        assert np.array_equal(actual, expected)
    assert fp32_serialized.size == sum(counts)


def test_conversion_rejects_non_native_precision_width():
    serialized = np.zeros(226, dtype=np.float32)
    try:
        convert_layout(
            serialized,
            source_precision_bytes=8,
            target_precision_bytes=4,
            input_dim=1,
            hidden_dim=8,
            num_layers=1,
            num_actions=2,
        )
    except ValueError as exc:
        assert "source_precision_bytes" in str(exc)
    else:
        raise AssertionError("invalid precision width was accepted")
